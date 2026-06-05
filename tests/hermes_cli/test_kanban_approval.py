import json
import sys
from pathlib import Path

from hermes_cli import kanban_approval as ka


def test_create_request_no_send_writes_request_and_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = ka.create_request(
        task_id="t_abc123",
        title="Approve test gate",
        message="Only approve the harmless test gate.",
        send_telegram=False,
    )

    req_path = Path(result["path"])
    assert req_path.exists()
    data = json.loads(req_path.read_text())
    assert data["status"] == "pending"
    assert data["approval_surface"] == "telegram-native-callback"
    assert data["kanban_task_id"] == "t_abc123"
    assert data["command"][:3] == ["hermes", "kanban", "unblock"]

    cb_path = tmp_path / "approval-bridge" / "telegram-callbacks" / f"{result['callback_id']}.json"
    assert json.loads(cb_path.read_text())["request_id"] == result["id"]


def test_create_request_dedupes_pending_same_task_title(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = ka.create_request(
        task_id="t_abc123",
        title="Same gate",
        message="first",
        send_telegram=False,
    )
    second = ka.create_request(
        task_id="t_abc123",
        title="Same gate",
        message="second",
        send_telegram=False,
    )

    assert second["id"] == first["id"]
    assert second["deduped"] is True
    assert len(list((tmp_path / "approval-bridge" / "requests").glob("*.json"))) == 1


def test_defer_is_idempotent_and_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    marker = tmp_path / "marker"
    result = ka.create_request(
        task_id="t_abc123",
        title="Defer gate",
        message="do not run",
        command=[sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"],
        send_telegram=False,
    )

    first = ka.resolve_callback(result["callback_id"], "d", actor="test-user")
    second = ka.resolve_callback(result["callback_id"], "a", actor="test-user")

    assert first.status == "deferred"
    assert first.executed is False
    assert second.already_resolved is True
    assert second.status == "deferred"
    assert not marker.exists()


def test_approve_executes_once_and_replay_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    marker = tmp_path / "marker"
    result = ka.create_request(
        task_id="t_abc123",
        title="Approve gate",
        message="run once",
        command=[sys.executable, "-c", f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text((p.read_text() if p.exists() else '') + 'x')"],
        send_telegram=False,
    )

    first = ka.resolve_callback(result["callback_id"], "a", actor="test-user")
    second = ka.resolve_callback(result["callback_id"], "a", actor="test-user")

    assert first.status == "completed"
    assert first.executed is True
    assert first.exit_code == 0
    assert second.already_resolved is True
    assert second.status == "completed"
    assert marker.read_text() == "x"
