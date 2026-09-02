import json
import subprocess
from types import SimpleNamespace


def test_handoff_task_requires_callback_without_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    from tools.handoff_tools import _handle_handoff

    result = json.loads(
        _handle_handoff(
            {
                "mode": "handoff_task",
                "target_profile": "default",
                "prompt": "Do the thing.",
            }
        )
    )

    assert result["success"] is False
    assert "requires origin_delivery" in result["error"]


def test_handoff_task_creates_task_and_notify_subscription(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    from hermes_cli import kanban_db as kb
    from tools.handoff_tools import _handle_handoff

    result = json.loads(
        _handle_handoff(
            {
                "mode": "handoff_task",
                "target_profile": "default",
                "prompt": "Run durable work and report back.",
                "origin_delivery": {
                    "platform": "tui",
                    "chat_id": "session-key-1",
                    "delivery_mode": "notify",
                    "metadata": {"origin": "test"},
                },
                "idempotency_key": "test-handoff-task",
            },
            session_id="origin-session-1",
        )
    )

    assert result["success"] is True
    assert result["mode"] == "handoff_task"
    assert result["callback_registered"] is True
    assert result["origin_session_id"] == "origin-session-1"

    conn = kb.connect()
    try:
        task = kb.get_task(conn, result["task_id"])
        assert task is not None
        assert task.assignee == "default"
        assert task.session_id == "origin-session-1"
        assert "callback_required" in (task.body or "")
        subs = kb.list_notify_subs(conn, result["task_id"])
        assert len(subs) == 1
        assert subs[0]["platform"] == "tui"
        assert subs[0]["chat_id"] == "session-key-1"
        assert subs[0]["delivery_mode"] == "notify"
        metadata = subs[0]["delivery_metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata["origin"] == "test"
    finally:
        conn.close()


def test_handoff_profile_runs_target_profile_session(monkeypatch):
    from tools import handoff_tools

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="target final", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        handoff_tools,
        "_latest_handoff_profile_session",
        lambda target_profile, *, started_after: "session123",
    )
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "source-task")

    result = json.loads(
        handoff_tools._handle_handoff(
            {
                "mode": "handoff_profile",
                "target_profile": "default",
                "prompt": "Open immediate session.",
                "max_runtime_seconds": 17,
                "toolsets": "file,terminal",
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "final"
    assert result["final_result"] == "target final"
    assert result["session_id"] == "session123"
    assert result["session_link"] == "@session:default/session123"
    assert captured["timeout"] == 17
    assert captured["cmd"][:4] == ["hermes", "--profile", "default", "chat"]
    assert "--source" in captured["cmd"]
    assert "handoff_profile" in captured["cmd"]
    assert "--toolsets" in captured["cmd"]
    assert "HERMES_SESSION_PLATFORM" not in captured["env"]
    assert "HERMES_KANBAN_TASK" not in captured["env"]


def test_handoff_profile_returns_blocked_result_on_target_failure(monkeypatch):
    from tools import handoff_tools

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="partial output",
            stderr="missing capability",
        ),
    )
    monkeypatch.setattr(
        handoff_tools,
        "_latest_handoff_profile_session",
        lambda target_profile, *, started_after: None,
    )

    result = json.loads(
        handoff_tools._handle_handoff(
            {
                "mode": "handoff_profile",
                "target_profile": "default",
                "prompt": "Open immediate session.",
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "missing capability"
