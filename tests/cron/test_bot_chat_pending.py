"""Only never-started cron delivery may wait for a CLI owner's release."""
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB


@pytest.mark.parametrize("error", [None, subprocess.TimeoutExpired("hermes", 1)])
def test_cli_owner_deferral_and_attempt_fence(tmp_path, monkeypatch, error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None and lease is not None
    run = Mock(side_effect=error, return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(delivery.subprocess, "run", run)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/bin/hermes")
    job = {"id": "job", "execution_id": "execution"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        queue.drain()
        run.assert_not_called()
        lease.release()
        queue.drain()
        assert run.call_count == 1
        expected = "ambiguous" if error else "settled"
        assert queue.read_pending(key)["status"] == expected
        queue.drain()
        delivery._deliver_to_bot_chat(job, "output", "")
        assert run.call_count == 1
    finally:
        lease.release()
        db.close()


def test_delivery_exception_retains_attempt_and_continues_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    blocked_parent = tmp_path / "blocked"
    blocked_home = blocked_parent / "recipient"
    blocked_home.mkdir(parents=True)
    for home in (blocked_home, tmp_path):
        db = SessionDB(db_path=home / "state.db")
        db.create_session(session_id="chat", source="cli")
        db.set_session_title("chat", "Bot Chat")
        db.close()
    queue.defer("b" * 64, {"id": "bad"}, "bad output", "", blocked_home)
    queue.defer("a" * 64, {"id": "good"}, "good output", "", tmp_path)
    calls = []
    original_is_dir = Path.is_dir
    armed = False

    def resolve_cli(_):
        nonlocal armed
        armed = True
        return "/bin/hermes"

    def is_dir(self):
        if armed and self == blocked_home:
            raise PermissionError("target traversal denied after discovery")
        return original_is_dir(self)

    def run(*args, **kwargs):
        calls.append(kwargs["env"]["HERMES_HOME"])
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(delivery.shutil, "which", resolve_cli)
    monkeypatch.setattr(delivery.subprocess, "run", run)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    queue.drain()
    assert queue.read_pending("b" * 64)["status"] == "ambiguous"
    assert "PermissionError" in queue.read_pending("b" * 64)["error"]
    assert queue.read_pending("a" * 64)["status"] == "settled"
    queue.drain()
    assert calls == [str(tmp_path)]


def test_pending_queue_uses_admission_order_and_keeps_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "job"}
    queue.defer("f" * 64, job, "older", "", tmp_path)
    queue.defer("a" * 64, job, "newer", "", tmp_path)
    seen = []

    def interrupted(job, content, profile, **kwargs):
        seen.append(content)
        raise KeyboardInterrupt

    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        queue.drain()
    assert seen == ["older"]
    assert queue.read_pending("f" * 64)["status"] == "claimed"
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    queue.drain()
    queue.drain()
    assert seen == ["older", "newer"]
