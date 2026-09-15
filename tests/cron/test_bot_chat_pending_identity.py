"""A deferred delivery keeps its original destination and admission identity."""
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tools.bot_live_delivery import read_delivery_result


@pytest.mark.parametrize("recipient", ["cli", "desktop", "renamed"])
def test_deferred_destination_does_not_follow_root_changes(tmp_path, monkeypatch, recipient):
    source = tmp_path / "source"
    home = tmp_path / "original" / "profiles" / "beta"
    other = tmp_path / "other" / "profiles" / "beta"
    home.mkdir(parents=True)
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _: home)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="cli", config={}, registry_home=home)
    assert refusal is None
    job = {"id": "job", "execution_id": "execution"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "beta")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:beta"]["delivery_id"]
    finally:
        lease.release()
        db.close()
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _: other)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(delivery.subprocess, "run", run)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/bin/hermes")
    if recipient == "desktop":
        lease, refusal = try_acquire_active_session(
            session_id="chat", surface="desktop", config={}, registry_home=home,
            metadata={"bot_live_delivery_consumer": True, "live_session_id": "live"})
        assert refusal is None
    elif recipient == "renamed":
        home.rename(home.with_name("renamed"))
    try:
        with monkeypatch.context() as changed:
            changed.setenv("HERMES_HOME", str(tmp_path / "new-source"))
            queue.drain(source / "cron" / "bot_chat_pending")
            queue.drain(source / "cron" / "bot_chat_pending")
        if recipient == "cli":
            assert run.call_count == 1
            argv = run.call_args.args[0]
            assert "-p" not in argv
            assert Path(run.call_args.kwargs["env"]["HERMES_HOME"]) == home
        elif recipient == "desktop":
            run.assert_not_called()
            receipt = read_delivery_result(home, key)
            assert receipt is not None and receipt["status"] == "queued"
            assert queue.read_pending(key)["status"] == "transferred"
        else:
            run.assert_not_called()
            assert not home.exists()
            assert queue.read_pending(key)["status"] == "ambiguous"
        assert read_delivery_result(other, key) is None
    finally:
        lease.release()


def test_corrupt_record_is_retained_without_blocking_other_admissions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    queue.defer("a" * 64, {"id": "job"}, "first", "", tmp_path)
    broken = tmp_path / "cron" / "bot_chat_pending" / "broken.json"
    broken.write_text("{", encoding="utf-8")
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    queue.drain()
    queue.defer("b" * 64, {"id": "next"}, "second", "", tmp_path)
    queue.drain()
    assert seen == ["first", "second"]
    assert broken.read_text(encoding="utf-8") == "{"
