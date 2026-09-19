"""``hermes send --to tui:<session_key>`` — the out-of-process way into a Desktop/TUI conversation.

No messaging adapter reaches a live Desktop/TUI session, so a script or cron job that has to notify
the conversation a delegation started in has nowhere to send. The durable per-session mailbox
(``tools/bot_live_delivery``) is that place; the session's own poller claims it.
"""
import pytest

from hermes_cli import send_cmd


def _parse(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    send_cmd.register_send_subparser(subparsers)
    return parser.parse_args(["send", *argv])


def _live_home(tmp_path, monkeypatch, session_key="desk-1", live_session_id="tab-7"):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id=session_key, source="desktop")
    lease, refusal = try_acquire_active_session(
        session_id=session_key, surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": live_session_id, "bot_live_delivery_consumer": True})
    assert refusal is None and lease is not None
    return db, lease


def test_send_queues_into_the_live_session_mailbox(tmp_path, monkeypatch, capsys):
    from tools.bot_live_delivery import read_delivery_result

    db, lease = _live_home(tmp_path, monkeypatch)
    try:
        with pytest.raises(SystemExit) as exited:
            send_cmd.cmd_send(_parse(["--to", "tui:desk-1", "HERM-634: run started"]))
        assert exited.value.code == 0, capsys.readouterr()
        assert "queued" in capsys.readouterr().out
        tickets = sorted((tmp_path / "runtime" / "bot_live_delivery").glob("*.json"))
        assert len(tickets) == 1
        record = read_delivery_result(tmp_path, tickets[0].stem)
        assert record is not None
        assert record["message"] == "HERM-634: run started"
        assert record["status"] == "queued"
        # Pinned to the live owner: the lease is what makes the session's poller the only consumer.
        assert (record["session_id"], record["lease_id"], record["live_session_id"]) == (
            "desk-1", lease.lease_id, "tab-7")
    finally:
        lease.release()
        db.close()


def test_send_without_a_live_owner_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exited:
        send_cmd.cmd_send(_parse(["--to", "tui:ghost", "hello"]))
    assert exited.value.code == 1
    assert "no live" in capsys.readouterr().err.lower()
    assert not (tmp_path / "runtime").exists()
