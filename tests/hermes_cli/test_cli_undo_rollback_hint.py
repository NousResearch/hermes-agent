"""/undo with checkpoints on names the files the undone turns changed and points at /rollback (no command to paste);
/rollback then lists the checkpoints and restoring the right one works, keeping the turn /undo kept."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from tests.hermes_cli.test_cli_init import _make_cli
from tools.checkpoint_manager import CheckpointManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    (project / ".git").mkdir(parents=True)
    (project / "app.js").write_text("hello\n")
    cli = _make_cli()
    monkeypatch.setenv("TERMINAL_CWD", str(project))  # after _make_cli: loading the CLI config sets it
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db, cli.session_id = db, "20261105_010000_bbbbbb"
    db.create_session(cli.session_id, source="cli")
    cli._prefill_input_buffer = MagicMock()
    cli._confirm_destructive_slash = MagicMock(return_value=True)
    yield cli, db, project
    db.close()


def _next_second():
    """Git author stamps are whole seconds: start the next turn in a later one."""
    start = int(time.time())
    while int(time.time()) == start:
        time.sleep(0.02)  # health: allow HX013 -- whole-second git stamps: the next second IS the case


def _session(cli, db, project, *, enabled):
    """A kept turn that took a checkpoint but wrote nothing, then two undone turns that write: the first one's
    attempt finds nothing changed and reuses the kept turn's snapshot, the second takes a new one."""
    mgr = CheckpointManager(enabled=enabled)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    for text, write in (("kept: run the tests", None), ("edit 1", ("app.js", "turn 1\n")),
                        ("edit 2", ("notes.md", "turn 2\n"))):
        _next_second()
        db.append_message(cli.session_id, "user", text, timestamp=time.time())
        mgr.new_turn()
        mgr.ensure_checkpoint(str(project), f"before {text}")
        if write:
            (project / write[0]).write_text(write[1])
        db.append_message(cli.session_id, "assistant", f"did {text}", timestamp=time.time())
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    return mgr


def test_undo_lists_the_changed_files_then_rollback_restores_and_keeps_the_kept_turn(setup, capsys):
    cli, db, project = setup
    mgr = _session(cli, db, project, enabled=True)

    cli._cmd_undo("/undo 2")
    out = capsys.readouterr().out
    assert "Files changed in the undone turns: app.js, notes.md. Run /rollback to see checkpoints" in out, out
    assert "/rollback " not in out.replace("Run /rollback to", ""), out  # no hash, no number

    cli._handle_rollback_command("/rollback")
    listed = capsys.readouterr().out
    checkpoints = mgr.list_checkpoints(str(project))
    oldest = checkpoints[-1]["short_hash"]  # the one edit 1 found (it reused it)
    assert len(checkpoints) == 2 and oldest in listed, listed

    cli._handle_rollback_command(f"/rollback {oldest}")  # by the hash /rollback listed

    assert (project / "app.js").read_text() == "hello\n"
    kept = ["kept: run the tests", "did kept: run the tests"]
    assert [m["content"] for m in cli.conversation_history] == kept
    assert [m["content"] for m in db.get_messages_as_conversation(cli.session_id)] == kept

