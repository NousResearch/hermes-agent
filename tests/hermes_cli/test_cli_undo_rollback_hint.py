"""/undo with checkpoints on names the /rollback that restores the undone turns' files; off, it prints main's output."""

import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from tests.hermes_cli.test_cli_init import _make_cli
from tools.checkpoint_manager import CheckpointManager

_PRINTED = re.compile(r"To restore them: (/rollback [0-9a-f]{8,40})\b")


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


def _turn(cli, db, mgr, project, text, write=None):
    """One chat turn: the user row, the checkpoint attempt before the turn's first write, the write."""
    db.append_message(cli.session_id, "user", text, timestamp=time.time())
    mgr.new_turn()
    mgr.ensure_checkpoint(str(project), f"before {text}")
    if write:
        (project / write[0]).write_text(write[1])
    db.append_message(cli.session_id, "assistant", f"did {text}", timestamp=time.time())


def _session(cli, db, project, *, enabled):
    """A kept turn whose checkpoint saw the original tree but wrote nothing, then two undone turns: the
    first one's checkpoint attempt finds nothing changed and REUSES that snapshot, the second takes a new one."""
    mgr = CheckpointManager(enabled=enabled)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    _turn(cli, db, mgr, project, "kept: run the tests")
    _turn(cli, db, mgr, project, "edit 1", ("app.js", "turn 1\n"))
    _turn(cli, db, mgr, project, "edit 2", ("notes.md", "turn 2\n"))
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    return mgr


def test_undo_hint_round_trip_restores_the_first_undone_turns_files_and_keeps_the_kept_turn(setup, capsys):
    """The printed command, run as printed through the real /rollback handler, brings the files back to what
    the FIRST undone turn found (it reused the kept turn's snapshot), even after another checkpoint was taken
    in between, and leaves the kept turn in the transcript."""
    cli, db, project = setup
    mgr = _session(cli, db, project, enabled=True)
    assert len(mgr.list_checkpoints(str(project))) == 2  # the reused attempt made no commit

    cli._cmd_undo("/undo 2")
    out = capsys.readouterr().out
    command = _PRINTED.search(out)
    assert command, out
    assert "app.js, notes.md" in out
    (project / "scratch.txt").write_text("later\n")  # the next checkpoint shifts every list position
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(project), "a later turn")

    cli._handle_rollback_command(command.group(1))

    assert (project / "app.js").read_text() == "hello\n"
    kept = ["kept: run the tests", "did kept: run the tests"]
    assert [m["content"] for m in cli.conversation_history] == kept
    assert [m["content"] for m in db.get_messages_as_conversation(cli.session_id)] == kept


def test_undo_with_checkpoints_off_prints_no_rollback_line(setup, capsys):
    cli, db, project = setup
    _session(cli, db, project, enabled=False)

    cli._cmd_undo("/undo 2")

    out = capsys.readouterr().out
    assert "Undid 2 turns" in out
    assert "/rollback" not in out


def test_undo_gives_no_command_once_the_undone_spans_snapshot_was_pruned(setup, capsys):
    """The count cap dropped the snapshot the first undone turn found (with the project's metadata unwritable,
    so nothing could note the drop): the newest one would restore only part of the undone edits."""
    from tools import checkpoint_manager as cm
    from tools.checkpoint_pruning import Pruner

    cli, db, project = setup
    _session(cli, db, project, enabled=True)
    meta = cm._project_meta_path(cm._store_path(), cm._project_hash(str(project)))
    meta.chmod(0o444)
    try:
        Pruner(cm._run_git, cm._store_path(), str(project), cm._GIT_TIMEOUT, cm._dir_size_bytes,
               cm._REFS_PREFIX).trim(cm._ref_name(cm._project_hash(str(project))), 1)
        cli._cmd_undo("/undo 2")
    finally:
        meta.chmod(0o644)

    out = capsys.readouterr().out
    assert "Run /rollback to list checkpoints" in out and "notes.md" in out, out
    assert "To restore them" not in out, out
