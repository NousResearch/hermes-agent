"""/undo with checkpoints on names the /rollback that restores the undone turns' files; off, it prints main's output."""

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


def _run_two_writing_turns(cli, db, project, *, enabled):
    """An earlier turn (kept), then two turns that each took a checkpoint before writing a file."""
    mgr = CheckpointManager(enabled=enabled)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    start = time.time() - 60
    db.append_message(cli.session_id, "user", "read only", timestamp=start)
    db.append_message(cli.session_id, "assistant", "done", timestamp=start + 1)
    for n, name in ((1, "app.js"), (2, "notes.md")):
        db.append_message(cli.session_id, "user", f"edit {n}", timestamp=start + 10 * n)
        mgr.new_turn()
        mgr.ensure_checkpoint(str(project), f"before write {n}")
        (project / name).write_text(f"turn {n}\n")
        db.append_message(cli.session_id, "assistant", f"edited {n}", timestamp=start + 10 * n + 1)
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    return mgr


def test_undo_names_the_rollback_number_of_the_oldest_undone_checkpoint(setup, capsys):
    cli, db, project = setup
    mgr = _run_two_writing_turns(cli, db, project, enabled=True)

    cli._cmd_undo("/undo 2")

    out = capsys.readouterr().out
    assert "To restore them: /rollback 2" in out
    assert "app.js, notes.md" in out
    # Pasting it restores the snapshot taken before the first undone turn wrote.
    target = cli._resolve_checkpoint_ref("2", mgr.list_checkpoints(str(project)))
    assert mgr.restore(str(project), target, safe=False)["success"]
    assert (project / "app.js").read_text() == "hello\n"


def test_undo_with_checkpoints_off_prints_no_rollback_line(setup, capsys):
    cli, db, project = setup
    _run_two_writing_turns(cli, db, project, enabled=False)

    cli._cmd_undo("/undo 2")

    out = capsys.readouterr().out
    assert "Undid 2 turns" in out
    assert "/rollback" not in out


def test_undo_gives_no_number_once_pruning_dropped_an_undone_checkpoint(setup, capsys):
    """The size cap drops the oldest snapshot even below max_snapshots: the span's first checkpoint is gone,
    so the newest one would restore only part of the undone edits. No number then."""
    from tools import checkpoint_manager as cm
    from tools.checkpoint_pruning import Pruner

    cli, db, project = setup
    _run_two_writing_turns(cli, db, project, enabled=True)
    pruner = Pruner(cm._run_git, cm._store_path(), str(project), cm._GIT_TIMEOUT, lambda _p: 2, cm._REFS_PREFIX)
    assert pruner.drop_one_round(1)

    cli._cmd_undo("/undo 2")

    out = capsys.readouterr().out
    assert "/rollback" in out and "notes.md" in out
    assert "/rollback 1" not in out and "/rollback 2" not in out
