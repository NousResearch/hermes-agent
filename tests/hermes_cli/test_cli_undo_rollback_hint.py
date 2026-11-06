"""/undo with checkpoints on names the /rollback that restores the undone turns' files; off, it prints main's output."""

import re
import threading
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


def _turn(cli, db, mgr, project, text, write=None, *, take=True, busy=False):
    """One chat turn: the user row, the checkpoint attempt before the turn's first write (bound to the turn by its
    row's stamp, as ``agent/turn_context.py`` binds it), the write. ``take=False``: the write went through a tool
    that takes no checkpoint (a non-destructive terminal command, a script). ``busy``: another process holds the
    store lock during the attempt."""
    stamp = time.time()
    db.append_message(cli.session_id, "user", text, timestamp=stamp)
    mgr.new_turn()
    mgr.turn_key = repr(stamp)
    if busy:
        with _store_held():
            assert not mgr.ensure_checkpoint(str(project), f"before {text}")
    elif take:
        mgr.ensure_checkpoint(str(project), f"before {text}")
    if write:
        (project / write[0]).write_text(write[1])
    db.append_message(cli.session_id, "assistant", f"did {text}", timestamp=time.time())


class _store_held:
    """The checkpoint store lock held by another holder (a take or prune elsewhere) for the block's duration."""

    def __enter__(self):
        from tools.checkpoint_manager import _resolve_checkpoint_base
        from tools.checkpoint_pruning import store_lock

        held, self.release = threading.Event(), threading.Event()

        def hold():
            with store_lock(_resolve_checkpoint_base()):
                held.set()
                self.release.wait()
        self.thread = threading.Thread(target=hold)
        self.thread.start()
        held.wait()

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join()


def _next_second():
    """Git author stamps are whole seconds: start the next turn in a later one."""
    start = int(time.time())
    while int(time.time()) == start:
        time.sleep(0.02)  # health: allow HX013 -- whole-second git stamps: the next second IS the case


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


def test_a_busy_first_undone_take_gives_no_command_that_restores_the_wrong_files(setup, capsys):
    """The first undone turn's checkpoint attempt meets a busy store (never waited for), then it edits app.js;
    the next turn takes a checkpoint that already holds that edit. A printed command must restore app.js."""
    cli, db, project = setup
    mgr = CheckpointManager(enabled=True)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    _turn(cli, db, mgr, project, "kept")
    _next_second()
    _turn(cli, db, mgr, project, "edit 1", ("app.js", "t1\n"), busy=True)
    _next_second()
    _turn(cli, db, mgr, project, "edit 2", ("b.txt", "t2\n"))
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)

    cli._cmd_undo("/undo 2")
    out = capsys.readouterr().out
    if command := _PRINTED.search(out):
        cli._handle_rollback_command(command.group(1))
    assert not command or (project / "app.js").read_text() == "hello\n", out
    assert "Run /rollback to list checkpoints" in out, out


def test_two_turns_sharing_a_stamp_give_no_hash_for_the_later_turns_take(setup, capsys):
    """Platform event times can be whole seconds: two undone turns share their user rows' stamp. The first edits
    with no take, the second takes one that already holds that edit; a printed command must restore app.js."""
    from tools.checkpoint_takes import bind_checkpoint_turn

    cli, db, project = setup
    mgr = CheckpointManager(enabled=True)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    _turn(cli, db, mgr, project, "kept")
    _next_second()
    stamp = float(int(time.time()))
    for text, path, take in (("edit 1", "app.js", False), ("edit 2", "b.txt", True)):
        db.append_message(cli.session_id, "user", text, timestamp=stamp)
        bind_checkpoint_turn(SimpleNamespace(_checkpoint_mgr=mgr, _persist_user_message_timestamp=stamp), {})
        mgr.new_turn()
        if take:
            mgr.ensure_checkpoint(str(project), f"before {text}")
        (project / path).write_text(f"{text}\n")
        db.append_message(cli.session_id, "assistant", f"did {text}", timestamp=time.time())
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)

    cli._cmd_undo("/undo 2")
    out = capsys.readouterr().out
    if command := _PRINTED.search(out):
        cli._handle_rollback_command(command.group(1))
    assert not command or (project / "app.js").read_text() == "hello\n", out
    assert "Run /rollback to list checkpoints" in out, out


def test_a_first_undone_turn_that_took_no_checkpoint_gets_no_hash(setup, capsys):
    """The first undone turn edits through a tool that takes no checkpoint; the next turn's take holds that edit."""
    cli, db, project = setup
    mgr = CheckpointManager(enabled=True)
    cli.agent = SimpleNamespace(_checkpoint_mgr=mgr)
    _turn(cli, db, mgr, project, "kept")
    _next_second()
    _turn(cli, db, mgr, project, "sed app.js via a script", ("app.js", "t1\n"), take=False)
    _next_second()
    _turn(cli, db, mgr, project, "edit 2", ("b.txt", "t2\n"))
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)

    cli._cmd_undo("/undo 2")

    out = capsys.readouterr().out
    assert "To restore them" not in out and "Run /rollback to list checkpoints" in out, out
