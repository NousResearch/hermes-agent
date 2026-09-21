"""`hermes sessions optimize|optimize-storage|prune` refuse while another process holds state.db (#110054).

Running the storage rewrite underneath a fleet of live gateways put every agent into the retired-WAL
refusal; the command now runs the same fail-closed holder scan doctor/repair use, names each holder as
``PID N (command)`` and exits non-zero, with ``--force`` as the operator override. Driven through the
production entry point ``cmd_sessions`` with a REAL second process holding the store.
"""

import argparse
import subprocess
import sys
from argparse import Namespace

import pytest

import hermes_cli.sessions_cmd as sessions_cmd

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="holder scan is unavailable on Windows")


_HOLDER = (
    "import sqlite3, sys, time\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT count(*) FROM sqlite_master')\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
)


@pytest.fixture
def state_db(monkeypatch, tmp_path):
    import hermes_state
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)
    seed = SessionDB(db_path=db_path)
    seed.create_session("seed", "cli")
    seed.append_message("seed", "user", "hello")
    seed.close()
    return db_path


@pytest.fixture
def foreign_holder(state_db):
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(state_db)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.stdin.close()
            proc.wait(timeout=10)


def _args(action, force):
    common = dict(sessions_action=action, force=force)
    if action == "prune":
        return Namespace(dry_run=False, yes=True, never_active=False, include_archived=False,
                         include_pinned=False, **common,
                         **{name: None for name in sessions_cmd._FILTER_ARGS})
    if action == "optimize-storage":
        return Namespace(no_vacuum=False, yes=True, **common)
    return Namespace(**common)


def _sessions_subparsers():
    parser = argparse.ArgumentParser()
    from hermes_cli.subcommands.sessions import build_sessions_parser

    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    sessions = subparsers.choices["sessions"]
    nested = [a for a in sessions._actions if isinstance(a, argparse._SubParsersAction)][0]
    return nested.choices


@pytest.mark.parametrize("action", sorted(sessions_cmd._HELD_STORE_ACTIONS))
def test_store_rewrites_refuse_and_name_the_holder_until_forced(action, state_db, foreign_holder, capsys):
    assert sessions_cmd.cmd_sessions(_args(action, force=False)) == 1
    out = capsys.readouterr().out
    assert f"PID {foreign_holder.pid} (" in out
    assert f"Refusing `hermes sessions {action}`" in out and "--force" in out
    # The gate scans the store the command actually opened, not some other resolver's file: no
    # gated action can point the command at another database, so the default resolver IS the
    # operated-on path, and the refusal names it. (Offline commands such as set-journal-mode
    # take --db legitimately; they never go through this gate.)
    assert str(state_db) in out
    assert not [opt for name, sub in _sessions_subparsers().items()
                if name in sessions_cmd._HELD_STORE_ACTIONS for a in sub._actions
                for opt in a.option_strings if opt in ("--db", "--db-path", "--database")]

    assert sessions_cmd.cmd_sessions(_args(action, force=True)) != 1
    assert "Refusing" not in capsys.readouterr().out


def test_prune_preview_passes_the_delete_waits_for_a_quiet_store(state_db, foreign_holder, capsys):
    # A preview never rewrites anything, so it is answered even while the holder lives.
    prune_preview = _args("prune", force=False)
    prune_preview.dry_run = True
    prune_preview.yes = False
    assert sessions_cmd.cmd_sessions(prune_preview) is None
    assert "Refusing" not in capsys.readouterr().out
    prune_preview.dry_run = False
    prune_preview.yes = True
    assert sessions_cmd.cmd_sessions(prune_preview) == 1
    assert "Refusing `hermes sessions prune`" in capsys.readouterr().out
    # Control: once the holder exits the same command runs.
    foreign_holder.stdin.close()
    foreign_holder.wait(timeout=10)
    assert sessions_cmd.cmd_sessions(prune_preview) is None
    assert "Refusing" not in capsys.readouterr().out
