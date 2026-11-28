"""CLI wiring for the live-claim/reclaim --force flags (t_1f3b0108).

Guards against a flag that exists in --help but never reaches kanban_db,
which a pure kanban_db unit test would not catch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_parser


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    return home


def _parse(argv):
    root = argparse.ArgumentParser()
    kanban_parser.build_parser(root.add_subparsers(dest="cmd"))
    return root.parse_args(["kanban"] + argv)


def test_force_flags_are_parsed(kanban_home):
    assert _parse(["reclaim", "t_x"]).force is False
    assert _parse(["reclaim", "t_x", "--force"]).force is True
    assert _parse(["block", "t_x", "why"]).force is False
    assert _parse(["block", "t_x", "why", "--force"]).force is True


def test_cli_reclaim_passes_force_through(kanban_home, monkeypatch):
    seen = {}

    def fake_reclaim(conn, task_id, *, reason=None, force=False, signal_fn=None,
                     with_reason=False):
        seen["force"] = force
        return (True, None) if with_reason else True

    monkeypatch.setattr(kb, "reclaim_task", fake_reclaim)
    kc._cmd_reclaim(_parse(["reclaim", "t_x"]))
    assert seen["force"] is False
    kc._cmd_reclaim(_parse(["reclaim", "t_x", "--force"]))
    assert seen["force"] is True


def test_cli_block_passes_force_through(kanban_home, monkeypatch):
    seen = {}
    real = kb.block_task

    def spy(conn, task_id, **kw):
        seen["force"] = kw.get("force")
        return real(conn, task_id, **kw)

    monkeypatch.setattr(kb, "block_task", spy)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="w")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    kc._cmd_block(_parse(["block", tid, "because", "--force"]))
    assert seen["force"] is True
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_cli_block_reports_a_live_claim_instead_of_raising(kanban_home, monkeypatch):
    """A LiveClaimError must become an operator message, not a traceback."""
    def boom(conn, task_id, **kw):
        raise kb.LiveClaimError(task_id)

    monkeypatch.setattr(kb, "block_task", boom)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="w")
    rc = kc._cmd_block(_parse(["block", tid, "because"]))
    assert rc != 0
