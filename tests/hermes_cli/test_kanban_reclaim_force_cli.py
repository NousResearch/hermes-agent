"""CLI wiring check for t_1f3b0108: the --force flags actually reach kanban_db.

Guards against the flag existing in --help but never being plumbed through,
which a pure kanban_db unit test would not catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


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
    import argparse

    root = argparse.ArgumentParser()
    kc.build_parser(root.add_subparsers(dest="cmd"))
    return root.parse_args(["kanban"] + argv)


def test_reclaim_force_flag_is_parsed_and_plumbed(kanban_home, monkeypatch):
    seen = {}

    def fake_reclaim(conn, task_id, *, reason=None, force=False, signal_fn=None):
        seen["force"] = force
        return True

    monkeypatch.setattr(kb, "reclaim_task", fake_reclaim)
    assert _parse(["reclaim", "t_x"]).force is False
    kc._cmd_reclaim(_parse(["reclaim", "t_x"]))
    assert seen["force"] is False
    kc._cmd_reclaim(_parse(["reclaim", "t_x", "--force"]))
    assert seen["force"] is True


def test_complete_and_block_force_flags_are_parsed(kanban_home):
    assert _parse(["complete", "t_x"]).force is False
    assert _parse(["complete", "t_x", "--force"]).force is True
    assert _parse(["block", "t_x", "why"]).force is False
    assert _parse(["block", "t_x", "why", "--force"]).force is True


def test_cli_complete_passes_force_through(kanban_home, monkeypatch):
    seen = {}
    real = kb.complete_task

    def spy(conn, task_id, **kw):
        seen["force"] = kw.get("force")
        return real(conn, task_id, **kw)

    monkeypatch.setattr(kb, "complete_task", spy)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="w")
    kc._cmd_complete(_parse(["complete", tid, "--force"]))
    assert seen["force"] is True
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"


def test_cli_block_passes_force_through(kanban_home, monkeypatch):
    seen = {}
    real = kb.block_task

    def spy(conn, task_id, **kw):
        seen["force"] = kw.get("force")
        return real(conn, task_id, **kw)

    monkeypatch.setattr(kb, "block_task", spy)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="w")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    kc._cmd_block(_parse(["block", tid, "because", "--force"]))
    assert seen["force"] is True
