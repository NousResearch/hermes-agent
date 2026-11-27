"""Deferred Kanban workspace cleanup across chains and shared ancestors."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Use real SQLite and scratch paths in an isolated home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("grandparent_running", [False, True])
def test_deferred_scratch_sweep_recurses_to_grandparent(kanban_home, grandparent_running):
    """Sweep terminal ancestors in A->B->C, but preserve a running grandparent.

    Regression for the non-recursive parent sweep: when leaf C completed it swept only its
    direct parent B, leaving A's scratch dir leaked on disk forever even though A had become
    eligible (its only child B was now terminal). The sweep must cascade up the whole chain.
    """
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="A grandparent")
        if grandparent_running:
            assert kb.claim_task(conn, a) is not None
        b = kb.create_task(conn, title="B parent")
        c = kb.create_task(conn, title="C leaf")
        kb.link_tasks(conn, a, b)  # B depends on A
        kb.link_tasks(conn, b, c)  # C depends on B
        a_ws = kbw.resolve_workspace(kb.get_task(conn, a))
        b_ws = kbw.resolve_workspace(kb.get_task(conn, b))
        c_ws = kbw.resolve_workspace(kb.get_task(conn, c))
        kbw.set_workspace_path(conn, a, a_ws)
        kbw.set_workspace_path(conn, b, b_ws)
        kbw.set_workspace_path(conn, c, c_ws)
        marker = a_ws / "active-work.txt"
        marker.write_text("grandparent work", encoding="utf-8")

        # B's cleanup remains deferred while leaf C is active.
        if not grandparent_running:
            kb.complete_task(conn, a, result="handoff A")
        assert kb.archive_task(conn, b)
        assert a_ws.exists() and b_ws.exists(), "deferred while leaf C is still active"

        # Leaf C completes -> the sweep must cascade C -> B -> A.
        assert kb.archive_task(conn, c)

    assert not c_ws.exists(), "leaf scratch dir cleaned up"
    assert not b_ws.exists(), "direct parent scratch dir swept"
    if grandparent_running:
        assert marker.read_text(encoding="utf-8") == "grandparent work"
    else:
        assert not a_ws.exists(), "terminal grandparent scratch dir must also be swept"


def test_deferred_scratch_sweep_handles_diamond_dag(kanban_home):
    """A diamond DAG (A->B, A->C, B->D, C->D) reaps every eligible scratch dir, and the
    shared ancestor A is reached through both paths without breaking the cascade."""
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="A root")
        b = kb.create_task(conn, title="B")
        c = kb.create_task(conn, title="C")
        d = kb.create_task(conn, title="D leaf")
        kb.link_tasks(conn, a, b)
        kb.link_tasks(conn, a, c)
        kb.link_tasks(conn, b, d)
        kb.link_tasks(conn, c, d)
        ws = {}
        for tid in (a, b, c, d):
            w = kbw.resolve_workspace(kb.get_task(conn, tid))
            kbw.set_workspace_path(conn, tid, w)
            ws[tid] = w

        # Complete the ancestors first; each is deferred while leaf D is still active.
        kb.complete_task(conn, a, result="a")
        kb.complete_task(conn, b, result="b")
        kb.complete_task(conn, c, result="c")
        assert ws[a].exists() and ws[b].exists() and ws[c].exists(), "deferred while D active"

        # D completes -> the cascade reaps B, C, and the shared ancestor A.
        kb.complete_task(conn, d, result="d")

    for tid in (a, b, c, d):
        assert not ws[tid].exists(), f"scratch dir for {tid} must be swept once the DAG is terminal"
