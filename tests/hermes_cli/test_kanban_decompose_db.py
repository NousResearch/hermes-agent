"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


# ---------------------------------------------------------------------------
# Board-targeted decomposition (regression pin for the decomposer board-
# routing gap: decomposed children used to inherit the parent's board, so a
# cover card for an unwatched/exempted board landed there — invisible to
# every watcher).
#
# A board is not a column — it is the DB file a row lives in. Targeting a
# board at decompose time creates the children and their links (including
# the root-wait edges) in the TARGET board's DB while the root stays on its
# own board, deliberately in ``triage``: flipping it to ``todo`` there would
# gate it by links the parent board's sweep cannot see, while the target
# board's sweep cannot see the root's row at all — either way the promotion
# graph corrupts or a zombie orchestrator gets dispatched (both proven live
# during development). A ``triage`` root is not dispatchable and not
# sweep-eligible, so the hazard is closed by construction; a repeat
# decompose of the still-triage root is an idempotent no-op. Omitting the
# argument preserves the parent-board behavior.
# ---------------------------------------------------------------------------


def _create_triage_on_board(board, title):
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=title, triage=True)


_DECOMPOSE_CHILDREN = [
    {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
    {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
]


def test_decompose_without_board_argument_keeps_parent_board(kanban_home):
    """Backwards compat: omitting the board argument decomposes on the
    parent's board exactly as before."""
    kb.create_board("watched")  # exists but must stay empty
    tid = _create_triage_on_board("default", "rough idea")
    with kbc.connect_closing() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=_DECOMPOSE_CHILDREN, author="decomposer",
        )
    assert child_ids is not None and len(child_ids) == 2

    with kbc.connect_closing() as conn:
        root = kb.get_task(conn, tid)
        assert root is not None and root.status == "todo"
        assert root.assignee == "orchestrator"
        for cid in child_ids:
            assert kb.get_task(conn, cid) is not None
    with kbc.connect_closing(board="watched") as conn:
        for cid in (tid, *child_ids):
            assert kb.get_task(conn, cid) is None, "nothing may land on an untargeted board"


def test_decompose_board_argument_places_children_on_target_board(kanban_home):
    """board="watched": the children and their links land on the watched
    board; the root stays on its own board, gated and un-dispatchable, in
    triage — with the audit trail. (Fails against a kernel without the board
    argument: TypeError unexpected keyword argument 'board'.)"""
    kb.create_board("watched")
    tid = _create_triage_on_board("default", "rough idea")
    with kbc.connect_closing() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=_DECOMPOSE_CHILDREN, author="decomposer", board="watched",
        )
    assert child_ids is not None and len(child_ids) == 2

    with kbc.connect_closing(board="watched") as conn:
        # Children live on the target board with a coherent graph state:
        # the first has no internal parents -> ready on recompute_ready.
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])
        assert c0 is not None and c0.status == "ready"
        assert c1 is not None and c1.status == "todo"
        # The root is linked under every child on THIS board — the promotion
        # graph lives where the children are visible — even though its own
        # row stays on the parent board.
        for cid in child_ids:
            edge = conn.execute(
                "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
                (cid, tid),
            ).fetchone()
            assert edge is not None, "root must be gated by every child on the target board"

    with kbc.connect_closing() as conn:
        # The root stays home in triage: not dispatchable there, and the
        # parent-board sweep cannot vacuously promote it (its gating links
        # live on the target board).
        root = kb.get_task(conn, tid)
        assert root is not None and root.status == "triage"
        for cid in child_ids:
            assert kb.get_task(conn, cid) is None, "children must not stay on the parent board"
        # The audit trail stays with the root.
        assert any("Decomposed into" in (c.body or "") for c in kb.list_comments(conn, tid))
        assert any(ev.kind == "decomposed" for ev in kb.list_events(conn, tid))


def test_decompose_cross_board_is_idempotent_on_repeat(kanban_home):
    """A cross-board root stays in triage, so --all sweeps re-find it; a
    repeat decompose must be a no-op returning None, not a duplicate batch."""
    kb.create_board("watched")
    tid = _create_triage_on_board("default", "rough idea")
    with kbc.connect_closing() as conn:
        first = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=_DECOMPOSE_CHILDREN, author="decomposer", board="watched",
        )
        assert first is not None and len(first) == 2
        repeat = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=_DECOMPOSE_CHILDREN, author="decomposer", board="watched",
        )
        assert repeat is None
        events = kb.list_events(conn, tid)
    assert sum(1 for ev in events if ev.kind == "decomposed") == 1
    with kbc.connect_closing(board="watched") as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
        assert n["n"] == 2, "the repeat must not create a second child batch"


def test_decompose_unknown_target_board_raises_valueerror(kanban_home):
    """Unknown target board: raise-only ValueError from the kernel (the CLI
    maps it to a clean failure)."""
    tid = _create_triage_on_board("default", "rough idea")
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError):
            kb.decompose_triage_task(
                conn, tid, root_assignee="orchestrator",
                children=_DECOMPOSE_CHILDREN, author="decomposer", board="no-such-board",
            )




