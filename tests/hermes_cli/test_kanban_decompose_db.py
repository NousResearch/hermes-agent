"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
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
        child_ids = decompose_triage_task(
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
        child_ids = decompose_triage_task(
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

    assert comments
    assert any(ev.kind == "decomposed" for ev in events)


def test_decomposed_children_are_reported_on_the_root(kanban_home):
    """The root must be able to name the cards it was split into.

    The dependency edges run child -> root (that is what holds the root in
    ``todo`` until the graph finishes), so ``child_ids`` is empty on the root
    and ``parent_ids`` lists the subtasks. The lineage is in the ``decomposed``
    event; this pins that ``decomposed_children`` reads it back with titles and
    live statuses, in fan-out order, and that the two views stay distinct.
    """
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="project view")
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "spec it", "assignee": "researcher", "parents": []},
                {"title": "build it", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None

        # The dependency view is inverted by design...
        assert kb.child_ids(conn, tid) == []
        assert sorted(kb.parent_ids(conn, tid)) == sorted(child_ids)

        # ...and the lineage view answers the question a reader actually asks.
        subtasks = kb.decomposed_children(conn, tid)
        assert [s["id"] for s in subtasks] == child_ids
        assert [s["title"] for s in subtasks] == ["spec it", "build it"]
        assert [s["status"] for s in subtasks] == ["ready", "todo"]

        # A child is not itself a root, and an ordinary card has no subtasks.
        assert kb.decomposed_children(conn, child_ids[0]) == []
        plain = kb.create_task(conn, title="standalone")
        assert kb.decomposed_children(conn, plain) == []

        # Status is read live, not frozen into the event.
        kb.complete_task(conn, child_ids[0], result="done")
        subtasks = kb.decomposed_children(conn, tid)
        assert subtasks[0]["status"] == "done"


def test_graph_completion_leaves_one_rollup_comment_on_the_root(kanban_home):
    """The parent card must carry what its subtasks did, once.

    A reader of the parent has no way to see a subtask's handoff otherwise:
    ``child_ids`` is empty by design, and the summary only ever lived in the
    root worker's prompt. The roll-up is written at the moment the graph
    completes, and exactly once — two completions cannot stack it.
    """
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="project view")
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "spec it", "assignee": "researcher", "parents": []},
                {"title": "build it", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None

        # First completion: the graph is not finished, so no roll-up yet.
        kb.complete_task(conn, child_ids[0], result="spec frozen in docs/x.md")
        assert not [c for c in kb.list_comments(conn, tid) if "decompose-rollup" in c.body]

        # Second completion finishes the graph.
        kb.complete_task(conn, child_ids[1], result="endpoint + UI shipped")
        rollups = [c for c in kb.list_comments(conn, tid) if "decompose-rollup" in c.body]
        assert len(rollups) == 1
        body = rollups[0].body
        assert "2/2 subtasks are done" in body
        for cid in child_ids:
            assert cid in body
        assert "spec frozen in docs/x.md" in body
        assert "endpoint + UI shipped" in body
        assert any(ev.kind == "decompose_rollup" for ev in kb.list_events(conn, tid))

        # Idempotent: the same graph re-notified (a reopen-and-recomplete, or two
        # racing completions) must not stack a second roll-up.
        assert kb.complete_task(conn, child_ids[1], result="again", force=True) is False
        kb._record_decompose_rollup(conn, child_ids[1])
        rollups = [c for c in kb.list_comments(conn, tid) if "decompose-rollup" in c.body]
        assert len(rollups) == 1




