"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
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
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decomposed_children_inherit_root_priority(kanban_home):
    """Regression: children were inserted without a priority column, so
    they landed at the SQL default 0 and queued behind every card on a
    board whose live band sits far above 0. Dispatch is strictly
    highest-priority-first, so every auto-decomposed subtree was starved
    and autonomous orchestration looked like it never fired.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="high-band root", triage=True, priority=97)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "child A", "assignee": "researcher"},
                {"title": "child B", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        priorities = [kb.get_task(conn, cid).priority for cid in child_ids]

    assert root.priority == 97
    assert priorities == [97, 97], (
        "auto-decomposed children must inherit the root's priority so the "
        f"subtree dispatches in the same band; got {priorities}"
    )


def test_decomposed_child_can_override_priority(kanban_home):
    """Inheritance is the default, not a ceiling: an explicit per-child
    priority still wins so a decomposer can deprioritize one leaf."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", triage=True, priority=90)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "inherits"},
                {"title": "explicit", "priority": 12},
            ],
            author="decomposer",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        priorities = [kb.get_task(conn, cid).priority for cid in child_ids]

    assert priorities == [90, 12]


def test_decomposed_children_inherit_zero_priority_root(kanban_home):
    """A root genuinely at 0 still yields children at 0 — the fix is
    inheritance, not a hardcoded floor."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", triage=True, priority=0)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[{"title": "child A"}],
            author="decomposer",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        assert kb.get_task(conn, child_ids[0]).priority == 0




