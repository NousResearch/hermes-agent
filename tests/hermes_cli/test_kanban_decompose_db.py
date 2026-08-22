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


def test_manual_decompose_children_stay_todo_across_recompute_passes(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="review before dispatch")
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "groom first", "assignee": "patch"},
                {"title": "groom dependent", "assignee": "patch", "parents": [0]},
            ],
            author="product-owner",
            auto_promote=False,
        )
        assert child_ids is not None
        first_id, dependent_id = child_ids
        first = kb.get_task(conn, first_id)
        dependent = kb.get_task(conn, dependent_id)
        assert first is not None and first.status == "todo"
        assert dependent is not None and dependent.status == "todo"

        assert kb.recompute_ready(conn) == 0
        assert kb.recompute_ready(conn) == 0
        first = kb.get_task(conn, first_id)
        assert first is not None and first.status == "todo"

        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (first_id,))
        assert kb.recompute_ready(conn) == 0
        dependent = kb.get_task(conn, dependent_id)
        assert dependent is not None and dependent.status == "todo"

        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (dependent_id,))
        assert kb.recompute_ready(conn) == 0
        dependent = kb.get_task(conn, dependent_id)
        assert dependent is not None and dependent.status == "blocked"


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




