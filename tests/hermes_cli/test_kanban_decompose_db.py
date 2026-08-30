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


def test_decompose_creates_children_and_promotes_root(kanban_home, all_assignees_spawnable):
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


def test_create_known_assignee_not_parked(kanban_home):
    """A real assignee (``default`` is Agent Smith, always spawnable) does NOT
    get triage-parked."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="good", assignee="default")
        task = kb.get_task(conn, tid)
    assert tid is not None
    assert task is not None
    assert task.status != "triage"


@pytest.mark.real_assignees
def test_create_unknown_assignee_parked_in_triage(kanban_home):
    """An assignee that is not a real profile parks the card in triage with a
    comment, and the bogus name is preserved on the row (not rejected)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="phantom", assignee="engineer")
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.assignee == "engineer"
    assert any("unknown assignee" in (c.body or "") for c in comments)


@pytest.mark.real_assignees
def test_decompose_unknown_assignee_child_parked_in_triage(kanban_home):
    """A decomposer-spawned child naming a phantom profile parks in triage AND
    carries a comment, exactly like a top-level phantom card (the 2026-08-30
    incident produced 12 'engineer' + 12 'orchestrator' junk children)."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
    children = [{"title": "build it", "assignee": "engineer", "parents": []}]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    kid = child_ids[0]
    with kb.connect() as conn:
        child = kb.get_task(conn, kid)
        comments = kb.list_comments(conn, kid)
    assert child is not None
    assert child.status == "triage"
    assert child.assignee == "engineer"
    assert any("unknown assignee" in (c.body or "") for c in comments)
