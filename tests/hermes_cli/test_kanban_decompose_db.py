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


def test_decompose_children_marked_and_stay_todo_when_auto_promote_false(kanban_home):
    """#79608: decompose-created children carry created_from_decompose=1 and,
    with auto_promote=False, remain 'todo' after decompose (no recompute)."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee=None,
            children=[{"title": "child A"}, {"title": "child B"}],
            author="decomposer",
            auto_promote=False,
        )
    assert child_ids is not None

    with kb.connect() as conn:
        for cid in child_ids:
            task = kb.get_task(conn, cid)
            assert task.status == "todo", f"{cid} should stay todo after decompose"
            assert task.created_from_decompose is True, f"{cid} should be marked"


def test_dispatcher_tick_respects_auto_promote_false(kanban_home):
    """#79608: the dispatcher's per-tick recompute_ready must NOT promote
    decompose children while auto_promote_children=false — the manual-review
    gate. Without skip_decompose_children the old behavior (promote) holds."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee=None,
            children=[{"title": "child A"}, {"title": "child B"}],
            author="decomposer",
            auto_promote=False,
        )
    assert child_ids is not None

    with kb.connect() as conn:
        # auto_promote_children=false → dispatcher passes skip=True.
        res = kb.dispatch_once(
            conn, dry_run=False, max_spawn=10, skip_decompose_children=True
        )
        assert res.promoted == 0
        for cid in child_ids:
            assert kb.get_task(conn, cid).status == "todo", (
                f"{cid} must stay todo while auto_promote_children=false"
            )


def test_dispatcher_tick_promotes_decompose_children_by_default(kanban_home):
    """#79608: with auto_promote_children=true (default) the dispatcher's
    per-tick recompute promotes parent-free decompose children as before."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee=None,
            children=[{"title": "child A"}, {"title": "child B"}],
            author="decomposer",
            auto_promote=False,
        )
    assert child_ids is not None

    with kb.connect() as conn:
        # Default skip_decompose_children=False → promotion happens.
        res = kb.dispatch_once(conn, dry_run=False, max_spawn=10)
        assert res.promoted == len(child_ids)
        for cid in child_ids:
            assert kb.get_task(conn, cid).status == "ready"


def test_parent_gated_decompose_child_promotes_after_parent_done(kanban_home):
    """#79608: the manual-review gate applies ONLY to parent-free decompose
    children. A dependency-gated decompose child whose parent completes must
    still promote under skip_decompose_children=True (issue: 'existing
    behavior for dependency-gated children is unaffected')."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee=None,
            children=[
                {"title": "research", "assignee": "researcher", "parents": []},
                {"title": "build", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
            auto_promote=False,
        )
    assert child_ids is not None
    c_research, c_build = child_ids

    # Human review pass on the parent-free child: promote → claim → complete.
    with kb.connect() as conn:
        ok, _ = kb.promote_task(conn, c_research, actor="human", reason="reviewed")
        assert ok
        assert kb.claim_task(conn, c_research, claimer="worker") is not None
        assert kb.complete_task(conn, c_research) is True
        assert kb.get_task(conn, c_research).status == "done"

    with kb.connect() as conn:
        # Dispatcher tick with auto_promote_children=false (skip=True): the
        # parent-gated build child must still promote once its parent is done.
        res = kb.dispatch_once(conn, dry_run=False, max_spawn=10, skip_decompose_children=True)
        assert kb.get_task(conn, c_build).status == "ready"
        assert kb.get_task(conn, c_build).created_from_decompose is True
        assert res.promoted == 0  # only build promoted via parents-path; root still gated


def test_manual_promote_clears_decompose_marker(kanban_home):
    """#79608: a human promote IS the review pass — it clears the decompose
    marker so a reviewed child that later returns to 'todo' (claim demote on
    an undone parent, link_tasks, unblock_task) is not gated again."""
    with kb.connect() as conn:
        tid = _create_triage(conn)
        cid = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee=None,
            children=[{"title": "child A"}],
            author="decomposer",
            auto_promote=False,
        )[0]

        t = kb.get_task(conn, cid)
        assert t.created_from_decompose is True
        ok, _ = kb.promote_task(conn, cid, actor="human", reason="reviewed")
        assert ok
        t = kb.get_task(conn, cid)
        assert t.status == "ready"
        assert t.created_from_decompose is False

        # Reviewed child returns to todo (link an undone parent → claim demote).
        parent = kb.create_task(conn, title="new parent", assignee="x")
        kb.link_tasks(conn, parent, cid)
        assert kb.claim_task(conn, cid, claimer="worker") is None  # parents_not_done
        assert kb.get_task(conn, cid).status == "todo"
        assert kb.get_task(conn, cid).created_from_decompose is False
        kb.complete_task(conn, parent)

        # Under skip=True, the demoted (previously reviewed) child must promote.
        kb.dispatch_once(conn, dry_run=False, max_spawn=10, skip_decompose_children=True)
        assert kb.get_task(conn, cid).status == "ready"




