"""P2a domain primitives: task_kind, non-blocking task containment, epic hierarchy.

Covers the canonical task model additions and public epic primitives in
``hermes_cli.kanban_db``. Behavioural contracts only — no snapshot tests.

Key invariants under test:
  * ``task_kind`` validates to {task, bug, spike, subtask, gate} and defaults
    to ``task`` (schema roundtrip + backfill).
  * ``parent_task_id`` is NON-BLOCKING containment: it must never surface in
    dependency queries (``parent_ids`` / ``child_ids``) or block readiness.
  * Epic hierarchy (``parent_epic_id``) + task-to-epic attachment
    (``tasks.epic_id``) validate same-board identity, parent existence/type,
    and cycles; they fail closed with ``ValueError``.
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


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# task_kind
# ---------------------------------------------------------------------------


def test_task_kind_defaults_to_task(conn):
    tid = kb.create_task(conn, title="plain")
    task = kb.get_task(conn, tid)
    assert task.task_kind == "task"


def test_task_kind_roundtrip(conn):
    parent = kb.create_task(conn, title="subtask parent")
    for kind in ("task", "bug", "spike", "subtask", "gate"):
        parent_task_id = parent if kind == "subtask" else None
        tid = kb.create_task(
            conn,
            title=kind,
            task_kind=kind,
            parent_task_id=parent_task_id,
        )
        assert kb.get_task(conn, tid).task_kind == kind


def test_task_kind_rejects_unknown_value(conn):
    with pytest.raises(ValueError):
        kb.create_task(conn, title="bad", task_kind="story")


def test_subtask_requires_parent_before_write(conn):
    with pytest.raises(ValueError, match="subtask.*parent_task_id"):
        kb.create_task(conn, title="orphan subtask", task_kind="subtask")
    assert kb.list_tasks(conn) == []


def test_task_kind_backfills_existing_rows(kanban_home, conn):
    # Simulate a pre-P2a row: insert directly without task_kind / parent_task_id.
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('t_legacy', 'legacy', 'todo', 1, 'scratch')"
        )
    # Re-open forces the migration/backfill pass.
    conn.close()
    c2 = kb.connect()
    try:
        legacy = kb.get_task(c2, "t_legacy")
        assert legacy is not None
        assert legacy.task_kind == "task"
        assert legacy.parent_task_id is None
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# parent_task_id — non-blocking containment
# ---------------------------------------------------------------------------


def test_containment_roundtrip_and_no_dependency_blocking(conn):
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parent_task_id=parent)

    child_task = kb.get_task(conn, child)
    assert child_task.parent_task_id == parent
    # Non-blocking: a subtask is immediately ready, NOT parked in todo.
    assert child_task.status == "ready"

    # Containment is invisible to the dependency surface.
    assert kb.parent_ids(conn, child) == []
    assert kb.child_ids(conn, parent) == []
    assert kb.get_task_parent(conn, child).id == parent
    assert kb.get_subtask_children(conn, parent) == [child]
    assert kb.get_task_parent(conn, parent) is None


def test_containment_does_not_block_readiness_after_parent_done(conn):
    parent = kb.create_task(conn, title="parent", assignee="p")
    child = kb.create_task(conn, title="child", parent_task_id=parent, assignee="c")
    assert kb.get_task(conn, child).status == "ready"

    # Complete the parent — containment must not demote/re-promote the child.
    assert kb.complete_task(conn, parent)
    assert kb.get_task(conn, child).status == "ready"


def test_containment_rejects_unknown_parent(conn):
    with pytest.raises(ValueError):
        kb.create_task(conn, title="orphan", parent_task_id="t_does_not_exist")


def test_containment_rejects_cross_board_parent(conn, kanban_home):
    kb.create_board("other")
    other_conn = kb.connect(board="other")
    try:
        other_tid = kb.create_task(other_conn, title="on other board")
    finally:
        other_conn.close()
    # Same id must NOT resolve on the default board's connection.
    with pytest.raises(ValueError):
        kb.create_task(conn, title="cross board child", parent_task_id=other_tid)


def test_containment_set_parent_rejects_cycle(conn):
    a = kb.create_task(conn, title="a")
    b = kb.create_task(conn, title="b", parent_task_id=a)
    # b is contained by a; making a contained by b closes a cycle.
    with pytest.raises(ValueError):
        kb.set_task_parent(conn, a, b)
    with pytest.raises(ValueError):
        kb.set_task_parent(conn, a, a)


def test_containment_clear_parent(conn):
    parent = kb.create_task(conn, title="p")
    child = kb.create_task(conn, title="c", parent_task_id=parent)
    assert kb.clear_task_parent(conn, child)
    assert kb.get_task(conn, child).parent_task_id is None


def test_update_task_is_atomic_and_preserves_containment_semantics(conn):
    parent = kb.create_task(conn, title="parent")
    epic_id = kb.create_epic(conn, title="update epic")
    task_id = kb.create_task(conn, title="before")

    updated = kb.update_task(
        conn,
        task_id,
        title="after",
        body="canonical body",
        priority=4,
        task_kind="subtask",
        parent_task_id=parent,
        epic_id=epic_id,
    )
    assert updated is not None
    assert updated.title == "after"
    assert updated.body == "canonical body"
    assert updated.priority == 4
    assert updated.task_kind == "subtask"
    assert updated.parent_task_id == parent
    assert updated.epic_id == epic_id
    assert kb.parent_ids(conn, task_id) == []
    assert kb.child_ids(conn, parent) == []

    with pytest.raises(ValueError):
        kb.update_task(
            conn,
            task_id,
            title="must roll back",
            parent_task_id="missing-parent",
        )
    assert kb.get_task(conn, task_id).title == "after"


def test_transition_task_routes_status_changes_through_domain_rules(conn):
    task_id = kb.create_task(conn, title="transition", initial_status="backlog")
    assert kb.transition_task(conn, task_id, "triage")
    assert kb.get_task(conn, task_id).status == "triage"
    assert kb.transition_task(conn, task_id, "ready")
    assert kb.get_task(conn, task_id).status == "ready"
    assert kb.transition_task(conn, task_id, "blocked", reason="waiting")
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.transition_task(conn, task_id, "ready")
    assert kb.get_task(conn, task_id).status == "ready"

    with pytest.raises(ValueError, match="running"):
        kb.transition_task(conn, task_id, "running")
    with pytest.raises(ValueError, match="unknown status"):
        kb.transition_task(conn, task_id, "not-a-status")

    parent = kb.create_task(conn, title="unfinished parent")
    child = kb.create_task(
        conn,
        title="dependency child",
        parents=[parent],
        initial_status="backlog",
    )
    assert kb.transition_task(conn, child, "ready") is False
    blocked_child = kb.get_task(conn, child)
    assert blocked_child is not None
    assert blocked_child.status == "backlog"


def test_transition_task_routes_structured_lifecycle_verbs(conn):
    done_id = kb.create_task(conn, title="done path")
    assert kb.transition_task(conn, done_id, "done", result="complete")
    done = kb.get_task(conn, done_id)
    assert done is not None
    assert done.status == "done"
    assert done.result == "complete"

    scheduled_id = kb.create_task(conn, title="scheduled path")
    assert kb.transition_task(
        conn,
        scheduled_id,
        "scheduled",
        reason="not before tomorrow",
    )
    scheduled = kb.get_task(conn, scheduled_id)
    assert scheduled is not None
    assert scheduled.status == "scheduled"
    assert kb.transition_task(conn, scheduled_id, "ready")

    review_id = kb.create_task(conn, title="review path", assignee="octacon")
    assert kb.transition_task(
        conn,
        review_id,
        "review",
        summary="ready for QA",
        reviewer="quan",
        force_review=True,
    )
    review = kb.get_task(conn, review_id)
    assert review is not None
    assert review.status == "review"
    assert review.assignee == "quan"
    assert kb.transition_task(conn, review_id, "todo")
    reopened = kb.get_task(conn, review_id)
    assert reopened is not None
    assert reopened.status == "ready"
    assert reopened.assignee == "octacon"

    archived_id = kb.create_task(conn, title="archive path")
    assert kb.transition_task(conn, archived_id, "archived")
    archived = kb.get_task(conn, archived_id)
    assert archived is not None
    assert archived.status == "archived"


# ---------------------------------------------------------------------------
# Epic primitives — hierarchy
# ---------------------------------------------------------------------------


def test_epic_create_roundtrip(conn):
    eid = kb.create_epic(conn, title="Checkout v2", description="desc")
    epic = kb.get_epic(conn, eid)
    assert epic is not None
    assert epic.title == "Checkout v2"
    assert epic.description == "desc"
    assert epic.parent_epic_id is None
    assert epic.status == "active"
    assert epic.board_slug == "default"
    assert eid in [e.id for e in kb.list_epics(conn)]


def test_epic_sub_epic_hierarchy(conn):
    parent = kb.create_epic(conn, title="Platform")
    child = kb.create_epic(conn, title="Billing", parent_epic_id=parent)
    assert kb.get_epic(conn, child).parent_epic_id == parent
    assert child in [e.id for e in kb.list_epics(conn, parent_epic_id=parent)]


def test_epic_rejects_unknown_parent(conn):
    with pytest.raises(ValueError):
        kb.create_epic(conn, title="orphan", parent_epic_id="epic_nope")


def test_epic_rejects_cycle(conn):
    a = kb.create_epic(conn, title="a")
    b = kb.create_epic(conn, title="b", parent_epic_id=a)
    # b is under a; making a a child of b closes a hierarchy cycle.
    with pytest.raises(ValueError):
        kb.set_epic_parent(conn, a, b)
    with pytest.raises(ValueError):
        kb.set_epic_parent(conn, a, a)  # self-parent


def test_epic_rejects_cross_board_identity(conn):
    # An epic claiming a different board must not accept tasks on this board.
    eid = kb.create_epic(conn, title="foreign", board_slug="other-board")
    tid = kb.create_task(conn, title="t")
    with pytest.raises(ValueError):
        kb.set_task_epic(conn, tid, eid)


# ---------------------------------------------------------------------------
# Task-to-epic attachment
# ---------------------------------------------------------------------------


def test_task_epic_attachment_roundtrip(conn):
    eid = kb.create_epic(conn, title="Epic A")
    tid = kb.create_task(conn, title="task under epic", epic_id=eid)
    assert kb.get_task(conn, tid).epic_id == eid

    # Attach an existing task via the primitive.
    other = kb.create_task(conn, title="attach later")
    assert kb.set_task_epic(conn, other, eid)
    assert kb.get_task(conn, other).epic_id == eid

    # list_tasks(epic_id=...) returns both.
    found = {t.id for t in kb.list_tasks(conn, epic_id=eid)}
    assert found == {tid, other}

    assert kb.clear_task_epic(conn, other)
    assert kb.get_task(conn, other).epic_id is None


def test_task_epic_attachment_on_non_default_board(kanban_home):
    kb.create_board("other")
    other_conn = kb.connect(board="other")
    try:
        eid = kb.create_epic(other_conn, title="Other epic", board_slug="other")
        tid = kb.create_task(other_conn, title="Other task", board="other")
        assert kb.set_task_epic(other_conn, tid, eid, board="other")
        task = kb.get_task(other_conn, tid)
        assert task is not None
        assert task.epic_id == eid
    finally:
        other_conn.close()


def test_task_epic_rejects_unknown_epic(conn):
    tid = kb.create_task(conn, title="t")
    with pytest.raises(ValueError):
        kb.set_task_epic(conn, tid, "epic_nope")


def test_epic_task_counts(conn):
    eid = kb.create_epic(conn, title="Epic A")
    kb.create_task(conn, title="done", epic_id=eid)
    kb.create_task(conn, title="open", epic_id=eid)
    counts = kb.epic_task_counts(conn)
    assert counts[eid]["total"] == 2
