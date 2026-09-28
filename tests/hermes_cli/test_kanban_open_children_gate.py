"""Regression for the falsely-completed-decomposed-parent failure mode.

fm #37 (LGTM 14:55:18 -> merge 18:33:52, a 3h38m34s post-review stall) and the
"parent cards show done when merely decomposed" complaint share one root
cause: ``complete_task`` never checked whether a task's own fanned-out
children (``kanban_create`` with no dependency edge back to the creator —
the ordinary orchestrator pattern) had actually finished. A worker could
``kanban_complete`` immediately after spawning delivery work and the board
would report the request done while nothing had shipped.

These tests pin ``kanban_db.unsatisfied_decomposed_children`` /
``OpenChildrenError``:

* a task cannot complete while a child it spawned (``creator_task_id``) is
  still open,
* it CAN complete once every such child reaches done/archived,
* the legitimate "spawn a review/QA child that depends on me, then complete
  myself to release it" pattern (child linked as a dependent via
  ``parents=[self]``) is NOT blocked — that's the mechanism the gate must
  not break,
* ``force=True`` is the only bypass (explicit operator override, never a
  worker-reachable one — see tests/tools/test_kanban_open_children_tool.py),
* recompute_ready still promotes a child gated on the parent once the
  parent finishes (no deadlock introduced by the new column).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def test_complete_blocked_by_open_spawned_child(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    child_id = kb.create_task(
        conn, title="delivery work", assignee="builder",
        creator_task_id=parent_id,
    )

    with pytest.raises(kb.OpenChildrenError) as excinfo:
        kb.complete_task(conn, parent_id, summary="decomposed into subtasks")

    assert excinfo.value.task_id == parent_id
    assert excinfo.value.open_children == [(child_id, "ready")]
    task = kb.get_task(conn, parent_id)
    assert task is not None
    assert task.status != "done", "complete_task must not mutate the task when it raises"


def test_complete_succeeds_once_spawned_children_are_done(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    child_id = kb.create_task(
        conn, title="delivery work", assignee="builder",
        creator_task_id=parent_id,
    )
    assert kb.unsatisfied_decomposed_children(conn, parent_id) == [(child_id, "ready")]

    assert kb.complete_task(conn, child_id, summary="shipped")
    assert kb.unsatisfied_decomposed_children(conn, parent_id) == []

    assert kb.complete_task(conn, parent_id, summary="decomposed and delivered")
    assert kb.get_task(conn, parent_id).status == "done"


def test_archived_spawned_child_does_not_block(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    child_id = kb.create_task(
        conn, title="abandoned", assignee="builder", creator_task_id=parent_id,
    )
    kb.archive_task(conn, child_id)

    assert kb.unsatisfied_decomposed_children(conn, parent_id) == []
    assert kb.complete_task(conn, parent_id, summary="decomposed; one child archived")


def test_review_child_depending_on_creator_does_not_block_completion(conn):
    """The documented orchestrator pattern: spawn a review/QA child that
    DEPENDS on this task (parents=[self]) and complete this task to release
    it. This is the release mechanism, not the falsely-completed-parent bug
    -- the gate must not confuse the two."""
    impl_id = kb.create_task(conn, title="implementation", assignee="anika")
    review_id = kb.create_task(
        conn, title="review", assignee="meera",
        creator_task_id=impl_id, parents=[impl_id],
    )
    assert kb.get_task(conn, review_id).status == "todo"

    assert kb.unsatisfied_decomposed_children(conn, impl_id) == []
    assert kb.complete_task(conn, impl_id, summary="implementation done, review it")

    kb.recompute_ready(conn)
    assert kb.get_task(conn, review_id).status == "ready"


def test_force_overrides_the_open_children_gate(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    kb.create_task(
        conn, title="delivery work", assignee="builder", creator_task_id=parent_id,
    )
    assert kb.complete_task(
        conn, parent_id, summary="operator override", force=True,
    )
    assert kb.get_task(conn, parent_id).status == "done"


def test_grandchildren_and_unrelated_creators_are_not_conflated(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    child_id = kb.create_task(
        conn, title="mid", assignee="builder", creator_task_id=parent_id,
    )
    # Spawned BY the child, not the parent -- must not count against the parent.
    grandchild_id = kb.create_task(
        conn, title="leaf", assignee="builder", creator_task_id=child_id,
    )
    assert kb.unsatisfied_decomposed_children(conn, parent_id) == [(child_id, "ready")]

    assert kb.complete_task(conn, grandchild_id, summary="leaf done")
    assert kb.complete_task(conn, child_id, summary="mid done")
    assert kb.unsatisfied_decomposed_children(conn, parent_id) == []
    assert kb.complete_task(conn, parent_id, summary="all delivered")


def test_creator_task_id_persists_on_the_row(conn):
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    child_id = kb.create_task(
        conn, title="delivery work", assignee="builder", creator_task_id=parent_id,
    )
    task = kb.get_task(conn, child_id)
    assert task is not None
    assert task.creator_task_id == parent_id
