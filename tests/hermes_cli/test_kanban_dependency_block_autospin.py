
from hermes_cli import kanban_db_connect
"""A dependency block with nothing to wait on must not auto-spin the card.

``block_task(kind="dependency")`` routes to ``todo`` so ``recompute_ready``
promotes the card once its parents finish. When the card has **no** unfinished
parent, that promotion fires immediately: the reefmind ``t_5e4df497`` card
emitted ``dependency_wait`` at 12:02:24 and was ``promoted`` back to ``ready``
at 12:02:32, re-running the identical verification as a new run. Nothing was
ever delivered to the operator because the card never entered a human bucket.

A dependency wait is only a dependency wait when there is a real unsatisfied
parent link. Otherwise it is a block, and it must be routed and notified as one.
"""

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "dep-block.db"))
    kanban_db_connect.init_db()
    c = kanban_db_connect.connect()
    yield c
    c.close()


def _kinds(conn, tid):
    return [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        ).fetchall()
    ]


def test_dependency_block_without_parents_lands_in_blocked_not_todo(conn):
    tid = kb.create_task(conn, title="implement", assignee="orchestrator")
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))

    assert kb.block_task(
        conn, tid,
        reason="review-handoff: PR #3165 needs an independent reviewer",
        kind="dependency",
    ) is True

    task = kb.get_task(conn, tid)
    assert task.status == "blocked", (
        "a dependency block with no unsatisfied parent would be promoted "
        "straight back to ready from todo, respinning the same work; it must "
        f"land in the human bucket instead. Got status={task.status!r}"
    )
    assert "blocked" in _kinds(conn, tid), (
        "the stop must emit a notifiable `blocked` event, not a silent "
        f"`dependency_wait`. Got {_kinds(conn, tid)!r}"
    )


def test_dependency_block_preserves_the_reason_and_provenance(conn):
    tid = kb.create_task(conn, title="implement", assignee="orchestrator")
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    kb.block_task(conn, tid, reason="review-handoff: needs a reviewer", kind="dependency")

    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    import json
    payload = json.loads(row["payload"])
    assert payload.get("reason") == "review-handoff: needs a reviewer"
    assert payload.get("kind") == "dependency", "the declared block kind is provenance"
    assert payload.get("routed_from") == "dependency", (
        "the notice needs to know this arrived as a dependency wait with no "
        "dependency to wait on"
    )


def test_dependency_block_with_an_unfinished_parent_still_waits_in_todo(conn):
    parent = kb.create_task(conn, title="upstream", assignee="backend")
    tid = kb.create_task(conn, title="implement", assignee="orchestrator")
    kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))

    assert kb.block_task(conn, tid, reason="waiting on upstream", kind="dependency") is True

    task = kb.get_task(conn, tid)
    assert task.status == "todo", (
        "a real dependency wait must keep using the parent-gating path so "
        f"recompute_ready promotes it when the parent finishes; got {task.status!r}"
    )
    assert "dependency_wait" in _kinds(conn, tid)
    assert "blocked" not in _kinds(conn, tid)


def test_dependency_block_with_a_done_parent_does_not_auto_spin(conn):
    parent = kb.create_task(conn, title="upstream", assignee="backend")
    kb.complete_task(conn, parent, summary="done")
    tid = kb.create_task(conn, title="implement", assignee="orchestrator")
    kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))

    kb.block_task(conn, tid, reason="review-handoff: needs a reviewer", kind="dependency")

    task = kb.get_task(conn, tid)
    assert task.status == "blocked", (
        "every parent is already done, so there is nothing to wait for — this "
        f"is a block, not a dependency wait; got {task.status!r}"
    )
