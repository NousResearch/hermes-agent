"""Gate: empty-evidence parent completion refused while open, evidence-less
children exist (t_ca1c4a33 / loop-gate-noop).

``kanban_db.complete_task`` must refuse a completion whose own evidence
(``result`` + ``summary``) is empty while >=1 child is neither ``done`` nor
``archived`` and itself has no evidence (blank task-row ``result`` and blank
latest ``task_runs.summary``), naming every blocking child. Carve-outs:
``metadata['superseded_by']`` and a non-empty ``metadata['allow_open_children']``
reason. Non-empty parent evidence, all-done children, and childless tasks
never gate. All tests run on a scratch ``kanban.db`` fixture — never the
live ``~/.hermes/kanban.db``.
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


def _parent_with_open_empty_child(conn):
    parent_id = kb.create_task(conn, title="parent", assignee="planner")
    # Parent still open -> child parks in ``todo`` with no runs, no result.
    child_id = kb.create_task(conn, title="child", assignee="builder", parents=[parent_id])
    assert kb.get_task(conn, child_id).status == "todo"
    return parent_id, child_id


def test_open_child_with_no_evidence_blocks_empty_completion(conn):
    parent_id, child_id = _parent_with_open_empty_child(conn)

    with pytest.raises(kb.OpenChildrenError) as excinfo:
        kb.complete_task(conn, parent_id)
    msg = str(excinfo.value)
    assert child_id in msg
    assert f"{child_id} (todo)" in msg
    assert parent_id in msg

    parent = kb.get_task(conn, parent_id)
    assert parent is not None and parent.status != "done"


def test_allow_open_children_metadata_override_passes(conn):
    parent_id, _child_id = _parent_with_open_empty_child(conn)

    assert kb.complete_task(
        conn, parent_id,
        metadata={"allow_open_children": "children will start next tick"},
    )
    assert kb.get_task(conn, parent_id).status == "done"


def test_superseded_by_metadata_override_passes(conn):
    parent_id, _child_id = _parent_with_open_empty_child(conn)

    assert kb.complete_task(conn, parent_id, metadata={"superseded_by": "t_deadbeef"})
    assert kb.get_task(conn, parent_id).status == "done"


def test_all_children_done_allows_legitimate_empty_close(conn):
    # Build the all-done graph the only order the kernel allows: close the
    # childless parent, add + close the child, then reopen the parent so the
    # empty close is replayed against an open parent whose children are all done.
    parent_id = kb.create_task(conn, title="parent", assignee="planner")
    assert kb.complete_task(conn, parent_id)  # no children yet -> never gated
    child_id = kb.create_task(conn, title="child", assignee="builder", parents=[parent_id])
    assert kb.complete_task(conn, child_id)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', completed_at = NULL WHERE id = ?",
            (parent_id,),
        )

    assert kb.complete_task(conn, parent_id) is True
    assert kb.get_task(conn, parent_id).status == "done"


def test_nonempty_parent_evidence_never_rejected(conn):
    parent_id, _child_id = _parent_with_open_empty_child(conn)

    assert kb.complete_task(conn, parent_id, result="real evidence") is True
    assert kb.get_task(conn, parent_id).status == "done"


def test_integration_scratch_db_lists_both_children_and_schema_unchanged(conn):
    parent_id = kb.create_task(conn, title="parent", assignee="planner")
    child_a = kb.create_task(conn, title="child a", assignee="builder", parents=[parent_id])
    child_b = kb.create_task(
        conn, title="child b", assignee="writer", parents=[parent_id],
        initial_status="blocked",
    )

    def schema():
        return [
            tuple(row) for row in conn.execute(
                "SELECT name, sql FROM sqlite_master ORDER BY name"
            ).fetchall()
        ]

    schema_before = schema()

    with pytest.raises(kb.OpenChildrenError) as excinfo:
        kb.complete_task(conn, parent_id)
    msg = str(excinfo.value)
    assert f"{child_a} (todo)" in msg
    assert f"{child_b} (blocked)" in msg
    assert msg.index(child_a) != msg.index(child_b)  # both listed, not one fused id

    assert kb.complete_task(
        conn, parent_id,
        metadata={"allow_open_children": "supersede-close; children re-scoped next tick"},
    )
    assert kb.get_task(conn, parent_id).status == "done"

    # No migration: gate is additive logic only.
    assert schema() == schema_before
