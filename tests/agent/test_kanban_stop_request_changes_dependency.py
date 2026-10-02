from __future__ import annotations

import pytest

from agent.kanban_stop import KanbanStopTarget, kanban_stop_target
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.mark.parametrize(
    ("add_dependency", "expected_status"),
    [(False, "ready"), (True, "todo")],
    ids=("ready-landing", "dependency-gated-todo-landing"),
)
def test_request_changes_terminal_handoff_accepts_real_landing_status(
    tmp_path, monkeypatch, add_dependency, expected_status,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="reviewed work", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:regression")
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready for review",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:regression")
        assert review is not None

        if add_dependency:
            parent_id = kb.create_task(
                conn,
                title="unfinished prerequisite",
                assignee="builder",
                initial_status="blocked",
            )
            kb.link_tasks(
                conn,
                parent_id,
                task_id,
                expected_child_run_id=review.current_run_id,
            )
            assert conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, task_id),
            ).fetchone()[0] == 1

        assert kb.get_task(conn, task_id).status == "running"
        ok, detail = kb.request_changes(
            conn,
            task_id,
            reason="review requests another implementation pass",
            expected_run_id=review.current_run_id,
        )
        assert ok, detail

        landed = kb.get_task(conn, task_id)
        run = kb.get_run(conn, review.current_run_id)
        assert landed.status == expected_status
        assert landed.current_run_id is None
        assert run.outcome == "changes_requested"
        assert run.ended_at is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    target = kanban_stop_target()
    assert target == KanbanStopTarget(
        task_id=task_id,
        run_id=review.current_run_id,
        status=expected_status,
        terminal_handoff_accepted=True,
    )
