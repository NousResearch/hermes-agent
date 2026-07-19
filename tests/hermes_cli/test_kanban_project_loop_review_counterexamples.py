from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_project_loop as project_loop
from hermes_cli.kanban_project_loop import mark_launched_rounds


def _configured_loop(conn, tmp_path):
    verify = kb.create_task(
        conn,
        title="Final owner verify",
        assignee="default",
        workspace_kind="dir",
        workspace_path=str(tmp_path),
    )
    kb.configure_project_loop(
        conn,
        project_key="expert-rollout",
        goal="Ship the expert workflow",
        acceptance_criteria=["All blockers are closed"],
        verify_task_id=verify,
        max_rounds=3,
        max_tasks=12,
    )
    return verify


def test_orphan_running_run_does_not_activate_awaiting_round(tmp_path):
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="continue_bounded",
            next_steps=[{"key": "retry", "title": "Retry launch"}],
        )
        work = next(
            task
            for task in kb.list_project_loop_round_tasks(conn, "expert-rollout")
            if task.role == "work"
        )
        conn.execute(
            "INSERT INTO task_runs "
            "(task_id, profile, status, worker_pid, started_at) "
            "VALUES (?, 'default', 'running', 4321, 1)",
            (work.task_id,),
        )

        assert mark_launched_rounds(conn) == 0
        assert kb.get_project_loop(conn, "expert-rollout").status == "awaiting_launch"


def test_current_verify_rejects_missing_outcome_before_completion(tmp_path):
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)

        with pytest.raises(
            ValueError, match="project-loop Verify completion requires outcome"
        ):
            kb.complete_task(conn, verify)

        assert kb.get_task(conn, verify).status == "ready"
        assert kb.get_project_loop(conn, "expert-rollout").status == "active"


def test_current_verify_rejects_invalid_outcome_before_completion(tmp_path):
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)

        with pytest.raises(
            ValueError, match="project-loop Verify completion requires outcome"
        ):
            kb.complete_task(conn, verify, outcome="completed")

        assert kb.get_task(conn, verify).status == "ready"
        assert kb.get_project_loop(conn, "expert-rollout").status == "active"


def test_verify_completion_rolls_back_when_reconcile_fails_then_retries(
    tmp_path, monkeypatch
):
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        claimed = kb.claim_task(conn, verify, claimer="review-counterexample")
        assert claimed is not None
        run_id = kb.get_task(conn, verify).current_run_id
        assert run_id is not None
        original_store_receipt = project_loop._store_receipt

        def fail_receipt(*_args, **_kwargs):
            raise RuntimeError("simulated reconcile failure")

        monkeypatch.setattr(project_loop, "_store_receipt", fail_receipt)
        with pytest.raises(RuntimeError, match="simulated reconcile failure"):
            kb.complete_task(
                conn,
                verify,
                outcome="continue_bounded",
                metadata={
                    "project_loop": {
                        "next_steps": [{"key": "retry", "title": "Retry launch"}]
                    }
                },
                expected_run_id=run_id,
            )

        task = kb.get_task(conn, verify)
        assert task.status == "running"
        assert task.current_run_id == run_id
        run = kb.latest_run(conn, verify)
        assert run.status == "running"
        assert run.ended_at is None
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "active"
        assert state.rounds_used == 0
        assert kb.list_project_loop_round_tasks(conn, "expert-rollout") == []
        assert conn.execute(
            "SELECT COUNT(*) FROM project_loop_reconciliations"
        ).fetchone()[0] == 0

        monkeypatch.setattr(project_loop, "_store_receipt", original_store_receipt)
        assert kb.complete_task(
            conn,
            verify,
            outcome="continue_bounded",
            metadata={
                "project_loop": {
                    "next_steps": [{"key": "retry", "title": "Retry launch"}]
                }
            },
            expected_run_id=run_id,
        )
        assert kb.get_task(conn, verify).status == "done"
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "awaiting_launch"
        assert state.rounds_used == 1


def test_owner_gate_completion_rolls_back_when_reconcile_fails_then_retries(
    tmp_path, monkeypatch
):
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        gate = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="owner_judgment_required",
        ).owner_gate_task_id
        assert gate is not None
        original_store_receipt = project_loop._store_receipt

        def fail_receipt(*_args, **_kwargs):
            raise RuntimeError("simulated owner reconcile failure")

        monkeypatch.setattr(project_loop, "_store_receipt", fail_receipt)
        with pytest.raises(RuntimeError, match="simulated owner reconcile failure"):
            kb.complete_task(conn, gate, outcome="goal_complete")

        assert kb.get_task(conn, gate).status == "blocked"
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "owner_gate"
        assert state.current_owner_gate_task_id == gate

        monkeypatch.setattr(project_loop, "_store_receipt", original_store_receipt)
        assert kb.complete_task(conn, gate, outcome="goal_complete")
        assert kb.get_task(conn, gate).status == "done"
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "complete"
        assert state.current_owner_gate_task_id is None
