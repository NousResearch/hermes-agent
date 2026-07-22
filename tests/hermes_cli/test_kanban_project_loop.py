from __future__ import annotations

from hermes_cli import kanban_db as kb


def _board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return path


def _configured_loop(conn, tmp_path, *, max_rounds=3, max_tasks=12):
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
        goal="Ship the expert workflow without unresolved blockers",
        acceptance_criteria=[
            "All expert blockers are closed",
            "A real dispatched run is observed for every continuation round",
        ],
        verify_task_id=verify,
        max_rounds=max_rounds,
        max_tasks=max_tasks,
    )
    return verify


def _three_blockers():
    return [
        {"key": "source-gap", "title": "Close source coverage gap", "assignee": "default"},
        {"key": "qa-gap", "title": "Close QA evidence gap", "assignee": "default"},
        {"key": "ops-gap", "title": "Close operator handoff gap", "assignee": "default"},
    ]


def test_verify_blockers_create_and_dispatch_bounded_next_round(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    spawned = []

    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.complete_task(
            conn,
            verify,
            outcome="continue_bounded",
            metadata={"project_loop": {"next_steps": _three_blockers()}},
        )

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "awaiting_launch"
        assert state.rounds_used == 1
        assert state.tasks_created == 4

        round_tasks = kb.list_project_loop_round_tasks(
            conn, "expert-rollout", round_no=1
        )
        blockers = [task for task in round_tasks if task.role == "work"]
        next_verify = [task for task in round_tasks if task.role == "verify"]
        assert {task.step_key for task in blockers} == {
            "source-gap",
            "qa-gap",
            "ops-gap",
        }
        assert len(next_verify) == 1
        assert set(kb.parent_ids(conn, next_verify[0].task_id)) == {
            task.task_id for task in blockers
        }

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4321,
            max_spawn=3,
        )
        assert len(result.spawned) == 1
        first = result.spawned[0][0]
        assert first in {task.task_id for task in blockers}
        assert kb.complete_task(
            conn, first, expected_run_id=kb.get_task(conn, first).current_run_id
        )
        second_result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 4322,
            max_spawn=3,
        )
        assert len(second_result.spawned) == 1
        assert second_result.spawned[0][0] != first
        assert kb.get_project_loop(conn, "expert-rollout").status == "active"


def test_project_loop_reconcile_is_idempotent(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        first = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="continue_bounded",
            next_steps=_three_blockers(),
        )
        second = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="continue_bounded",
            next_steps=_three_blockers(),
        )

        assert first.created_task_ids
        assert second.created_task_ids == first.created_task_ids
        assert len(kb.list_project_loop_round_tasks(conn, "expert-rollout")) == 4
        assert kb.get_project_loop(conn, "expert-rollout").rounds_used == 1


def test_continue_without_next_steps_fails_closed(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.complete_task(
            conn,
            verify,
            outcome="continue_bounded",
            metadata={"project_loop": {"next_steps": []}},
        )

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.last_decision == "stop"
        assert state.stop_reason == "continue_bounded requires at least one next step"
        assert kb.list_project_loop_round_tasks(conn, "expert-rollout") == []


def test_malformed_next_steps_fail_closed(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.complete_task(
            conn,
            verify,
            outcome="continue_bounded",
            metadata={"project_loop": {"next_steps": [{"key": "missing-title"}]}},
        )

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.last_decision == "stop"
        assert state.stop_reason.startswith("invalid continue_bounded next steps:")


def test_owner_judgment_creates_exactly_one_gate(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        first = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="owner_judgment_required",
            owner_question="Choose whether the residual risk is acceptable.",
        )
        second = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="owner_judgment_required",
            owner_question="Choose whether the residual risk is acceptable.",
        )

        assert first.owner_gate_task_id == second.owner_gate_task_id
        gate = kb.get_task(conn, first.owner_gate_task_id)
        assert gate is not None
        assert gate.status == "blocked"
        assert gate.block_kind == "needs_input"
        gates = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            ("project-loop:expert-rollout:owner-gate:0",),
        ).fetchall()
        assert len(gates) == 1


def _create_owner_gate(conn, tmp_path):
    verify = _configured_loop(conn, tmp_path)
    result = kb.reconcile_project_loop(
        conn,
        project_key="expert-rollout",
        verify_task_id=verify,
        decision="owner_judgment_required",
        owner_question="Choose whether the residual risk is acceptable.",
    )
    assert result.owner_gate_task_id is not None
    return result.owner_gate_task_id


def test_owner_gate_goal_complete_is_consumed_once(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        assert kb.complete_task(conn, gate, outcome="goal_complete")
        first = kb.reconcile_owner_gate(
            conn, gate_task_id=gate, decision="goal_complete"
        )
        second = kb.reconcile_owner_gate(conn, gate_task_id=gate, decision="stop")

        assert first == second
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "complete"
        assert state.current_owner_gate_task_id is None


def test_owner_gate_stop_resumes_loop_to_stopped(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        assert kb.complete_task(conn, gate, outcome="stop")

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.last_decision == "stop"
        assert state.current_owner_gate_task_id is None


def test_owner_gate_continue_preserves_budgets_and_next_steps_guard(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        assert kb.complete_task(
            conn,
            gate,
            outcome="continue_bounded",
            metadata={"project_loop": {"next_steps": _three_blockers()}},
        )

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "awaiting_launch"
        assert state.rounds_used == 1
        assert state.tasks_created == 4

    fail_closed_path = tmp_path / "fail-closed.db"
    kb._INITIALIZED_PATHS.discard(str(fail_closed_path.resolve()))
    with kb.connect(fail_closed_path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        assert kb.complete_task(conn, gate, outcome="continue_bounded")
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.stop_reason == "continue_bounded requires at least one next step"


def test_owner_gate_rejects_non_decision_outcome_without_consuming(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        try:
            kb.complete_task(conn, gate, outcome="completed")
        except ValueError as exc:
            assert "owner gate completion requires outcome" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("invalid owner decision was accepted")
        assert kb.get_task(conn, gate).status == "blocked"
        assert kb.get_project_loop(conn, "expert-rollout").status == "owner_gate"


def test_spawn_failure_does_not_activate_awaiting_round(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="continue_bounded",
            next_steps=[
                {"key": "retry", "title": "Retry launch", "assignee": "default"}
            ],
        )

        def fail_spawn(_task, _workspace):
            raise RuntimeError("spawn failed")

        kb.dispatch_once(conn, spawn_fn=fail_spawn, max_spawn=1)

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "awaiting_launch"
        run = conn.execute(
            "SELECT * FROM task_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["outcome"] == "spawn_failed"


def test_round_and_task_budgets_stop_instead_of_looping(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path, max_rounds=1, max_tasks=3)
        result = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="continue_bounded",
            next_steps=_three_blockers(),
        )

        assert result.decision == "stop"
        assert result.created_task_ids == []
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.rounds_used == 0
        assert state.tasks_created == 0
        assert "task budget" in state.stop_reason


def test_goal_complete_is_the_only_successful_terminal_close(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        result = kb.reconcile_project_loop(
            conn,
            project_key="expert-rollout",
            verify_task_id=verify,
            decision="goal_complete",
        )

        assert result.decision == "goal_complete"
        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "complete"
        assert state.last_decision == "goal_complete"


def test_terminal_project_loop_cannot_be_reconfigured_or_reconciled(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.complete_task(conn, verify, outcome="stop")
        replacement = kb.create_task(conn, title="replacement Verify")

        try:
            kb.configure_project_loop(
                conn,
                project_key="expert-rollout",
                goal="silently reopen",
                acceptance_criteria=["must not happen"],
                verify_task_id=replacement,
                max_rounds=3,
                max_tasks=12,
            )
        except ValueError as exc:
            assert "terminal loops cannot be reconfigured" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("terminal project loop was reconfigured")

        try:
            kb.reconcile_project_loop(
                conn,
                project_key="expert-rollout",
                verify_task_id=verify,
                decision="continue_bounded",
                next_steps=_three_blockers(),
                _receipt_task_id="late-reconcile",
            )
        except ValueError as exc:
            assert "only active loops can reconcile" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("terminal project loop continued")


def test_archiving_current_verify_stops_loop_and_clears_control_reference(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.archive_task(conn, verify)

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.current_verify_task_id is None
        assert "control task archived" in state.stop_reason


def test_failed_repeat_archive_does_not_mutate_terminal_loop(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        verify = _configured_loop(conn, tmp_path)
        assert kb.archive_task(conn, verify)
        before = kb.get_project_loop(conn, "expert-rollout")

        assert not kb.archive_task(conn, verify)
        assert kb.get_project_loop(conn, "expert-rollout") == before


def test_deleting_current_owner_gate_stops_loop_and_cleans_relations(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        gate = _create_owner_gate(conn, tmp_path)
        assert kb.delete_task(conn, gate)

        state = kb.get_project_loop(conn, "expert-rollout")
        assert state.status == "stopped"
        assert state.current_owner_gate_task_id is None
        assert "control task deleted" in state.stop_reason
        assert conn.execute(
            "SELECT COUNT(*) FROM project_loop_tasks WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0


def test_ordinary_card_does_not_pay_project_loop_tax(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        task_id = kb.create_task(conn, title="ordinary single-round card")
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        assert kb.complete_task(
            conn,
            task_id,
            outcome="continue_bounded",
            metadata={"project_loop": {"next_steps": _three_blockers()}},
        )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
        assert conn.execute("SELECT COUNT(*) FROM project_loops").fetchone()[0] == 0
