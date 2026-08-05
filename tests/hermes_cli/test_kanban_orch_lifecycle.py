"""M3 T17-T19: lifecycle totality and cancel cascade fencing."""

import pytest

from hermes_cli.kanban_orch_lifecycle import (
    AcceptedRun,
    DeliveryAttempt,
    DeliveryObligation,
    LifecycleError,
    Node,
    Request,
    StageLease,
    StaleCallback,
    SynthesisResult,
    Task,
    TaskRun,
    TRANSITION_TABLE,
    accept_lane_run,
    accept_worker_success,
    apply_transition,
    cancel_cascade,
    cancellation_satisfied,
    complete_cancelled_node,
    finish_cancellation,
    is_terminal,
    optional_cancel_drained,
    request_optional_cancellation,
    supersede_request,
    transition_allowed,
    validate_result_provenance,
)


def test_transition_table_total() -> None:
    """Every §7.2 edge is accepted; terminal block/unblock/retry are rejected."""
    for source, event, target in TRANSITION_TABLE:
        request = Request("board", "tenant", "orch", source)
        resume = source if target == "blocked" else (
            "decomposing" if source == "blocked" else None
        )
        resolved_target = resume if target == "resume_state" else target
        assert transition_allowed(
            source, event, resolved_target,
            resume_state=(resume if source == "blocked" else None),
        )
        applied = apply_transition(
            request, event, resolved_target,
            resume_state=(resume if source == "blocked" else None),
        )
        assert applied.state == resolved_target
        assert applied.lifecycle_revision == 1

    for terminal in ("completed", "failed", "cancelled"):
        for event, target in (("block", "blocked"), ("unblock", "waiting_lanes"), ("retry", "waiting_lanes")):
            assert not transition_allowed(terminal, event, target)
            with pytest.raises(LifecycleError):
                apply_transition(Request("board", "tenant", "orch", terminal), event, target)


def test_optional_cancel_drain() -> None:
    """Unaccepted optional work is fenced, then must reach a terminal node state."""
    nodes = [
        Node("board", "tenant", "orch", "required", True, "accepted"),
        Node("board", "tenant", "orch", "optional-running", False, "running"),
        Node("board", "tenant", "orch", "optional-ready", False, "ready"),
    ]
    assert request_optional_cancellation(nodes) == 2
    assert not optional_cancel_drained(nodes)
    complete_cancelled_node(nodes[1])
    complete_cancelled_node(nodes[2])
    assert optional_cancel_drained(nodes)


def test_cancel_cascade_fencing() -> None:
    """Cancel is exact-scope, fences callbacks, and preserves started-send ambiguity."""
    request = Request("board", "tenant", "orch", "waiting_lanes", lifecycle_revision=7)
    tasks = [
        Task("board", "tenant", "orch", "planned", "planned"),
        Task("board", "tenant", "orch", "running", "running"),
        Task("other", "tenant", "orch", "outside", "planned"),
    ]
    runs = [TaskRun("board", "tenant", "orch", "running", "run-1")]
    nodes = [
        Node("board", "tenant", "orch", "planned-node", True, "planned"),
        Node("board", "tenant", "orch", "running-node", True, "running"),
        Node("other", "tenant", "orch", "outside-node", True, "planned"),
    ]
    leases = [
        StageLease("board", "tenant", "orch", "decomposition"),
        StageLease("other", "tenant", "orch", "decomposition"),
    ]
    obligations = [
        DeliveryObligation("board", "tenant", "orch", "pending", "pending"),
        DeliveryObligation("board", "tenant", "orch", "claimed-no-send", "claimed"),
        DeliveryObligation("board", "tenant", "orch", "claimed-started", "claimed"),
        DeliveryObligation("other", "tenant", "orch", "outside", "pending"),
    ]
    attempts = [DeliveryAttempt("board", "tenant", "orch", "claimed-started", "send-1", "started")]

    result = cancel_cascade(
        request,
        tasks=tasks,
        task_runs=runs,
        nodes=nodes,
        leases=leases,
        obligations=obligations,
        attempts=attempts,
        now=1234,
    )
    assert request.state == "cancelling"
    assert (result.old_revision, result.new_revision) == (7, 8)
    assert (result.old_cancel_epoch, result.new_cancel_epoch) == (0, 1)
    assert tasks[0].status == "cancelled"
    assert tasks[1].cancellation_requested_at == 1234
    assert tasks[2].status == "planned"  # exact-scope fence
    assert runs[0].cancellation_epoch == 1
    assert nodes[0].state == "cancelled"
    assert nodes[1].state == "cancellation_requested"
    assert leases[0].state == "revoked" and leases[0].epoch == 2
    assert leases[1].state == "active"
    assert obligations[0].state == "cancelled"
    assert obligations[1].state == "cancelled"  # no started attempt yet
    assert obligations[2].state == "claimed"
    assert obligations[2].duplicate_possible is True
    assert obligations[3].state == "pending"
    assert result.retained_obligation_ids == ("claimed-started",)

    with pytest.raises(StaleCallback):
        accept_worker_success(request, callback_revision=7, callback_cancel_epoch=0)

    # Drain the active task/node and resolve the started send ambiguity.
    runs[0].ended_at = 1235
    nodes[1].state = "cancelled"
    attempts[0].state = "unknown"
    obligations[2].state = "dead_letter"
    assert cancellation_satisfied(
        request, tasks=tasks, task_runs=runs, nodes=nodes, leases=leases,
        obligations=obligations, attempts=attempts,
    )
    finish_cancellation(
        request, tasks=tasks, task_runs=runs, nodes=nodes, leases=leases,
        obligations=obligations, attempts=attempts,
    )
    assert request.state == "cancelled"
    assert request.lifecycle_revision == 9

    with pytest.raises(StaleCallback):
        accept_worker_success(request, callback_revision=8, callback_cancel_epoch=1)


def test_terminal_state_stored() -> None:
    """Terminal states are stored correctly and no transitions out are accepted."""
    for terminal in ("completed", "failed", "cancelled"):
        request = Request("board", "tenant", "orch", terminal, lifecycle_revision=10)
        assert is_terminal(request.state)
        # No transitions out of terminal
        for event in ("block", "unblock", "retry", "claim_decomposition"):
            assert not transition_allowed(terminal, event, "waiting_lanes")
        with pytest.raises(LifecycleError):
            apply_transition(request, "claim_decomposition", "decomposing")


def test_exact_lane_acceptance() -> None:
    """T15: accepted run current required lane + stable plan epoch."""
    request = Request("board", "tenant", "orch", "waiting_lanes", lifecycle_revision=5)
    node = Node("board", "tenant", "orch", "lane-1", True, "running", plan_version=1)

    # Valid acceptance
    run = accept_lane_run(
        request, node,
        plan_version=1,
        run_id="run-1",
        task_id="task-1",
        outcome_digest="d" * 64,
        callback_revision=5,
        callback_cancel_epoch=0,
    )
    assert node.state == "accepted"
    assert run.plan_version == 1
    assert run.outcome_digest == "d" * 64
    assert run.node_key == "lane-1"

    # Already-accepted node rejected
    with pytest.raises(LifecycleError, match="node_already_accepted"):
        accept_lane_run(
            request, node,
            plan_version=1,
            run_id="run-2",
            task_id="task-2",
            outcome_digest="d" * 64,
            callback_revision=5,
            callback_cancel_epoch=0,
        )

    # Stale plan version rejected
    node2 = Node("board", "tenant", "orch", "lane-2", True, "running", plan_version=2)
    with pytest.raises(LifecycleError, match="stale_plan_version"):
        accept_lane_run(
            request, node2,
            plan_version=1,
            run_id="run-3",
            task_id="task-3",
            outcome_digest="d" * 64,
            callback_revision=5,
            callback_cancel_epoch=0,
        )

    # Wrong request state (not waiting_lanes or synthesizing)
    request2 = Request("board", "tenant", "orch", "submitted")
    node3 = Node("board", "tenant", "orch", "lane-3", True, "running", plan_version=1)
    with pytest.raises(LifecycleError, match="accept_not_authorized"):
        accept_lane_run(
            request2, node3,
            plan_version=1,
            run_id="run-4",
            task_id="task-4",
            outcome_digest="d" * 64,
            callback_revision=0,
            callback_cancel_epoch=0,
        )

    # Stale callback (wrong revision) rejected
    request3 = Request("board", "tenant", "orch", "waiting_lanes", lifecycle_revision=3)
    node4 = Node("board", "tenant", "orch", "lane-4", True, "running", plan_version=1)
    with pytest.raises(StaleCallback):
        accept_lane_run(
            request3, node4,
            plan_version=1,
            run_id="run-5",
            task_id="task-5",
            outcome_digest="d" * 64,
            callback_revision=2,
            callback_cancel_epoch=0,
        )


def test_result_producer_provenance() -> None:
    """T16: synthesis result bound to completed synthesis attempt."""
    result = SynthesisResult(
        board="board",
        tenant="tenant",
        orch_id="orch",
        result_digest="d" * 64,
        synthesis_attempt_id=42,
        plan_version=1,
    )

    # Valid: completed attempt matches
    validate_result_provenance(result, completed_attempt_id=42, current_plan_version=1)

    # Unbound: attempt ID mismatch
    with pytest.raises(LifecycleError, match="result_producer_unbound"):
        validate_result_provenance(result, completed_attempt_id=99, current_plan_version=1)

    # Stale plan version
    with pytest.raises(LifecycleError, match="stale_plan_version"):
        validate_result_provenance(result, completed_attempt_id=42, current_plan_version=2)


def test_supersession_generation() -> None:
    """Supersession: generation+1, new orch_id, predecessor must be terminal."""
    predecessor = Request(
        "board", "tenant", "orch-old", "failed",
        lifecycle_revision=5, generation=1,
    )

    successor = supersede_request(predecessor, new_orch_id="orch-new")
    assert successor.generation == 2
    assert successor.orch_id == "orch-new"
    assert successor.state == "submitted"
    assert successor.lifecycle_revision == 0
    assert successor.cancel_epoch == 0

    # Non-terminal predecessor raises
    active = Request("board", "tenant", "orch-active", "waiting_lanes", generation=1)
    with pytest.raises(LifecycleError, match="supersession_predecessor_not_terminal"):
        supersede_request(active, new_orch_id="orch-new2")

    # Cancelled can also be superseded
    cancelled = Request("board", "tenant", "orch-cancelled", "cancelled", generation=3)
    successor2 = supersede_request(cancelled, new_orch_id="orch-new3")
    assert successor2.generation == 4
