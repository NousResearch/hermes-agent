"""Sequential routing stops at finite boundaries and trusts only host verification."""

import json

import pytest

from agent.engineering_workflow import (
    WorkflowLimits,
    WorkerOutcome,
    run_engineering_workflow,
)


PLAN = json.dumps({
    "objective": "Repair parser",
    "constraints": ["Preserve API"],
    "steps": ["Edit parser"],
    "acceptance_criteria": ["Unit check passes"],
})
ROUTES = {
    stage: {"provider": "provider-a", "model": model}
    for stage, model in (
        ("planner", "plan"),
        ("worker", "work"),
        ("reviewer", "review"),
    )
}
CATALOGUE = {
    "providers": [
        {
            "slug": "provider-a",
            "models": ["plan", "work", "review"],
            "authenticated": True,
        }
    ]
}


def harness(
    exit_codes,
    *,
    limits=WorkflowLimits(),
    catalogue=None,
    worker_outcome=None,
    worker_turns=0,
    digests=None,
    feedback="bounded worker feedback",
    plans=None,
    stop_after_worker=False,
):
    stages = []
    stopped_state = {"requested": False}
    plan_answers = iter(plans) if plans else None
    receipts_seen = []
    catalogue_state = catalogue or [CATALOGUE]

    def infer(stage, route, payload):
        stages.append((stage, route.model, payload))
        return (
            (next(plan_answers) if plan_answers else PLAN)
            if stage != "worker"
            else '{"status":"READY"}'
        )

    def execute_worker(worker_text, context, next_worker, plan):
        assert worker_text == '{"status":"READY"}'
        for _ in range(worker_turns):
            next_worker(feedback)
        stopped_state["requested"] = stop_after_worker
        return worker_outcome or WorkerOutcome(status="READY", summary="edited")

    def verify(context):
        receipts_seen.append(context)
        from agent.engineering_workflow import VerificationReceipt

        return [
            VerificationReceipt(
                run_id=context.run_id,
                workspace_id=context.workspace_id,
                attempt_id=context.attempt_id,
                revision=context.revision,
                snapshot_digest=context.snapshot_digest,
                check_id="unit",
                exit_code=exit_codes[min(len(receipts_seen) - 1, len(exit_codes) - 1)],
                complete=True,
                timed_out=False,
            )
        ]

    result = run_engineering_workflow(
        objective="Repair parser",
        assignments=ROUTES,
        catalogue_reader=lambda: catalogue_state[
            min(len(stages), len(catalogue_state) - 1)
        ],
        infer=infer,
        execute_worker=execute_worker,
        verify=verify,
        snapshot_digest=iter(digests).__next__ if digests else lambda: "snapshot-1",
        workspace_id="ws-1",
        check_ids=("unit",),
        limits=limits,
        stop_requested=lambda: stopped_state["requested"],
    )
    return result, stages, receipts_seen


def test_failed_check_enters_reviewer_then_new_worker_attempt():
    result, stages, receipts = harness([1, 0])
    assert result.status == "DONE"
    assert [stage for stage, _, _ in stages] == [
        "planner",
        "worker",
        "reviewer",
        "worker",
    ]
    assert [item.revision for item in receipts] == [1, 2]
    assert receipts[0].attempt_id != receipts[1].attempt_id
    assert len(stages) == result.stage_calls == 4


@pytest.mark.parametrize(
    "limits,expected_stages",
    [
        (
            WorkflowLimits(max_attempts=1, max_replans=2, max_stage_calls=10),
            ["planner", "worker"],
        ),
        (
            WorkflowLimits(max_attempts=3, max_replans=0, max_stage_calls=10),
            ["planner", "worker"],
        ),
        (
            WorkflowLimits(max_attempts=3, max_replans=2, max_stage_calls=2),
            ["planner", "worker"],
        ),
    ],
)
def test_all_retry_and_stage_budgets_stop_before_extra_model_calls(
    limits, expected_stages
):
    result, stages, _ = harness([1, 0], limits=limits)
    assert result.status == "STOP"
    assert [stage for stage, _, _ in stages] == expected_stages


def test_worker_design_blocker_never_runs_verification_or_reviewer():
    result, stages, receipts = harness(
        [0],
        worker_outcome=WorkerOutcome(
            status="BLOCKED",
            summary="Persistence contract conflicts",
            decision_required="Allow schema migration?",
        ),
    )
    assert result.status == "BLOCKED"
    assert result.decision_required == "Allow schema migration?"
    assert [stage for stage, _, _ in stages] == ["planner", "worker"]
    assert receipts == []


def test_picker_change_after_admission_blocks_before_worker_inference():
    old = CATALOGUE
    removed = {
        "providers": [{**CATALOGUE["providers"][0], "models": ["plan", "review"]}]
    }
    result, stages, receipts = harness([0], catalogue=[old, removed])
    assert result.status == "BLOCKED"
    assert [stage for stage, _, _ in stages] == ["planner"]
    assert receipts == []


def test_worker_round_trip_cannot_bypass_total_stage_budget():
    result, stages, receipts = harness(
        [0],
        limits=WorkflowLimits(max_stage_calls=2),
        worker_turns=1,
    )
    assert result.status == "STOP"
    assert [stage for stage, _, _ in stages] == ["planner", "worker"]
    assert receipts == []


def test_source_change_after_receipt_blocks_done():
    result, stages, receipts = harness([0], digests=["snapshot-1", "snapshot-2"])
    assert result.status == "BLOCKED"
    assert [stage for stage, _, _ in stages] == ["planner", "worker"]
    assert len(receipts) == 1


def test_oversized_worker_reentry_cannot_expand_handoff():
    result, stages, receipts = harness([0], worker_turns=1, feedback="x" * 20_000)
    assert result.status == "BLOCKED"
    assert [stage for stage, _, _ in stages] == ["planner", "worker"]
    assert receipts == []


def test_planner_cannot_change_operator_objective():
    changed = json.dumps({**json.loads(PLAN), "objective": "Different task"})
    result, stages, receipts = harness([0], plans=[changed])
    assert result.status == "BLOCKED"
    assert result.reason == "objective_changed"
    assert [stage for stage, _, _ in stages] == ["planner"]
    assert receipts == []


def test_reviewer_cannot_change_operator_objective():
    changed = json.dumps({**json.loads(PLAN), "objective": "Different task"})
    result, stages, receipts = harness([1, 0], plans=[PLAN, changed])
    assert result.status == "BLOCKED"
    assert result.reason == "objective_changed"
    assert [stage for stage, _, _ in stages] == ["planner", "worker", "reviewer"]
    assert len(receipts) == 1


@pytest.mark.parametrize(
    "limits",
    [
        WorkflowLimits(max_attempts=6),
        WorkflowLimits(max_replans=5),
        WorkflowLimits(max_stage_calls=65),
    ],
)
def test_unreasonably_large_budget_is_rejected(limits):
    with pytest.raises(ValueError):
        harness([0], limits=limits)


def test_interrupt_after_worker_stops_before_host_check():
    result, stages, receipts = harness([0], stop_after_worker=True)
    assert result.status == "STOP"
    assert result.reason == "interrupt"
    assert [stage for stage, _, _ in stages] == ["planner", "worker"]
    assert receipts == []
