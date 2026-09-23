"""Contracts for the sequential engineering workflow's handoffs and host receipts."""

import json

import pytest

from agent.engineering_workflow import (
    HandoffError,
    ReceiptError,
    VerificationContext,
    VerificationReceipt,
    parse_handoff,
    verify_receipts,
)


PLAN = {
    "objective": "Fix the failing parser",
    "constraints": ["Keep the public interface"],
    "steps": ["Add a regression test", "Repair the parser"],
    "acceptance_criteria": ["The parser suite passes"],
}


def test_handoff_accepts_only_bounded_structured_plan():
    plan = parse_handoff(json.dumps(PLAN))
    assert plan.objective == PLAN["objective"]
    assert plan.steps == tuple(PLAN["steps"])
    assert plan.acceptance_criteria == tuple(PLAN["acceptance_criteria"])


@pytest.mark.parametrize("payload", [
    '{"objective":"one","objective":"two","constraints":[],"steps":[],"acceptance_criteria":[]}',
    json.dumps({**PLAN, "api_key": "synthetic-secret"}),
    json.dumps({**PLAN, "steps": "Repair the parser"}),
    json.dumps({**PLAN, "status": "DONE"}),
    json.dumps({**PLAN, "steps": []}),
    "{" + " " * 20_000 + "}",
    "not json",
])
def test_handoff_rejects_ambiguous_or_unbounded_data(payload):
    with pytest.raises(HandoffError):
        parse_handoff(payload)


CTX = VerificationContext(
    run_id="run-1", workspace_id="ws-1", attempt_id="try-1", revision=2,
    snapshot_digest="abc123", check_ids=("unit", "lint"),
)


def receipt(check_id, **changes):
    values = dict(
        run_id=CTX.run_id, workspace_id=CTX.workspace_id,
        attempt_id=CTX.attempt_id, revision=CTX.revision,
        snapshot_digest=CTX.snapshot_digest, check_id=check_id,
        exit_code=0, complete=True, timed_out=False,
    )
    values.update(changes)
    return VerificationReceipt(**values)


def test_host_receipts_must_match_the_exact_attempt_and_snapshot():
    assert verify_receipts(CTX, [receipt("unit"), receipt("lint")]) is True
    assert verify_receipts(CTX, [receipt("unit", exit_code=1), receipt("lint")]) is False


@pytest.mark.parametrize("bad", [
    [receipt("unit")],
    [receipt("unit"), receipt("unit"), receipt("lint")],
    [receipt("unit", run_id="foreign"), receipt("lint")],
    [receipt("unit", workspace_id="foreign"), receipt("lint")],
    [receipt("unit", attempt_id="old"), receipt("lint")],
    [receipt("unit", revision=1), receipt("lint")],
    [receipt("unit", snapshot_digest="old"), receipt("lint")],
    [receipt("unit", timed_out=True), receipt("lint")],
    [receipt("unit", complete=False), receipt("lint")],
    [receipt("unit", exit_code=True), receipt("lint")],
    [receipt("unit", exit_code="0"), receipt("lint")],
])
def test_host_receipts_fail_closed_on_missing_stale_or_invalid_proof(bad):
    with pytest.raises(ReceiptError):
        verify_receipts(CTX, bad)

from agent.engineering_workflow import ModelRouteError, admit_stage_routes, revalidate_route


ROUTES = {
    "planner": {"provider": "provider-a", "model": "plan-model"},
    "worker": {"provider": "provider-a", "model": "work-model"},
    "reviewer": {"provider": "provider-b", "model": "review-model"},
}
CATALOGUE = {
    "providers": [
        {"slug": "provider-a", "models": ["plan-model", "work-model"], "authenticated": True},
        {"slug": "provider-b", "models": ["review-model"], "authenticated": True},
    ],
}


def test_stage_routes_only_admit_exact_available_picker_pairs():
    routes = admit_stage_routes(ROUTES, CATALOGUE)
    assert routes["planner"].model == "plan-model"
    assert routes["reviewer"].provider == "provider-b"
    for stage in routes:
        revalidate_route(routes[stage], CATALOGUE)


@pytest.mark.parametrize("changed", [
    {"providers": CATALOGUE["providers"][1:]},
    {"providers": [{**CATALOGUE["providers"][0], "models": ["work-model"]}, CATALOGUE["providers"][1]]},
    {"providers": [{**CATALOGUE["providers"][0], "authenticated": False}, CATALOGUE["providers"][1]]},
    {"providers": [{**CATALOGUE["providers"][0], "unavailable_models": ["plan-model"]}, CATALOGUE["providers"][1]]},
])
def test_admitted_model_is_rechecked_before_each_stage(changed):
    route = admit_stage_routes(ROUTES, CATALOGUE)["planner"]
    with pytest.raises(ModelRouteError):
        revalidate_route(route, changed)


@pytest.mark.parametrize("invalid", [
    {**ROUTES, "planner": {"provider": "provider-a", "model": "invented"}},
    {**ROUTES, "planner": {"provider": "auto", "model": "plan-model"}},
    {**ROUTES, "planner": {"provider": "moa", "model": "plan-model"}},
    {**ROUTES, "planner": {"provider": "provider-a", "model": "plan-model", "api_key": "x"}},
    {"worker": ROUTES["worker"], "reviewer": ROUTES["reviewer"]},
])
def test_stage_route_input_cannot_invent_or_redirect_model(invalid):
    with pytest.raises(ModelRouteError):
        admit_stage_routes(invalid, CATALOGUE)
