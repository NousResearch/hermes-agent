import pytest

from agent.crew_route_contract import (
    ROUTE_POLICY_VERSION,
    RouteContractError,
    WorkIdentity,
    resolve_route,
    sunny_routes,
    validate_work_batch,
)


def _work(**overrides):
    values = {
        "bucket_key": "coding",
        "pool_member": "zoro",
        "business": "sunny",
        "account": "crew-primary",
        "board": "board-a",
        "project": "project-a",
        "worktree": "/worktrees/a",
        "idempotency_key": "idem-a",
        "parent_link": "parent-a",
        "route_policy_version": ROUTE_POLICY_VERSION,
        "provenance": "firstmate:approved-plan",
    }
    values.update(overrides)
    return values


def test_routes_are_exact_inert_identities_and_never_clamp_effort():
    assert sunny_routes()["validated-plan-worker"].as_tuple() == (
        "openai-codex", "gpt-5.6-sol", "low",
    )
    invalid = (
        {"provider": "unknown", "model_id": "gpt-5.6-sol", "effort": "low"},
        {"provider": "openai-codex", "model_id": "Hermes", "effort": "xhigh"},
        {"provider": "openai-codex", "model_id": "gpt-5.5", "effort": "max"},
        {"provider": "openai-codex", "model_id": "gpt-5.6-sol"},
        {"provider": "openai-codex", "model_id": "gpt-5.6-sol",
         "model": "gpt-5.6-luna", "effort": "low"},
    )
    for route in invalid:
        with pytest.raises(RouteContractError):
            resolve_route(route)


def test_work_provenance_and_scope_fail_closed():
    item = WorkIdentity.from_mapping(_work())
    assert validate_work_batch([item], allowed_parent_links={"parent-a"}) == (item,)
    with pytest.raises(RouteContractError, match="untrusted work provenance"):
        WorkIdentity.from_mapping(_work(provenance="caller:claimed"))
    other = WorkIdentity.from_mapping(_work(idempotency_key="idem-b", business="other"))
    with pytest.raises(RouteContractError, match="cross-scope"):
        validate_work_batch([item, other], allowed_parent_links={"parent-a"})
