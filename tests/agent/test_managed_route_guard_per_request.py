"""Unit coverage for the common per-request managed-route guard
(``agent.managed_route_guard``) — the seam ``begin_iteration`` calls before
EVERY iteration of a managed turn's tool loop, not just the turn's first
request (that first-request check is cli.py's
``_enforce_kanban_routing_receipt``, already covered by
``tests/hermes_cli/test_kanban_worker_route_enforcement.py``).

Pure/unit here (real receipt store, in-memory `agent` stand-in, no network):
proves the guard's own decision logic — no-op when unwired, blocks on
revoked policy / diverged route, passes through a matching route. The real
provider-boundary + no-leak behavior (this guard actually stopping a live
HTTP request) is covered by
``tests/hermes_cli/test_kanban_worker_per_request_guard_integration.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.model_selection import select
from agent.model_selection_store import activate_policy, persist_receipt, publish_policy


def _policy(route_id="fake-route", model="fake-model", provider="custom-fake", endpoint="http://x/v1"):
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": route_id, "route_revision": 1, "provider": provider,
            "model": model, "endpoint": endpoint, "maker": "test",
            "model_family": model, "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": [route_id], "shallow": [route_id]}},
    }


def _requirements(execution_id="t_guard_unit"):
    return {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": execution_id, "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 0, "reserve_tokens": 0, "reasoning": "high",
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }


def _agent(receipt_id, hermes_home, *, provider="custom-fake", model="fake-model",
           endpoint="http://x/v1", reasoning_config=None):
    return SimpleNamespace(
        _managed_routing_receipt_id=receipt_id, _managed_routing_home=str(hermes_home),
        requested_provider=provider, provider=provider, model=model, base_url=endpoint,
        reasoning_config=reasoning_config or {"enabled": True, "effort": "high"},
    )


def _receipted(hermes_home, **route_kwargs):
    policy = _policy(**route_kwargs)
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    decision = select(_requirements(), policy, {}, now=1000)
    return persist_receipt(hermes_home, decision), policy, record["revision"]


def test_unwired_agent_is_a_complete_noop(tmp_path):
    from agent.managed_route_guard import enforce_managed_route_per_request

    agent = SimpleNamespace()  # no _managed_routing_receipt_id at all
    assert enforce_managed_route_per_request(agent) is None


def test_matching_route_passes_every_iteration(tmp_path):
    from agent.managed_route_guard import enforce_managed_route_per_request

    receipt_id, _, _ = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    for _ in range(3):  # simulate several tool-loop iterations of the same turn
        assert enforce_managed_route_per_request(agent) is None


def test_routine_policy_edit_after_pin_never_blocks_the_live_run(tmp_path):
    """A ROUTINE policy edit (publish + activate a new revision, e.g. a requalification or
    an unrelated route tweak, WITHOUT an explicit emergency revocation) after request 1 must
    NOT block request 2 of the same live run: pinned routes affect only NEW attempts, not the
    live one (plan §12/§13; root AGENTS binding note on emergency revocation semantics)."""
    from agent.managed_route_guard import enforce_managed_route_per_request

    receipt_id, policy, revision = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    assert enforce_managed_route_per_request(agent) is None  # request 1: fine

    # Routine edit: publish + activate a new revision of the SAME route, no explicit
    # revocation call. This must be transparent to the already-pinned live run.
    new_policy = dict(policy)
    new_policy["revision"] = revision + 1
    record = publish_policy(tmp_path, new_policy, approval_ref="operator:routine-edit")
    activate_policy(tmp_path, "kanban-default", record["revision"])

    assert enforce_managed_route_per_request(agent) is None  # request 2: still fine


def test_explicit_emergency_revocation_blocks_the_next_request(tmp_path):
    """An explicit emergency revocation (NOT a routine publish/activate) after request 1
    must block request 2 — the only mechanism allowed to kill a live run mid-turn."""
    from agent.managed_route_guard import enforce_managed_route_per_request
    from agent.model_selection_store import revoke_route

    receipt_id, policy, _ = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    assert enforce_managed_route_per_request(agent) is None  # request 1: fine

    revoke_route(
        tmp_path, "kanban-default", route_id=policy["routes"][0]["route_id"],
        reason="incident", approval_ref="operator:revoke",
    )

    reason = enforce_managed_route_per_request(agent)  # request 2: must block
    assert reason == "stale_or_revoked_decision"


def test_revocation_durably_blocks_a_receipt_persisted_after_it(tmp_path):
    """Parent-reproduced bypass: a NEW receipt persisted (same policy/route) AFTER an
    explicit emergency revocation, without any republish/reapproval, must ALSO be blocked --
    revocation is durable authorization state about the route, not something a fresh receipt
    timestamp can silently escape (root AGENTS.md / task: "durable authorization state
    rather than receipt timestamps")."""
    from agent.model_selection_store import persist_receipt, revoke_route

    receipt_old, policy, _ = _receipted(tmp_path)

    revoke_route(
        tmp_path, "kanban-default", route_id=policy["routes"][0]["route_id"],
        reason="probe", approval_ref="operator:test",
    )

    # A NEW attempt/receipt for the SAME still-active policy/route, minted strictly AFTER
    # the revocation -- no republish, no reactivate, no readmission.
    decision_new = select(_requirements("new-attempt"), policy, {}, now=10_000_000)
    receipt_new = persist_receipt(tmp_path, decision_new)

    from agent.managed_route_runtime import enforce_worker_route
    from agent.model_selection_types import RoutingBlocked

    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            tmp_path, receipt_new,
            actual_provider=policy["routes"][0]["provider"],
            actual_model=policy["routes"][0]["model"],
            actual_endpoint=policy["routes"][0]["endpoint"], actual_reasoning="high",
        )


def test_same_second_revocation_still_blocks_a_receipt_from_the_same_second(tmp_path, monkeypatch):
    """Same-second/clock-skew must not authorize a revived route: ordering must use the
    monotonic revocation-event id, never wall-clock seconds where a revoke and a later
    receipt can share `created_at`."""
    import agent.model_selection_store as store

    receipt_old, policy, _ = _receipted(tmp_path)

    frozen = [12345]
    monkeypatch.setattr(store.time, "time", lambda: frozen[0])

    store.revoke_route(
        tmp_path, "kanban-default", route_id=policy["routes"][0]["route_id"],
        reason="incident-same-second", approval_ref="operator:test",
    )
    decision_new = select(_requirements("new-attempt-same-second"), policy, {}, now=frozen[0])
    receipt_new = store.persist_receipt(tmp_path, decision_new)

    from agent.managed_route_runtime import enforce_worker_route
    from agent.model_selection_types import RoutingBlocked

    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            tmp_path, receipt_new,
            actual_provider=policy["routes"][0]["provider"],
            actual_model=policy["routes"][0]["model"],
            actual_endpoint=policy["routes"][0]["endpoint"], actual_reasoning="high",
        )


def test_explicit_readmit_restores_new_attempts_but_not_automatically(tmp_path):
    """Only an explicit `readmit_route` call clears a durable revocation -- a fresh receipt
    alone (proven blocked above) never does. After readmission, a NEW attempt's receipt is
    accepted again."""
    from agent.model_selection_store import is_route_revoked, persist_receipt, readmit_route, revoke_route

    receipt_old, policy, _ = _receipted(tmp_path)
    route_id = policy["routes"][0]["route_id"]

    revoke_route(tmp_path, "kanban-default", route_id=route_id, reason="incident", approval_ref="operator:test")
    assert is_route_revoked(tmp_path, "kanban-default", route_id) is not None

    readmit_route(
        tmp_path, "kanban-default", route_id=route_id,
        reason="incident resolved", approval_ref="operator:readmit",
    )
    assert is_route_revoked(tmp_path, "kanban-default", route_id) is None

    decision_new = select(_requirements("post-readmit-attempt"), policy, {}, now=10_000_001)
    receipt_new = persist_receipt(tmp_path, decision_new)

    from agent.managed_route_runtime import enforce_worker_route

    enforce_worker_route(
        tmp_path, receipt_new,
        actual_provider=policy["routes"][0]["provider"],
        actual_model=policy["routes"][0]["model"],
        actual_endpoint=policy["routes"][0]["endpoint"], actual_reasoning="high",
    )  # must not raise


def test_client_rebuild_diverging_from_receipt_blocks(tmp_path):
    """A mid-turn client rebuild/rotation that ends up on a DIFFERENT model than
    the receipted route must be caught here, not silently sent."""
    from agent.managed_route_guard import enforce_managed_route_per_request

    receipt_id, _, _ = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    assert enforce_managed_route_per_request(agent) is None
    agent.model = "some-other-model"  # simulates fallback/rotation drift
    reason = enforce_managed_route_per_request(agent)
    assert reason == "stale_or_revoked_decision"


def test_missing_origin_home_fails_closed(tmp_path):
    from agent.managed_route_guard import enforce_managed_route_per_request

    receipt_id, _, _ = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    agent._managed_routing_home = None
    assert enforce_managed_route_per_request(agent) == "missing_routing_origin_home"


def test_clear_managed_route_receipt_makes_agent_unmanaged_again(tmp_path):
    from agent.managed_route_guard import clear_managed_route_receipt, enforce_managed_route_per_request

    receipt_id, _, _ = _receipted(tmp_path)
    agent = _agent(receipt_id, tmp_path)
    clear_managed_route_receipt(agent)
    assert enforce_managed_route_per_request(agent) is None
