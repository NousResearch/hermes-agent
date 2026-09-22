"""Origin-profile-scoped policy/receipt storage (design §3.C, §12).

E2E against a real temp Hermes home + real sqlite file -- no mocks (root
rubric: "E2E validation, not just green unit mocks" for file/network I/O).
"""
from __future__ import annotations

from typing import Any

import pytest

from agent.model_selection_types import RoutingBlocked


def _policy(revision=1):
    return dict(schema_version=1, policy_id="p1", revision=revision, approval_ref="operator:fixture",
                routes=[dict(route_id="a", route_revision=1, provider="openai", model="a",
                             endpoint="https://api.openai.com/v1", maker="openai",
                             model_family="fixture", status="approved",
                             allowed_roles=["builder"], capabilities=["tool_use"],
                             verified_input_budget=100000, allowed_reasoning=["high"],
                             qualifications=["deep"], assessment="fixture", evidence=[])],
                rankings={"builder": {"deep": ["a"]}})


def test_publish_is_immutable_and_never_overwrites_a_revision(tmp_path):
    from agent.model_selection_store import publish_policy, list_policy_revisions

    home = tmp_path / "home"
    home.mkdir()
    rec = publish_policy(home, _policy(), approval_ref="operator:conv-1")
    assert rec["policy_id"] == "p1" and rec["revision"] == 1

    # Re-publishing the identical content is idempotent (no duplicate row).
    publish_policy(home, _policy(), approval_ref="operator:conv-1")
    assert len(list_policy_revisions(home, "p1")) == 1

    # A conflicting republish of the SAME revision is rejected, never silently
    # overwritten.
    mutated = _policy()
    mutated["routes"][0]["status"] = "suspended"
    with pytest.raises(RoutingBlocked):
        publish_policy(home, mutated, approval_ref="operator:conv-1")


def test_activate_selects_exactly_one_active_revision(tmp_path):
    from agent.model_selection_store import publish_policy, activate_policy, get_active_policy

    home = tmp_path / "home"
    home.mkdir()
    publish_policy(home, _policy(revision=1), approval_ref="operator:conv-1")
    publish_policy(home, _policy(revision=2), approval_ref="operator:conv-2")
    assert get_active_policy(home, "p1") is None  # publish != activation

    activate_policy(home, "p1", 1)
    assert get_active_policy(home, "p1")["revision"] == 1
    activate_policy(home, "p1", 2)
    assert get_active_policy(home, "p1")["revision"] == 2


def test_receipt_persistence_is_idempotent_and_rejects_conflicting_reuse(tmp_path):
    from agent.model_selection import select
    from agent.model_selection_store import persist_receipt, get_receipt, append_outcome, list_outcomes

    home = tmp_path / "home"
    home.mkdir()
    policy = _policy()
    req = dict(schema_version=1, role="builder", execution_kind="kanban", execution_id="t_1",
               attempt_id="1", slot_id="", task_class="cross-component",
               required_capabilities=["tool_use"], input_tokens=100, reserve_tokens=100,
               reasoning="high", provenance={"frozen_sha": "a" * 40, "verified_by": "parent",
               "complete": True, "contributors": []})
    decision = select(req, policy, {}, now=1000)

    receipt_id = persist_receipt(home, decision)
    # Re-persisting the SAME decision under the same key is idempotent.
    assert persist_receipt(home, decision) == receipt_id
    fetched = get_receipt(home, receipt_id)
    assert fetched["selected"]["route_id"] == "a"

    append_outcome(home, receipt_id, "routing_selected", {"route_id": "a"})
    append_outcome(home, receipt_id, "routing_started", {"pid": 123})
    outcomes = list_outcomes(home, receipt_id)
    assert [o["kind"] for o in outcomes] == ["routing_selected", "routing_started"]
    assert [o["seq"] for o in outcomes] == [1, 2]

    # A conflicting decision reusing the same (execution_kind, execution_id,
    # attempt_id, slot_id) key must be rejected, not silently overwritten.
    other_policy = _policy()
    other_policy["routes"][0]["route_id"] = "b"
    other_policy["rankings"] = {"builder": {"deep": ["b"]}}
    other_policy["routes"][0]["provider"] = "anthropic"
    other_decision = select(req, other_policy, {}, now=1000)
    with pytest.raises(RoutingBlocked):
        persist_receipt(home, other_decision)


def test_publish_rejects_malformed_policy_before_creating_store(tmp_path):
    from agent.model_selection_store import publish_policy

    home = tmp_path / "home"
    home.mkdir()
    malformed = _policy()
    routes: Any = malformed["routes"]
    routes[0]["status"] = "enabled"

    with pytest.raises(RoutingBlocked, match="status"):
        publish_policy(home, malformed, approval_ref="operator:conv-1")
    assert not (home / "model_routing.db").exists()


@pytest.mark.parametrize("damage", [
    "missing_revision", "policy_content", "receipt_content", "receipt_binding",
    "policy_schema", "receipt_schema", "duplicate_policy_key", "duplicate_receipt_key",
])
def test_guard_requires_intact_retained_policy_and_receipt(tmp_path, damage):
    import sqlite3
    from agent.model_selection import select
    from agent.managed_route_runtime import enforce_worker_route
    from agent.model_selection_store import publish_policy, activate_policy, persist_receipt

    policy = _policy()
    publish_policy(tmp_path, policy, approval_ref=policy["approval_ref"])
    activate_policy(tmp_path, "p1", 1)
    requirements = dict(schema_version=1, role="builder", execution_kind="kanban", execution_id="t_1",
                        attempt_id="1", task_class="cross-component", required_capabilities=[],
                        input_tokens=100, reserve_tokens=100, reasoning="high",
                        provenance=dict(frozen_sha="", verified_by="fixture", complete=True, contributors=[]))
    receipt = persist_receipt(tmp_path, select(requirements, policy, {}, now=1000))
    publish_policy(tmp_path, _policy(2), approval_ref="operator:new")
    activate_policy(tmp_path, "p1", 2)
    actual = dict(actual_provider="openai", actual_model="a", actual_endpoint="https://api.openai.com/v1",
                  actual_reasoning="high", record_outcome=False)
    enforce_worker_route(tmp_path, receipt, **actual)  # newer active policy must not unpin the run
    mutations = {
        "missing_revision": "DELETE FROM policy_revisions WHERE revision=1",
        "policy_content": "UPDATE policy_revisions SET content_json=json_set(content_json,'$.routes[0].model','changed') WHERE revision=1",
        "receipt_content": "UPDATE routing_receipts SET decision_json=json_set(decision_json,'$.requirements.input_tokens',1)",
        "receipt_binding": "UPDATE routing_receipts SET execution_id='t_other'",
        "policy_schema": "UPDATE policy_revisions SET content_json=json_set(content_json,'$.schema_version',99) WHERE revision=1",
        "receipt_schema": "UPDATE routing_receipts SET decision_json=json_set(decision_json,'$.schema_version',99)",
        "duplicate_policy_key": "UPDATE policy_revisions SET content_json=replace(content_json,'\"schema_version\":1','\"schema_version\":99,\"schema_version\":1') WHERE revision=1",
        "duplicate_receipt_key": "UPDATE routing_receipts SET decision_json=replace(decision_json,'\"schema_version\":1','\"schema_version\":99,\"schema_version\":1')",
    }
    with sqlite3.connect(tmp_path / "model_routing.db") as conn:
        conn.execute(mutations[damage])
    with pytest.raises(RoutingBlocked):
        enforce_worker_route(tmp_path, receipt, **actual)
