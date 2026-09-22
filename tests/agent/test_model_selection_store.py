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
