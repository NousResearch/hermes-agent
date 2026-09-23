"""Execution-bound classification authority, independent of model intake (SC3)."""
from copy import deepcopy

import pytest

from agent.managed_route_runtime import resolve_route
from agent.model_selection_store import activate_policy, get_receipt, publish_policy
from agent.model_selection_types import RoutingBlocked


def requirements():
    return {
        "schema_version": 1, "role": "builder", "execution_kind": "delegation",
        "execution_id": "child", "attempt_id": "1", "slot_id": "",
        "task_class": "established-pattern", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 2000, "reasoning": "high",
        "provenance": {"frozen_sha": "", "verified_by": "parent", "complete": True,
                       "contributors": []},
    }


def publish(home, policy_id="test"):
    routes = []
    for quality in ("shallow", "deep"):
        routes.append({
            "route_id": quality, "route_revision": 1, "provider": "openai",
            "model": quality, "endpoint": "https://api.openai.com/v1", "maker": "openai",
            "model_family": "fixture", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 100000, "allowed_reasoning": ["high"],
            "qualifications": [quality], "assessment": "fixture", "evidence": ["fixture"],
        })
    publish_policy(home, {"schema_version": 1, "policy_id": policy_id, "revision": 1,
                         "routes": routes, "rankings": {"builder": {
                             "shallow": ["shallow"], "deep": ["deep"]}}},
                   approval_ref="operator:fixture")
    activate_policy(home, policy_id, 1)


def test_persisted_classification_controls_only_its_exact_execution(tmp_path):
    from agent.model_selection_classification import attest_classification

    publish(tmp_path)
    req = requirements()
    record = attest_classification(
        tmp_path, req, authority="parent",
        attester={"task_id": "parent", "run_id": "7", "role": "orchestrator"},
        complete=True, risk_flags=[], evidence=["scope:reviewed-diff"], expected_version=0,
    )
    selected = resolve_route(tmp_path, "test", req, now=100)
    decision = get_receipt(tmp_path, selected["receipt_id"])
    assert decision is not None
    assert selected["model"] == "shallow"
    assert decision["requirements"]["classification"] == record

    other = deepcopy(req)
    other["attempt_id"] = "2"
    assert resolve_route(tmp_path, "test", other, now=100)["model"] == "deep"

    # A later attestation is a new immutable version, not a rewrite of this run's receipt.
    elevated = attest_classification(
        tmp_path, req, authority="operator", attester={"approval_ref": "operator:incident"},
        complete=True, risk_flags=["persistence"], evidence=["scope:new-evidence"],
        expected_version=record["version"],
    )
    assert elevated["version"] > record["version"]
    retained = get_receipt(tmp_path, selected["receipt_id"])
    assert retained is not None
    assert retained["requirements"]["classification"] == record
    with pytest.raises(RoutingBlocked, match="authority"):
        attest_classification(
            tmp_path, req, authority="parent",
            attester={"task_id": "parent", "run_id": "7", "role": "orchestrator"},
            complete=True, risk_flags=[], evidence=["scope:lower"],
            expected_version=elevated["version"],
        )
    with pytest.raises(RoutingBlocked, match="version"):
        attest_classification(
            tmp_path, req, authority="operator", attester={"approval_ref": "operator:stale"},
            complete=True, risk_flags=[], evidence=["scope:stale"], expected_version=0,
        )


def test_attestation_cannot_be_reused_for_changed_scope_or_parent_risk_downgrade(tmp_path):
    from agent.model_selection_classification import attest_classification

    publish(tmp_path)
    req = requirements()
    parent = {"task_id": "parent", "run_id": "7", "role": "orchestrator"}
    record = attest_classification(
        tmp_path, req, authority="parent", attester=parent, complete=True,
        risk_flags=["auth"], evidence=["scope:auth-path"], expected_version=0,
    )
    later = attest_classification(
        tmp_path, req, authority="parent", attester=parent, complete=True,
        risk_flags=[], evidence=["scope:proposal"], expected_version=record["version"],
    )
    assert later["risk_flags"] == ["auth"]
    assert resolve_route(tmp_path, "test", req, now=100)["model"] == "deep"

    different_scope = deepcopy(req)
    different_scope["required_capabilities"] = ["tool_use"]
    with pytest.raises(RoutingBlocked, match="scope"):
        resolve_route(tmp_path, "test", different_scope, now=100)


@pytest.mark.parametrize("complete,flags,proposed", [
    (False, [], []), (True, ["migration"], []), (True, [], ["auth"]),
])
def test_incomplete_or_risky_scope_cannot_lower_quality(tmp_path, complete, flags, proposed):
    from agent.model_selection_classification import attest_classification
    from hermes_cli.kanban_model_routing import validate_routing_requirements

    publish(tmp_path)
    intake = validate_routing_requirements({"risk_flags": proposed}) or {}
    req = requirements()
    req["risk_flags"] = intake.get("risk_flags", [])
    attest_classification(
        tmp_path, req, authority="parent",
        attester={"task_id": "parent", "run_id": "7", "role": "orchestrator"},
        complete=complete, risk_flags=flags, evidence=["scope:reviewed"], expected_version=0,
    )
    assert resolve_route(tmp_path, "test", req, now=100)["model"] == "deep"
    for field in ("classification", "authority", "mandatory_quality", "attester"):
        with pytest.raises(ValueError, match="unknown fields"):
            validate_routing_requirements({field: "operator"})


@pytest.mark.parametrize("adapter", ["kanban", "delegation", "moa"])
@pytest.mark.parametrize("proposed", [[], ["auth"]])
def test_adapters_load_attestation_without_dropping_model_risk(tmp_path, monkeypatch, adapter, proposed):
    from pathlib import Path
    from types import SimpleNamespace
    from agent.model_selection_classification import attest_classification

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    publish(tmp_path, "kanban-default")
    intake = {"task_class": "established-pattern", "input_tokens": 1000,
              "reserve_tokens": 2000, "risk_flags": proposed}
    spec = {"routing_role": "builder", "routing_requirements": intake, "reasoning_effort": "high"}
    conn = None
    if adapter == "kanban":
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli.kanban_model_routing import _build_requirements, resolve_task_route

        kb.init_db()
        conn = kbc.connect()
        task_id = kb.create_task(conn, title="classified scope", assignee="builder", **spec)
        task = kb.claim_task(conn, task_id, claimer="dispatcher")
        assert task is not None
        req = _build_requirements(task, frozen_sha="", verified_by="parent")
        resolve = lambda: resolve_task_route(tmp_path, conn, task, now=100, frozen_sha="", verified_by="parent")
    elif adapter == "delegation":
        from tools.delegate_tool_routing import _build_requirements, resolve_delegation_route

        spec["_delegation_id"] = "child"
        req = _build_requirements(spec, intake, role="builder", task_index=0, attempt_id="1")
        resolve = lambda: resolve_delegation_route(spec, SimpleNamespace(), 0, attempt_id="1")
    else:
        from agent.moa_model_routing import _build_requirements, resolve_moa_slot_route

        req = _build_requirements(spec, intake, role="builder", execution_id="cohort", attempt_id="1", slot_id="reference")
        resolve = lambda: resolve_moa_slot_route(spec, execution_id="cohort", attempt_id="1", slot_id="reference", hermes_home=str(tmp_path))
    try:
        attest_classification(
            tmp_path, req, authority="parent", attester={"task_id": "parent", "run_id": "7", "role": "orchestrator"},
            complete=True, risk_flags=[], evidence=["scope:reviewed"], expected_version=0,
        )
        resolution = resolve()
        assert resolution is not None
        assert resolution["model"] == ("deep" if proposed else "shallow")
        decision = get_receipt(tmp_path, resolution["receipt_id"])
        assert decision is not None
        assert decision["requirements"]["risk_flags"] == proposed
        assert decision["requirements"]["classification"]["execution"]["attempt_id"] == req["attempt_id"]
    finally:
        if conn is not None:
            conn.close()


def test_missing_retained_attestation_blocks_managed_startup(tmp_path):
    from agent.managed_route_runtime import enforce_worker_route
    from agent.model_selection_classification import attest_classification
    from agent.model_selection_store import _connect

    publish(tmp_path)
    req = requirements()
    attest_classification(
        tmp_path, req, authority="parent", attester={"task_id": "parent", "run_id": "7", "role": "orchestrator"},
        complete=True, risk_flags=[], evidence=["scope:reviewed"], expected_version=0,
    )
    selected = resolve_route(tmp_path, "test", req, now=100)
    actual = dict(actual_provider=selected["provider"], actual_model=selected["model"],
                  actual_endpoint=selected["endpoint"], actual_reasoning=selected["reasoning_effort"])
    enforce_worker_route(tmp_path, selected["receipt_id"], **actual)
    with _connect(tmp_path) as conn:
        conn.execute("DELETE FROM routing_classifications")
    with pytest.raises(RoutingBlocked, match="classification"):
        enforce_worker_route(tmp_path, selected["receipt_id"], **actual)


def test_nested_attestation_cannot_clear_parent_mandatory_floor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.model_selection_classification import attest_classification
    from tools.delegate_tool_routing import _build_requirements, resolve_delegation_route

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    publish(tmp_path)
    req = requirements()
    req["execution_id"] = "parent"
    req["provenance"]["frozen_sha"] = "fixture-sha"
    attest_classification(
        tmp_path, req, authority="operator", attester={"approval_ref": "operator:auth"},
        complete=True, risk_flags=["auth"], evidence=["scope:auth"], expected_version=0,
    )
    parent_route = resolve_route(tmp_path, "test", req, now=100)
    parent = SimpleNamespace(_managed_routing_home=tmp_path,
                             _managed_routing_receipt_id=parent_route["receipt_id"])
    intake = {"task_class": "established-pattern", "input_tokens": 1000,
              "reserve_tokens": 2000, "risk_flags": [], "provenance": req["provenance"]}
    task = {"_delegation_id": "nested", "routing_requirements": intake, "reasoning_effort": "high"}
    child_req = _build_requirements(task, intake, role="builder", task_index=0, attempt_id="1")
    attest_classification(
        tmp_path, child_req, authority="parent",
        attester={"task_id": "parent", "run_id": "1", "role": "builder"},
        complete=True, risk_flags=[], evidence=["scope:child"], expected_version=0,
    )
    child = resolve_delegation_route(task, parent, 0, attempt_id="1")
    assert child is not None
    receipt = get_receipt(tmp_path, child["receipt_id"])
    assert receipt is not None
    assert receipt["requirements"]["quality"] == "deep"
    assert receipt["requirements"]["inherited_risk_flags"] == ["auth"]
    assert receipt["requirements"]["risk_flags"] == []
