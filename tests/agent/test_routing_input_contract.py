"""Unknown input is not empty input at any managed adapter boundary."""
from types import SimpleNamespace

import pytest

from agent.model_selection_store import activate_policy, publish_policy
from agent.model_selection_types import RoutingBlocked


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    policy = {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "routes": [{
            "route_id": "local", "route_revision": 1, "provider": "custom",
            "model": "local", "endpoint": "http://127.0.0.1:1/v1", "maker": "test",
            "model_family": "local", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["medium"], "qualifications": ["deep"],
            "assessment": "local fixture only", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["local"]}},
    }
    publish_policy(tmp_path, policy, approval_ref="test-only")
    activate_policy(tmp_path, "kanban-default", 1)
    return tmp_path


@pytest.mark.parametrize("adapter", ["kanban", "delegation", "moa"])
@pytest.mark.parametrize("intake", [None, {"input_tokens": 0, "reserve_tokens": 0}])
def test_unknown_adapter_budget_blocks_before_receipt(policy_home, adapter, intake):
    from agent.moa_model_routing import resolve_moa_slot_route
    from hermes_cli.kanban_model_routing import resolve_task_route
    from tools.delegate_tool_routing import resolve_delegation_route

    task = {"goal": "Nonempty task", "routing_role": "builder", "routing_requirements": intake}
    with pytest.raises(RoutingBlocked, match="missing_input_estimate"):
        if adapter == "kanban":
            from hermes_cli import kanban_db as kb
            from hermes_cli.kanban_db_connect import connect
            kb.init_db()
            with connect() as conn:
                tid = kb.create_task(conn, title="Nonempty task", routing_role="builder",
                                     routing_requirements=intake)
                resolve_task_route(policy_home, conn, kb.get_task(conn, tid),
                                   now=1, frozen_sha="", verified_by="test")
        elif adapter == "delegation":
            resolve_delegation_route(task, SimpleNamespace(), 0)
        else:
            resolve_moa_slot_route(task, execution_id="turn", slot_id="aggregator")


@pytest.mark.parametrize("value", [False, True, "10", [], {}, -1, 1.5])
def test_selector_rejects_non_integer_estimates_before_comparison(policy_home, value):
    from agent.model_selection import select
    from agent.model_selection_store import get_active_policy
    req = dict(schema_version=1, role="builder", execution_kind="delegation",
               execution_id="task", attempt_id="1", task_class="", required_capabilities=[],
               input_tokens=value, reserve_tokens=1000, reasoning="medium",
               provenance=dict(frozen_sha="", verified_by="test", complete=True, contributors=[]))
    with pytest.raises(RoutingBlocked, match="schema_invalid"):
        select(req, get_active_policy(policy_home, "kanban-default"), {}, 1)


def test_explicitly_empty_input_is_distinct_from_unknown(policy_home):
    from agent.model_selection import select
    from agent.model_selection_store import get_active_policy
    req = dict(schema_version=1, role="builder", execution_kind="diagnostic",
               execution_id="empty", attempt_id="1", task_class="", required_capabilities=[],
               input_tokens=0, input_empty=True, reserve_tokens=1000, reasoning="medium",
               provenance=dict(frozen_sha="", verified_by="test", complete=True, contributors=[]))
    decision = select(req, get_active_policy(policy_home, "kanban-default"), {}, 1)
    assert decision["requirements"]["input_tokens"] == 0
    assert decision["requirements"]["input_empty"] is True


@pytest.mark.parametrize("field,value", [("routes", None), ("routes", [None]),
                                       ("rankings", []), ("revision", True)])
def test_malformed_policy_is_a_typed_blocker(policy_home, field, value):
    from agent.model_selection import select
    from agent.model_selection_store import get_active_policy

    policy = get_active_policy(policy_home, "kanban-default")
    policy[field] = value
    req = dict(schema_version=1, role="builder", execution_kind="diagnostic",
               execution_id="shape", attempt_id="1", task_class="", required_capabilities=[],
               input_tokens=1000, reserve_tokens=1000, reasoning="medium",
               provenance=dict(frozen_sha="", verified_by="test", complete=True, contributors=[]))
    with pytest.raises(RoutingBlocked, match="schema_invalid"):
        select(req, policy, {}, 1)
