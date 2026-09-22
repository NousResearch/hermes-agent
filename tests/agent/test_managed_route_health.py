"""Real provider results feed the next routed attempt, isolated by home."""

import time

import pytest


def test_receipted_provider_failure_excludes_route_only_in_own_profile(tmp_path, monkeypatch):
    from agent.managed_route_health import observe_request
    from agent.managed_route_runtime import resolve_route
    from agent.model_selection_store import activate_policy, get_receipt, publish_policy
    import httpx
    from openai import RateLimitError

    routes = [dict(route_id=name, route_revision=1, provider="custom", model=name,
                   endpoint="http://127.0.0.1:1/v1", maker=name, model_family=name,
                   status="approved", allowed_roles=["builder"], capabilities=[],
                   verified_input_budget=100000, allowed_reasoning=["high"],
                   qualifications=["deep"], assessment="fixture", evidence=[])
              for name in ("preferred", "alternate")]
    publish_policy(tmp_path, dict(schema_version=1, policy_id="test", revision=1, routes=routes,
                   rankings={"builder": {"deep": [r["route_id"] for r in routes]}}), approval_ref="fixture")
    activate_policy(tmp_path, "test", 1)
    req = dict(schema_version=1, role="builder", execution_kind="delegate",
               execution_id="task", attempt_id="1", task_class="cross-component",
               required_capabilities=[], input_tokens=1000, reserve_tokens=2000, reasoning="high",
               target_profile="worker-a",
               provenance=dict(frozen_sha="", verified_by="host", complete=True, contributors=[]))
    now = int(time.time())
    first = resolve_route(tmp_path, "test", req, now=now)
    response = httpx.Response(429, headers={"retry-after": "600"},
                              request=httpx.Request("POST", routes[0]["endpoint"]))
    with pytest.raises(RateLimitError):
        with observe_request(tmp_path, first["receipt_id"]):
            raise RateLimitError("private prompt must not be persisted", response=response, body=None)
    second = resolve_route(tmp_path, "test", dict(req, attempt_id="2"), now=now + 1)
    assert second["model"] == "alternate"
    other = resolve_route(tmp_path, "test", dict(req, target_profile="worker-b", attempt_id="3"), now=now + 1)
    assert other["model"] == "preferred"
    again = resolve_route(tmp_path, "test", dict(req, attempt_id="4"), now=now + 1)
    assert again["model"] == "alternate"
    assert "private prompt" not in str(get_receipt(tmp_path, again["receipt_id"]))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    del req["target_profile"]
    implicit = resolve_route(tmp_path, "test", dict(req, attempt_id="5"), now=now + 1)
    assert get_receipt(tmp_path, implicit["receipt_id"])["requirements"]["target_profile"] == str(tmp_path.resolve())


def test_managed_moa_stream_records_failure_during_consumption(tmp_path, monkeypatch):
    """The native MoA aggregator returns a lazy stream, so route health is not
    known when ``call_llm`` returns.  A failure raised while consuming that
    stream must be attributed to the receipted route rather than recorded as a
    successful request (or omitted entirely)."""
    from agent.managed_route_runtime import resolve_route
    from agent.model_selection_store import activate_policy, list_outcomes, publish_policy
    from agent.moa_loop import MoAChatCompletions
    import agent.moa_loop as moa_loop

    route = dict(
        route_id="managed-aggregator", route_revision=1, provider="custom",
        model="managed-model", endpoint="http://127.0.0.1:1/v1", maker="test-maker",
        model_family="managed", status="approved", allowed_roles=["moaaggregator"],
        capabilities=[], verified_input_budget=100000, allowed_reasoning=["high"],
        qualifications=["deep"], assessment="fixture", evidence=[],
    )
    policy = dict(
        schema_version=1, policy_id="test", revision=1, routes=[route],
        rankings={"moaaggregator": {"deep": [route["route_id"]]}},
    )
    publish_policy(tmp_path, policy, approval_ref="fixture")
    activate_policy(tmp_path, "test", 1)
    requirements = dict(
        schema_version=1, role="moaaggregator", execution_kind="moa",
        execution_id="stream-health", attempt_id="1", slot_id="aggregator",
        task_class="cross-component", required_capabilities=[], input_tokens=1000,
        reserve_tokens=2000, reasoning="high", target_profile="worker-a",
        provenance=dict(frozen_sha="", verified_by="host", complete=True, contributors=[]),
    )
    resolution = resolve_route(tmp_path, "test", requirements, now=int(time.time()))
    resolution["routing_home"] = tmp_path

    class BrokenStream:
        def __iter__(self):
            yield object()
            raise TimeoutError("stream disconnected after provider acceptance")

    monkeypatch.setattr(
        moa_loop, "_slot_runtime_managed",
        lambda *args, **kwargs: (
            {"provider": "custom", "model": "managed-model", "base_url": route["endpoint"],
             "reasoning_effort": "high"},
            resolution,
        ),
    )
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: BrokenStream())
    monkeypatch.setattr(
        "agent.moa_model_routing.enforce_moa_slot_route", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "agent.managed_route_budget.enforce_input_budget", lambda *args, **kwargs: None,
    )

    facade = MoAChatCompletions.__new__(MoAChatCompletions)
    facade._agent = None
    facade.preset_name = "managed"
    facade._pending_trace = None
    facade._plan_aggregator_cache = lambda messages, tools, guidance, runtime: (messages, tools)
    prepared = {
        "aggregator": {"routing_role": "moaaggregator"},
        "messages": [{"role": "user", "content": "perform the managed review"}],
        "guidance": "",
        "aggregator_temperature": None,
    }

    stream = facade._call_prepared_aggregator(
        prepared, {"stream": True, "stream_options": {"include_usage": True}},
    )
    with pytest.raises(TimeoutError, match="stream disconnected"):
        list(stream)

    health = [
        event["payload"] for event in list_outcomes(tmp_path, resolution["receipt_id"])
        if event["kind"] == "routing_health"
    ]
    assert health and health[-1]["status"] == "outage"
