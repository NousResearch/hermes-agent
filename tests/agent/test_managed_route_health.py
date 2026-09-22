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
