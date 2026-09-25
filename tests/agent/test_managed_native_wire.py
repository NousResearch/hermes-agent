"""Managed native wire validation after protocol conversion (A1/Q4/SYS2)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@pytest.fixture
def native_endpoint(request):
    class Handler(BaseHTTPRequestHandler):
        requests = []

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reported = getattr(request, "param", "matching")
            reported = payload.get("model") if reported == "matching" else reported
            if self.path.rstrip("/").endswith(("/messages", "/responses")):
                self.requests.append(payload)
            response = {"id": "msg_fixture", "type": "message", "role": "assistant",
                        "model": reported, "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 1}}
            if self.path.rstrip("/").endswith("/responses"):
                response = {"id": "resp-local", "object": "response", "status": "completed",
                            "model": reported, "output": [{"type": "message", "id": "msg-local",
                            "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}]}],
                            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}}
                created = {"type": "response.created", "response": {"id": "resp-local", "model": response.pop("model")}}
                body = ("event: response.created\ndata: " + json.dumps(created) +
                    "\n\nevent: response.output_text.delta\ndata: " + json.dumps(
                    {"type": "response.output_text.delta", "delta": "done", "output_index": 0,
                     "content_index": 0, "item_id": "msg-local"}) +
                    "\n\nevent: response.completed\ndata: " + json.dumps(
                    {"type": "response.completed", "response": response}) + "\n\n").encode()
            elif payload.get("stream"):
                events = [
                    {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
                body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
            else:
                body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if payload.get("stream") else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _receipt(home, url, model, provider, effort="high"):
    from agent.model_selection import select
    from agent.model_selection_store import publish_policy, activate_policy, persist_receipt

    policy = {"schema_version": 1, "policy_id": "native", "revision": 1,
              "approval_ref": "fixture", "routes": [{
                  "route_id": "native", "route_revision": 1, "provider": provider,
                  "model": model, "endpoint": url, "maker": "anthropic", "model_family": "claude",
                  "status": "approved", "allowed_roles": ["builder"], "capabilities": [],
                  "verified_input_budget": 200000, "allowed_reasoning": [effort],
                  "qualifications": ["deep"], "assessment": "fixture", "evidence": ["fixture"],
              }], "rankings": {"builder": {"deep": ["native"]}}}
    req = {"schema_version": 1, "execution_kind": "delegation", "execution_id": "native",
           "attempt_id": "1", "slot_id": "", "role": "builder", "task_class": "cross-component",
           "required_capabilities": [], "input_tokens": 1000, "reserve_tokens": 8192,
           "reasoning": effort, "provenance": {"frozen_sha": "fixture", "verified_by": "fixture",
                                                 "complete": True, "contributors": []}}
    publish_policy(home, policy, approval_ref="fixture")
    activate_policy(home, "native", 1)
    return persist_receipt(home, select(req, policy, {}, now=1))


@pytest.mark.parametrize("mutation", [None, "model", "reasoning", "reserve"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("api_mode,model,provider", [("anthropic_messages", "claude-sonnet-4-6", "anthropic"),
                                                   ("codex_responses", "gpt-5-codex", "openai-compat")])
def test_native_final_payload_matches_receipt(tmp_path, monkeypatch, native_endpoint, mutation, streaming, with_history, api_mode, model, provider):
    from run_agent import AIAgent
    from hermes_cli import middleware

    url, requests = native_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(api_key="fixture-key", provider=provider, model=model, base_url=url,
                    api_mode=api_mode, max_iterations=1, enabled_toolsets=[], quiet_mode=True,
                    skip_context_files=True, skip_memory=True, save_trajectories=False,
                    reasoning_config={"enabled": True, "effort": "high"})
    agent._disable_streaming = not streaming
    agent._managed_routing_home = tmp_path
    agent._managed_routing_receipt_id = _receipt(tmp_path, url, model, provider)
    original = middleware.run_llm_execution_middleware

    def alter(payload, send, **context):
        def final(request):
            request = dict(request)
            if mutation == "model":
                request["model"] = "unapproved-model"
            elif mutation == "reasoning":
                if api_mode == "anthropic_messages":
                    request["thinking"] = {"type": "disabled"}
                request.pop("output_config", None)
                request.pop("reasoning", None)
            elif mutation == "reserve":
                request["max_output_tokens" if api_mode == "codex_responses" else "max_tokens"] = 300000
            return send(request)
        return original(payload, final, **context)

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", alter)
    history = [
        {"role": "user", "content": "Inspect fixture text"},
        {"role": "assistant", "content": "Inspecting", "tool_calls": [{"id": "call_fixture", "type": "function",
            "function": {"name": "fixture_reader", "arguments": '{"path":"fixture.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_fixture", "name": "fixture_reader", "content": "fixture text"},
    ] if with_history else []
    try:
        result = agent.run_conversation("bounded native request", conversation_history=history)
        if mutation is None:
            assert len(requests) == 1, result
            assert requests[0]["reasoning" if api_mode == "codex_responses" else "output_config"]["effort"] == "high"
            from agent.model_selection_store import list_outcomes
            observations = [event["payload"] for event in list_outcomes(tmp_path, agent._managed_routing_receipt_id)
                            if event["kind"] == "routing_wire_validated"]
            assert len(observations) == 1
            assert observations[0]["model"] == requests[0]["model"]
            assert observations[0]["reasoning"] == "high"
            assert "bounded native request" not in json.dumps(observations)
            assert "fixture-key" not in json.dumps(observations)
        else:
            assert requests == []
            assert result["failed"]
    finally:
        agent.close()


@pytest.mark.parametrize("api_mode,model,provider", [("anthropic_messages", "claude-sonnet-4-6", "custom"),
                                                   ("codex_responses", "gpt-5-codex", "openai-compat")])
@pytest.mark.parametrize("mutation", [False, True])
def test_auxiliary_native_conversion_is_checked(tmp_path, monkeypatch, native_endpoint, api_mode, model, provider, mutation):
    from agent import auxiliary_client, anthropic_adapter
    from agent.managed_route_aux_wire import managed_aux_wire_scope
    from agent.model_selection_types import RoutingBlocked

    url, requests = native_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipt_id = _receipt(tmp_path, url, model, provider)
    reached = []
    if api_mode == "anthropic_messages":
        original = anthropic_adapter.build_anthropic_kwargs

        def altered(*args, **kwargs):
            result = original(*args, **kwargs)
            reached.append(True)
            if mutation:
                result["model"] = "unapproved-model"
            return result

        monkeypatch.setattr(anthropic_adapter, "build_anthropic_kwargs", altered)
    else:
        original = auxiliary_client._CodexCompletionsAdapter._build_responses_kwargs

        def altered(*args, **kwargs):
            result, wire_model, timeout = original(*args, **kwargs)
            reached.append(True)
            if mutation:
                result["model"] = "unapproved-model"
            return result, wire_model, timeout

        monkeypatch.setattr(auxiliary_client._CodexCompletionsAdapter, "_build_responses_kwargs", altered)

    def call():
        with managed_aux_wire_scope(tmp_path, receipt_id):
            return auxiliary_client.call_llm(
                provider=provider, model=model, base_url=url, api_key="fixture-key", api_mode=api_mode,
                messages=[{"role": "user", "content": "bounded auxiliary native request"}],
                reasoning_config={"enabled": True, "effort": "high"}, max_tokens=8192,
                extra_body={"reasoning": {"enabled": True, "effort": "high"}},
            )

    if mutation:
        with pytest.raises(RoutingBlocked):
            call()
        assert not requests
    else:
        response = call()
        assert response.choices[0].message.content == "done"
        assert len(requests) == 1
    assert reached, "the real native conversion must run before final validation"


@pytest.mark.parametrize("entrypoint", ["agent", "auxiliary"])
@pytest.mark.parametrize("model,effort,wire_effort", [
    ("claude-sonnet-4-6", "xhigh", "max"),
    ("claude-haiku-4-5", "high", None),
])
def test_actual_reasoning_conversion_cannot_change_managed_contract(
    tmp_path, monkeypatch, native_endpoint, entrypoint, model, effort, wire_effort,
):
    from contextlib import nullcontext
    from agent import auxiliary_client
    from agent.managed_route_aux_wire import managed_aux_wire_scope
    from agent.model_selection_types import RoutingBlocked
    from run_agent import AIAgent

    url, requests = native_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipt_id = _receipt(tmp_path, url, model, "custom", effort)
    reasoning = {"enabled": True, "effort": effort}
    for managed in (False, True):
        requests.clear()
        if entrypoint == "auxiliary":
            def call():
                with managed_aux_wire_scope(tmp_path, receipt_id) if managed else nullcontext():
                    return auxiliary_client.call_llm(
                        provider="custom", model=model, base_url=url, api_key="fixture-key",
                        api_mode="anthropic_messages", max_tokens=8192,
                        messages=[{"role": "user", "content": "actual conversion probe"}],
                        reasoning_config=reasoning,
                        extra_body={"reasoning": reasoning},
                    )
            if managed:
                with pytest.raises(RoutingBlocked):
                    call()
            else:
                assert call().choices[0].message.content == "done"
        else:
            agent = AIAgent(
                provider="custom", model=model, base_url=url, api_key="fixture-key",
                api_mode="anthropic_messages", reasoning_config=reasoning,
                max_iterations=1, enabled_toolsets=[], quiet_mode=True,
                skip_context_files=True, skip_memory=True, save_trajectories=False,
            )
            if managed:
                agent._managed_routing_home = tmp_path
                agent._managed_routing_receipt_id = receipt_id
            try:
                result = agent.run_conversation("actual conversion probe")
                if managed:
                    assert result["failed"]
                else:
                    assert "done" in result["final_response"]
            finally:
                agent.close()
        if managed:
            assert requests == [], "real transport coercion must block before content leaves"
        else:
            assert len(requests) == 1
            assert requests[0].get("output_config", {}).get("effort") == wire_effort


@pytest.mark.parametrize("native_endpoint,identity_status", [
    ("matching", "matching"), (None, "missing"), ("Different-Model", "changed"),
], indirect=["native_endpoint"])
@pytest.mark.parametrize("boundary", ["normal", "streaming", "summary"])
@pytest.mark.parametrize("api_mode,model,provider", [
    ("anthropic_messages", "claude-sonnet-4-6", "anthropic"),
    ("codex_responses", "gpt-5-codex", "openai-compat"),
])
def test_native_success_records_reported_identity_without_enforcement(
    tmp_path, monkeypatch, native_endpoint, identity_status, boundary, api_mode, model, provider,
):
    from agent.chat_completion_helpers import handle_max_iterations
    from agent.model_selection_store import get_receipt, is_route_revoked, list_outcomes
    from run_agent import AIAgent

    url, requests = native_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipt_id = _receipt(tmp_path, url, model, provider)
    original_receipt = get_receipt(tmp_path, receipt_id)
    agent = AIAgent(
        api_key="fixture-key", provider=provider, model=model, base_url=url,
        api_mode=api_mode, max_iterations=1, enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, save_trajectories=False,
        reasoning_config={"enabled": True, "effort": "high"},
    )
    agent._disable_streaming = boundary != "streaming"
    agent._managed_routing_home = tmp_path
    agent._managed_routing_receipt_id = receipt_id
    try:
        if boundary == "summary":
            text = handle_max_iterations(agent, [{"role": "user", "content": "private fixture prompt"}], 1)
        else:
            text = agent.run_conversation("private fixture prompt")["final_response"]
        assert "done" in text
        assert len(requests) == 1
        health = [e["payload"] for e in list_outcomes(tmp_path, receipt_id) if e["kind"] == "routing_health"]
        assert len(health) == 1
        expected = {"matching": model, "missing": None, "changed": "Different-Model"}[identity_status]
        assert health[0]["reported_model"] == expected
        assert health[0]["identity_status"] == identity_status
        assert health[0]["status"] == "healthy"
        assert health[0]["replay_safe"] is False
        assert get_receipt(tmp_path, receipt_id) == original_receipt
        assert is_route_revoked(tmp_path, "native", "native") is None
        assert "private fixture prompt" not in json.dumps(health)
        assert "fixture-key" not in json.dumps(health)
    finally:
        agent.close()


@pytest.mark.parametrize("native_endpoint,identity_status", [
    ("matching", "matching"), (None, "missing"), ("Different-Model", "changed"),
], indirect=["native_endpoint"])
@pytest.mark.parametrize("boundary", ["reference", "synthesis", "aggregator_stream"])
@pytest.mark.parametrize("api_mode,model,provider", [
    ("anthropic_messages", "claude-sonnet-4-6", "custom"),
    ("codex_responses", "gpt-5-codex", "openai-compat"),
])
def test_moa_native_identity_survives_auxiliary_conversion(
    tmp_path, monkeypatch, native_endpoint, identity_status, boundary, api_mode, model, provider,
):
    from agent.model_selection_store import _connect
    from agent.moa_loop import MoAChatCompletions, _run_reference, aggregate_moa_context
    from hermes_cli import runtime_provider

    url, requests = native_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _receipt(tmp_path, url, model, provider)
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kwargs: dict(
        provider=provider, base_url=url, api_key="fixture-key", api_mode=api_mode,
        request_overrides={"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}},
        command=None, args=[],
    ))
    slot = dict(provider=provider, model=model, routing_role="builder", routing_policy_id="native",
                reasoning_effort="high", routing_requirements=dict(input_tokens=1000, reserve_tokens=8192))
    messages = [{"role": "user", "content": "private native MoA prompt"}]
    if boundary == "reference":
        _, text, _ = _run_reference(slot, messages, execution_id="identity", slot_id="reference-0")
        assert text == "done"
    elif boundary == "synthesis":
        text = aggregate_moa_context(user_prompt=messages[0]["content"], api_messages=messages,
                                     reference_models=[], aggregator=slot)
        assert "done" in text
    else:
        facade = MoAChatCompletions.__new__(MoAChatCompletions)
        facade._agent = None
        facade.preset_name = "identity"
        facade._pending_trace = None
        facade._plan_aggregator_cache = lambda messages, tools, guidance, runtime: (messages, tools)
        chunks = list(facade._call_prepared_aggregator(
            dict(aggregator=slot, messages=messages, guidance="", aggregator_temperature=None),
            dict(stream=True, max_tokens=8192),
        ))
        assert chunks
    assert len(requests) == 1
    with _connect(tmp_path) as conn:
        health = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload_json FROM routing_outcomes WHERE kind='routing_health'")]
    assert len(health) == 1
    expected = {"matching": model, "missing": None, "changed": "Different-Model"}[identity_status]
    assert health[0]["reported_model"] == expected
    assert health[0]["identity_status"] == identity_status
    assert health[0]["status"] == "healthy"
    assert health[0]["replay_safe"] is False
    assert "private native MoA prompt" not in json.dumps(health)
