"""Final-wire remaining acceptance cells. Inert transports only.

Physical boundaries: HTTPX handle_request and botocore http_session.send.
No credentials, no 180s waits, no provider network.
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from agent.conversation_compression import (
    ProviderBoundRequestOverLimit,
    estimate_provider_bound_request_pressure,
    refuse_over_limit_provider_dispatch,
)
from agent.final_wire_admission import (
    COVERED_MAIN,
    KNOWN_R,
    PROVIDER_DEFAULT_UNRESOLVED,
    FinalAttemptIdentity,
    ProviderBoundInvalidAccounting,
    ProviderBoundUnsupportedAccounting,
    admit_final_json,
    bind_attempt_identity,
    current_attempt_identity,
    estimate_schema_tokens,
    intercepted_openai_class,
    project_final_body,
    unwrap_local_refusal,
    wrap_botocore_runtime_client,
    wrap_httpx_client_transports,
)
from agent.model_metadata import _estimate_tools_tokens_rough, estimate_request_tokens_rough
from tests.agent.test_provider_bound_dispatch_guard import history, run_local
from tests.agent.test_relay_llm import relay_turn  # noqa: F401
from tests.run_agent.test_413_compression import agent  # noqa: F401


WINDOW = 1000


def _identity(**overrides):
    values = dict(
        purpose=COVERED_MAIN,
        family="chat_completions",
        model="test/model",
        endpoint="https://inert.invalid/v1",
        window=WINDOW,
        correlation_id="attempt-1",
    )
    values.update(overrides)
    return FinalAttemptIdentity(**values)


def _chat_ok():
    return {
        "id": "inert",
        "object": "chat.completion",
        "created": 0,
        "model": "test/model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }


class CountingTransport(httpx.BaseTransport):
    def __init__(self):
        self.calls = []

    def handle_request(self, request):
        self.calls.append(request)
        return httpx.Response(200, json=_chat_ok())


class CountingSession:
    def __init__(self):
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append(request)
        return SimpleNamespace(status_code=200, headers={}, content=b"{}")


def _tool_defs(size):
    return [{"type": "function", "function": {
        "name": "review_inert", "description": "inert tool",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "description": "x" * size}}},
    }}]


def test_safe_botocore_delegate_exactly_once():
    session = CountingSession()
    client = SimpleNamespace(_endpoint=SimpleNamespace(http_session=session), meta=SimpleNamespace(events=None))
    wrap_botocore_runtime_client(client)
    request = SimpleNamespace(
        body=json.dumps({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}),
        url="https://inert.invalid/v1/model/test%2Fmodel/converse",
    )
    with bind_attempt_identity(_identity(family="bedrock_converse")):
        client._endpoint.http_session.send(request)
    assert len(session.calls) == 1


def test_over_limit_botocore_delegate_zero():
    session = CountingSession()
    client = SimpleNamespace(_endpoint=SimpleNamespace(http_session=session), meta=SimpleNamespace(events=None))
    wrap_botocore_runtime_client(client)
    request = SimpleNamespace(url="https://inert.invalid/v1/model/test%2Fmodel/converse", body=json.dumps({
        "messages": [{"role": "user", "content": [{"text": "b" * 4400}]}],
        "inferenceConfig": {"maxTokens": 200},
    }))
    with bind_attempt_identity(_identity(family="bedrock_converse")):
        with pytest.raises(ProviderBoundRequestOverLimit):
            client._endpoint.http_session.send(request)
    assert session.calls == []


def test_native_tool_schemas_counted_not_zero():
    tools = _tool_defs(4400)
    anthropic = [{"name": "review_inert", "description": "inert tool", "input_schema": tools[0]["function"]["parameters"]}]
    converse = [{"toolSpec": {"name": "review_inert", "description": "inert", "inputSchema": {"json": tools[0]["function"]["parameters"]}}}]
    gemini = [{"functionDeclarations": [{"name": "review_inert", "description": "inert", "parameters": tools[0]["function"]["parameters"]}]}]
    openai_n = estimate_schema_tokens(tools)
    anth_n = estimate_schema_tokens(anthropic)
    conv_n = estimate_schema_tokens(converse)
    gem_n = estimate_schema_tokens(gemini)
    assert openai_n > 1000
    assert anth_n > 1000
    assert conv_n > 1000
    assert gem_n > 1000
    pressure = estimate_provider_bound_request_pressure({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": anthropic,
    })
    assert pressure > 1000


def test_mutated_nested_schema_invalidates_content_cache():
    tools = _tool_defs(20)
    first = _estimate_tools_tokens_rough(tools)
    tools[0]["function"]["parameters"]["properties"]["value"]["description"] = "c" * 4400
    stale = _estimate_tools_tokens_rough(tools)
    fresh = _estimate_tools_tokens_rough(deepcopy(tools))
    middle = _tool_defs(20) + _tool_defs(20) + _tool_defs(20)
    before = _estimate_tools_tokens_rough(middle)
    middle[1]["function"]["parameters"]["properties"]["value"]["description"] = "m" * 4400
    after = _estimate_tools_tokens_rough(middle)
    assert first < 100
    assert stale == fresh
    assert fresh > 1000
    assert after > before
    assert after > 1000


def test_unknown_context_representation_is_not_zero():
    with pytest.raises(ProviderBoundUnsupportedAccounting):
        estimate_provider_bound_request_pressure("not-a-mapping")


def test_known_r_participates_in_shared_window_rule():
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "x" * 3200}], "max_tokens": 200}
    pressure = estimate_provider_bound_request_pressure(payload)
    assert 800 < pressure < 1000
    agent = SimpleNamespace(api_mode="chat_completions", provider="openai", model="x",
                            context_compressor=SimpleNamespace(context_length=1000), session_id="s")
    with pytest.raises(ProviderBoundRequestOverLimit):
        refuse_over_limit_provider_dispatch(agent, payload)
    snap = project_final_body(payload, _identity())
    assert snap.reservation_state == KNOWN_R
    assert snap.resolved_r == 200


def test_omitted_optional_cap_is_unresolved_not_zero():
    payload = {"model": "test/model", "input": [{"role": "user", "content": "hi"}]}
    snap = project_final_body(payload, _identity(family="codex_responses"))
    assert snap.reservation_state == PROVIDER_DEFAULT_UNRESOLVED
    assert snap.resolved_r is None
    agent = SimpleNamespace(api_mode="codex_responses", provider="openai-codex", model="gpt-5.5",
                            context_compressor=SimpleNamespace(context_length=1000), session_id="s")
    refuse_over_limit_provider_dispatch(agent, payload)


def test_invalid_unknown_w_and_malformed_required_cap():
    agent = SimpleNamespace(api_mode="chat_completions", provider="openai", model="x",
                            context_compressor=SimpleNamespace(context_length=0), session_id="s")
    with pytest.raises((ProviderBoundInvalidAccounting, ProviderBoundRequestOverLimit)):
        refuse_over_limit_provider_dispatch(agent, {"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises((ProviderBoundInvalidAccounting, ProviderBoundRequestOverLimit)):
        admit_final_json(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "nope"},
            _identity(family="anthropic_messages"),
        )
    with pytest.raises((ProviderBoundInvalidAccounting, ProviderBoundRequestOverLimit)):
        admit_final_json(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "max_completion_tokens": 99},
            _identity(),
        )


def test_local_refusal_restored_when_sdk_retries_zero_and_nonzero(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: sleeps.append(1))
    cls = intercepted_openai_class(OpenAI)
    for retries in (0, 2):
        inner = CountingTransport()
        http_client = httpx.Client(transport=inner)
        wrap_httpx_client_transports(http_client, covered=True)
        sdk = cls(
            api_key="inert-not-a-secret",
            base_url="https://inert.invalid/v1",
            max_retries=retries,
            http_client=http_client,
        )
        monkeypatch.setattr(sdk, "_calculate_retry_timeout", lambda *a, **k: 0)
        payload = {"model": "test/model", "messages": [{"role": "user", "content": "b" * 4400}]}
        sleeps.clear()
        with bind_attempt_identity(_identity()):
            with pytest.raises(ProviderBoundRequestOverLimit):
                sdk.chat.completions.create(**payload)
        assert inner.calls == []
        assert sleeps == []
        sdk.close()
        http_client.close()


def test_genuine_remote_failure_still_retries(monkeypatch):
    attempts = {"n": 0}

    class Flaky(httpx.BaseTransport):
        def handle_request(self, request):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise httpx.ConnectError("transient")
            return httpx.Response(200, json=_chat_ok())

    inner = Flaky()
    http_client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(http_client, covered=True)
    cls = intercepted_openai_class(OpenAI)
    sdk = cls(api_key="inert-not-a-secret", base_url="https://inert.invalid/v1",
              max_retries=2, http_client=http_client)
    monkeypatch.setattr(sdk, "_calculate_retry_timeout", lambda *a, **k: 0)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    with bind_attempt_identity(_identity()):
        result = sdk.chat.completions.create(model="test/model", messages=[{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content == "ok"
    assert attempts["n"] == 2
    sdk.close()
    http_client.close()


def test_lazy_concurrent_retry_fallback_attempt_identity_does_not_leak():
    seen = []

    def worker(name, window):
        with bind_attempt_identity(_identity(correlation_id=name, window=window, model=name)):
            seen.append((current_attempt_identity().correlation_id, current_attempt_identity().window, current_attempt_identity().model))

    t1 = threading.Thread(target=worker, args=("a", 1000))
    t2 = threading.Thread(target=worker, args=("b", 2000))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert ("a", 1000, "a") in seen
    assert ("b", 2000, "b") in seen
    assert current_attempt_identity() is None


def test_string_responses_input_is_not_empty_list():
    pressure = estimate_provider_bound_request_pressure({
        "instructions": "sys",
        "input": "u" * 4000,
    })
    assert pressure > 500


def test_instructions_and_system_counted():
    chat = estimate_provider_bound_request_pressure({
        "messages": [{"role": "user", "content": "hi"}],
        "system": "s" * 4000,
    })
    responses = estimate_provider_bound_request_pressure({
        "input": [{"role": "user", "content": "hi"}],
        "instructions": "s" * 4000,
    })
    assert chat > 1000
    assert responses > 1000


def test_preflight_partial_compaction_diagnostic_is_truthful(agent):
    a = agent
    a.tools = []
    a.context_compressor.context_length = WINDOW
    a.context_compressor.threshold_tokens = 500
    calls = a.client.chat.completions.create
    from tests.run_agent.test_413_compression import _mock_response
    calls.return_value = _mock_response(content="inert")
    before = history(6000)

    def compact(rows, *args, **kwargs):
        a.context_compressor.last_prompt_tokens = -1
        a.context_compressor.awaiting_real_usage_after_compression = True
        return [{"role": "user", "content": "p" * 4400}], a._cached_system_prompt

    with (
        patch.object(a, "_compress_context", side_effect=compact) as comp,
        patch.object(a.context_compressor, "get_active_compression_failure_cooldown", return_value=None),
        patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"),
        patch.object(a, "_cleanup_task_resources"), patch.object(a, "_emit_status") as statuses,
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10),
    ):
        result = a.run_conversation("hello", conversation_history=before)
    texts = [str(c.args[0]) for c in statuses.call_args_list if c.args]
    diagnostic = str(result.get("final_response", "")) + " " + " ".join(texts)
    assert comp.call_count >= 1
    assert calls.call_count == 0
    assert result.get("failed") is True
    assert "No messages were dropped" not in diagnostic


def test_codex_managed_relay_growth_refuses_at_httpx(agent, relay_turn):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    a = agent
    a.api_mode = "codex_responses"
    a.provider = "openai-codex"
    a.model = "gpt-5.5"
    a.session_id = "session-1"
    a.context_compressor.context_length = WINDOW
    relay, turn = relay_turn
    inner = CountingTransport()
    http_client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(http_client, covered=True)
    sdk = intercepted_openai_class(OpenAI)(
        api_key="inert-not-a-secret",
        base_url="https://inert.invalid/v1",
        max_retries=0,
        http_client=http_client,
    )
    payload = {"model": a.model, "input": [{"role": "user", "content": "hi"}], "instructions": "safe"}

    def grow(name, request, annotated):
        assert annotated is not None
        annotated.instructions = "r" * 4400
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request("final-wire-relay-growth", 1, False, grow)
    try:
        try:
            _dispatch_nonstreaming_api_request(a, payload, make_client=lambda *args, **kwargs: sdk)
        except Exception as exc:
            refusal = unwrap_local_refusal(exc)
            assert isinstance(refusal, ProviderBoundRequestOverLimit), exc
        else:
            raise AssertionError("Relay-grown Codex request was admitted")
    finally:
        relay.intercepts.deregister_llm_request("final-wire-relay-growth")
        sdk.close()
        http_client.close()
    assert inner.calls == []
