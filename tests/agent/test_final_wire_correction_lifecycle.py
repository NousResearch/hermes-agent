"""Correction refusals survive real dispatch/lazy/Relay/summary cleanup paths."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from agent import final_wire_admission as admission
from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request, handle_max_iterations
from tests.run_agent.test_413_compression import agent  # noqa: F401
from tests.agent.test_final_wire_boundary_regressions import relay_turn  # noqa: F401
from tests.agent.test_final_wire_review_regressions import identity
from tests.agent.test_final_wire_current_stream import Pool, body, replace_stream


def mutate(request, defect):
    final = json.loads(request.content)
    if defect == "stream":
        final["messages"] = [{"role": "user", "content": "x" * 4400}]
    elif defect == "model":
        final["model"] = "inert-smaller-model"
    else:
        final["max_tokens"] = 1
        final["max_completion_tokens"] = True
    replace_stream(request, final)


@pytest.mark.parametrize("defect", ["stream", "model", "cap"])
def test_main_summary_refusal_does_not_retry_sleep_or_leak_binding(agent, monkeypatch, defect):
    a = agent
    a.model, a.provider, a.api_mode, a.base_url = identity().model, "openai", "chat_completions", identity().endpoint
    a._base_url_lower = a.base_url
    a.context_compressor.context_length = 1000
    a.max_tokens = 1
    a.reasoning_config = None
    pool, sleeps, attempts = Pool(), [], []
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    monkeypatch.setattr("time.sleep", lambda *args: sleeps.append((sys._getframe(1).f_code.co_name, args)))
    def hook(request):
        attempts.append(admission.current_attempt_identity())
        mutate(request, defect)
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=a.base_url, http_client=http, max_retries=2)
        monkeypatch.setattr(a, "_ensure_primary_openai_client", lambda **kw: sdk)
        monkeypatch.setattr(a, "_supports_reasoning_extra_body", lambda: False)
        handle_max_iterations(a, [{"role": "assistant", "content": "inert work"}], 1)
    assert len(attempts) == 1
    assert attempts[0].window == 1000
    assert pool.sent == []
    # Existing worker polling is not provider retry backoff.
    assert not any(name == "_sleep_for_retry" for name, args in sleeps)
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("defect", ["stream", "model", "cap"])
def test_real_primary_failure_cleans_up_without_fallback_or_sleep(agent, monkeypatch, defect):
    a = agent
    a.tools = []
    a.model, a.provider, a.api_mode, a.base_url = identity().model, "openai", "chat_completions", identity().endpoint
    a._base_url_lower = a.base_url
    a.context_compressor.context_length = 1000
    a.context_compressor.threshold_tokens = 950
    a.max_tokens = 1
    pool, sleeps = Pool(), []
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    monkeypatch.setattr("time.sleep", lambda *args: sleeps.append((sys._getframe(1).f_code.co_name, args)))
    with httpx.Client(transport=transport, event_hooks={"request": [lambda request: mutate(request, defect)]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=a.base_url, http_client=http, max_retries=2)
        a.client = sdk
        monkeypatch.setattr(a, "_ensure_primary_openai_client", lambda **kw: sdk)
        monkeypatch.setattr(a, "_create_request_openai_client", lambda **kw: sdk)
        monkeypatch.setattr(a, "_close_request_openai_client", lambda *args, **kw: None)
        with (patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"),
              patch.object(a, "_cleanup_task_resources") as cleanup,
              patch.object(a, "_try_activate_fallback") as fallback):
            a.run_conversation("hi")
    assert pool.sent == []
    fallback.assert_not_called()
    cleanup.assert_called_once()
    # Existing worker polling is not provider retry backoff.
    assert not any(name == "_sleep_for_retry" for name, args in sleeps)
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


def test_real_relay_model_override_cannot_reuse_bound_window(agent, relay_turn, monkeypatch):
    a = agent
    a.api_mode, a.provider, a.model, a.base_url = "codex_responses", "openai-codex", "inert-original-model", identity().endpoint
    a.context_compressor.context_length = 1000
    relay, turn = relay_turn
    sent, sleeps = [], []
    monkeypatch.setattr("time.sleep", lambda *args: sleeps.append((sys._getframe(1).f_code.co_name, args)))
    def change(name, request, annotated):
        assert annotated is not None
        annotated.model = "inert-smaller-model"
        return relay.LLMRequestInterceptOutcome(request, annotated)
    with httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json={}))) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=a.base_url, http_client=http, max_retries=2)
        relay.intercepts.register_llm_request("correction-model-change", 1, False, change)
        try:
            with pytest.raises(admission.ProviderBoundInvalidAccounting):
                _dispatch_nonstreaming_api_request(a, {"model": a.model, "input": "hi"}, make_client=lambda *a, **kw: sdk)
        finally:
            relay.intercepts.deregister_llm_request("correction-model-change")
    assert sent == []
    # Existing worker polling is not provider retry backoff.
    assert not any(name == "_sleep_for_retry" for name, args in sleeps)
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


def test_concurrent_final_model_change_is_isolated_from_matching_attempt():
    def send(changed):
        pool = Pool()
        transport = httpx.HTTPTransport(trust_env=False)
        transport._pool = pool
        ident = replace(identity(), window=1000 if changed else 2000, correlation_id=str(changed))
        with httpx.Client(transport=transport) as http:
            admission.wrap_httpx_client_transports(http)
            with admission.bind_attempt_identity(ident):
                final = body("hi")
                if changed:
                    final["model"] = "inert-unknown"
                    with pytest.raises(admission.ProviderBoundInvalidAccounting):
                        http.post(ident.endpoint + "/chat/completions", json=final)
                else:
                    http.post(ident.endpoint + "/chat/completions", json=final)
        assert admission.current_attempt_identity() is None
        assert admission.current_local_refusal() is None
        return pool.sent
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(send, [True, False]))
    assert results == [[], [body("hi")]]
