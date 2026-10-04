"""Main summary/recovery remain covered; auxiliary designation stays distinct."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from agent.chat_completion_helpers import handle_max_iterations, _dispatch_nonstreaming_api_request
from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.final_wire_admission import (
    current_attempt_identity, current_local_refusal, intercepted_openai_class,
    wrap_httpx_client_transports, COVERED_MAIN, NOT_COVERED_AUXILIARY,
)
from tests.run_agent.test_413_compression import agent  # noqa: F401


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("growth", [False, True])
def test_actual_main_terminal_summary_physical_identity(agent, monkeypatch, growth, retry):
    a = agent
    a.provider, a.api_mode, a.model, a.base_url = "openai", "chat_completions", "inert", "https://inert.invalid"
    a._base_url_lower = a.base_url
    a.context_compressor.context_length = 1000
    a.max_tokens = 1
    a.reasoning_config = None
    seen, identities = [], []
    def hook(request):
        identities.append(current_attempt_identity())
        if growth:
            body = json.loads(request.content)
            body["messages"] = [{"role": "user", "content": "x" * 4400}]
            request.stream = httpx.ByteStream(json.dumps(body).encode())
            if hasattr(request, "_content"):
                del request._content
            request.read()
    def receive(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "inert", "object": "chat.completion", "created": 0, "model": "inert", "choices": [{"index": 0, "message": {"role": "assistant", "content": "" if retry and len(seen) == 1 else "summary"}, "finish_reason": "stop"}]})
    with httpx.Client(transport=httpx.MockTransport(receive), event_hooks={"request": [hook]}) as http:
        wrap_httpx_client_transports(http)
        sdk = intercepted_openai_class(OpenAI)(api_key="inert", base_url=a.base_url, http_client=http, max_retries=2)
        monkeypatch.setattr(a, "_ensure_primary_openai_client", lambda **kw: sdk)
        monkeypatch.setattr(a, "_supports_reasoning_extra_body", lambda: False)
        text = handle_max_iterations(a, [{"role": "assistant", "content": "inert work"}], 1)
        expected = 2 if retry and not growth else 1
        assert len(seen) == (0 if growth else expected)
        assert len(identities) == expected
        assert all(ident is not None and ident.purpose == COVERED_MAIN and ident.window == 1000 and ident.family == "chat_completions" for ident in identities)
        assert ("summary" == text) if not growth else ("missing attempt identity" not in text)
        assert current_attempt_identity() is None
        assert current_local_refusal() is None


@pytest.mark.parametrize("prewrapped", [False, True])
def test_actual_credential_swap_promotes_supplied_aux_transport(agent, monkeypatch, prewrapped):
    a = agent
    a.provider, a.api_mode, a.model, a.base_url = "openai", "chat_completions", "inert", "https://inert.invalid"
    a.context_compressor.context_length = 1000
    seen = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={}))) as http:
        if prewrapped:
            wrap_httpx_client_transports(http, covered=False)
        a._client_kwargs = {"api_key": "inert", "base_url": a.base_url, "http_client": http}
        monkeypatch.setattr("run_agent.OpenAI", intercepted_openai_class(OpenAI))
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
        a._swap_credential(SimpleNamespace(id="inert-entry", runtime_api_key="inert-next", runtime_base_url=a.base_url))
        with pytest.raises(ProviderBoundRequestOverLimit):
            _dispatch_nonstreaming_api_request(a, {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "extra_body": {"messages": [{"role": "user", "content": "x" * 4400}]}}, make_client=lambda *args, **kw: a.client)
        assert seen == []
        assert current_attempt_identity() is None


def test_deliberate_not_covered_aux_transport_has_no_main_binding_requirement():
    seen = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))) as http:
        wrap_httpx_client_transports(http, covered=False)
        assert NOT_COVERED_AUXILIARY != COVERED_MAIN
        http.post("https://inert.invalid", json={"auxiliary": "inert"})
    assert len(seen) == 1
    assert current_attempt_identity() is None
