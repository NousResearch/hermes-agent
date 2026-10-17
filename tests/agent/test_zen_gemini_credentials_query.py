"""Native Gemini credentials and query options survive runtime/client rebuilding."""

import json

import httpx
import pytest

from agent.gemini_native_adapter import GeminiNativeClient
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home
from run_agent import AIAgent


def capture_http(monkeypatch):
    requests = []

    def send(_client, request, **_kwargs):
        requests.append(request)
        payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize("construction", ["resolved", "routed"])
def test_named_key_command_is_evaluated_for_each_native_request(monkeypatch, caplog, construction):
    (get_hermes_home() / "config.yaml").write_text(
        "providers:\n  zen-command:\n    base_url: https://opencode.ai/zen/v1\n"
        "    key_cmd: contract-token\n", encoding="utf-8",
    )
    tokens = iter(["rotating-contract-one", "rotating-contract-two", "rotating-contract-three"])
    calls = []

    def credential():
        token = next(tokens)
        calls.append(token)
        return token

    monkeypatch.setattr("agent.command_token_source.build_command_token_provider", lambda *_: credential)
    requests = capture_http(monkeypatch)
    runtime = resolve_runtime_provider(requested="custom:zen-command", target_model="gemini-4-argon")
    assert runtime["api_key"] is credential
    options = ({k: runtime[k] for k in ("provider", "requested_provider", "api_mode", "api_key", "base_url")}
               if construction == "resolved" else {"provider": "custom:zen-command"})
    agent = AIAgent(**options, model="gemini-4-argon", quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        assert calls == []
        for stream in (False, True, False):
            client = agent._create_request_openai_client(reason="rotating-key")
            result = client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "OK"}],
                                                    stream=stream)
            if stream:
                list(result)
            agent._close_request_openai_client(client, reason="request_complete")
        assert [r.headers["x-goog-api-key"] for r in requests] == calls
        assert len(calls) == 3
        assert not any(token in caplog.text for token in calls)
    finally:
        agent.close()


@pytest.mark.parametrize("invalid", [" ", None, 7])
def test_invalid_callable_key_fails_closed_without_http(monkeypatch, invalid):
    requests = capture_http(monkeypatch)
    client = GeminiNativeClient(api_key=lambda: invalid, base_url="https://opencode.ai/zen/v1")
    try:
        with pytest.raises(RuntimeError, match="(?i)(key|credential)"):
            client.chat.completions.create(model="gemini-4-argon", messages=[{"role": "user", "content": "OK"}])
        assert requests == []
    finally:
        client.close()


def test_fallback_client_options_survive_rebuild_and_primary_restore(monkeypatch):
    from agent.client_lifecycle import _swap_fallback_clients
    from agent.route_binding import reinstall_runtime_snapshot

    requests = capture_http(monkeypatch)
    agent = AIAgent(api_key="test-key", base_url="https://opencode.ai/zen/v1?tenant=primary",
                    provider="opencode-zen", model="gemini-4-argon", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    fallback = GeminiNativeClient(api_key=lambda: "fallback-command-key", base_url="https://opencode.ai/zen/v1",
                                  default_query={"tenant": "fallback"}, default_headers={"x-contract": "fallback"})
    primary = agent.client
    try:
        _swap_fallback_clients(agent, fallback, "opencode-zen", agent.model, fallback.base_url, "chat_completions")
        client = agent._create_request_openai_client(reason="fallback-query")
        client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "OK"}])
        agent._close_request_openai_client(client, reason="request_complete")
        assert requests[-1].url.params["tenant"] == "fallback"
        assert requests[-1].headers["x-contract"] == "fallback"
        assert requests[-1].headers["x-goog-api-key"] == "fallback-command-key"
        reinstall_runtime_snapshot(agent, agent._primary_runtime)
        client = agent._create_request_openai_client(reason="primary-query")
        client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "OK"}])
        agent._close_request_openai_client(client, reason="request_complete")
        assert requests[-1].url.params["tenant"] == "primary"
    finally:
        agent.close()
        primary.close()
        fallback.close()


@pytest.mark.parametrize("rotation", ["pool", "refresh"])
def test_query_credentials_do_not_follow_route_rotation(monkeypatch, rotation):
    from types import SimpleNamespace

    requests = capture_http(monkeypatch)
    agent = AIAgent(api_key="test-key", base_url="https://opencode.ai/zen/v1?key=old-query-secret",
                    provider="opencode-zen", model="gemini-4-argon", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    try:
        entry = SimpleNamespace(runtime_api_key="new-host-key", runtime_base_url="https://example.invalid/v1",
                                id="other-route")
        if rotation == "pool":
            assert agent._swap_credential(entry)
        else:
            assert agent._adopt_openai_credentials(entry.runtime_api_key, entry.runtime_base_url, reason="refresh")
        client = agent._create_request_openai_client(reason="new-host")
        client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "OK"}])
        agent._close_request_openai_client(client, reason="request_complete")
        assert requests[-1].url.host == "example.invalid"
        assert "old-query-secret" not in str(requests[-1].url)
        assert "key" not in requests[-1].url.params
    finally:
        agent.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("construction", ["raw", "factory", "main"])
def test_native_query_and_headers_survive_every_request(monkeypatch, stream, construction):
    from agent.agent_runtime_helpers import create_openai_client
    from types import SimpleNamespace

    requests = []

    def send(_client, request, **_kwargs):
        requests.append(request)
        continuation = len(requests) == 1
        candidate = {"content": {"parts": [{"text": "OK"}]},
                     "finishReason": "CONTINUATION" if continuation else "STOP"}
        if continuation:
            candidate["continuationToken"] = "next-hop"
        payload = {"candidates": [candidate]}
        if stream:
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", send)
    headers = {"x-contract": "preserved"}
    kwargs = {"api_key": "test-key", "base_url": "https://opencode.ai/zen?tenant=sample&key=query-secret"}
    agent = None
    if construction == "main":
        (get_hermes_home() / "config.yaml").write_text("model:\n  default_headers:\n    x-contract: preserved\n",
                                                     encoding="utf-8")
        agent = AIAgent(**kwargs, provider="opencode-zen", model="gemini-4-argon", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        client = agent._create_request_openai_client(reason="query-contract")
    else:
        if construction == "factory":
            kwargs["base_url"] = "https://opencode.ai/zen"
            kwargs["default_query"] = {"tenant": "sample", "key": "query-secret", "alt": "json"}
            shell = SimpleNamespace(provider="opencode-zen", model="gemini-4-argon",
                                    _client_log_context=lambda: "query-contract",
                                    _build_keepalive_http_client=lambda *a, **k: None)
            client = create_openai_client(shell, {**kwargs, "default_headers": headers}, reason="query", shared=False)
        else:
            client = GeminiNativeClient(**kwargs, default_headers=headers)
    try:
        result = client.chat.completions.create(model="gemini-4-argon", messages=[{"role": "user", "content": "OK"}],
                                                stream=stream, max_tokens=65536)
        if stream:
            list(result)
        assert len(requests) == 2
        for request in requests:
            method = "streamGenerateContent" if stream else "generateContent"
            assert request.url.path == f"/zen/v1/models/gemini-4-argon:{method}"
            assert request.url.params["tenant"] == "sample"
            assert request.url.params["key"] == "query-secret"
            assert request.headers["x-contract"] == "preserved"
            if stream:
                assert request.url.params.get_list("alt") == ["sse"]
        assert json.loads(requests[-1].content)["continuationToken"] == "next-hop"
    finally:
        if agent:
            agent._close_request_openai_client(client, reason="request_complete")
            agent.close()
        else:
            client.close()
