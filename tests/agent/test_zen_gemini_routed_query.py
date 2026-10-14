"""Named-custom router queries survive main clients, rebuilds and primary restore."""

import json

import httpx
import pytest

from agent.gemini_native_adapter import GeminiNativeClient
from agent.route_binding import bind_route_entry, reinstall_primary_runtime
from hermes_constants import get_hermes_home
from run_agent import AIAgent


def _capture_http(monkeypatch):
    requests = []

    def send(_client, request, **_kwargs):
        requests.append(request)
        continuation = "continuationToken" not in json.loads(request.content)
        candidate = {
            "content": {"parts": [{"text": "OK"}]},
            "finishReason": "CONTINUATION" if continuation else "STOP",
        }
        if continuation:
            candidate["continuationToken"] = "dummy-next-hop"
        payload = {"candidates": [candidate]}
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200, request=request, headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(payload)}\n\n",
            )
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


def _assert_completion(client, model, requests, stream, tenant):
    assert isinstance(client, GeminiNativeClient)
    before = len(requests)
    result = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "OK"}],
        stream=stream, max_tokens=65536,
    )
    if stream:
        list(result)
    else:
        assert result.choices[0].message.content == "OKOK"
    sent = requests[before:]
    assert len(sent) == 2
    method = "streamGenerateContent" if stream else "generateContent"
    for request in sent:
        assert request.url.host == "opencode.ai"
        assert request.url.path == f"/zen/v1/models/{model}:{method}"
        assert request.url.params.get("tenant") == tenant
        assert request.url.params.get_list("key") == [f"dummy-query-{tenant}"]
        assert request.url.params.get_list("alt") == ["sse" if stream else "json"]
        assert request.headers["x-contract"] == tenant
        assert request.headers["x-model-contract"] == "preserved"
    assert json.loads(sent[-1].content)["continuationToken"] == "dummy-next-hop"


@pytest.mark.parametrize("credential_kind", ["static", "callable"])
@pytest.mark.parametrize("stream", [False, True])
def test_router_query_survives_main_rebuilds_and_fallback_restore(monkeypatch, caplog, credential_kind, stream):
    issued = []

    def credential():
        token = f"dummy-command-key-{len(issued)}"
        issued.append(token)
        return token

    auth = ("    key_cmd: dummy-key-command\n" if credential_kind == "callable"
            else "    api_key: dummy-static-primary-key\n")
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  default_headers:\n    x-model-contract: preserved\n"
        "providers:\n"
        "  zen-query:\n"
        "    base_url: https://opencode.ai/zen/v1?tenant=primary&key=dummy-query-primary&alt=json\n"
        "    extra_headers:\n      x-contract: primary\n" + auth +
        "  zen-fallback-query:\n"
        "    base_url: https://opencode.ai/zen/v1?tenant=fallback&key=dummy-query-fallback&alt=json\n"
        "    api_key: dummy-static-fallback-key\n"
        "    extra_headers:\n      x-contract: fallback\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.command_token_source.build_command_token_provider", lambda *_: credential)
    requests = _capture_http(monkeypatch)
    # No resolved credentials/base_url: exercise the real named-custom router at construction.
    agent = AIAgent(
        provider="custom:zen-query", model="gemini-4-argon", quiet_mode=True,
        skip_context_files=True, skip_memory=True,
    )
    retired = []
    try:
        assert issued == []
        _assert_completion(agent.client, agent.model, requests, stream, "primary")
        expected_query = {"tenant": "primary", "key": "dummy-query-primary", "alt": "json"}
        assert agent._client_kwargs["default_query"] == expected_query
        assert agent._primary_runtime["client_kwargs"]["default_query"] == expected_query
        if credential_kind == "callable":
            assert agent.api_key is credential
            assert agent._primary_runtime["client_kwargs"]["api_key"] is credential

        retired.append(agent.client)
        assert agent._replace_primary_openai_client(reason="routed-query-rebuild")
        assert agent.client is not retired[-1]
        _assert_completion(agent.client, agent.model, requests, stream, "primary")
        for _ in range(2):
            client = agent._create_request_openai_client(reason="routed-query-request")
            try:
                _assert_completion(client, agent.model, requests, stream, "primary")
            finally:
                agent._close_request_openai_client(client, reason="routed-query-rebuild")

        primary = agent._primary_runtime
        retired.append(agent.client)
        assert bind_route_entry(agent, {}, "custom:zen-fallback-query", agent.model)
        fallback = agent._create_request_openai_client(reason="routed-query-fallback")
        try:
            _assert_completion(fallback, agent.model, requests, stream, "fallback")
        finally:
            agent._close_request_openai_client(fallback, reason="request_complete")
        retired.append(agent.client)
        reinstall_primary_runtime(
            agent, primary, primary["provider"], primary["model"],
            lambda pool: pool.provider == primary["provider"], lambda: None, prefetched=True,
        )
        assert agent._fallback_activated is False
        _assert_completion(agent.client, agent.model, requests, stream, "primary")
        restored = agent._create_request_openai_client(reason="routed-query-restored")
        try:
            assert restored is not fallback
            _assert_completion(restored, agent.model, requests, stream, "primary")
        finally:
            agent._close_request_openai_client(restored, reason="request_complete")

        primary_keys = [r.headers["x-goog-api-key"] for r in requests if r.url.params["tenant"] == "primary"]
        if credential_kind == "callable":
            assert primary_keys == issued
            assert len(set(issued)) == len(issued)
        else:
            assert primary_keys == ["dummy-static-primary-key"] * len(primary_keys)
            assert issued == []
        assert all(r.headers["x-goog-api-key"] == "dummy-static-fallback-key"
                   for r in requests if r.url.params["tenant"] == "fallback")
        for token in issued + ["dummy-static-primary-key", "dummy-static-fallback-key",
                               "dummy-query-primary", "dummy-query-fallback"]:
            assert token not in caplog.text
    finally:
        agent.close()
        for client in retired:
            client.close()
