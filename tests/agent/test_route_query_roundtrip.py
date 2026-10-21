"""A relay's URL query reaches the wire exactly as declared, and the live route reports it back.

The query can select a tenant, so ``?team=a&team=b&blank=`` is a route of its own: the SDK request
must carry every pair in order (a repeated key stays repeated, a blank value stays present), and the
URL Hermes rebuilds from the live client for identity decisions must name that same route. Only
HTTPX delivery is intercepted; config, runtime resolution, ``AIAgent`` and the SDKs are real.
"""

import json
from urllib.parse import parse_qsl

import httpx
import pytest
import hermes_yaml as yaml

MODEL = "claude-sonnet-4-6"
KEY = "opaque-relay-key"
ROTATED_KEY = "opaque-relay-key-rotated"
QUERIES = pytest.mark.parametrize("query", ["team=a&team=b&blank=", "team=a"], ids=["repeated-and-blank", "scalar"])
TRANSPORT_PATH = {"anthropic_messages": "/gw/v1/messages", "chat_completions": "/gw/chat/completions"}


def _response(request):
    body = json.loads(request.content or b"{}")
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(200, request=request, json={
            "id": "cc", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    message = {
        "id": "msg", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if body.get("stream"):
        events = [{"type": "message_start", "message": message}, {"type": "message_stop"}]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, request=request, content=data, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, request=request, json=message)


@pytest.fixture
def relay(tmp_path, monkeypatch):
    """``configure(query, transport)`` writes a ``custom:relay`` entry at ``/gw?<query>``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    sent = []

    def send(client, request, **kwargs):
        sent.append(request)
        return _response(request)

    async def async_send(client, request, **kwargs):
        return send(client, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)

    def configure(query, transport, *, primary="relay", pool=False):
        url = f"https://relay.example.com/gw?{query}"
        providers = {
            "relay": {"api": url, "key_env": "TEST_RELAY_KEY", "transport": transport,
                      "capabilities": {"anthropic_oauth_proxy": True}},
            "other": {"api": "https://other.example.com", "key_env": "TEST_RELAY_KEY", "transport": transport},
        }
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": f"custom:{primary}", "default": MODEL}, "providers": providers,
        }), encoding="utf-8")
        if pool:
            (tmp_path / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"custom:relay": [
                {"id": "first", "access_token": KEY, "base_url": url, "priority": 0},
                {"id": "second", "access_token": ROTATED_KEY, "base_url": url, "priority": 1},
            ]}}), encoding="utf-8")
        return url

    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()
    yield sent, configure
    shutdown_cached_clients()


def _agent(requested, **extra):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent
    runtime = resolve_runtime_provider(requested=requested, target_model=MODEL)
    return AIAgent(
        model=MODEL, provider=runtime["provider"], requested_provider=runtime.get("requested_provider"),
        api_key=runtime["api_key"], base_url=runtime["base_url"], api_mode=runtime["api_mode"],
        capabilities=runtime.get("capabilities"), enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, **extra,
    )


def _send(agent):
    hello = [{"role": "user", "content": "hello"}]
    if agent.api_mode == "anthropic_messages":
        agent._anthropic_client.messages.create(model=MODEL, max_tokens=32, messages=hello)
    else:
        agent.client.chat.completions.create(model=MODEL, max_tokens=32, messages=hello)


def _live_route(agent):
    from agent.auxiliary_oauth import client_route_url
    from agent.turn_context import live_route_base_url
    if agent.api_mode == "anthropic_messages":
        return client_route_url(agent._anthropic_client)
    return live_route_base_url(agent)


def _assert_declared(request, query, transport):
    assert request.url.path == TRANSPORT_PATH[transport]
    assert request.url.params.multi_items() == parse_qsl(query, keep_blank_values=True)


def _assert_route(url, reported):
    from hermes_cli.route_identity import same_provider_endpoint
    assert same_provider_endpoint(url, reported), reported


@QUERIES
@pytest.mark.parametrize("transport", ["anthropic_messages", "chat_completions"])
def test_primary_query_survives_wire_rebuild_and_rotation(relay, query, transport):
    """First request, a rebuilt client, and a pool rotation onto the same URL all send the declared
    query, and the route rebuilt from the live client after each step is the declared one."""
    sent, configure = relay
    url = configure(query, transport, pool=True)
    from agent.credential_pool import load_pool
    pool = load_pool("custom:relay")
    assert pool.select().id == "first"
    agent = _agent("custom:relay", credential_pool=pool)
    try:
        _send(agent)
        _assert_declared(sent[-1], query, transport)
        _assert_route(url, _live_route(agent))
        if transport == "anthropic_messages":
            agent._rebuild_anthropic_client()
        else:
            assert agent._replace_primary_openai_client(reason="test_rebuild")
        _send(agent)
        _assert_declared(sent[-1], query, transport)
        recovered, _ = agent._recover_with_credential_pool(status_code=429, has_retried_429=True)
        assert recovered and agent.api_key == ROTATED_KEY
        _send(agent)
        _assert_declared(sent[-1], query, transport)
        _assert_route(url, _live_route(agent))
    finally:
        if getattr(agent, "_anthropic_client", None) is not None:
            agent._anthropic_client.close()


@QUERIES
@pytest.mark.parametrize("transport", ["anthropic_messages", "chat_completions"])
def test_fallback_onto_the_relay_sends_and_reports_its_declared_query(relay, query, transport):
    sent, configure = relay
    url = configure(query, transport, primary="other")
    agent = _agent("custom:other", fallback_model=[{"provider": "custom:relay", "model": MODEL}])
    try:
        assert agent._try_activate_fallback()
        _send(agent)
        _assert_declared(sent[-1], query, transport)
        _assert_route(url, _live_route(agent))
    finally:
        if getattr(agent, "_anthropic_client", None) is not None:
            agent._anthropic_client.close()


@QUERIES
@pytest.mark.parametrize("transport", ["anthropic_messages", "chat_completions"])
@pytest.mark.asyncio
async def test_auxiliary_sync_and_async_calls_send_the_declared_query(relay, query, transport):
    from agent.auxiliary_client import async_call_llm, call_llm
    sent, configure = relay
    configure(query, transport)
    hello = [{"role": "user", "content": "hello"}]
    call_llm(task="compression", provider="custom:relay", model=MODEL, messages=hello, max_tokens=32)
    _assert_declared(sent[-1], query, transport)
    await async_call_llm(task="compression", provider="custom:relay", model=MODEL, messages=hello, max_tokens=32)
    _assert_declared(sent[-1], query, transport)
