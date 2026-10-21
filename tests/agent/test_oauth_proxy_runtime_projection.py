"""Every runtime snapshot an auxiliary call is built from carries the whole live route.

A named relay resolves to ``provider: custom`` plus ``requested_provider`` (its name), its full
endpoint (tenant query included), the effective model, the declared capability map and the
conversation id. ``anthropic_oauth_proxy`` decides Bearer vs ``x-api-key``, the OAuth beta and the
``x-claude-code-session-id`` header, so a producer that hands an auxiliary call only part of that
(the compressor's summary dispatch, the compression feasibility probe, the review fork, the TUI
``llm.oneshot`` snapshot) re-decides the wire policy on a weaker identity than the main turn used.

Each case runs a real ``AIAgent`` built from real config through its public entry point and reads
the request the SDK actually serialized. An ambient context runtime saying the opposite is bound
around the call: the explicit snapshot a producer passes must be complete on its own.
"""

import json
import time

import httpx
import pytest
import hermes_yaml as yaml

URL = "https://relay.example.com"
OTHER = "https://other.example.com"
MODEL_A = "claude-sonnet-4-6"
MODEL_B = "claude-haiku-4-6"
KEY = "opaque-relay-key"
SESSION = "projection-conversation"
HEADER = "x-claude-code-session-id"
OAUTH_BETA = "oauth-2025-04-20"


def _response(request):
    body = json.loads(request.content or b"{}")
    if request.url.path.endswith("/chat/completions"):
        payload = {
            "id": "cc_test", "object": "chat.completion", "created": 0, "model": body.get("model", MODEL_A),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if body.get("stream"):
            chunks = [
                {"id": "cc_test", "object": "chat.completion.chunk", "created": 0, "model": MODEL_A,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                {"id": "cc_test", "object": "chat.completion.chunk", "created": 0, "model": MODEL_A,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, request=request, content=data, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, request=request, json=payload)
    message = {
        "id": "msg_test", "type": "message", "role": "assistant", "model": body.get("model", MODEL_A),
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if body.get("stream"):
        events = [
            {"type": "message_start", "message": dict(message, content=[])},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, request=request, content=data, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, request=request, json=message)


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    for stale in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(stale, raising=False)
    config = {
        "model": {"provider": "custom:relay", "default": MODEL_A},
        "providers": {
            "relay": {
                "api": URL, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
                "capabilities": {"anthropic_oauth_proxy": True},
                "models": {MODEL_B: {"anthropic_oauth_proxy": False}},
            },
            "other": {"api": OTHER, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages"},
        },
        "auxiliary": {"compression": {"provider": "auto"}},
    }
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        return _response(request)

    async def async_send(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)

    def write(**changes):
        for path, value in changes.items():
            node = config
            *parents, leaf = path.split("__")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    write()
    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()
    yield type("Relay", (), {"requests": requests, "write": staticmethod(write)})
    shutdown_cached_clients()


def build_agent(requested="custom:relay", model=MODEL_A, **kwargs):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    rt = resolve_runtime_provider(requested=requested, target_model=model)
    return AIAgent(
        model=model, provider=rt["provider"], requested_provider=rt["requested_provider"],
        base_url=rt["base_url"], api_key=rt["api_key"], api_mode=rt["api_mode"],
        capabilities=rt.get("capabilities"), enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, session_id=SESSION, **kwargs,
    )


def contrary_ambient(oauth: bool):
    """Bind a context runtime that says the opposite: a complete snapshot must not need it."""
    from agent.auxiliary_client import scoped_runtime_main
    return scoped_runtime_main({
        "provider": "custom", "requested_provider": "custom:relay", "base_url": URL, "model": MODEL_A,
        "api_key": KEY, "api_mode": "anthropic_messages", "session_id": "ambient-conversation",
        "capabilities": {"anthropic_oauth_proxy": not oauth},
    })


def assert_messages_wire(request, oauth: bool, *, session=True, host="relay.example.com", model=None):
    assert request.url.host == host
    assert request.headers.get("authorization") == (f"Bearer {KEY}" if oauth else None)
    assert request.headers.get("x-api-key") == (None if oauth else KEY)
    assert (OAUTH_BETA in request.headers.get("anthropic-beta", "")) is oauth
    assert request.headers.get(HEADER) == (SESSION if oauth and session else None)
    if model:
        assert json.loads(request.content)["model"] == model


def summarize(agent):
    return agent.context_compressor._call_summary_llm("summarize this conversation", time.monotonic())


@pytest.mark.parametrize("oauth", [True, False])
def test_compression_feasibility_snapshot_builds_the_relays_client(relay, oauth):
    """``_current_main_runtime`` feeds the feasibility probe; the client it builds is the relay's."""
    from agent.auxiliary_client import get_text_auxiliary_client

    relay.write(providers__relay__capabilities={"anthropic_oauth_proxy": oauth})
    agent = build_agent()
    with contrary_ambient(oauth):
        client, model = get_text_auxiliary_client("compression", main_runtime=agent._current_main_runtime())
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=8)
    assert_messages_wire(relay.requests[-1], oauth, session=False, model=MODEL_A)


@pytest.mark.parametrize("oauth", [True, False])
def test_compressor_summary_dispatch_keeps_the_relays_wire_policy(relay, oauth):
    """The real summary request: Bearer + OAuth beta + conversation header exactly when declared."""
    relay.write(providers__relay__capabilities={"anthropic_oauth_proxy": oauth})
    agent = build_agent()
    with contrary_ambient(oauth):
        assert summarize(agent) == "ok"
    assert_messages_wire(relay.requests[-1], oauth, model=MODEL_A)


@pytest.mark.parametrize("relay_level", [True, False])
def test_compressor_summary_on_another_model_answers_from_the_named_owner(relay, relay_level):
    """``auxiliary.compression.model`` names another model on the main relay: its own declaration
    (looked up under the relay's name, which only ``requested_provider`` carries) decides."""
    relay.write(
        providers__relay__capabilities={"anthropic_oauth_proxy": relay_level},
        providers__relay__models={MODEL_B: {"anthropic_oauth_proxy": not relay_level}},
        auxiliary__compression__model=MODEL_B,
    )
    agent = build_agent()
    with contrary_ambient(relay_level):
        assert summarize(agent) == "ok"
    assert_messages_wire(relay.requests[-1], not relay_level, model=MODEL_B)


@pytest.mark.parametrize("primary,fallback", [("custom:other", "custom:relay"), ("custom:relay", "custom:other")])
def test_compressor_follows_fallback_and_restore(relay, primary, fallback):
    """Fallback and the next turn's restore both re-point the compressor's route, in both directions."""
    hosts = {"custom:relay": ("relay.example.com", True), "custom:other": ("other.example.com", False)}
    agent = build_agent(requested=primary, fallback_model=[{"provider": fallback, "model": MODEL_A}])

    def summary_on(route):
        host, oauth = hosts[route]
        with contrary_ambient(oauth):
            assert summarize(agent) == "ok"
        assert_messages_wire(relay.requests[-1], oauth, host=host)

    try:
        summary_on(primary)
        assert agent._try_activate_fallback()
        summary_on(fallback)
        assert agent._restore_primary_runtime() is True
        summary_on(primary)
    finally:
        agent._anthropic_client.close()


def test_compressor_follows_a_model_switch_onto_the_relay(relay):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    agent = build_agent(requested="custom:other")
    rt = resolve_runtime_provider(requested="custom:relay", target_model=MODEL_A)
    try:
        agent.switch_model(MODEL_A, "custom:relay", api_key=rt["api_key"], base_url=rt["base_url"],
                           api_mode=rt["api_mode"], capabilities=rt.get("capabilities"))
        with contrary_ambient(True):
            assert summarize(agent) == "ok"
        assert_messages_wire(relay.requests[-1], True)
    finally:
        agent._anthropic_client.close()


def _chat_relay(relay, oauth):
    relay.write(providers__relay__api=URL + "?team=a", providers__relay__transport="chat_completions",
                providers__relay__capabilities={"anthropic_oauth_proxy": oauth})
    agent = build_agent()
    agent.client.chat.completions.create(model=MODEL_A, messages=[{"role": "user", "content": "hi"}], max_tokens=8)
    assert relay.requests[-1].url.params.get_list("team") == ["a"]
    return agent


def assert_chat_wire(request, oauth):
    assert request.url.host == "relay.example.com"
    assert request.url.params.get_list("team") == ["a"]
    assert request.headers.get(HEADER) == (SESSION if oauth else None)


@pytest.mark.parametrize("oauth", [True, False])
def test_tui_oneshot_keeps_the_tenant_query(relay, oauth):
    """``llm.oneshot`` with a live session: the snapshot keeps ``?team=a`` beside the header."""
    from tui_gateway import server

    agent = _chat_relay(relay, oauth)
    sid = "projection-tui"
    server._sessions[sid] = {"agent": agent, "profile_home": None}
    try:
        reply = server._methods["llm.oneshot"]("r1", {
            "session_id": sid, "task": "compression", "instructions": "hello", "input": "review",
        })
    finally:
        server._sessions.pop(sid, None)
    assert reply["result"]["text"] == "ok"
    assert_chat_wire(relay.requests[-1], oauth)


@pytest.mark.parametrize("oauth", [True, False])
def test_compressor_summary_keeps_the_tenant_query(relay, oauth):
    agent = _chat_relay(relay, oauth)
    with contrary_ambient(oauth):
        assert summarize(agent) == "ok"
    assert_chat_wire(relay.requests[-1], oauth)


@pytest.mark.parametrize("oauth", [True, False])
def test_micro_summary_keeps_the_tenant_query(relay, oauth):
    agent = _chat_relay(relay, oauth)
    with contrary_ambient(oauth):
        assert agent.context_compressor._micro_summarize_one("user: hi\nassistant: hello") == "ok"
    assert_chat_wire(relay.requests[-1], oauth)


def test_main_model_retry_after_a_failed_summary_model_keeps_the_tenant_query(relay, monkeypatch):
    """A failed ``auxiliary.compression.model`` falls back to the main model by naming the main
    route explicitly; that explicit route is the live one, query included."""
    relay.write(auxiliary__compression__model=MODEL_B)
    agent = _chat_relay(relay, True)
    delivered = httpx.Client.send

    def send(client, request, **kwargs):
        if json.loads(request.content or b"{}").get("model") == MODEL_B:
            relay.requests.append(request)
            return httpx.Response(404, request=request, json={"error": {"message": "model not found"}})
        return delivered(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    turns = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    with contrary_ambient(True):
        assert agent.context_compressor._generate_summary(turns)
    assert json.loads(relay.requests[-2].content)["model"] == MODEL_B
    assert json.loads(relay.requests[-1].content)["model"] == MODEL_A
    assert_chat_wire(relay.requests[-1], True)


def test_review_fork_keeps_the_named_owner(relay):
    """The background-review / ``/btw`` fork inherits the relay's name, so its own summary call
    on another model still answers from the relay's per-model declaration."""
    from agent.background_review import build_cache_parity_fork

    relay.write(
        providers__relay__capabilities={"anthropic_oauth_proxy": False},
        providers__relay__models={MODEL_B: {"anthropic_oauth_proxy": True}},
        auxiliary__compression__model=MODEL_B,
    )
    agent = build_agent()
    fork, _, routed = build_cache_parity_fork(agent, max_iterations=1)
    try:
        assert routed is False
        with contrary_ambient(True):
            assert summarize(fork) == "ok"
        assert_messages_wire(relay.requests[-1], True, model=MODEL_B)
    finally:
        fork.close()


@pytest.mark.parametrize("oauth", [True, False])
def test_turn_start_title_keeps_the_tenant_query(relay, oauth):
    """The background titler is handed the same projection: its request keeps ``?team=a``."""
    from unittest.mock import MagicMock, patch

    from agent import turn_context
    from agent.auxiliary_client import call_llm

    agent = _chat_relay(relay, oauth)
    agent._session_db, agent._session_db_created, agent.platform = MagicMock(), True, "cli"
    with patch("agent.title_generator.maybe_auto_title") as titler:
        turn_context._maybe_title_session_at_turn_start(agent, [{"role": "user", "content": "Fix the login"}])
    main_runtime = titler.call_args.kwargs["main_runtime"]
    with contrary_ambient(oauth):
        call_llm(task="title_generation", main_runtime=main_runtime,
                 messages=[{"role": "user", "content": "title"}], max_tokens=8)
    assert_chat_wire(relay.requests[-1], oauth)
