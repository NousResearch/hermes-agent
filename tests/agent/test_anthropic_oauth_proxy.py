"""An explicit proxy capability controls auth and payloads across primary/aux routes."""

import json
from types import SimpleNamespace

import httpx
import pytest
import yaml

MODEL = "claude-sonnet-4-6"
URL = "https://relay.example.com"
KEY = "opaque-relay-key"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"provider": "custom:relay", "default": MODEL},
            "providers": {
                "relay": {
                    "api": URL,
                    "key_env": "TEST_RELAY_KEY",
                    "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": True},
                }
            },
        }),
        encoding="utf-8",
    )
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if json.loads(request.content).get("stream"):
            events = [
                {"type": "message_start", "message": message},
                {"type": "message_stop"},
            ]
            data = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
            return httpx.Response(
                200,
                request=request,
                content=data,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, request=request, json=message)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


def assert_wire(request, enabled, tools, host="relay.example.com", key=KEY):
    body = json.loads(request.content)
    assert request.url.host == host
    assert request.headers.get("authorization") == (
        f"Bearer {key}" if enabled else None
    )
    assert request.headers.get("x-api-key") == (None if enabled else key)
    assert ("oauth-2025-04-20" in request.headers.get("anthropic-beta", "")) == enabled
    assert ("Claude Code" in json.dumps(body.get("system", []))) == enabled
    if tools:
        assert body["tools"][0]["name"].startswith("mcp__") == enabled


@pytest.mark.parametrize(
    "native_label", [False, True], ids=["custom-provider", "native-label-custom-url"]
)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("tools", [TOOLS, []], ids=["tools", "text-only"])
def test_primary_proxy_auth_and_payload_are_opt_in(relay, enabled, tools, native_label):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.agent_init import _init_anthropic_client
    from agent.anthropic_adapter import build_anthropic_kwargs

    runtime = resolve_runtime_provider(requested="custom:relay", target_model=MODEL)
    if native_label:
        runtime.update(provider="anthropic", api_key="sk-ant-oat01-test-relay")
    capabilities = dict(runtime["capabilities"], anthropic_oauth_proxy=enabled)
    agent = SimpleNamespace(
        provider=runtime["provider"], capabilities=capabilities, quiet_mode=True
    )
    _init_anthropic_client(agent, runtime["api_key"], runtime["base_url"], 5)
    try:
        kwargs = build_anthropic_kwargs(
            model=MODEL,
            messages=[
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant", "content": "hello back",
                    "anthropic_content_blocks": [
                        {"type": "thinking", "thinking": "A greeting", "signature": "relay-signature"},
                        {"type": "text", "text": "hello back"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            tools=tools,
            max_tokens=32,
            reasoning_config=None,
            is_oauth=agent._is_anthropic_oauth,
            base_url=URL,
        )
        agent._anthropic_client.messages.create(**kwargs)
        assert_wire(relay[-1], enabled, tools, key=runtime["api_key"])
        wire_messages = json.loads(relay[-1].content)["messages"]
        signatures = [
            block["signature"]
            for message in wire_messages for block in message["content"]
            if isinstance(block, dict) and "signature" in block
        ]
        assert signatures == (["relay-signature"] if enabled else [])
    finally:
        agent._anthropic_client.close()


@pytest.mark.parametrize("route", ["custom:relay", "auto", "custom:other"])
def test_auxiliary_runtime_capability_survives_routing_and_cache(
    relay, route, tmp_path
):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.auxiliary_client import _get_cached_client

    runtime = resolve_runtime_provider(requested="custom:relay", target_model=MODEL)
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["providers"]["other"] = {
        "api": "https://other.example.com",
        "key_env": "TEST_RELAY_KEY",
        "transport": "anthropic_messages",
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Same route/credential/model, different session capability: must not reuse the wrong client.
    for enabled in (False, True, False):
        current = dict(
            runtime,
            model=MODEL,
            requested_provider="custom:relay",
            capabilities={"anthropic_oauth_proxy": enabled},
        )
        client, model = _get_cached_client(route, MODEL, main_runtime=current)
        assert client is not None
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=32
        )
        assert_wire(
            relay[-1],
            enabled and route != "custom:other",
            [],
            host="other.example.com"
            if route == "custom:other"
            else "relay.example.com",
        )


def test_fallback_follows_destination_oauth_policy(relay, tmp_path):
    from run_agent import AIAgent
    from hermes_cli.runtime_provider import resolve_runtime_provider

    path = tmp_path / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["providers"]["other"] = {
        "api": "https://other.example.com",
        "key_env": "TEST_RELAY_KEY",
        "transport": "anthropic_messages",
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = resolve_runtime_provider(requested="custom:other", target_model=MODEL)
    agent = AIAgent(
        model=MODEL,
        provider=runtime["provider"],
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        api_mode=runtime["api_mode"],
        capabilities=runtime.get("capabilities"),
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        fallback_model=[
            {
                "provider": "custom:relay",
                "model": MODEL,
                "api_mode": "anthropic_messages",
            },
            {
                "provider": "custom:other",
                "model": MODEL,
                "api_mode": "anthropic_messages",
            },
        ],
    )
    try:
        assert agent._try_activate_fallback()
        assert agent._is_anthropic_oauth is True
        client = agent._build_direct_anthropic_client(KEY, agent.base_url)
        try:
            client.messages.create(
                model=MODEL,
                max_tokens=32,
                messages=[{"role": "user", "content": "hello"}],
            )
            assert relay[-1].headers.get("authorization") == f"Bearer {KEY}"
        finally:
            client.close()
        assert agent._try_activate_fallback()
        assert agent._is_anthropic_oauth is False
        assert agent.capabilities.get("anthropic_oauth_proxy", False) is False
    finally:
        agent._anthropic_client.close()


@pytest.mark.parametrize("enabled", [True, False])
def test_custom_pool_rotation_preserves_oauth_wire_policy(relay, tmp_path, enabled):
    """Regression for #64896: _swap_credential must retain the route's opt-in."""
    from agent.anthropic_adapter import build_anthropic_kwargs
    from agent.credential_pool import load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    rotated_key = "opaque-relay-key-rotated"
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"custom:relay": [
            {"id": "first", "access_token": KEY, "base_url": URL, "priority": 0},
            {"id": "second", "access_token": rotated_key, "base_url": URL, "priority": 1},
        ]},
    }), encoding="utf-8")
    pool = load_pool("custom:relay")
    entry = pool.select()
    assert entry.id == "first"
    runtime = resolve_runtime_provider(requested="custom:relay", target_model=MODEL)
    agent = AIAgent(
        model=MODEL,
        provider=runtime["provider"],
        api_key=entry.runtime_api_key,
        base_url=runtime["base_url"],
        api_mode=runtime["api_mode"],
        capabilities={"anthropic_oauth_proxy": enabled},
        credential_pool=pool,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        for key in (KEY, rotated_key):
            if key == rotated_key:
                recovered, _ = agent._recover_with_credential_pool(
                    status_code=429, has_retried_429=True,
                )
                assert recovered
                assert pool.current().id == "second"
                assert agent.api_key == rotated_key
            kwargs = build_anthropic_kwargs(
                model=MODEL,
                messages=[{"role": "user", "content": "hello"}],
                tools=TOOLS,
                max_tokens=32,
                reasoning_config=None,
                is_oauth=agent._is_anthropic_oauth,
                base_url=agent.base_url,
            )
            agent._anthropic_client.messages.create(**kwargs)
            assert_wire(relay[-1], enabled, TOOLS, key=key)
        agent._rebuild_anthropic_client()
        agent._anthropic_client.messages.create(**kwargs)
        assert_wire(relay[-1], enabled, TOOLS, key=rotated_key)
    finally:
        agent._anthropic_client.close()


def test_resume_resolves_same_provider_model_capabilities(relay):
    from cli import HermesCLI

    class Destination:
        def switch_model(self, **kwargs):
            self.capabilities = kwargs.get("capabilities")

    cli = object.__new__(HermesCLI)
    cli.model = "claude-opus-4-6"
    cli.provider = cli.requested_provider = "custom:relay"
    cli.api_key, cli.base_url, cli.api_mode = KEY, URL, "anthropic_messages"
    cli.agent = Destination()
    cli._console_print = lambda message: None
    cli._restore_session_model({
        "model": MODEL,
        "model_config": json.dumps({
            "provider": "custom:relay",
            "base_url": URL,
            "api_mode": "anthropic_messages",
        }),
    })
    assert cli.agent.capabilities == {"anthropic_oauth_proxy": True}
