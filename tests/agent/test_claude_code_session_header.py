"""An OAuth-proxy relay gets a per-conversation ``x-claude-code-session-id``.

Such a relay fronts several Anthropic subscriptions and tells conversations
apart by that header alone: with every request unlabelled it keeps them all on
whichever account it is currently using, and the other subscriptions idle. The
header is opt-in — it rides only on routes that declared
``capabilities.anthropic_oauth_proxy``.
"""

import json
from types import SimpleNamespace

import httpx
import pytest
import yaml

MODEL = "claude-sonnet-4-6"
URL = "https://relay.example.com"
KEY = "opaque-relay-key"
HEADER = "x-claude-code-session-id"


@pytest.fixture
def relay(tmp_path, monkeypatch):
    """A custom ``anthropic_messages`` provider whose proxy capability is a parameter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    requests = []

    def write_config(enabled: bool):
        provider: dict = {
            "api": URL,
            "key_env": "TEST_RELAY_KEY",
            "transport": "anthropic_messages",
        }
        if enabled:
            provider["capabilities"] = {"anthropic_oauth_proxy": True}
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {"provider": "custom:relay", "default": MODEL},
                "providers": {"relay": provider},
            }),
            encoding="utf-8",
        )

    def send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    return SimpleNamespace(requests=requests, write_config=write_config)


def build_agent(capabilities, session_id):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    runtime = resolve_runtime_provider(requested="custom:relay", target_model=MODEL)
    return AIAgent(
        model=MODEL,
        provider=runtime["provider"],
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        api_mode=runtime["api_mode"],
        capabilities=capabilities,
        session_id=session_id,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_session_header_reaches_the_relay_only_when_declared(relay, enabled):
    """The wire carries the session id on a proxy route, and nothing on a plain one."""
    relay.write_config(enabled)
    agent = build_agent({"anthropic_oauth_proxy": True} if enabled else {}, "20260915_190000_abc123")
    try:
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
        agent._anthropic_client.messages.create(**kwargs)
    finally:
        agent._anthropic_client.close()

    sent = relay.requests[-1].headers.get(HEADER)
    assert sent == ("20260915_190000_abc123" if enabled else None)


def test_each_conversation_gets_its_own_id_and_keeps_it_across_turns(relay):
    """Distinct sessions must differ — that difference is what the relay balances on."""
    relay.write_config(True)
    seen = []
    for session_id in ("20260915_190000_aaa", "20260915_190100_bbb"):
        agent = build_agent({"anthropic_oauth_proxy": True}, session_id)
        try:
            for _ in range(2):  # two turns of one conversation
                kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
                seen.append(kwargs["extra_headers"][HEADER])
        finally:
            agent._anthropic_client.close()

    first, second = seen[:2], seen[2:]
    assert len(set(first)) == 1 and len(set(second)) == 1, "one conversation must keep one id"
    assert first[0] != second[0], "different conversations must not share a pin"


def test_auxiliary_calls_join_the_main_turn_conversation(relay):
    """A title/compression call must not pin a second account for the same conversation."""
    from agent.auxiliary_client import _build_call_kwargs, reset_runtime_main, set_runtime_main

    relay.write_config(True)
    token = set_runtime_main(
        "custom:relay", MODEL, requested_provider="custom:relay", base_url=URL,
        api_key=KEY, api_mode="anthropic_messages", session_id="20260915_190000_abc123",
        capabilities={"anthropic_oauth_proxy": True},
    )
    try:
        kwargs = _build_call_kwargs(
            "custom:relay", MODEL, [{"role": "user", "content": "summarize"}], base_url=URL,
        )
    finally:
        reset_runtime_main(token)

    assert kwargs.get("extra_headers", {}).get(HEADER) == "20260915_190000_abc123"


def test_unroutable_session_ids_are_replaced_by_a_stable_digest():
    """A relay validates the header's shape; an id it would drop must still identify the session."""
    from agent.claude_code_session import claude_code_session_id

    messy = "cron job/2026 09 15"
    first, second = claude_code_session_id(messy), claude_code_session_id(messy)
    assert first == second
    assert first.startswith("hermes_")
    assert first.replace("hermes_", "").isalnum()
    assert claude_code_session_id("other/session") != first
