"""Router virtual-provider mode tests: agent init installs RouterClient with
pinned chat_completions, and mid-session switch_model does the same.

Sibling of tests/agent/test_moa_switch_api_mode.py — the router facade only
speaks chat.completions; any other api_mode would dispatch .responses.create
against the router://local placeholder.
"""

from __future__ import annotations

import types

import pytest


def _make_fake_agent():
    """A minimal stand-in carrying only the attributes switch_model touches."""
    agent = types.SimpleNamespace()
    agent.model = "minimax-m3"
    agent.provider = "opencode-go"
    agent.api_mode = "anthropic_messages"
    agent.api_key = "old-key"
    agent.base_url = "https://old.example/v1"
    agent.client = object()
    agent._client_kwargs = {"base_url": "https://old.example/v1"}
    agent._config_context_length = 123456
    agent._transport_cache = {}
    agent.quiet_mode = True
    agent.platform = "whatsapp"
    return agent


@pytest.mark.parametrize(
    "incoming_api_mode",
    ["codex_responses", "anthropic_messages", "chat_completions", ""],
)
def test_switch_to_router_pins_chat_completions(monkeypatch, incoming_api_mode):
    from agent import agent_runtime_helpers as arh

    monkeypatch.setattr(arh, "load_pool", lambda *a, **k: None, raising=False)

    agent = _make_fake_agent()
    try:
        arh.switch_model(
            agent,
            new_model="default",
            new_provider="router",
            api_key="router-virtual-provider",
            base_url="router://local",
            api_mode=incoming_api_mode,
        )
    except Exception:
        # switch_model does post-swap work (compressor, pool, runtime) that may
        # raise against a fake agent. The runtime-field swap — including the
        # api_mode pin in the router branch — happens before any of that.
        pass

    assert agent.provider == "router"
    assert agent.base_url == "router://local"
    assert agent.api_mode == "chat_completions", (
        f"Router switch left api_mode={agent.api_mode!r}; the primary call "
        "would dispatch against router://local instead of "
        "RouterClient.chat.completions."
    )
    assert type(agent.client).__name__ == "RouterClient"
    # The decision relay must be wired on the switch path too — without it,
    # routing notes silently disappear after a mid-session /model switch
    # onto the router (agent_init and switch_model share the helper).
    assert agent.client.chat.completions.decision_callback is not None


def test_switch_relay_reaches_tool_progress_callback(monkeypatch):
    """The relay installed by switch_model forwards router.decision events to
    the agent's tool_progress_callback with the tui_gateway arg shape."""
    from agent import agent_runtime_helpers as arh

    monkeypatch.setattr(arh, "load_pool", lambda *a, **k: None, raising=False)

    agent = _make_fake_agent()
    seen: list[tuple] = []
    agent.tool_progress_callback = lambda *args, **kwargs: seen.append((args, kwargs))
    try:
        arh.switch_model(
            agent,
            new_model="default",
            new_provider="router",
            api_key="router-virtual-provider",
            base_url="router://local",
            api_mode="chat_completions",
        )
    except Exception:
        pass

    agent.client.chat.completions.decision_callback(
        "router.decision", label="lmstudio:m", tier="simple", classifier_label="c", cached=False
    )
    assert seen, "relay did not reach tool_progress_callback"
    args, kwargs = seen[0]
    assert args[0] == "router.decision"
    assert args[1] == "lmstudio:m"
    assert args[2] == "simple"


def test_router_client_implements_conversation_loop_protocol():
    """The duck-typed protocol conversation_loop.py consumes must exist on
    RouterClient exactly as it does on MoAClient: consume_reference_usage,
    consume_and_save_trace, last_aggregator_slot."""
    from agent.router_loop import RouterClient

    client = RouterClient("default")
    assert hasattr(client, "consume_reference_usage")
    assert hasattr(client, "consume_and_save_trace")
    assert hasattr(client, "last_aggregator_slot")
    usage, cost = client.consume_reference_usage()
    assert usage.input_tokens == 0
    assert cost is None
    assert client.last_aggregator_slot is None
    # No pending trace — must be a silent no-op.
    client.consume_and_save_trace("session-x")


def test_runtime_provider_resolves_router_virtual():
    from hermes_cli.runtime_provider import resolve_runtime_provider

    rt = resolve_runtime_provider(requested="router")
    assert rt["provider"] == "router"
    assert rt["api_mode"] == "chat_completions"
    assert rt["base_url"] == "router://local"
    assert rt["api_key"] == "router-virtual-provider"


def test_router_provider_row_in_picker():
    from hermes_cli.inventory import _router_provider_row

    row = _router_provider_row("")
    assert row is not None
    assert row["slug"] == "router"
    assert row["name"] == "Model Router"
    assert row["auth_type"] == "virtual"
    assert row["authenticated"] is True
    assert "default" in row["models"]

    current = _router_provider_row("router")
    assert current["is_current"] is True
