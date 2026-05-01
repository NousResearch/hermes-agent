"""Tests for plugin-driven per-turn model/provider overrides.

These complement test_pre_gateway_dispatch.py. They cover the override
mechanism added in this change:

- pre_gateway_dispatch plugins can set MessageEvent.model_override,
  MessageEvent.provider_override, and MessageEvent.model_tier_hint via
  the hook result dict.
- The dispatcher reads those fields and applies them per-turn via
  GatewayRunner._apply_turn_model_override.
- Overrides apply to the current dispatch only; they do not persist
  to the next turn (the next MessageEvent starts with all overrides
  None).
- Override coexists with rewrite/skip actions on the same hook call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform = Platform.WHATSAPP):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


def test_no_override_preserves_existing_model_and_provider():
    """When no plugin sets override fields, _apply_turn_model_override
    must return the original model/runtime_kwargs unchanged."""
    runner, _ = _make_runner()

    base_model = "gpt-4o-mini"
    base_runtime = {"provider": "openai", "api_key": "sk-test", "base_url": "https://api.openai.com/v1"}

    out_model, out_runtime = runner._apply_turn_model_override(
        base_model,
        base_runtime,
        model_override=None,
        provider_override=None,
        model_tier_hint=None,
        session_key="test-session",
    )

    assert out_model == base_model
    assert out_runtime is base_runtime  # exact same object, not a copy
    assert out_runtime == {"provider": "openai", "api_key": "sk-test", "base_url": "https://api.openai.com/v1"}


def test_override_swaps_model_and_provider_for_one_turn(monkeypatch):
    """A plugin-set model_override + provider_override must swap both
    via resolve_runtime_provider for the current turn."""
    runner, _ = _make_runner()

    fake_resolved = {
        "provider": "anthropic",
        "api_key": "sk-ant-test",
        "base_url": "https://api.anthropic.com",
        "api_mode": "messages",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    def _fake_resolve(requested=None):
        assert requested == "anthropic"
        return fake_resolved

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve)

    base_model = "gpt-4o-mini"
    base_runtime = {"provider": "openai", "api_key": "sk-test", "base_url": "https://api.openai.com/v1"}

    out_model, out_runtime = runner._apply_turn_model_override(
        base_model,
        base_runtime,
        model_override="claude-sonnet-4-6",
        provider_override="anthropic",
        model_tier_hint="frontier",
        session_key="test-session",
    )

    assert out_model == "claude-sonnet-4-6"
    assert out_runtime is not base_runtime  # must be a copy, not in-place mutation
    assert out_runtime["provider"] == "anthropic"
    assert out_runtime["api_key"] == "sk-ant-test"
    assert out_runtime["base_url"] == "https://api.anthropic.com"
    # Original runtime is untouched.
    assert base_runtime["provider"] == "openai"
    assert base_runtime["api_key"] == "sk-test"


def test_override_does_not_persist_to_next_turn(monkeypatch):
    """Override applies only to the turn whose MessageEvent set it.
    A second call with no override must return the original model/runtime."""
    runner, _ = _make_runner()

    def _fake_resolve(requested=None):
        return {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "base_url": "https://api.anthropic.com",
            "api_mode": "messages",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve)

    base_model = "gpt-4o-mini"
    base_runtime = {"provider": "openai", "api_key": "sk-test", "base_url": "https://api.openai.com/v1"}

    # Turn 1: override sets a different model+provider.
    turn1_model, turn1_runtime = runner._apply_turn_model_override(
        base_model,
        base_runtime,
        model_override="claude-sonnet-4-6",
        provider_override="anthropic",
        session_key="persist-test-session",
    )
    assert turn1_model == "claude-sonnet-4-6"
    assert turn1_runtime["provider"] == "anthropic"

    # Turn 2: no override (typical: a normal user turn after the
    # overridden one). Must return the *base* model/runtime, NOT what
    # turn 1 produced. The override is per-MessageEvent; nothing
    # should leak to subsequent calls.
    turn2_model, turn2_runtime = runner._apply_turn_model_override(
        base_model,
        base_runtime,
        model_override=None,
        provider_override=None,
        session_key="persist-test-session",
    )
    assert turn2_model == base_model
    assert turn2_runtime is base_runtime
    assert turn2_runtime["provider"] == "openai"

    # Also confirm: a fresh MessageEvent starts with all overrides None.
    fresh = _make_event("turn 2 message")
    assert fresh.model_override is None
    assert fresh.provider_override is None
    assert fresh.model_tier_hint is None


@pytest.mark.asyncio
async def test_hook_can_set_override_alongside_rewrite(monkeypatch):
    """A plugin may return {'action': 'rewrite', 'text': ..., 'model_override': ...}.
    Both effects must apply: text is rewritten AND override fields land on the event."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    captured = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{
                "action": "rewrite",
                "text": "REWRITTEN",
                "model_override": "claude-sonnet-4-6",
                "provider_override": "anthropic",
                "model_tier_hint": "frontier",
            }]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        captured["text"] = event.text
        captured["model_override"] = event.model_override
        captured["provider_override"] = event.provider_override
        captured["model_tier_hint"] = event.model_tier_hint
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    # Rewrite took effect.
    assert captured["text"] == "REWRITTEN"
    # Override fields landed on the event for the dispatcher to read.
    assert captured["model_override"] == "claude-sonnet-4-6"
    assert captured["provider_override"] == "anthropic"
    assert captured["model_tier_hint"] == "frontier"
