"""Plugin slash commands must see the originating channel via session contextvars.

Regression test for the /context vs /compress channel-scope mismatch: plugin
commands run BEFORE the agent-run path that normally binds the session context
(``_set_session_env``), so a handler that resolves "the current turn" had no way
to know which channel it was invoked from and fell back to a global most-recent
lookup — leaking another channel's telemetry. The gateway now binds
platform/chat_id into session contextvars around plugin-command dispatch.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str, platform: Platform, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="tester",
            chat_id=chat_id,
            chat_name="ops",
            user_name="tester",
            chat_type="group",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {platform: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompt_pending = {}
    return runner


@pytest.mark.asyncio
async def test_plugin_command_sees_channel_context(monkeypatch):
    """A plugin command handler reads the invoking channel from session env."""
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "*")

    captured: dict = {}

    def _probe_handler(_raw_args: str) -> str:
        from gateway.session_context import get_session_env

        captured["platform"] = get_session_env("HERMES_SESSION_PLATFORM", "")
        captured["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID", "")
        captured["chat_name"] = get_session_env("HERMES_SESSION_CHAT_NAME", "")
        captured["session_key"] = get_session_env("HERMES_SESSION_KEY", "")
        return "ok"

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: _probe_handler if name == "context" else None,
    )

    runner = _make_runner(Platform.DISCORD)
    result = await runner._handle_message(
        _make_event("/context", Platform.DISCORD, "111222333")
    )

    assert result == "ok"
    assert captured["platform"] == "discord"
    assert captured["chat_id"] == "111222333"
    assert captured["chat_name"] == "ops"
    # session_key must be bound too (resident-agent / transcript resolution
    # for /context's non-fixed breakdown depends on it).
    assert captured["session_key"]


@pytest.mark.asyncio
async def test_plugin_command_context_cleared_after_dispatch(monkeypatch):
    """Session context is reset after the handler returns (no leak to next turn)."""
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "*")

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: (lambda _a: "ok") if name == "context" else None,
    )

    runner = _make_runner(Platform.DISCORD)
    await runner._handle_message(_make_event("/context", Platform.DISCORD, "111222333"))

    from gateway.session_context import get_session_env

    # clear_session_vars sets the contextvars to "" (explicitly cleared).
    assert get_session_env("HERMES_SESSION_CHAT_ID", "SENTINEL") == ""


@pytest.mark.asyncio
async def test_plugin_command_handler_exception_still_clears_context(monkeypatch):
    """A raising handler must not leak bound channel context (finally clears it)."""
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "*")

    def _boom(_raw_args: str) -> str:
        raise RuntimeError("handler blew up")

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda name: _boom if name == "context" else None,
    )

    runner = _make_runner(Platform.DISCORD)
    # Dispatch swallows handler exceptions (logged, falls through); must not raise.
    await runner._handle_message(_make_event("/context", Platform.DISCORD, "111222333"))

    from gateway.session_context import get_session_env

    assert get_session_env("HERMES_SESSION_CHAT_ID", "SENTINEL") == ""
