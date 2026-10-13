"""Focused regressions for exact per-channel slash-command caps."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.slash_access import channel_command_denial


CHANNEL_RULES = {
    "signal": {
        "synthetic-group-id": {
            "allowed_slash_commands": ["/whoami", "status"],
            "deny_message": "Not enabled in this room.",
        }
    }
}


def _source(*, chat_id="synthetic-group-id", chat_type="group", user_id="member"):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
    )


def test_channel_cap_uses_exact_platform_chat_id_and_canonical_command():
    config = GatewayConfig(channel_command_access=CHANNEL_RULES)
    assert channel_command_denial(config, _source(), "whoami") is None
    assert (
        channel_command_denial(config, _source(), "stop") == "Not enabled in this room."
    )
    # Display names, other IDs, and other platforms cannot select this rule.
    assert channel_command_denial(config, _source(chat_id="another-id"), "stop") is None
    assert (
        channel_command_denial(
            config,
            SessionSource(platform=Platform.DISCORD, chat_id="synthetic-group-id"),
            "stop",
        )
        is None
    )


def test_malformed_config_and_missing_immutable_chat_id_fail_closed():
    assert (
        channel_command_denial(
            GatewayConfig(channel_command_access="bad"), _source(), "status"
        )
        == "This command is not enabled in this channel."
    )
    assert (
        channel_command_denial(
            GatewayConfig(channel_command_access={"signal": []}), _source(), "status"
        )
        == "This command is not enabled in this channel."
    )
    assert (
        channel_command_denial(
            GatewayConfig(channel_command_access=CHANNEL_RULES),
            _source(chat_id=None),
            "status",
        )
        == "This command is not enabled in this channel."
    )
    malformed_rule = {
        "signal": {"synthetic-group-id": {"allowed_slash_commands": "status"}}
    }
    assert (
        channel_command_denial(
            GatewayConfig(channel_command_access=malformed_rule), _source(), "status"
        )
        == "This command is not enabled in this channel."
    )


def _runner(*, allowed_commands, platform_extra=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SIGNAL: PlatformConfig(enabled=True, extra=platform_extra or {})
        },
        channel_command_access={
            "signal": {
                "synthetic-group-id": {
                    "allowed_slash_commands": allowed_commands,
                    "deny_message": "Not enabled in this room.",
                }
            }
        },
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.SIGNAL: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False
    )
    runner.session_store = MagicMock()
    now = datetime.now()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:signal:group:synthetic-group-id",
        session_id="synthetic-session",
        created_at=now,
        updated_at=now,
        platform=Platform.SIGNAL,
        chat_type="group",
        total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _event(text):
    return MessageEvent(text=text, source=_source(), message_id="synthetic-message-id")


@pytest.mark.asyncio
async def test_denied_command_is_stopped_before_pending_intercepts_and_hooks():
    runner = _runner(allowed_commands=["whoami"])
    runner._hm_pending_reply_intercepts = AsyncMock(
        return_value="would-have-intercepted"
    )
    runner._hm_command_hooks = AsyncMock(return_value=(True, "hook-ran", None))
    result = await runner._handle_message(_event("/stop"))
    assert result == "Not enabled in this room."
    runner._hm_pending_reply_intercepts.assert_not_awaited()
    runner._hm_command_hooks.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_cap_does_not_grant_around_existing_group_admin_policy():
    runner = _runner(
        allowed_commands=["restart"],
        platform_extra={"group_allow_admin_from": ["synthetic-admin"]},
    )
    result = await runner._handle_message(_event("/restart"))
    assert result is not None and "admin-only" in result


@pytest.mark.asyncio
async def test_allowed_channel_command_and_unconfigured_chat_preserve_behavior():
    runner = _runner(allowed_commands=["whoami"])
    result = await runner._handle_message(_event("/whoami"))
    assert result and "Tier: unrestricted" in result
    other = _source(chat_id="unconfigured-group")
    result = await runner._handle_message(
        MessageEvent(text="/whoami", source=other, message_id="synthetic-2")
    )
    assert result and "Tier: unrestricted" in result
