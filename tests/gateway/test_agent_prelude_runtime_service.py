"""DEAD path: not imported by gateway/run.py — contract-only unit tests.
See gateway/RUNTIME_SERVICES.md.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_prelude_runtime_service import (
    GatewayAgentPreludeResult,
    append_discord_voice_channel_context,
    build_agent_start_hook_context,
    run_gateway_agent_prelude,
)
from gateway.config import Platform


def test_append_discord_voice_channel_context_only_for_discord_with_context():
    adapter = SimpleNamespace(get_voice_channel_context=lambda guild_id: "VC state")

    result = append_discord_voice_channel_context(
        "base",
        platform=Platform.DISCORD,
        guild_id=123,
        adapter=adapter,
    )

    assert result == "base\n\nVC state"


def test_append_discord_voice_channel_context_skips_when_not_available():
    result = append_discord_voice_channel_context(
        "base",
        platform=Platform.QQ_NAPCAT,
        guild_id=123,
        adapter=SimpleNamespace(get_voice_channel_context=lambda guild_id: "VC state"),
    )

    assert result == "base"


def test_build_agent_start_hook_context_truncates_message_preview():
    payload = build_agent_start_hook_context(
        platform=Platform.QQ_NAPCAT,
        user_id="u1",
        session_id="sess-1",
        message_text="x" * 600,
    )

    assert payload["platform"] == Platform.QQ_NAPCAT.value
    assert payload["user_id"] == "u1"
    assert payload["session_id"] == "sess-1"
    assert len(payload["message"]) == 500


@pytest.mark.asyncio
async def test_run_gateway_agent_prelude_emits_start_hook_without_context_expansion():
    hooks = SimpleNamespace(emit=AsyncMock())
    hook_ctx = {"session_id": "sess-1"}

    result = await run_gateway_agent_prelude(
        hooks=hooks,
        hook_ctx=hook_ctx,
        message_text="hello",
        should_expand_context_references=False,
    )

    assert result == GatewayAgentPreludeResult(
        hook_ctx=hook_ctx,
        message_text="hello",
        blocked=False,
    )
    hooks.emit.assert_awaited_once_with("agent:start", hook_ctx)


@pytest.mark.asyncio
async def test_run_gateway_agent_prelude_applies_expanded_context_message():
    hooks = SimpleNamespace(emit=AsyncMock())
    hook_ctx = {"session_id": "sess-1"}

    result = await run_gateway_agent_prelude(
        hooks=hooks,
        hook_ctx=hook_ctx,
        message_text="@file:test.py",
        should_expand_context_references=True,
        expand_context_references=AsyncMock(
            return_value=SimpleNamespace(
                message_text="expanded content",
                blocked_warning=None,
            )
        ),
    )

    assert result == GatewayAgentPreludeResult(
        hook_ctx=hook_ctx,
        message_text="expanded content",
        blocked=False,
    )
    hooks.emit.assert_awaited_once_with("agent:start", hook_ctx)


@pytest.mark.asyncio
async def test_run_gateway_agent_prelude_sends_blocked_warning_and_short_circuits():
    hooks = SimpleNamespace(emit=AsyncMock())
    send_blocked_warning = AsyncMock()
    hook_ctx = {"session_id": "sess-1"}

    result = await run_gateway_agent_prelude(
        hooks=hooks,
        hook_ctx=hook_ctx,
        message_text="@file:../../etc/passwd",
        should_expand_context_references=True,
        expand_context_references=AsyncMock(
            return_value=SimpleNamespace(
                message_text="@file:../../etc/passwd",
                blocked_warning="Context injection refused.",
            )
        ),
        send_blocked_warning=send_blocked_warning,
    )

    assert result == GatewayAgentPreludeResult(
        hook_ctx=hook_ctx,
        message_text="@file:../../etc/passwd",
        blocked=True,
    )
    hooks.emit.assert_awaited_once_with("agent:start", hook_ctx)
    send_blocked_warning.assert_awaited_once_with("Context injection refused.")
