"""Production-path unit tests for agent_turn_error_runtime_service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_turn_error_runtime_service import (
    GatewayAgentTurnErrorResult,
    handle_gateway_agent_turn_error,
)
from gateway.config import Platform
from gateway.session import SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _runner(*, transcript=None):
    adapter = SimpleNamespace(stop_typing=AsyncMock())
    store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=list(transcript or [])),
        append_to_transcript=AsyncMock(),
    )
    return SimpleNamespace(
        _adapter_for_source=MagicMock(return_value=adapter),
        _thread_metadata_for_source=MagicMock(return_value={}),
        _reply_anchor_for_event=MagicMock(return_value=None),
        async_session_store=store,
    )


async def _call(runner, exc, **kw):
    defaults = dict(
        runner=runner,
        event=SimpleNamespace(message_id="m1"),
        source=_source(),
        session_entry=SimpleNamespace(session_id="sid"),
        session_key="sk",
        history=[{"role": "user", "content": "x"}] * 3,
        message_text="hello",
        persist_user_message="hello",
        persist_user_timestamp=None,
        exc=exc,
        logger=MagicMock(),
    )
    defaults.update(kw)
    return await handle_gateway_agent_turn_error(**defaults)


@pytest.mark.asyncio
async def test_error_401_includes_api_key_hint():
    exc = Exception("auth failed")
    exc.status_code = 401
    runner = _runner()
    result = await _call(runner, exc)
    assert isinstance(result, GatewayAgentTurnErrorResult)
    assert "API key" in result.response or "claude /login" in result.response
    assert result.response.startswith("Sorry, I encountered")


@pytest.mark.asyncio
async def test_error_400_large_history_returns_compact_hint():
    exc = Exception("payload")
    exc.status_code = 400
    history = [{"role": "user", "content": f"m{i}"} for i in range(60)]
    runner = _runner()
    result = await _call(runner, exc, history=history)
    assert "Session too large" in result.response
    assert "/compact" in result.response
    assert "/reset" in result.response


@pytest.mark.asyncio
async def test_error_persists_user_when_not_already_in_transcript():
    exc = Exception("boom")
    runner = _runner(transcript=[])
    result = await _call(runner, exc)
    assert "unexpected error" in result.response
    runner.async_session_store.append_to_transcript.assert_awaited_once()
    entry = runner.async_session_store.append_to_transcript.await_args.args[1]
    assert entry["role"] == "user"
    assert entry["content"] == "hello"
    assert entry["message_id"] == "m1"


@pytest.mark.asyncio
async def test_error_skips_persist_when_user_already_present():
    exc = Exception("boom")
    runner = _runner(
        transcript=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    await _call(runner, exc)
    runner.async_session_store.append_to_transcript.assert_not_awaited()


@pytest.mark.asyncio
async def test_error_429_usage_limit_with_reset_seconds():
    class _Body:
        def json(self):
            return {
                "error": {
                    "type": "usage_limit_reached",
                    "resets_in_seconds": 7200,
                }
            }

    exc = Exception("rate")
    exc.status_code = 429
    exc.response = _Body()
    runner = _runner()
    result = await _call(runner, exc)
    assert "usage limit" in result.response
    assert "resets in ~2h" in result.response
