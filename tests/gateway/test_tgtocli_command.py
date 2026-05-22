"""Tests for /tgtocli gateway slash command."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/tgtocli", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_id="20260519_101112_abcdef12", title="Phone work"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id=session_id,
        session_key="agent:main:telegram:dm:67890",
    )
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = title
    return runner


@pytest.mark.asyncio
async def test_tgtocli_rejects_non_telegram_sources():
    runner = _make_runner()
    result = await runner._handle_tgtocli_command(_make_event(platform=Platform.DISCORD))

    assert "/tgtocli" in result
    assert "Telegram" in result
    runner.session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_tgtocli_returns_resume_instructions_for_current_telegram_session():
    runner = _make_runner(session_id="20260519_101112_abcdef12")

    result = await runner._handle_tgtocli_command(_make_event())

    assert "Open this Telegram session in CLI manually" in result
    assert "hermes --resume 20260519_101112_abcdef12" in result
    assert "tmux new-session -s hermes-tgtocli-abcdef12" in result
    assert "does not switch an already-open CLI" in result
    assert "Phone work" in result


@pytest.mark.asyncio
async def test_tgtocli_does_not_spawn_tmux_or_subprocess():
    runner = _make_runner(session_id="20260519_101112_abcdef12")

    result = await runner._handle_tgtocli_command(_make_event())

    assert "CLI resume started" not in result
    assert "tmux attach" not in result
    assert "hermes --resume 20260519_101112_abcdef12" in result
