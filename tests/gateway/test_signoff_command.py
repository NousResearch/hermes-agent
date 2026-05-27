"""Tests for gateway /signoff command."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str = "/signoff") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="thread",
        thread_id="c1",
    )
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    return runner


@pytest.mark.asyncio
async def test_signoff_command_runs_wtsignoff_and_returns_output():
    runner = _make_runner()
    proc = SimpleNamespace(
        returncode=0,
        communicate=AsyncMock(return_value=(b"signed off\n", b"")),
    )

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as create_proc:
        result = await runner._handle_signoff_command(_make_event("/signoff --since today"))

    create_proc.assert_awaited_once()
    args = create_proc.await_args.args
    assert args[:2] == ("/Users/admin/.local/bin/wtsignoff", "--since")
    assert args[2] == "today"
    assert result == "signed off"


@pytest.mark.asyncio
async def test_signoff_command_reports_missing_tool():
    runner = _make_runner()

    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)):
        result = await runner._handle_signoff_command(_make_event("/signoff"))

    assert "wtsignoff was not found" in result
