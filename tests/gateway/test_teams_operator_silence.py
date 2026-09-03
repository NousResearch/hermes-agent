"""Teams must not post operator/setup text into chats."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, teams_skips_operator_sends
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _teams() -> Platform:
    return Platform("teams")


def _source(*, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=_teams(),
        chat_id="19:group-throwaway",
        chat_type=chat_type,
        user_id="29:user-throwaway",
        user_name="tester",
    )


def _runner():
    platform = _teams()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={})}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="n1"))
    adapter.send_private_notice = AsyncMock(
        return_value=SendResult(success=True, message_id="p1")
    )
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.pairing_store.generate_code.return_value = "ABCD2345"
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner, adapter


def test_teams_skips_operator_sends_predicate():
    assert teams_skips_operator_sends(_teams()) is True
    assert teams_skips_operator_sends(Platform.SLACK) is False
    assert teams_skips_operator_sends(Platform.TELEGRAM) is False


@pytest.mark.anyio
async def test_teams_skips_platform_notice():
    runner, adapter = _runner()
    await runner._deliver_platform_notice(_source(), "setup notice")
    adapter.send.assert_not_awaited()
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.anyio
async def test_teams_skips_pairing_send_even_when_pair_is_configured(monkeypatch):
    for key in (
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TEAMS_ALLOWED_USERS",
        "TEAMS_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    runner, adapter = _runner()
    runner.config.platforms[_teams()].extra["unauthorized_dm_behavior"] = "pair"
    runner._is_user_authorized_for_source = lambda source, **kwargs: False

    result = await runner._handle_message(
        MessageEvent(
            text="hello",
            message_id="m1",
            source=_source(),
        )
    )

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.anyio
async def test_teams_sethome_is_refused_and_does_not_persist(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.save_env_value",
        lambda key, value: saved.__setitem__(key, value),
    )
    persist = MagicMock()
    monkeypatch.setattr("gateway.slash_commands.persist_home_channel", persist)

    runner, adapter = _runner()
    runner._handle_set_home_command = GatewayRunner._handle_set_home_command.__get__(
        runner, GatewayRunner
    )
    event = MessageEvent(
        text="/sethome",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m-home",
    )

    result = await runner._handle_set_home_command(event)

    persist.assert_not_called()
    assert saved == {}
    assert runner.config.get_home_channel(_teams()) is None
    adapter.send.assert_not_awaited()
    assert not result
