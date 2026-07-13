from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


class FakeRunner(GatewaySlashCommandsMixin):
    pass


def _event() -> MessageEvent:
    return MessageEvent(
        text="/restart",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1003589561528",
            chat_type="group",
            user_id="6218150306",
            thread_id="17",
        ),
        message_id="123",
        platform_update_id=456,
    )


def test_restart_new_session_flag_reads_platform_config():
    runner = object.__new__(FakeRunner)
    runner.adapters = {
        Platform.TELEGRAM: SimpleNamespace(
            config=PlatformConfig(
                enabled=True,
                token="fake",
                extra={"restart_starts_new_session": True},
            )
        )
    }
    assert runner._restart_starts_new_session(_event()) is True


@pytest.mark.asyncio
async def test_restart_resets_session_before_requesting_restart(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")

    runner = object.__new__(FakeRunner)
    runner._is_stale_restart_redelivery = MagicMock(return_value=False)
    runner._restart_requested = False
    runner._draining = False
    runner._restart_starts_new_session = MagicMock(return_value=True)
    runner._handle_reset_command = AsyncMock(return_value="new session")
    runner._running_agent_count = MagicMock(return_value=0)
    runner.request_restart = MagicMock()

    await runner._handle_restart_command(_event())

    runner._handle_reset_command.assert_awaited_once()
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)
    assert (
        runner._handle_reset_command.await_args_list[0].args[0].text
        == runner._is_stale_restart_redelivery.call_args_list[0].args[0].text
    )
