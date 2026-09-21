"""First-contact home-channel notices stay on the operator side of a shared gateway."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource


def _runner(*, admins=()):
    runner = GatewayTurnMixin()
    runner.async_session_store = SimpleNamespace(has_any_sessions=AsyncMock(return_value=True))
    runner.config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"allow_admin_from": list(admins)},
            )
        }
    )
    runner._deliver_platform_notice = AsyncMock()
    return runner


def _source(user_id="patient-1"):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
    )


@pytest.mark.asyncio
async def test_public_first_contact_does_not_receive_home_channel_notice(monkeypatch):
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    runner = _runner()

    await runner._hmwa_first_contact_notes(_source(), [], [])

    runner._deliver_platform_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_admin_receives_home_channel_notice(monkeypatch):
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    runner = _runner(admins=("operator-1",))
    source = _source("operator-1")

    await runner._hmwa_first_contact_notes(source, [], [])

    runner._deliver_platform_notice.assert_awaited_once()
    delivered_source, message = runner._deliver_platform_notice.await_args.args
    assert delivered_source is source
    assert "No home channel is set for Whatsapp" in message