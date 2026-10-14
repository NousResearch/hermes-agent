"""The one-time "No home channel is set" prompt goes to the operator, not to senders an adapter authorized via a role."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(get_home_channel=lambda platform: None)
    store = SimpleNamespace(has_any_sessions=AsyncMock(return_value=True))
    monkeypatch.setattr(GatewayRunner, "async_session_store", property(lambda self: store))
    runner._deliver_platform_notice = AsyncMock()
    return runner


@pytest.mark.parametrize("role_authorized, prompted", [(False, True), (True, False)])
def test_home_channel_prompt_skips_role_authorized_senders(monkeypatch, tmp_path, role_authorized, prompted):
    runner = _runner(monkeypatch, tmp_path)
    source = SessionSource(platform=Platform.DISCORD, chat_id="c", chat_type="dm", user_id="u", role_authorized=role_authorized)
    asyncio.run(runner._hmwa_first_contact_notes(source, [], []))
    assert runner._deliver_platform_notice.await_count == (1 if prompted else 0)
