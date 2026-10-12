"""Slack status calls fall back per client without changing legacy values."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.slack.adapter as slack_adapter
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
async def test_status_falls_back_per_client_without_losing_legacy_values(monkeypatch):
    monkeypatch.setattr(slack_adapter, "_AGENT_SESSIONS_SUPPORTED", True)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    legacy_set_status = AsyncMock()
    adapter._app = SimpleNamespace(
        client=SimpleNamespace(assistant_threads_setStatus=legacy_set_status)
    )
    adapter.config.typing_status_text = "investigating the incident"

    await adapter.send_typing("C123", metadata={"thread_id": "parent_ts"})
    await adapter.stop_typing("C123", metadata={"thread_id": "parent_ts"})

    assert legacy_set_status.await_args_list == [
        call(
            channel_id="C123",
            thread_ts="parent_ts",
            status="investigating the incident",
        ),
        call(channel_id="C123", thread_ts="parent_ts", status=""),
    ]