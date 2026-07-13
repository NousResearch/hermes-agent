from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.slack.adapter as _slack_mod
from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, SlackHistoryAccessError


_slack_mod.SLACK_AVAILABLE = True


def _adapter_with_single_workspace_client() -> tuple[SlackAdapter, AsyncMock]:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._running = True
    adapter._configured_workspace_count = 1
    client = AsyncMock()
    adapter._team_clients = {"T1": client}
    return adapter, client


@pytest.mark.asyncio
async def test_history_owner_discovers_only_member_channels_not_dms():
    adapter, client = _adapter_with_single_workspace_client()
    adapter.config.extra["history_cross_channel_user_ids"] = ["U12345678"]
    client.conversations_list.return_value = {
        "ok": True,
        "channels": [
            {"id": "C123456789", "name": "general", "is_member": True},
            {"id": "C999999999", "name": "other", "is_member": False},
            {"id": "D123456789", "name": "dm", "is_member": True},
        ],
    }

    result = await adapter.list_history_channels_for_agent(
        expected_team_id="T1",
        requester_user_id="U12345678",
    )

    assert result["channels"] == [
        {
            "id": "C123456789",
            "name": "general",
            "is_member": True,
            "is_private": False,
        }
    ]
    client.conversations_list.assert_awaited_once_with(
        types="public_channel,private_channel",
        exclude_archived=True,
        limit=50,
    )


@pytest.mark.asyncio
async def test_non_owner_cannot_discover_channels_before_slack_api_call():
    adapter, client = _adapter_with_single_workspace_client()
    adapter.config.extra["history_cross_channel_user_ids"] = ["U12345678"]

    with pytest.raises(SlackHistoryAccessError, match="cross_channel_not_allowed"):
        await adapter.list_history_channels_for_agent(
            expected_team_id="T1",
            requester_user_id="U87654321",
        )

    client.conversations_list.assert_not_awaited()


def test_history_owner_config_rejects_malformed_or_environment_only_ids(monkeypatch):
    adapter, _ = _adapter_with_single_workspace_client()
    adapter.config.extra["history_cross_channel_user_ids"] = ["not-a-member-id"]
    monkeypatch.setenv("SLACK_HISTORY_CROSS_CHANNEL_USER_IDS", "U12345678")

    assert adapter.allows_agent_cross_channel_history("U12345678") is False

    adapter.config.extra["history_cross_channel_user_ids"] = ["U12345678"]

    assert adapter.allows_agent_cross_channel_history("U12345678") is True
