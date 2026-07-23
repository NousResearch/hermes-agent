from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, load_gateway_config


def _load_with_yaml_dict(yaml_dict: dict):
    """Load a synthetic config.yaml without reading host profile state."""

    fake_home = Path("/tmp/fake_hermes_home_slack_history_config")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with (
        patch("gateway.config.get_hermes_home", return_value=fake_home),
        patch.object(Path, "exists", fake_exists),
        patch("builtins.open", create=True) as mock_file,
        patch("yaml.safe_load", return_value=yaml_dict),
    ):
        mock_file.return_value.__enter__ = lambda stream: stream
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        return load_gateway_config()


def test_top_level_slack_history_owner_list_reaches_selected_adapter_config():
    config = _load_with_yaml_dict(
        {
            "slack": {
                "history_cross_channel_user_ids": ["U12345678"],
            }
        }
    )

    assert config.platforms[Platform.SLACK].extra[
        "history_cross_channel_user_ids"
    ] == ["U12345678"]


def test_missing_slack_history_owner_list_does_not_create_a_cross_channel_grant():
    config = _load_with_yaml_dict({"slack": {"require_mention": True}})

    assert "history_cross_channel_user_ids" not in config.platforms[Platform.SLACK].extra


def test_top_level_slack_allowed_channels_is_profile_local(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "C_PROCESS_GLOBAL")

    config = _load_with_yaml_dict({"slack": {"allowed_channels": ["C_PROFILE_A"]}})

    assert config.platforms[Platform.SLACK].extra["allowed_channels"] == [
        "C_PROFILE_A"
    ]


def test_nested_slack_allowed_channels_is_profile_local(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "C_PROCESS_GLOBAL")

    config = _load_with_yaml_dict(
        {
            "gateway": {
                "platforms": {
                    "slack": {"allowed_channels": ["C_PROFILE_B"]},
                }
            }
        }
    )

    assert config.platforms[Platform.SLACK].extra["allowed_channels"] == [
        "C_PROFILE_B"
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        {"typo": ["C123456789"]},
        123,
        ["C123456789", "not-a-slack-channel"],
    ],
)
def test_real_config_loader_keeps_malformed_allowed_channels_fail_closed(malformed):
    from plugins.platforms.slack.adapter import SlackAdapter

    config = _load_with_yaml_dict({"slack": {"allowed_channels": malformed}})
    adapter = SlackAdapter(config.platforms[Platform.SLACK])

    assert adapter._parse_slack_allowed_channels() == (set(), False)


@pytest.mark.asyncio
async def test_mixed_malformed_history_owners_fail_closed_before_slack_calls(
    monkeypatch,
):
    import plugins.platforms.slack.adapter as slack_module
    from plugins.platforms.slack.adapter import SlackAdapter, SlackHistoryAccessError

    monkeypatch.setattr(slack_module, "SLACK_AVAILABLE", True)
    config = _load_with_yaml_dict(
        {
            "slack": {
                "history_cross_channel_user_ids": [
                    "U12345678",
                    {"bad": "entry"},
                ]
            }
        }
    )
    adapter = SlackAdapter(config.platforms[Platform.SLACK])
    adapter._app = MagicMock()
    adapter._running = True
    adapter._configured_workspace_count = 1
    client = AsyncMock()
    adapter._team_clients = {"T1": client}
    adapter._history_bot_team_ids = {"T1"}

    assert adapter.allows_agent_cross_channel_history("U12345678") is False
    with pytest.raises(SlackHistoryAccessError, match="cross_channel_not_allowed"):
        await adapter.list_history_channels_for_agent(
            expected_team_id="T1",
            requester_user_id="U12345678",
            active_channel_id="D123456789",
        )
    with pytest.raises(SlackHistoryAccessError, match="cross_channel_not_allowed"):
        await adapter.read_history_for_agent(
            channel_id="C123456789",
            expected_team_id="T1",
            active_channel_id="D123456789",
            requester_user_id="U12345678",
        )

    client.conversations_list.assert_not_awaited()
    client.conversations_info.assert_not_awaited()
    client.conversations_history.assert_not_awaited()
    client.conversations_replies.assert_not_awaited()
