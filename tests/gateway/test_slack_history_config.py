from pathlib import Path
from unittest.mock import MagicMock, patch

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
