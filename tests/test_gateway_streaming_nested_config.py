"""Regression test for #25676 — nested gateway.streaming config must be loaded."""
from pathlib import Path
from unittest.mock import patch, MagicMock



def _load_with_yaml_dict(yaml_dict: dict):
    """Patch filesystem so load_gateway_config() sees *yaml_dict* as config.yaml."""
    from gateway.config import load_gateway_config

    fake_home = Path("/tmp/fake_hermes_home_25676")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with patch("gateway.config.get_hermes_home", return_value=fake_home), \
         patch.object(Path, "exists", fake_exists), \
         patch("builtins.open", create=True) as mock_file:
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


class TestStreamingConfigNested:
    def test_top_level_streaming(self):
        cfg = _load_with_yaml_dict({"streaming": {"enabled": True, "transport": "draft"}})
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"

    def test_nested_gateway_streaming(self):
        """Regression for #25676."""
        cfg = _load_with_yaml_dict({"gateway": {"streaming": {"enabled": True, "transport": "draft"}}})
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"

    def test_top_level_takes_precedence(self):
        cfg = _load_with_yaml_dict({
            "streaming": {"enabled": True, "transport": "edit"},
            "gateway": {"streaming": {"enabled": False, "transport": "draft"}},
        })
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "edit"


class TestDiscordAutoRenameThreadsConfig:
    def test_default_config_keeps_discord_thread_rename_disabled(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["discord"]["auto_rename_threads"] == {
            "enabled": False,
            "mode": "session_title",
            "sync_title_command": False,
            "max_length": 100,
        }

    def test_top_level_discord_auto_rename_threads_bridges_to_platform_extra(self):
        from gateway.config import Platform

        cfg = _load_with_yaml_dict(
            {
                "discord": {
                    "auto_rename_threads": {
                        "enabled": True,
                        "mode": "session_title",
                        "sync_title_command": True,
                        "max_length": 80,
                    }
                }
            }
        )

        assert cfg.platforms[Platform.DISCORD].extra["auto_rename_threads"] == {
            "enabled": True,
            "mode": "session_title",
            "sync_title_command": True,
            "max_length": 80,
        }
