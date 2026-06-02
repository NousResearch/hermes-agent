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


class TestGatewayHomeChannelConfig:
    def test_top_level_home_channel_loaded_without_gateway_run_env_bridge(self, monkeypatch):
        """Direct config loads should see the same home target gateway.run sees."""
        monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("WHATSAPP_HOME_CHANNEL_NAME", raising=False)
        monkeypatch.delenv("WHATSAPP_HOME_CHANNEL_THREAD_ID", raising=False)

        cfg = _load_with_yaml_dict({
            "WHATSAPP_HOME_CHANNEL": "home-chat",
            "WHATSAPP_HOME_CHANNEL_NAME": "Nicholas",
            "WHATSAPP_HOME_CHANNEL_THREAD_ID": "topic-1",
        })

        from gateway.config import Platform

        home = cfg.get_home_channel(Platform.WHATSAPP)
        assert home is not None
        assert home.chat_id == "home-chat"
        assert home.name == "Nicholas"
        assert home.thread_id == "topic-1"

    def test_platform_home_channel_takes_precedence_over_top_level_legacy_key(self, monkeypatch):
        monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)

        cfg = _load_with_yaml_dict({
            "WHATSAPP_HOME_CHANNEL": "legacy-chat",
            "platforms": {
                "whatsapp": {
                    "home_channel": {
                        "platform": "whatsapp",
                        "chat_id": "platform-chat",
                        "name": "Platform Home",
                    },
                },
            },
        })

        from gateway.config import Platform

        home = cfg.get_home_channel(Platform.WHATSAPP)
        assert home is not None
        assert home.chat_id == "platform-chat"
        assert home.name == "Platform Home"
