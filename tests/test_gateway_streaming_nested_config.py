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


class TestStreamingModeAlias:
    """``streaming: {mode: ...}`` is an alias that also implies ``enabled``.

    Regression for a live config footgun: ``streaming: {mode: auto}`` was
    silently ignored (mode was never read), so streaming stayed disabled and
    the whole reply buffered before the first Telegram send.
    """

    def test_mode_auto_enables_streaming(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "auto"})
        assert sc.enabled is True
        assert sc.transport == "auto"

    def test_mode_edit_enables_streaming(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "edit"})
        assert sc.enabled is True
        assert sc.transport == "edit"

    def test_mode_off_disables_streaming(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "off"})
        assert sc.enabled is False
        assert sc.transport == "off"

    def test_mode_with_extra_keys_still_enables(self):
        """Real-world block: mode plus unrelated preloader_frames."""
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict(
            {"mode": "auto", "preloader_frames": ["a", "b"]}
        )
        assert sc.enabled is True
        assert sc.transport == "auto"

    def test_explicit_enabled_overrides_mode(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "auto", "enabled": False})
        assert sc.enabled is False
        # transport still resolves from mode
        assert sc.transport == "auto"

    def test_transport_takes_precedence_over_mode(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "off", "transport": "draft"})
        assert sc.transport == "draft"
        assert sc.enabled is True

    def test_empty_block_stays_disabled(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({})
        assert sc.enabled is False

    def test_transport_only_enables(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"transport": "edit"})
        assert sc.enabled is True
        assert sc.transport == "edit"
