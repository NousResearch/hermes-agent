"""Regression test for #25676 — nested gateway.streaming config must be loaded."""
from unittest.mock import patch

import hermes_yaml as yaml



def _load_with_yaml_dict(yaml_dict: dict, tmp_path):
    """Load a real config.yaml through the gateway's shared YAML reader."""
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(yaml_dict), encoding="utf-8")
    with patch("gateway.config.get_hermes_home", return_value=tmp_path):
        return load_gateway_config()


class TestStreamingConfigNested:
    def test_top_level_streaming(self, tmp_path):
        cfg = _load_with_yaml_dict({"streaming": {"enabled": True, "transport": "draft"}}, tmp_path)
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"


    def test_top_level_takes_precedence(self, tmp_path):
        cfg = _load_with_yaml_dict({
            "streaming": {"enabled": True, "transport": "edit"},
            "gateway": {"streaming": {"enabled": False, "transport": "draft"}},
        }, tmp_path)
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



    def test_explicit_enabled_overrides_mode(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "auto", "enabled": False})
        assert sc.enabled is False
        # transport still resolves from mode
        assert sc.transport == "auto"


    def test_empty_block_stays_disabled(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({})
        assert sc.enabled is False



class TestStreamingYamlBooleanQuirk:
    """YAML 1.1 parses bare ``off``/``on`` as booleans; ``mode``/``transport``
    must normalize those back to canonical string tokens.

    Regression for the review on PR #62873: bare ``mode: off`` arrived as
    Python ``False`` and stringified to ``"false"``, which is not ``"off"``,
    so streaming was enabled instead of honoring the advertised disable.
    """

    def test_mode_bare_off_boolean_disables(self):
        from gateway.config import StreamingConfig

        # yaml.safe_load("off") -> False
        sc = StreamingConfig.from_dict({"mode": False})
        assert sc.enabled is False
        assert sc.transport == "off"

    def test_mode_bare_on_boolean_enables(self):
        from gateway.config import StreamingConfig

        # yaml.safe_load("on") -> True
        sc = StreamingConfig.from_dict({"mode": True})
        assert sc.enabled is True
        assert sc.transport == "auto"


    def test_loader_normalizes_bare_yaml_off(self, tmp_path):
        """End-to-end through load_gateway_config(): unquoted ``mode: off``
        (a YAML boolean) must keep streaming disabled."""
        from gateway.config import load_gateway_config

        (tmp_path / "config.yaml").write_text("streaming:\n  mode: off\n", encoding="utf-8")
        with patch("gateway.config.get_hermes_home", return_value=tmp_path):
            cfg = load_gateway_config()
        assert cfg.streaming.enabled is False
        assert cfg.streaming.transport == "off"

