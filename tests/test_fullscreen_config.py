"""Tests for display.fullscreen config option (issue #639)."""
import pytest
from unittest.mock import patch
import cli as _cli_mod


# ─── load_cli_config defaults ─────────────────────────────────────────────────

class TestFullscreenDefault:
    def test_default_is_false(self):
        """fullscreen must default to False so existing behaviour is unchanged."""
        from cli import load_cli_config
        config = load_cli_config()
        assert config.get("display", {}).get("fullscreen") is False

    def test_default_config_has_fullscreen_key(self):
        """The display section must contain a fullscreen key."""
        from cli import load_cli_config
        config = load_cli_config()
        assert "fullscreen" in config.get("display", {})


# ─── hermes_cli/config.py DEFAULT_CONFIG ──────────────────────────────────────

class TestHermesConfigDefault:
    def test_hermes_default_config_has_fullscreen(self):
        from hermes_cli.config import load_config
        config = load_config()
        assert "fullscreen" in config.get("display", {})

    def test_hermes_default_fullscreen_is_false(self):
        from hermes_cli.config import load_config
        config = load_config()
        assert config["display"]["fullscreen"] is False
