"""Tests for the light-mode terminal detection + color remap in cli.py.

Covers the env-override path and the SkinConfig.get_color() wrapper that
the resize / light-mode salvage installs at module import time.  We don't
try to fake an OSC 11 reply — the env-override branch short-circuits
before the terminal query, which is the path most users hit.
"""

from __future__ import annotations


import pytest


@pytest.fixture
def cli_mod(monkeypatch):
    """Import cli with the light-mode cache cleared each test."""
    import cli as _cli

    # The module-level _install_skin_light_mode_hook() and import-time
    # _detect_light_mode() prime ran once at first import.  We just reset
    # the detection cache so the per-test env override takes effect.
    monkeypatch.setattr(_cli, "_LIGHT_MODE_CACHE", None)
    return _cli


class TestLightModeDetection:
    def test_hermes_light_env_true_forces_light(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True

    def test_hermes_light_env_false_forces_dark(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "0")
        # Also blank out other signals so nothing else flips it light.
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert cli_mod._detect_light_mode() is False


    def test_background_hex_hint_light(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.setenv("HERMES_TUI_BACKGROUND", "#FFFFFF")
        assert cli_mod._detect_light_mode() is True


    def test_colorfgbg_light_bg_slot(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.setenv("COLORFGBG", "0;15")  # bg slot 15 = light
        assert cli_mod._detect_light_mode() is True

    def test_cache_is_sticky(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True
        # Even if the env flips, the cached result wins until reset.
        monkeypatch.setenv("HERMES_LIGHT", "0")
        assert cli_mod._detect_light_mode() is True




class TestLightModeRemap:

    def test_remap_known_dark_color(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        # Force the detect cache to True for this test.
        cli_mod._LIGHT_MODE_CACHE = True
        assert cli_mod._maybe_remap_for_light_mode("#FFF8DC") == "#1A1A1A"
        assert cli_mod._maybe_remap_for_light_mode("#FFD700") == "#9A6B00"





class TestSkinConfigHook:
    """The salvage wraps SkinConfig.get_color at module import time so
    every skin color read goes through the light-mode remap.  Verify
    the hook installed and functions correctly.
    """

    def test_hook_installed(self, cli_mod):
        from hermes_cli.skin_engine import SkinConfig

        assert getattr(SkinConfig, "_hermes_light_mode_hook_installed", False) is True


    def test_skin_color_remaps_through_wrapper_in_light_mode(
        self, cli_mod, monkeypatch
    ):
        from hermes_cli.skin_engine import SkinConfig

        cli_mod._LIGHT_MODE_CACHE = True
        skin = SkinConfig(
            name="test",
            colors={"banner_text": "#FFF8DC", "response_border": "#FFD700"},
        )
        # The wrapper kicks in at get_color, not at construction time.
        assert skin.get_color("banner_text") == "#1A1A1A"
        assert skin.get_color("response_border") == "#9A6B00"

