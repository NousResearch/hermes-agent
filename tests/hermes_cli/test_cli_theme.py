"""Tests for hermes_cli.cli_theme — color, ANSI, and light-mode detection."""

from __future__ import annotations

import os
import sys

import pytest


def _load_cli_theme_module():
    """Import hermes_cli/cli_theme.py without triggering cli.py side effects."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "cli_theme_under_test",
        Path(__file__).resolve().parents[2] / "hermes_cli" / "cli_theme.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestHexToAnsi:
    def test_basic_color(self):
        m = _load_cli_theme_module()
        result = m._hex_to_ansi("#FF0000")
        assert "38;2;255;0;0" in result

    def test_bold_prefix(self):
        m = _load_cli_theme_module()
        result = m._hex_to_ansi("#FF0000", bold=True)
        assert result.startswith("\033[1;")

    def test_invalid_hex_returns_default(self):
        m = _load_cli_theme_module()
        result = m._hex_to_ansi("invalid")
        assert result  # should return a fallback, not crash


class TestLuminanceFromHex:
    def test_white(self):
        m = _load_cli_theme_module()
        assert m._luminance_from_hex("#FFFFFF") == pytest.approx(1.0, abs=0.01)

    def test_black(self):
        m = _load_cli_theme_module()
        assert m._luminance_from_hex("#000000") == pytest.approx(0.0, abs=0.01)

    def test_short_hex(self):
        m = _load_cli_theme_module()
        assert m._luminance_from_hex("#FFF") == pytest.approx(1.0, abs=0.01)

    def test_invalid_returns_none(self):
        m = _load_cli_theme_module()
        assert m._luminance_from_hex("not-a-color") is None

    def test_empty_returns_none(self):
        m = _load_cli_theme_module()
        assert m._luminance_from_hex("") is None


class TestDetectLightMode:
    def test_default_is_dark(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = None
        for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT", "HERMES_TUI_THEME", "HERMES_TUI_BACKGROUND", "COLORFGBG"):
            monkeypatch.delenv(var, raising=False)
        assert m._detect_light_mode() is False

    def test_env_override_true(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = None
        monkeypatch.setenv("HERMES_LIGHT", "true")
        assert m._detect_light_mode() is True

    def test_env_override_false(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = None
        monkeypatch.setenv("HERMES_LIGHT", "false")
        assert m._detect_light_mode() is False

    def test_theme_light(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = None
        for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HERMES_TUI_THEME", "light")
        assert m._detect_light_mode() is True

    def test_colorfgbg_light(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = None
        for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT", "HERMES_TUI_THEME", "HERMES_TUI_BACKGROUND"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("COLORFGBG", "0;15")  # bg=15 = light
        # Patch _query_osc11_background to avoid terminal I/O in tests
        monkeypatch.setattr(m, "_query_osc11_background", lambda: None)
        assert m._detect_light_mode() is True


class TestMaybeRemapForLightMode:
    def test_no_remap_in_dark_mode(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = False
        assert m._maybe_remap_for_light_mode("#FFF8DC") == "#FFF8DC"

    def test_remap_in_light_mode(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = True
        assert m._maybe_remap_for_light_mode("#FFF8DC") == "#1A1A1A"

    def test_remap_case_insensitive(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = True
        assert m._maybe_remap_for_light_mode("#fff8dc") == "#1A1A1A"

    def test_unknown_color_unchanged(self, monkeypatch):
        m = _load_cli_theme_module()
        m._LIGHT_MODE_CACHE = True
        assert m._maybe_remap_for_light_mode("#123456") == "#123456"


class TestConstants:
    def test_bold_is_ansi(self):
        m = _load_cli_theme_module()
        assert m._BOLD == "\033[1m"

    def test_rst_is_ansi(self):
        m = _load_cli_theme_module()
        assert m._RST == "\033[0m"

    def test_light_remap_table_is_non_empty(self):
        m = _load_cli_theme_module()
        assert len(m._LIGHT_MODE_REMAP) > 5
