"""Tests for the cua-driver --no-overlay policy.

cua-driver's cursor overlay rendering loop can consume CPU indefinitely when
idle (#28152, #47032).  Hermes passes ``--no-overlay`` to suppress it when the
``computer_use.no_overlay`` config is enabled (or auto-detected on Linux).

These assert the behavior contract (auto-detect on Linux, explicit override,
config failure fails safe toward overlay enabled), not specific config
snapshots.
"""

import sys
from unittest.mock import patch

from tools.computer_use import cua_backend


class TestNoOverlayFlag:
    def test_default_linux_disables(self):
        """Auto-detect: Linux => overlay disabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"):
            assert cua_backend._cua_no_overlay() is True

    def test_default_macos_enables(self):
        """Auto-detect: macOS => overlay enabled (visually useful)."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "darwin"):
            assert cua_backend._cua_no_overlay() is False

    def test_default_windows_enables(self):
        """Auto-detect: Windows => overlay enabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "win32"):
            assert cua_backend._cua_no_overlay() is False

    def test_explicit_true_overrides(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"no_overlay": True}}):
            assert cua_backend._cua_no_overlay() is True

    def test_explicit_false_overrides(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"no_overlay": False}}), \
             patch.object(sys, "platform", "linux"):
            # Explicit False overrides auto-detect on Linux.
            assert cua_backend._cua_no_overlay() is False

    def test_config_load_failure_fails_safe(self):
        """Unreadable config => default to overlay enabled."""
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("boom")):
            assert cua_backend._cua_no_overlay() is False

    def test_missing_section_enables(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"other": {}}):
            assert cua_backend._cua_no_overlay() is False


class TestMcpArgsOverlayFlag:
    def test_no_overlay_appended_when_enabled(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp", "--no-overlay"]

    def test_no_overlay_not_appended_when_disabled(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=False):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp"]

    def test_does_not_mutate_original_list(self):
        """The original args list must not be mutated."""
        original = ["mcp"]
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(original)
            assert "--no-overlay" in result
            assert "--no-overlay" not in original
