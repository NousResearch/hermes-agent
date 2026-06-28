"""Tests for the cua-driver --no-overlay policy.

cua-driver's cursor overlay rendering loop can consume CPU indefinitely when
idle (#28152, #47032).  Hermes passes ``--no-overlay`` to suppress it when the
``computer_use.no_overlay`` config is enabled (or auto-detected on headless
Linux / WSL2).

These assert the behavior contract (auto-detect on headless/WSL2, explicit
override, version probe), not specific config snapshots.
"""

import os
import sys
from unittest.mock import mock_open, patch

from tools.computer_use import cua_backend


class TestNoOverlayFlag:
    def test_default_linux_headless_disables(self):
        """Auto-detect: Linux without DISPLAY => overlay disabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            assert cua_backend._cua_no_overlay() is True

    def test_default_linux_desktop_enables(self):
        """Auto-detect: Linux with DISPLAY => overlay enabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._cua_no_overlay() is False

    def test_default_linux_wsl2_disables(self):
        """Auto-detect: WSL2 (microsoft in /proc/version) => overlay disabled."""
        fake_version = "Linux version 5.15.0 (Microsoft@Microsoft.com)"
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("builtins.open", mock_open(read_data=fake_version)):
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
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            # Explicit False overrides auto-detect on headless Linux.
            assert cua_backend._cua_no_overlay() is False

    def test_config_load_failure_falls_through_to_auto_detect(self):
        """Unreadable config => auto-detect."""
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("boom")), \
             patch.object(sys, "platform", "darwin"):
            assert cua_backend._cua_no_overlay() is False

    def test_missing_section_falls_through_to_auto_detect(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"other": {}}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._cua_no_overlay() is False


class TestDriverSupportsNoOverlay:
    def test_returns_true_when_help_shows_flag(self):
        fake_help = "Usage: cua-driver [OPTIONS] COMMAND\n  --no-overlay  Disable cursor overlay\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_help
            mock_run.return_value.stderr = ""
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is True

    def test_returns_false_when_help_lacks_flag(self):
        fake_help = "Usage: cua-driver [OPTIONS] COMMAND\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_help
            mock_run.return_value.stderr = ""
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is False

    def test_returns_false_on_subprocess_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is False


class TestMcpArgsOverlayFlag:
    def test_appended_when_enabled_and_supported(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp", "--no-overlay"]

    def test_not_appended_when_disabled(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=False), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp"]

    def test_not_appended_when_driver_unsupported(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=False):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp"]

    def test_does_not_mutate_original_list(self):
        original = ["mcp"]
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(original)
            assert "--no-overlay" in result
            assert "--no-overlay" not in original
