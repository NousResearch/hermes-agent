"""Tests for the KDE Plasma/X11 crash guard (#66392).

cua-driver 0.8.3's uinput virtual pointer can crash a KDE Plasma/Qt X11
session during MCP initialization. The guard blocks ``start()`` on
detected KDE Plasma / X11 hosts unless the user explicitly opts in via
``computer_use.linux_opt_in: true`` in config.yaml.

Non-Linux, non-KDE, and config-read-error paths all return True so the
guard never blocks production macOS/Windows use.
"""

import os
from unittest.mock import patch

from tools.computer_use import cua_backend


class TestIsLinux:
    def test_linux(self):
        with patch.object(cua_backend.sys, "platform", "linux"):
            assert cua_backend._is_linux() is True

    def test_darwin(self):
        with patch.object(cua_backend.sys, "platform", "darwin"):
            assert cua_backend._is_linux() is False

    def test_win32(self):
        with patch.object(cua_backend.sys, "platform", "win32"):
            assert cua_backend._is_linux() is False


class TestIsLinuxX11:
    def test_linux_with_display(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._is_linux_x11() is True

    def test_linux_no_display(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, clear=True):
            assert cua_backend._is_linux_x11() is False

    def test_darwin_ignores_display(self):
        with patch.object(cua_backend.sys, "platform", "darwin"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._is_linux_x11() is False


class TestKdePlasmaDetected:
    def test_linux_kde_full_session(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"KDE_FULL_SESSION": "true"}):
            assert cua_backend._kde_plasma_detected() is True

    def test_linux_xdg_current_desktop_kde(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "KDE"}):
            assert cua_backend._kde_plasma_detected() is True

    def test_linux_desktop_session_plasma(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"DESKTOP_SESSION": "plasma"}):
            assert cua_backend._kde_plasma_detected() is True

    def test_linux_non_kde_returns_false(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "GNOME"}):
            assert cua_backend._kde_plasma_detected() is False

    def test_non_linux_kde_env_returns_false(self):
        """Even with KDE env vars, non-Linux is not KDE for our purposes."""
        with patch.object(cua_backend.sys, "platform", "darwin"), \
             patch.dict(os.environ, {"KDE_FULL_SESSION": "true"}):
            assert cua_backend._kde_plasma_detected() is False

    def test_linux_no_desktop_vars_returns_false(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, clear=True):
            assert cua_backend._kde_plasma_detected() is False


class TestLinuxCuaAllowed:
    def test_non_linux_always_allowed(self):
        with patch.object(cua_backend.sys, "platform", "darwin"):
            assert cua_backend._linux_cua_allowed() is True

    def test_linux_non_kde_allowed(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "GNOME"}):
            assert cua_backend._linux_cua_allowed() is True

    def test_linux_kde_without_opt_in_blocked(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"KDE_FULL_SESSION": "true"}), \
             patch("hermes_cli.config.load_config", return_value={}):
            assert cua_backend._linux_cua_allowed() is False

    def test_linux_kde_with_opt_in_allowed(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"KDE_FULL_SESSION": "true"}), \
             patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"linux_opt_in": True}}):
            assert cua_backend._linux_cua_allowed() is True

    def test_config_read_error_fails_open(self):
        with patch.object(cua_backend.sys, "platform", "linux"), \
             patch.dict(os.environ, {"KDE_FULL_SESSION": "true"}), \
             patch("hermes_cli.config.load_config", side_effect=ImportError):
            assert cua_backend._linux_cua_allowed() is True


class TestCuaDriverBackendStartGuard:
    def test_start_raises_on_blocked(self):
        backend = cua_backend.CuaDriverBackend()
        with patch.object(cua_backend, "_linux_cua_allowed", return_value=False), \
             patch.object(backend, "_session"):
            import pytest
            with pytest.raises(RuntimeError, match="computer_use is blocked.*KDE"):
                backend.start()

    def test_start_proceeds_when_allowed(self):
        backend = cua_backend.CuaDriverBackend()
        with patch.object(cua_backend, "_linux_cua_allowed", return_value=True), \
             patch.object(backend, "_session"), \
             patch.object(cua_backend, "_maybe_nudge_update"):
            # Should not raise
            backend.start()
