"""Tests for WSL detection and WSL-aware gateway behavior."""
from gateway import systemd_runtime

from types import SimpleNamespace
from unittest.mock import patch, mock_open

import pytest

import hermes_cli.gateway as gateway
import hermes_constants
from hermes_platform.host import runtime as host_runtime


# =============================================================================
# is_wsl() in hermes_constants
# =============================================================================

class TestIsWsl:
    """Test the shared is_wsl() utility."""

    def setup_method(self):
        # Reset cached value between tests
        host_runtime._wsl_detected = None

    def test_detects_wsl2(self):
        fake_content = (
            "Linux version 5.15.146.1-microsoft-standard-WSL2 "
            "(gcc (GCC) 11.2.0) #1 SMP Thu Jan 11 04:09:03 UTC 2024\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_content)):
            assert hermes_constants.is_wsl() is True


    def test_no_proc_version(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert hermes_constants.is_wsl() is False


# =============================================================================
# supports_systemd_services() WSL integration
# =============================================================================

class TestSupportsSystemdServicesWSL:
    """Test that supports_systemd_services() handles WSL correctly."""

    @pytest.mark.linux_only
    def test_wsl_with_systemd(self, monkeypatch):
        """WSL + working systemd → True.

        Linux-gated: ``supports_systemd_services()`` short-circuits on
        ``is_linux()``, so off Linux this asserted nothing about systemd.
        """
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(
            gateway.shutil, "which", lambda _name: "/usr/bin/systemctl"
        )
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(gateway, "_wsl_systemd_operational", lambda: True)
        assert systemd_runtime.supports_services() is True

    @pytest.mark.linux_only
    def test_termux_still_excluded(self, monkeypatch):
        """Termux → False regardless of WSL status.

        Linux-gated: off Linux the ``not is_linux()`` arm returns False first,
        so the Termux exclusion itself would never be exercised.
        """
        monkeypatch.setattr(gateway, "is_termux", lambda: True)
        assert systemd_runtime.supports_services() is False


# =============================================================================
# WSL messaging in gateway commands
# =============================================================================

class TestGatewayCommandWSLMessages:
    """Test that WSL users see appropriate guidance."""

    @pytest.mark.linux_only
    def test_install_wsl_no_systemd(self, monkeypatch, capsys):
        """hermes gateway install on WSL without systemd shows guidance.

        Linux-gated: WSL *is* a Linux host, and the guidance branch sits after
        the macOS/Windows arms in ``gateway_command``. Reaching it on another
        host previously required stubbing ``is_macos``/``is_windows`` — on a
        real Windows host the unstubbed version would have run
        ``gateway_windows.install()`` against the user's real Startup folder.
        """
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(systemd_runtime, "supports_services", lambda: False)
        monkeypatch.setattr(gateway, "is_managed", lambda: False)

        args = SimpleNamespace(
            gateway_command="install", force=False, system=False,
            run_as_user=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            gateway.gateway_command(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "WSL detected" in out
        assert "hermes gateway run" in out



