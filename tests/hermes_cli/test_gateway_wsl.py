"""Tests for WSL detection and WSL-aware gateway behavior."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, mock_open

import pytest

import hermes_cli.gateway as gateway
import hermes_constants


# =============================================================================
# is_wsl() in hermes_constants
# =============================================================================

class TestIsWsl:
    """Test the shared is_wsl() utility."""

    def setup_method(self):
        # Reset cached value between tests
        hermes_constants._wsl_detected = None

    def test_detects_wsl2(self):
        fake_content = (
            "Linux version 5.15.146.1-microsoft-standard-WSL2 "
            "(gcc (GCC) 11.2.0) #1 SMP Thu Jan 11 04:09:03 UTC 2024\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_content)):
            assert hermes_constants.is_wsl() is True

    def test_detects_wsl1(self):
        fake_content = (
            "Linux version 4.4.0-19041-Microsoft "
            "(Microsoft@Microsoft.com) (gcc version 5.4.0) #1\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_content)):
            assert hermes_constants.is_wsl() is True

    def test_native_linux(self):
        fake_content = (
            "Linux version 6.5.0-44-generic (buildd@lcy02-amd64-015) "
            "(x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0) #44\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_content)):
            assert hermes_constants.is_wsl() is False

    def test_no_proc_version(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert hermes_constants.is_wsl() is False

    def test_result_is_cached(self):
        """After first detection, subsequent calls return the cached value."""
        hermes_constants._wsl_detected = True
        # Even with open raising, cached value is returned
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert hermes_constants.is_wsl() is True


# =============================================================================
# _wsl_systemd_operational() in gateway
# =============================================================================

class TestWslSystemdOperational:
    """Test the WSL systemd check."""

    def test_running(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=0, stdout="running\n", stderr=""
            ),
        )
        assert gateway._wsl_systemd_operational() is True

    def test_degraded(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stdout="degraded\n", stderr=""
            ),
        )
        assert gateway._wsl_systemd_operational() is True

    def test_starting(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stdout="starting\n", stderr=""
            ),
        )
        assert gateway._wsl_systemd_operational() is True

    def test_offline_no_systemd(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stdout="offline\n", stderr=""
            ),
        )
        assert gateway._wsl_systemd_operational() is False

    def test_systemctl_not_found(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            MagicMock(side_effect=FileNotFoundError),
        )
        assert gateway._wsl_systemd_operational() is False

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(
            gateway.subprocess, "run",
            MagicMock(side_effect=subprocess.TimeoutExpired("systemctl", 5)),
        )
        assert gateway._wsl_systemd_operational() is False


# =============================================================================
# supports_systemd_services() WSL integration
# =============================================================================

class TestSupportsSystemdServicesWSL:
    """Test that supports_systemd_services() handles WSL correctly."""

    def test_wsl_with_systemd(self, monkeypatch):
        """WSL + working systemd → True."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(gateway, "_wsl_systemd_operational", lambda: True)
        assert gateway.supports_systemd_services() is True

    def test_wsl_without_systemd(self, monkeypatch):
        """WSL + no systemd → False."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(gateway, "_wsl_systemd_operational", lambda: False)
        assert gateway.supports_systemd_services() is False

    def test_native_linux(self, monkeypatch):
        """Native Linux (not WSL) → True without checking systemd."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        assert gateway.supports_systemd_services() is True

    def test_termux_still_excluded(self, monkeypatch):
        """Termux → False regardless of WSL status."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: True)
        assert gateway.supports_systemd_services() is False


# =============================================================================
# WSL messaging in gateway commands
# =============================================================================

class TestGatewayCommandWSLMessages:
    """Test that WSL users see appropriate guidance."""

    def test_install_wsl_no_systemd(self, monkeypatch, capsys):
        """hermes gateway install on WSL without systemd shows guidance."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_macos", lambda: False)
        monkeypatch.setattr(gateway, "is_managed", lambda: False)
        # CRITICAL: also stub is_windows. Without this, running this test on a
        # real Windows host falls through to the is_windows() branch *before*
        # the WSL guidance branch, invoking gateway_windows.install() which
        # writes a Startup-folder .cmd into the real user's Startup folder
        # (NOT tmp_path) pointing at a now-vanished pytest fixture path.
        # The user then sees a broken Hermes_Gateway.cmd flash a cmd.exe
        # window on every login. See fix/windows-gateway-reliability.
        monkeypatch.setattr(gateway, "is_windows", lambda: False)

        args = SimpleNamespace(
            gateway_command="install", force=False, system=False,
            run_as_user=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            gateway.gateway_command(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "WSL detected" in out
        assert "systemd is not running" in out
        assert "hermes gateway run" in out
        assert "tmux" in out

    def test_start_wsl_no_systemd(self, monkeypatch, capsys):
        """hermes gateway start on WSL without systemd shows guidance."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_macos", lambda: False)
        # See test_install_wsl_no_systemd: stub is_windows so a Windows host
        # running this test does NOT actually spawn a detached gateway via
        # gateway_windows.start().
        monkeypatch.setattr(gateway, "is_windows", lambda: False)

        args = SimpleNamespace(gateway_command="start", system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway.gateway_command(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "WSL detected" in out
        assert "hermes gateway run" in out
        assert "wsl.conf" in out

    def test_status_wsl_running_manual(self, monkeypatch, capsys):
        """hermes gateway status on WSL with manual process shows WSL note."""
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_macos", lambda: False)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        # Stub is_windows so a Windows host running this test does NOT take
        # the Windows status branch (which reads gateway_windows.is_installed()).
        monkeypatch.setattr(gateway, "is_windows", lambda: False)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [12345])
        monkeypatch.setattr(gateway, "_runtime_health_lines", lambda: [])
        # Stub out the systemd unit path check
        monkeypatch.setattr(
            gateway, "get_systemd_unit_path",
            lambda system=False: SimpleNamespace(exists=lambda: False),
        )
        monkeypatch.setattr(
            gateway, "get_launchd_plist_path",
            lambda: SimpleNamespace(exists=lambda: False),
        )

        args = SimpleNamespace(gateway_command="status", deep=False, system=False)
        gateway.gateway_command(args)

        out = capsys.readouterr().out
        assert "WSL note" in out
        assert "tmux or screen" in out

    def test_status_wsl_not_running(self, monkeypatch, capsys):
        """hermes gateway status on WSL with no process shows WSL start advice."""
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_macos", lambda: False)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        # See test_status_wsl_running_manual.
        monkeypatch.setattr(gateway, "is_windows", lambda: False)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [])
        monkeypatch.setattr(gateway, "_runtime_health_lines", lambda: [])
        monkeypatch.setattr(
            gateway, "get_systemd_unit_path",
            lambda system=False: SimpleNamespace(exists=lambda: False),
        )
        monkeypatch.setattr(
            gateway, "get_launchd_plist_path",
            lambda: SimpleNamespace(exists=lambda: False),
        )

        args = SimpleNamespace(gateway_command="status", deep=False, system=False)
        gateway.gateway_command(args)

        out = capsys.readouterr().out
        assert "hermes gateway run" in out
        assert "tmux" in out
