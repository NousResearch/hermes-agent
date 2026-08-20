"""Tests for WSL detection and WSL-aware gateway behavior."""

import subprocess
from pathlib import Path
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


    def test_no_proc_version(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert hermes_constants.is_wsl() is False


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
        assert gateway.supports_systemd_services() is True

    @pytest.mark.linux_only
    def test_termux_still_excluded(self, monkeypatch):
        """Termux → False regardless of WSL status.

        Linux-gated: off Linux the ``not is_linux()`` arm returns False first,
        so the Termux exclusion itself would never be exercised.
        """
        monkeypatch.setattr(gateway, "is_termux", lambda: True)
        assert gateway.supports_systemd_services() is False


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
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
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
        assert "systemd is not running" in out
        assert "hermes gateway run" in out
        assert "tmux" in out


    @pytest.mark.linux_only
    def test_status_wsl_running_manual(self, monkeypatch, capsys):
        """hermes gateway status on WSL with manual process shows WSL note.

        Linux-gated for the same reason as the install case: the WSL note is
        printed only after the macOS/Windows service branches decline.
        """
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
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


class TestBuildWslInteropPaths:
    """Regression tests for issue #73163: _build_wsl_interop_paths() used to
    scrape EVERY /mnt/-prefixed entry from the current shell's PATH,
    persisting unrelated paths (Desktop app install dirs, git, node, a venv,
    etc.) into the generated systemd unit. Each of those extra entries is a
    Plan 9 filesystem round-trip on every PATH resolution the long-running
    gateway process does; accumulated over time this can exceed WSL's Plan 9
    service tolerance and crash the whole WSL VM. Only hardcoded Windows
    system paths and which()-resolved interop binaries should be persisted.
    """

    def test_non_windows_tool_paths_on_shell_path_are_not_scraped(self, monkeypatch):
        """The core regression: an unrelated /mnt/ path on the current
        shell's PATH (e.g. a Desktop app install dir) must NOT appear in
        the result, even though the old code captured every /mnt/ entry."""
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setenv(
            "PATH",
            "/usr/bin:/mnt/d/hermes-desktop/hermes/git/mingw64/bin:"
            "/mnt/d/hermes-desktop/hermes/node:/mnt/c/WINDOWS/system32",
        )
        monkeypatch.setattr(gateway.shutil, "which", lambda exe: None)
        with patch("hermes_cli.gateway.Path") as MockPath:
            MockPath.side_effect = lambda p: SimpleNamespace(
                exists=lambda: str(p) == "/mnt/c/WINDOWS/system32", parent=None
            )
            result = gateway._build_wsl_interop_paths([])

        assert "/mnt/d/hermes-desktop/hermes/git/mingw64/bin" not in result
        assert "/mnt/d/hermes-desktop/hermes/node" not in result

    def test_hardcoded_windows_system_paths_still_included(self, monkeypatch):
        """The hardcoded Windows interop paths (powershell, OpenSSH, WMI,
        system32) must still be included when they exist -- the fix only
        removes shell-PATH scraping, not the deliberate allowlist."""
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setattr(gateway.shutil, "which", lambda exe: None)

        real_exists = {
            "/mnt/c/WINDOWS/system32",
            "/mnt/c/WINDOWS",
            "/mnt/c/WINDOWS/System32/Wbem",
            "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
            "/mnt/c/WINDOWS/System32/OpenSSH/",
        }

        class _FakePath:
            def __init__(self, p):
                self._p = str(p)

            def exists(self):
                return self._p in real_exists

            @property
            def parent(self):
                return _FakePath(str(Path(self._p).parent))

            def __str__(self):
                return self._p

        with patch("hermes_cli.gateway.Path", _FakePath):
            result = gateway._build_wsl_interop_paths([])

        assert "/mnt/c/WINDOWS/system32" in result
        assert "/mnt/c/WINDOWS/System32/OpenSSH/" in result

    def test_which_resolved_interop_binaries_still_included(self, monkeypatch):
        """which()-resolved powershell.exe/cmd.exe/etc. directories must
        still be added -- which() searches the real PATH itself and needs
        no scraping loop to find them."""
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setenv("PATH", "/usr/bin")

        def _fake_which(exe):
            if exe == "powershell.exe":
                return "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe"
            return None

        monkeypatch.setattr(gateway.shutil, "which", _fake_which)

        class _FakePath:
            def __init__(self, p):
                self._p = str(p)

            def exists(self):
                return False

            @property
            def parent(self):
                return _FakePath("/".join(self._p.split("/")[:-1]))

            def __str__(self):
                return self._p

        with patch("hermes_cli.gateway.Path", _FakePath):
            result = gateway._build_wsl_interop_paths([])

        assert "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0" in result

    def test_non_wsl_returns_empty(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        assert gateway._build_wsl_interop_paths([]) == []

    def test_deduplicates_against_existing_path_entries(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_wsl", lambda: True)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setattr(gateway.shutil, "which", lambda exe: None)

        class _FakePath:
            def __init__(self, p):
                self._p = str(p)

            def exists(self):
                return self._p == "/mnt/c/WINDOWS/system32"

            @property
            def parent(self):
                return _FakePath(str(Path(self._p).parent))

            def __str__(self):
                return self._p

        with patch("hermes_cli.gateway.Path", _FakePath):
            result = gateway._build_wsl_interop_paths(["/mnt/c/WINDOWS/system32"])

        assert "/mnt/c/WINDOWS/system32" not in result
