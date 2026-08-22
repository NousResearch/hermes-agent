"""Tests for the Linux desktop sandbox-helper fixup and the userns probe.

``_desktop_linux_sandbox_fixup`` historically demanded a root-owned 4755
``chrome-sandbox`` on every Linux host and shelled out to ``sudo`` to get it
— which fails silently when the desktop entry launches ``hermes desktop``
without a TTY (#88032, #51327), and blocked the updater's relaunch gate
(#58593). On hosts where unprivileged user namespaces work, Chromium uses
its namespace sandbox and never consults the setuid helper, so the fixup now
probes for that capability first and skips the sudo path entirely.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from unittest.mock import patch

from hermes_cli import main as cli_main


class TestDesktopLinuxUsernsSandboxAvailable:
    def test_false_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert cli_main._desktop_linux_userns_sandbox_available() is False

    def test_false_when_unshare_is_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with patch.object(cli_main.shutil, "which", return_value=None):
            assert cli_main._desktop_linux_userns_sandbox_available() is False

    def test_true_when_probe_succeeds(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with patch.object(cli_main.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value.returncode = 0
            assert cli_main._desktop_linux_userns_sandbox_available() is True
        probe = run.call_args.args[0]
        assert probe[0] == "/usr/bin/unshare"
        assert "--user" in probe

    def test_false_when_probe_fails(self, monkeypatch):
        """EPERM from the kernel (userns disabled or AppArmor-restricted)."""
        monkeypatch.setattr(sys, "platform", "linux")
        with patch.object(cli_main.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value.returncode = 1
            assert cli_main._desktop_linux_userns_sandbox_available() is False

    def test_false_when_probe_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with patch.object(cli_main.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(
                 cli_main.subprocess,
                 "run",
                 side_effect=subprocess.TimeoutExpired(cmd="unshare", timeout=5),
             ):
            assert cli_main._desktop_linux_userns_sandbox_available() is False


class TestDesktopLinuxSandboxFixup:
    def _fake_packaged_app(self, tmp_path):
        """Unpacked-app layout with a non-root, non-setuid chrome-sandbox."""
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        sandbox = unpacked / "chrome-sandbox"
        sandbox.write_text("", encoding="utf-8")
        sandbox.chmod(0o755)
        return exe

    def test_userns_host_skips_sudo_and_succeeds(self, monkeypatch, tmp_path):
        """A user-owned helper must not trigger sudo when userns works.

        This is the .desktop-launch regression: no TTY means sudo cannot
        prompt, so reaching the sudo path at all kills the launch.
        """
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=True
             ), \
             patch.object(cli_main.subprocess, "run") as run:
            assert cli_main._desktop_linux_sandbox_fixup(exe) is True
        run.assert_not_called()

    def test_restricted_host_without_sudo_still_fails(self, monkeypatch, tmp_path):
        """The pre-existing strict path is preserved when userns is unusable."""
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(cli_main.shutil, "which", return_value=None):
            assert cli_main._desktop_linux_sandbox_fixup(exe) is False

    def test_root_owned_setuid_helper_short_circuits(self, monkeypatch, tmp_path):
        """A correctly configured helper wins before the userns probe runs."""
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        real_lstat = (exe.parent / "chrome-sandbox").lstat()

        class _RootSetuidStat:
            st_mode = stat.S_IFREG | 0o4755
            st_uid = 0

            def __getattr__(self, name):
                return getattr(real_lstat, name)

        with patch.object(cli_main.Path, "lstat", return_value=_RootSetuidStat()), \
             patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available"
             ) as probe:
            assert cli_main._desktop_linux_sandbox_fixup(exe) is True
        probe.assert_not_called()
