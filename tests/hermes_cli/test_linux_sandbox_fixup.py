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


class TestDesktopLinuxNeedsDisableSetuidSandbox:
    def _fake_packaged_app(self, tmp_path):
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        sandbox = unpacked / "chrome-sandbox"
        sandbox.write_text("", encoding="utf-8")
        sandbox.chmod(0o755)
        return exe

    def test_true_for_user_owned_helper_when_userns_works(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
            cli_main, "_desktop_linux_userns_sandbox_available", return_value=True
        ):
            assert cli_main._desktop_linux_needs_disable_setuid_sandbox(exe) is True

    def test_false_for_root_owned_setuid_helper(self, monkeypatch, tmp_path):
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
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=True
             ) as probe:
            assert cli_main._desktop_linux_needs_disable_setuid_sandbox(exe) is False
        probe.assert_not_called()

    def test_false_when_helper_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        assert cli_main._desktop_linux_needs_disable_setuid_sandbox(exe) is False


class TestDesktopLinuxSandboxFixupEscalation:
    """The escalation path when userns is unavailable and the helper needs setuid.

    On such hosts the fixup must still reach root, but a bare ``sudo`` can only
    prompt with a TTY. Launched from the .desktop entry there is none, so it
    must not be attempted; ``sudo -n`` (cached creds / NOPASSWD) and ``pkexec``
    (graphical polkit prompt) are the paths that can succeed there.
    """

    def _fake_packaged_app(self, tmp_path):
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        sandbox = unpacked / "chrome-sandbox"
        sandbox.write_text("", encoding="utf-8")
        sandbox.chmod(0o755)
        return exe

    def _which(self, available):
        return lambda name: f"/usr/bin/{name}" if name in available else None

    def test_no_tty_uses_pkexec_and_never_bare_sudo(self, monkeypatch, tmp_path):
        """Without a TTY, bare sudo must never run — it can only fail there."""
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(cli_main.shutil, "which",
                          side_effect=self._which({"sh", "sudo", "pkexec"})), \
             patch.object(cli_main.sys.stdin, "isatty", return_value=False), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1)
            cli_main._desktop_linux_sandbox_fixup(exe)

        argvs = [call.args[0] for call in run.call_args_list]
        assert any(a[0].endswith("pkexec") for a in argvs), "pkexec must be attempted"
        sudo_calls = [a for a in argvs if a[0].endswith("sudo")]
        assert sudo_calls, "sudo -n must be attempted"
        for a in sudo_calls:
            assert a[1] == "-n", f"bare sudo must not run without a TTY: {a}"

    def test_pkexec_success_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(cli_main.shutil, "which",
                          side_effect=self._which({"sh", "pkexec"})), \
             patch.object(cli_main.sys.stdin, "isatty", return_value=False), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            assert cli_main._desktop_linux_sandbox_fixup(exe) is True

    def test_tty_host_may_use_interactive_sudo(self, monkeypatch, tmp_path):
        """With a TTY, bare sudo is a legitimate second attempt."""
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(cli_main.shutil, "which",
                          side_effect=self._which({"sh", "sudo"})), \
             patch.object(cli_main.sys.stdin, "isatty", return_value=True), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1)
            assert cli_main._desktop_linux_sandbox_fixup(exe) is False

        argvs = [call.args[0] for call in run.call_args_list]
        assert any(a[0].endswith("sudo") and a[1] == "-n" for a in argvs)
        assert any(a[0].endswith("sudo") and a[1] != "-n" for a in argvs)

    def test_headless_failure_notifies_the_user(self, monkeypatch, tmp_path):
        """A launcher-started process shows no stdout — surface the reason."""
        monkeypatch.setattr(sys, "platform", "linux")
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 cli_main, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(cli_main.shutil, "which",
                          side_effect=self._which({"sh", "pkexec", "notify-send"})), \
             patch.object(cli_main.sys.stdin, "isatty", return_value=False), \
             patch.object(cli_main.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 1)
            assert cli_main._desktop_linux_sandbox_fixup(exe) is False

        argvs = [call.args[0] for call in run.call_args_list]
        assert any(
            a[0].endswith("notify-send") for a in argvs
        ), "must notify on headless failure"
