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

import contextlib
import io
import json
import stat
import subprocess
import sys
import threading
from unittest.mock import Mock, patch

import pytest

from hermes_cli import main_desktop

# Linux-only subject: run on the real Linux host instead of faking sys.platform.
pytestmark = pytest.mark.platforms("linux")


class TestDesktopLinuxUsernsSandboxAvailable:
    def test_false_when_unshare_is_missing(self, monkeypatch):
        with patch.object(main_desktop.shutil, "which", return_value=None):
            assert main_desktop._desktop_linux_userns_sandbox_available() is False

    def test_true_when_probe_succeeds(self, monkeypatch):
        with patch.object(main_desktop.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(main_desktop.subprocess, "run") as run:
            run.return_value.returncode = 0
            assert main_desktop._desktop_linux_userns_sandbox_available() is True
        probe = run.call_args.args[0]
        assert probe[0] == "/usr/bin/unshare"
        assert "--user" in probe

    def test_false_when_probe_fails(self, monkeypatch):
        """EPERM from the kernel (userns disabled or AppArmor-restricted)."""
        with patch.object(main_desktop.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(main_desktop.subprocess, "run") as run:
            run.return_value.returncode = 1
            assert main_desktop._desktop_linux_userns_sandbox_available() is False

    def test_false_when_probe_raises(self, monkeypatch):
        with patch.object(main_desktop.shutil, "which", return_value="/usr/bin/unshare"), \
             patch.object(
                 main_desktop.subprocess,
                 "run",
                 side_effect=subprocess.TimeoutExpired(cmd="unshare", timeout=5),
             ):
            assert main_desktop._desktop_linux_userns_sandbox_available() is False


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
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 main_desktop, "_desktop_linux_userns_sandbox_available", return_value=True
             ), \
             patch.object(main_desktop.subprocess, "run") as run:
            assert main_desktop._desktop_linux_sandbox_fixup(exe) is True
        run.assert_not_called()

    def test_restricted_host_without_sudo_still_fails(self, monkeypatch, tmp_path):
        """The pre-existing strict path is preserved when userns is unusable."""
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
                 main_desktop, "_desktop_linux_userns_sandbox_available", return_value=False
             ), \
             patch.object(main_desktop.shutil, "which", return_value=None):
            assert main_desktop._desktop_linux_sandbox_fixup(exe) is False

    def test_root_owned_setuid_helper_short_circuits(self, monkeypatch, tmp_path):
        """A correctly configured helper wins before the userns probe runs."""
        exe = self._fake_packaged_app(tmp_path)
        real_lstat = (exe.parent / "chrome-sandbox").lstat()

        class _RootSetuidStat:
            st_mode = stat.S_IFREG | 0o4755
            st_uid = 0

            def __getattr__(self, name):
                return getattr(real_lstat, name)

        with patch.object(main_desktop.Path, "lstat", return_value=_RootSetuidStat()), \
             patch.object(
                 main_desktop, "_desktop_linux_userns_sandbox_available"
             ) as probe:
            assert main_desktop._desktop_linux_sandbox_fixup(exe) is True
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
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(
            main_desktop, "_desktop_linux_userns_sandbox_available", return_value=True
        ):
            assert main_desktop._desktop_linux_needs_disable_setuid_sandbox(exe) is True

    def test_false_for_root_owned_setuid_helper(self, monkeypatch, tmp_path):
        exe = self._fake_packaged_app(tmp_path)
        real_lstat = (exe.parent / "chrome-sandbox").lstat()

        class _RootSetuidStat:
            st_mode = stat.S_IFREG | 0o4755
            st_uid = 0

            def __getattr__(self, name):
                return getattr(real_lstat, name)

        with patch.object(main_desktop.Path, "lstat", return_value=_RootSetuidStat()), \
             patch.object(
                 main_desktop, "_desktop_linux_userns_sandbox_available", return_value=True
             ) as probe:
            assert main_desktop._desktop_linux_needs_disable_setuid_sandbox(exe) is False
        probe.assert_not_called()

    def test_false_when_helper_missing(self, monkeypatch, tmp_path):
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        assert main_desktop._desktop_linux_needs_disable_setuid_sandbox(exe) is False


class TestDesktopLinuxSandboxFixupNoTty:
    """No-TTY sudo must fail fast, never hang the launch (#123927).

    When the .desktop entry (``Terminal=false``), autostart, or a detached
    updater relaunch reaches the sudo path — a user-owned ``chrome-sandbox``
    plus no usable user namespace (hardened kernels such as CachyOS's
    ``linux-cachyos-hardened``, ``user.max_user_namespaces=0``, containers)
    — sudo's password prompt can never be answered. The call used to inherit
    stdio with no ``-n`` and no timeout, so the icon click hung indefinitely.
    """

    def _fake_packaged_app(self, tmp_path):
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir(exist_ok=True)
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        sandbox = unpacked / "chrome-sandbox"
        sandbox.write_text("", encoding="utf-8")
        with contextlib.suppress(OSError):
            sandbox.chmod(0o755)
        return exe

    def test_sudo_is_non_interactive_without_a_tty(self, monkeypatch, tmp_path):
        """Without a TTY sudo gets ``-n`` + DEVNULL stdin + a timeout."""
        exe = self._fake_packaged_app(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return Mock(returncode=0)

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "stdin", io.StringIO())  # isatty() is False
        monkeypatch.setattr(
            main_desktop, "_desktop_linux_userns_sandbox_available", lambda: False)
        monkeypatch.setattr(main_desktop.shutil, "which", lambda _name: "/usr/bin/sudo")
        monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)
        assert main_desktop._desktop_linux_sandbox_fixup(exe) is True
        assert len(calls) == 2
        for cmd, kwargs in calls:
            assert cmd[0] == "/usr/bin/sudo"
            assert "-n" in cmd
            assert kwargs.get("stdin") == subprocess.DEVNULL
            assert kwargs.get("timeout") is not None

    def test_sudo_is_non_interactive_with_an_unusable_stdin(self, monkeypatch, tmp_path):
        """A detached launch (``sys.stdin is None``) or a closed stream still fails fast.

        CPython sets ``sys.stdin`` to ``None`` when the process has no stdin at all,
        and ``isatty()`` on a closed stream raises; both used to escape as an
        ``AttributeError``/``ValueError`` before the caller's ``--no-sandbox``
        fallback could run.
        """
        exe = self._fake_packaged_app(tmp_path)
        for state in ("missing", "closed"):
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                return Mock(returncode=0)

            if state == "missing":
                monkeypatch.setattr(sys, "stdin", None)
            else:
                closed = io.StringIO()
                closed.close()
                monkeypatch.setattr(sys, "stdin", closed)
            monkeypatch.setattr(sys, "platform", "linux")
            monkeypatch.setattr(
                main_desktop, "_desktop_linux_userns_sandbox_available", lambda: False)
            monkeypatch.setattr(main_desktop.shutil, "which", lambda _name: "/usr/bin/sudo")
            monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)
            assert main_desktop._desktop_linux_sandbox_fixup(exe) is True, state
            assert len(calls) == 2, state
            for cmd, kwargs in calls:
                assert "-n" in cmd, state
                assert kwargs.get("stdin") == subprocess.DEVNULL, state

    def test_sudo_stays_interactive_with_a_tty(self, monkeypatch, tmp_path):
        """A terminal launch keeps the password prompt (no -n)."""
        exe = self._fake_packaged_app(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return Mock(returncode=0)

        tty = Mock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "stdin", tty)
        monkeypatch.setattr(
            main_desktop, "_desktop_linux_userns_sandbox_available", lambda: False)
        monkeypatch.setattr(main_desktop.shutil, "which", lambda _name: "/usr/bin/sudo")
        monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)
        assert main_desktop._desktop_linux_sandbox_fixup(exe) is True
        assert len(calls) == 2
        for cmd, _kwargs in calls:
            assert "-n" not in cmd

    def test_blocking_sudo_returns_fast(self, tmp_path):
        """A sudo stuck at its password prompt must not hang the fixup.

        The fake honours ``-n`` (exit 1, like a real sudo with no cached
        credentials) and blocks otherwise (the pre-fix hang: a prompt with
        no TTY to answer it). Needs a POSIX shell, so Linux-only like the
        rest of this file.
        """
        exe = self._fake_packaged_app(tmp_path)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake = fake_bin / "sudo"
        fake.write_text(
            "#!/bin/sh\n"
            'for a in "$@"; do if [ "$a" = "-n" ]; then exit 1; fi; done\n'
            "sleep 30\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        outcome: dict = {}

        def target():
            outcome["result"] = main_desktop._desktop_linux_sandbox_fixup(exe)

        with patch.object(sys, "platform", "linux"), \
             patch.object(
                 main_desktop, "_desktop_linux_userns_sandbox_available",
                 return_value=False), \
             patch.object(main_desktop.shutil, "which", return_value=str(fake)), \
             patch.object(sys, "stdin", io.StringIO()):
            worker = threading.Thread(target=target, daemon=True)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), \
                "fixup hung in sudo with no TTY (icon click goes nowhere)"
            assert outcome.get("result") is False


class TestDesktopStickyNoSandbox:
    """The app's recorded fallback must reach argv, not only ``appendSwitch``.

    Electron applies the in-app switch after the zygote has spawned; on a host whose
    sandboxed zygote breaks shared memory the renderer dies pre-load (exit 132) and the
    window shows the recovery page (2026-10-13 → 2026-10-24, ~24.6k dumps).
    """

    def _write_marker(self, tmp_path, monkeypatch, payload: str):
        config_home = tmp_path / "config"
        (config_home / "Hermes").mkdir(parents=True)
        (config_home / "Hermes" / main_desktop._DESKTOP_SANDBOX_FALLBACK_MARKER).write_text(
            payload, encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    def _fake_packaged_app(self, tmp_path):
        unpacked = tmp_path / "linux-unpacked"
        unpacked.mkdir()
        exe = unpacked / "Hermes"
        exe.write_text("", encoding="utf-8")
        (unpacked / "chrome-sandbox").write_text("", encoding="utf-8")
        return exe

    def test_true_when_the_app_recorded_a_fallback(self, tmp_path, monkeypatch):
        self._write_marker(tmp_path, monkeypatch, json.dumps({"state": "fallback", "reason": "boot-loop"}))
        assert main_desktop._desktop_sticky_no_sandbox() is True

    def test_false_when_the_app_recorded_a_healthy_boot(self, tmp_path, monkeypatch):
        self._write_marker(tmp_path, monkeypatch, json.dumps({"state": "ok"}))
        assert main_desktop._desktop_sticky_no_sandbox() is False

    def test_false_without_a_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        assert main_desktop._desktop_sticky_no_sandbox() is False

    def test_false_for_a_malformed_marker(self, tmp_path, monkeypatch):
        self._write_marker(tmp_path, monkeypatch, "not json")
        assert main_desktop._desktop_sticky_no_sandbox() is False

    def test_launch_command_forwards_the_flag(self, tmp_path, monkeypatch):
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(main_desktop, "_desktop_linux_sandbox_fixup", return_value=True), \
             patch.object(main_desktop, "_desktop_linux_needs_disable_setuid_sandbox", return_value=False), \
             patch.object(main_desktop, "_desktop_sticky_no_sandbox", return_value=True):
            assert main_desktop._packaged_desktop_launch_command(exe) == [str(exe), "--no-sandbox"]

    def test_flag_is_not_duplicated(self, tmp_path, monkeypatch):
        """The userns fallback already appended it; the sticky marker must not add a second."""
        exe = self._fake_packaged_app(tmp_path)
        with patch.object(main_desktop, "_desktop_linux_sandbox_fixup", return_value=False), \
             patch.object(main_desktop, "_desktop_linux_needs_no_sandbox", return_value=True), \
             patch.object(main_desktop, "_desktop_linux_sandbox_helper_is_regular_file", return_value=True), \
             patch.object(main_desktop, "_desktop_sticky_no_sandbox", return_value=True):
            command = main_desktop._packaged_desktop_launch_command(exe)
        assert command.count("--no-sandbox") == 1
