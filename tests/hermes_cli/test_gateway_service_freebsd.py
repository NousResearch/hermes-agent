"""Tests for the FreeBSD rc.d dispatch of gateway service helpers.

Mirrors the systemd/launchd/Windows coverage: each freebsd_rc_* helper
shells out to service(8) / sysrc(8) with the right arguments, gating
via supports_freebsd_rc() picks the right escalator (sudo vs doas), and
the dispatcher lands in the right helper on FreeBSD hosts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


# ---------------------------------------------------------------------------
# supports_freebsd_rc() detector
# ---------------------------------------------------------------------------


class TestSupportsFreebsdRc:
    def _install_stubs(self, monkeypatch, *, platform, has_service, has_script,
                       is_root, escalators):
        """Common gate-input scaffolding.  ``escalators`` is a list of names
        found on PATH (in preference order)."""
        monkeypatch.setattr(gateway_cli.sys, "platform", platform)

        def fake_which(name):
            if name == "service":
                return "/usr/sbin/service" if has_service else None
            if name in escalators:
                return f"/usr/local/bin/{name}"
            return None

        monkeypatch.setattr(gateway_cli.shutil, "which", fake_which)
        monkeypatch.setattr(gateway_cli._freebsd_is_root.__wrapped__
                            if hasattr(gateway_cli._freebsd_is_root, "__wrapped__")
                            else gateway_cli, "_freebsd_is_root", lambda: is_root)
        monkeypatch.setattr(
            Path, "exists",
            lambda self: str(self) == str(gateway_cli.FREEBSD_RC_SCRIPT_PATH)
            and has_script,
        )

    def test_true_on_freebsd_with_script_and_sudo(self, monkeypatch):
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=True,
                            has_script=True, is_root=False, escalators=["sudo"])
        assert gateway_cli.supports_freebsd_rc() is True

    def test_true_on_freebsd_with_script_as_root_no_escalator(self, monkeypatch):
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=True,
                            has_script=True, is_root=True, escalators=[])
        assert gateway_cli.supports_freebsd_rc() is True

    def test_true_with_doas_but_no_sudo(self, monkeypatch):
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=True,
                            has_script=True, is_root=False, escalators=["doas"])
        assert gateway_cli.supports_freebsd_rc() is True

    def test_false_when_non_root_no_escalator(self, monkeypatch):
        # Port installed but user cannot reach service(8) / sysrc(8) — the
        # dispatcher must fall through to "not supported", not into a
        # helper that would print "Run as root: …" and return False.
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=True,
                            has_script=True, is_root=False, escalators=[])
        assert gateway_cli.supports_freebsd_rc() is False

    def test_false_on_linux(self, monkeypatch):
        self._install_stubs(monkeypatch, platform="linux", has_service=True,
                            has_script=True, is_root=True, escalators=["sudo"])
        assert gateway_cli.supports_freebsd_rc() is False

    def test_false_on_macos(self, monkeypatch):
        self._install_stubs(monkeypatch, platform="darwin", has_service=True,
                            has_script=True, is_root=True, escalators=["sudo"])
        assert gateway_cli.supports_freebsd_rc() is False

    def test_false_when_script_missing(self, monkeypatch):
        # Package isn't installed → rc.d script absent → False.
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=True,
                            has_script=False, is_root=True, escalators=["sudo"])
        assert gateway_cli.supports_freebsd_rc() is False

    def test_false_when_service_missing(self, monkeypatch):
        # Defense in depth — service(8) is base but the gate still checks.
        self._install_stubs(monkeypatch, platform="freebsd16", has_service=False,
                            has_script=True, is_root=True, escalators=["sudo"])
        assert gateway_cli.supports_freebsd_rc() is False


# ---------------------------------------------------------------------------
# _freebsd_privilege_escalator() preference order
# ---------------------------------------------------------------------------


class TestPrivilegeEscalator:
    def test_prefers_sudo_over_doas(self, monkeypatch):
        # Both present — sudo wins because it matches the Linux path
        # used elsewhere in the codebase.
        monkeypatch.setattr(
            gateway_cli.shutil, "which",
            lambda name: f"/usr/local/bin/{name}" if name in {"sudo", "doas"} else None,
        )
        assert gateway_cli._freebsd_privilege_escalator() == "sudo"

    def test_falls_back_to_doas(self, monkeypatch):
        monkeypatch.setattr(
            gateway_cli.shutil, "which",
            lambda name: "/usr/local/bin/doas" if name == "doas" else None,
        )
        assert gateway_cli._freebsd_privilege_escalator() == "doas"

    def test_returns_none_when_neither(self, monkeypatch):
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: None)
        assert gateway_cli._freebsd_privilege_escalator() is None


# ---------------------------------------------------------------------------
# freebsd_rc_* lifecycle helpers
# ---------------------------------------------------------------------------


def _record_calls(monkeypatch, *, root=False, escalator="sudo", rc=0):
    """Common scaffolding — capture argv, force root/escalator state."""
    calls = []

    def fake_run(cmd, check=True, **kwargs):
        calls.append(list(cmd))
        if rc != 0:
            import subprocess
            raise subprocess.CalledProcessError(rc, cmd)
        return SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway_cli, "_freebsd_is_root", lambda: root)
    monkeypatch.setattr(
        gateway_cli, "_freebsd_privilege_escalator", lambda: escalator,
    )
    return calls


class TestFreebsdRcInstall:
    def test_enables_rcvar_and_records_user(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_install(force=False, run_as_user="hermes")
        assert len(calls) == 1
        assert calls[0][:2] == ["sysrc", "hermes_gateway_enable=YES"]
        assert "hermes_gateway_user=hermes" in calls[0]

    def test_does_not_start_the_service(self, monkeypatch):
        # Install only flips rcvar; the dispatcher owns the start decision.
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_install(force=False, run_as_user="hermes")
        assert not any("start" in c and "service" in c[0] for c in calls)

    def test_uses_getpass_user_when_run_as_user_none(self, monkeypatch):
        import getpass
        monkeypatch.setattr(getpass, "getuser", lambda: "someone")
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_install(force=False, run_as_user=None)
        assert "hermes_gateway_user=someone" in calls[0]

    def test_prepends_sudo_when_not_root(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=False, escalator="sudo")
        gateway_cli.freebsd_rc_install(force=False, run_as_user="hermes")
        assert calls[0][0] == "sudo"
        assert calls[0][1] == "sysrc"

    def test_prepends_doas_when_only_doas_available(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=False, escalator="doas")
        gateway_cli.freebsd_rc_install(force=False, run_as_user="hermes")
        assert calls[0][0] == "doas"
        assert calls[0][1] == "sysrc"


class TestFreebsdRcStart:
    def test_calls_service_start_as_root(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_start()
        assert calls == [["service", "hermes_gateway", "start"]]

    def test_calls_service_start_via_sudo(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=False, escalator="sudo")
        gateway_cli.freebsd_rc_start()
        assert calls == [["sudo", "service", "hermes_gateway", "start"]]


class TestFreebsdRcStop:
    def test_calls_service_stop_as_root(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_stop()
        assert calls == [["service", "hermes_gateway", "stop"]]


class TestFreebsdRcRestart:
    def test_calls_service_restart_as_root(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_restart()
        assert calls == [["service", "hermes_gateway", "restart"]]


class TestFreebsdRcUninstall:
    def test_stops_then_removes_rcvar(self, monkeypatch):
        calls = _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_uninstall()
        # Order matters: stop before disable, so autoboot-off doesn't race
        # a running instance.
        assert calls[0] == ["service", "hermes_gateway", "stop"]
        assert calls[1] == ["sysrc", "-x", "hermes_gateway_enable"]

    def test_does_not_touch_rc_d_script(self, monkeypatch, capsys):
        # The rc.d script is pkg-owned and must be removed via `pkg delete`,
        # not by the CLI.  Uninstall should print that guidance.
        _record_calls(monkeypatch, root=True)
        gateway_cli.freebsd_rc_uninstall()
        out = capsys.readouterr().out
        assert "pkg delete" in out


class TestFreebsdRcStatus:
    def test_prints_remediation_using_available_escalator(self, monkeypatch, capsys):
        # service(8) exits non-zero when the service is not running.  The
        # helper must print a remediation line using whichever escalator
        # the host actually has.
        def fake_run(cmd, check=False, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli, "_freebsd_is_root", lambda: False)
        monkeypatch.setattr(
            gateway_cli, "_freebsd_privilege_escalator", lambda: "doas",
        )

        gateway_cli.freebsd_rc_status()
        out = capsys.readouterr().out
        assert "doas service hermes_gateway start" in out
        assert "doas sysrc hermes_gateway_enable=YES" in out

    def test_omits_remediation_when_service_running(self, monkeypatch, capsys):
        def fake_run(cmd, check=False, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli, "_freebsd_is_root", lambda: True)

        gateway_cli.freebsd_rc_status()
        out = capsys.readouterr().out
        assert "To start the gateway" not in out
