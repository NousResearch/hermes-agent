"""Tests for update-time kickstart of service-managed dashboard LaunchAgents (issue #44106 req 5-6).

Disjoint from test_update_stale_dashboard.py (clean review boundary); mirrors
its isolation patterns (function-level refresh, monkeypatched scan, ps stub).
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.dashboard_service import find_service_managed_dashboard_pids, get_dashboard_launchd_label
from hermes_cli.dashboard_procs import _kill_stale_dashboard_processes
from hermes_cli import dashboard_service
from hermes_cli import dashboard_procs


@pytest.fixture(autouse=True)
def _refresh_bindings():
    """Rebind like test_update_stale_dashboard.py does."""
    global find_service_managed_dashboard_pids
    global _kill_stale_dashboard_processes
    find_service_managed_dashboard_pids = dashboard_service.find_service_managed_dashboard_pids
    _kill_stale_dashboard_processes = dashboard_procs._kill_stale_dashboard_processes
    yield


# ------------------------------------------------------------------
# find_service_managed_dashboard_pids
# ------------------------------------------------------------------

class TestFindServiceManagedDashboardPids:
    """Discovery via plists + launchctl, not argv heuristics."""

    def test_non_macos_returns_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: False)
        assert find_service_managed_dashboard_pids() == {}

    def test_finds_our_plists_ignores_others(self, tmp_path, monkeypatch):
        # Create fake LaunchAgents dir at temp home (find_service_managed_dashboard_pids uses pwd home).
        fake_home = tmp_path / "home"
        launchd_dir = fake_home / "Library" / "LaunchAgents"
        launchd_dir.mkdir(parents=True)

        # Ours (valid)
        ours = launchd_dir / "ai.hermes.dashboard.plist"
        ours.write_bytes(plistlib.dumps({
            "Label": "ai.hermes.dashboard",
            "ProgramArguments": ["python", "-m", "hermes_cli.main", "dashboard"],
        }))
        # Named profile variant
        named = launchd_dir / "ai.hermes.dashboard-work.plist"
        named.write_bytes(plistlib.dumps({
            "Label": "ai.hermes.dashboard-work",
            "ProgramArguments": ["python", "-m", "hermes_cli.main", "--profile", "work", "dashboard"],
        }))
        # Unrelated gateway (ignored)
        gateway = launchd_dir / "ai.hermes.gateway.plist"
        gateway.write_bytes(plistlib.dumps({
            "Label": "ai.hermes.gateway",
        }))
        # Unparseable (skipped with diagnostic)
        bad = launchd_dir / "ai.hermes.dashboard-bad.plist"
        bad.write_text("<<< not xml <<>", encoding="utf-8")

        # Mock: patch the internal import site (where find_service_managed_dashboard_pids imports from).
        # Monkeypatch both import paths (function-level import inside dashboard_service and gateway).
        monkeypatch.setattr(
            "hermes_cli.gateway._launchd_print_service_pid",
            lambda domain, label: (True, 12345) if label == "ai.hermes.dashboard" else (
                (True, 12346) if label == "ai.hermes.dashboard-work" else (False, None)
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service._launchd_print_service_pid",
            lambda domain, label: (True, 12345) if label == "ai.hermes.dashboard" else (
                (True, 12346) if label == "ai.hermes.dashboard-work" else (False, None)
            ),
        )
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        monkeypatch.setattr("hermes_cli.gateway._launchd_domain", lambda: "gui/501")
        # Redirect pwd-based home resolution (find_service_managed_dashboard_pids uses pwd.getpwuid).
        import pwd as _pwd
        monkeypatch.setattr(
            _pwd, "getpwuid", lambda uid: type("FakePwd", (), {"pw_dir": str(fake_home)})()
        )
        # Fake home so glob hits our temp dir.

        result = find_service_managed_dashboard_pids()
        assert result == {12345: "ai.hermes.dashboard", 12346: "ai.hermes.dashboard-work"}

    def test_unparseable_plist_skipped_others_found(self, tmp_path, monkeypatch, capsys):
        d = tmp_path / "home" / "Library" / "LaunchAgents"
        d.mkdir(parents=True)
        good = d / "ai.hermes.dashboard.plist"
        good.write_bytes(plistlib.dumps({"Label": "ai.hermes.dashboard"}))
        bad = d / "ai.hermes.dashboard-corrupt.plist"
        bad.write_bytes(b"not xml")

        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        # Redirect pwd-based home resolution.
        import pwd as _pwd
        monkeypatch.setattr(
            _pwd, "getpwuid", lambda uid: type("FakePwd", (), {"pw_dir": str(tmp_path / "home")})()
        )
        monkeypatch.setattr("hermes_cli.gateway._launchd_domain", lambda: "gui/501")
        # Mock: patch the internal import site (where find_service_managed_dashboard_pids imports from).
        monkeypatch.setattr(
            "hermes_cli.gateway._launchd_print_service_pid",
            lambda domain, label: (True, 777) if label == "ai.hermes.dashboard" else (False, None),
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service._launchd_print_service_pid",
            lambda domain, label: (True, 777) if label == "ai.hermes.dashboard" else (False, None),
        )

        result = find_service_managed_dashboard_pids()
        out = capsys.readouterr().out
        assert "skipped unparseable" in out
        assert result == {777: "ai.hermes.dashboard"}


# ------------------------------------------------------------------
# _kill_stale_dashboard_processes integration
# ------------------------------------------------------------------

class TestKillStaleDashboardLaunchdIntegration:
    def test_kickstart_success_excludes_pid_and_adds_exclusion(self, monkeypatch, capsys):
        # restart_managed=True; managed_pids has the scanned pid.
        monkeypatch.setattr(
            "hermes_cli.main_dashboard._find_stale_dashboard_pids",
            lambda exclude_pids=None: [999],
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service.find_service_managed_dashboard_pids",
            lambda: {999: "ai.hermes.dashboard"},
        )
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        monkeypatch.setattr("hermes_cli.dashboard_procs.is_macos", lambda: True)
        monkeypatch.setattr("os.getuid", lambda: 42)  # windows-footgun: ok — stubbing the POSIX call, never executed on Windows

        # Mock successful kickstart.
        kickstart_calls = []
        original_run = subprocess.run

        def fake_run(args, **kw):
            if isinstance(args, list) and args[:2] == ["launchctl", "kickstart"]:
                kickstart_calls.append(args)
                return MagicMock(returncode=0, stdout="", stderr="")
            return original_run(args, **kw)

        monkeypatch.setattr("subprocess.run", fake_run)
        # Mock kill: never called for excluded pid.
        kill_called = []
        def fake_kill(pids, killed, failed):
            kill_called.extend(pids)
        monkeypatch.setattr("hermes_cli.dashboard_procs._kill_pids_posix", fake_kill)

        # Already restarted units starts empty.
        result = _kill_stale_dashboard_processes(restart_managed=True, already_restarted_units=set())
        assert 999 not in kill_called or kill_called == []  # excluded from kill
        assert ["launchctl", "kickstart", "-k", "gui/42/ai.hermes.dashboard"] in kickstart_calls
        out = capsys.readouterr().out
        assert "✓ kickstarted ai.hermes.dashboard (PID 999)" in out
        # After exclusion: pids removed, so matched should be empty (or just 999 removed).
        # Since we filter pids before kill, result matched should not include 999.
        assert 999 not in result.get("matched", [])
        # When all pids excluded (all kickstarted), "Stopping" line must be absent (F4).
        assert "⟲ Stopping" not in out

    def test_kickstart_failure_falls_through_to_kill_with_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "hermes_cli.main_dashboard._find_stale_dashboard_pids",
            lambda exclude_pids=None: [999],
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service.find_service_managed_dashboard_pids",
            lambda: {999: "ai.hermes.dashboard"},
        )
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        monkeypatch.setattr("hermes_cli.dashboard_procs.is_macos", lambda: True)
        monkeypatch.setattr("os.getuid", lambda: 42)  # windows-footgun: ok — stubbing the POSIX call, never executed on Windows

        # Kickstart always fails.
        def fail_run(args, **kw):
            if isinstance(args, list) and args[:2] == ["launchctl", "kickstart"]:
                raise subprocess.CalledProcessError(1, args, stderr="error")
            return MagicMock(returncode=0, stdout="", stderr="")
        monkeypatch.setattr("subprocess.run", fail_run)

        # Kill proceeds.
        kill_pids = []
        def capture_kill(pids, killed, failed):
            kill_pids.extend(pids)
            for p in pids:
                killed.append(p)
        monkeypatch.setattr("hermes_cli.dashboard_procs._kill_pids_posix", capture_kill)

        result = _kill_stale_dashboard_processes(restart_managed=True)
        assert 999 in kill_pids
        out = capsys.readouterr().out
        assert "launchctl kickstart -k gui/$(id -u)/ai.hermes.dashboard" in out
        assert "launchd will respawn via KeepAlive" in out
        assert 999 in result.get("killed", [])

    def test_stop_path_prints_keepalive_hint_for_managed_pid(self, monkeypatch, capsys):
        # restart_managed=False (e.g. --stop).
        monkeypatch.setattr(
            "hermes_cli.main_dashboard._find_stale_dashboard_pids",
            lambda exclude_pids=None: [888],
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service.find_service_managed_dashboard_pids",
            lambda: {888: "ai.hermes.dashboard-work"},
        )
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        monkeypatch.setattr("hermes_cli.dashboard_procs.is_macos", lambda: True)

        def capture_kill(pids, killed, failed):
            for p in pids:
                killed.append(p)
        monkeypatch.setattr("hermes_cli.dashboard_procs._kill_pids_posix", capture_kill)

        result = _kill_stale_dashboard_processes(restart_managed=False)
        out = capsys.readouterr().out
        assert "managed by launchd job ai.hermes.dashboard-work" in out
        assert "hermes dashboard service stop" in out
        assert 888 in result.get("killed", [])

    def test_kickstart_timeout_expired_falls_through_to_kill_with_hint(self, monkeypatch, capsys):
        # TimeoutExpired should degrade to raw-kill fallback + manual hint (F3).
        monkeypatch.setattr(
            "hermes_cli.main_dashboard._find_stale_dashboard_pids",
            lambda exclude_pids=None: [777],
        )
        monkeypatch.setattr(
            "hermes_cli.dashboard_service.find_service_managed_dashboard_pids",
            lambda: {777: "ai.hermes.dashboard"},
        )
        monkeypatch.setattr("hermes_cli.dashboard_service.is_macos", lambda: True)
        monkeypatch.setattr("hermes_cli.dashboard_procs.is_macos", lambda: True)
        monkeypatch.setattr("os.getuid", lambda: 42)  # windows-footgun: ok — stubbing the POSIX call, never executed on Windows

        import subprocess

        def timeout_run(args, **kw):
            if isinstance(args, list) and args[:2] == ["launchctl", "kickstart"]:
                raise subprocess.TimeoutExpired(args, timeout=30)
            return MagicMock(returncode=0, stdout="", stderr="")
        monkeypatch.setattr("subprocess.run", timeout_run)

        kill_pids = []
        def capture_kill(pids, killed, failed):
            kill_pids.extend(pids)
            for p in pids:
                killed.append(p)
        monkeypatch.setattr("hermes_cli.dashboard_procs._kill_pids_posix", capture_kill)

        result = _kill_stale_dashboard_processes(restart_managed=True)
        assert 777 in kill_pids
        out = capsys.readouterr().out
        assert "run manually: launchctl kickstart -k" in out
        assert "launchd will respawn via KeepAlive" in out
        assert 777 in result.get("killed", [])
