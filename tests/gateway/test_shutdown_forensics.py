"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from gateway import shutdown_forensics as sf


# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

class TestSignalName:

    def test_unknown_int_returns_signal_num_token(self):
        # Pick an integer extremely unlikely to ever be a real signal alias
        assert sf._signal_name(9999) == "signal#9999"


# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_handles_none_signal(self):
        ctx = sf.snapshot_shutdown_context(None)
        assert ctx["signal"] == "UNKNOWN"
        assert ctx["signal_num"] is None

    def test_includes_timestamps(self):
        before = time.time()
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        after = time.time()
        assert before <= ctx["ts"] <= after
        assert isinstance(ctx["ts_monotonic"], float)


    def test_under_systemd_false_without_invocation_id_and_normal_ppid(
        self, monkeypatch
    ):
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        # We can't actually change ppid; skip if we happen to be reaped
        # by init (e.g. running under tini).
        if os.getppid() == 1:
            pytest.skip("test process is reaped by init")
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["under_systemd"] is False


    def test_detects_takeover_marker_for_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}, "replacer_pid": 99999}}',
            encoding="utf-8",
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "takeover_marker" in ctx
        assert ctx["takeover_marker_for_self"] is True


# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:


    def test_context_as_json_handles_unserialisable_values(self):
        ctx = {"signal": "SIGTERM", "weird": object()}
        payload = sf.context_as_json(ctx)
        # default=str means objects get repr'd, JSON stays valid
        decoded = json.loads(payload)
        assert decoded["signal"] == "SIGTERM"
        assert "weird" in decoded


# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        log_path = tmp_path / "diag.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0

        # Wait briefly for the subprocess to write — bounded by its own timeout.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Wait a touch longer for the script to finish writing
                time.sleep(0.2)
                break
            time.sleep(0.1)

        # Reap the subprocess so it doesn't show up as a zombie.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert "SIGTERM" in contents


# ---------------------------------------------------------------------------
# parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf.parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf.parse_systemd_duration_to_us("3min") == 180 * 1_000_000


# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# _systemd_timeout_stop_us
# ---------------------------------------------------------------------------

class _FakeCompletedProcess:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


class TestSystemdTimeoutStopUs:
    """#103062: a scope reporting LoadState=not-found must be skipped -- its
    TimeoutStopUSec is systemd's compiled-in default (90s), not a real value."""

    def test_not_found_user_scope_falls_through_to_system_scope(self, monkeypatch):
        """The exact bug: --user has a stale/never-existed unit name (default
        90s, LoadState=not-found); the real, correctly-configured unit lives
        in the system scope (210s) and must be the value returned."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "--user" in cmd:
                return _FakeCompletedProcess(
                    "TimeoutStopUSec=1min 30s\nLoadState=not-found\n"
                )
            return _FakeCompletedProcess(
                "TimeoutStopUSec=3min 30s\nLoadState=loaded\n"
            )

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 210_000_000
        assert len(calls) == 2  # tried --user first, then fell through to system scope

    def test_not_found_in_both_scopes_returns_none(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess(
                "TimeoutStopUSec=1min 30s\nLoadState=not-found\n"
            )

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        assert sf._systemd_timeout_stop_us("nonexistent-xyz.service") is None

    def test_loaded_user_scope_is_trusted_and_short_circuits(self, monkeypatch):
        """A genuinely loaded --user unit must still win outright (the
        common case: hermes usually runs as a --user unit) -- the fix must
        not make every lookup fall through to the system scope."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeCompletedProcess(
                "TimeoutStopUSec=1min 30s\nLoadState=loaded\n"
            )

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 90_000_000
        assert len(calls) == 1  # never had to try the system scope

    def test_missing_loadstate_property_preserves_old_behaviour(self, monkeypatch):
        """Some environment where systemctl doesn't report LoadState at all
        (unexpected, but must degrade to the pre-fix behaviour rather than
        treating the value as untrustworthy)."""
        def fake_run(cmd, **kwargs):
            return _FakeCompletedProcess("TimeoutStopUSec=90s\n")

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 90_000_000

    def test_nonzero_returncode_scope_is_skipped(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if "--user" in cmd:
                return _FakeCompletedProcess("", returncode=1)
            return _FakeCompletedProcess(
                "TimeoutStopUSec=3min 30s\nLoadState=loaded\n"
            )

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        assert sf._systemd_timeout_stop_us("hermes-gateway.service") == 210_000_000

    def test_check_systemd_timing_alignment_no_longer_false_positives(self, monkeypatch, tmp_path):
        """End-to-end repro of the reported false 'Stale systemd unit' warning:
        a healthy system-scope unit (210s) shadowed by a not-found user-scope
        leftover must no longer report a mismatch."""
        monkeypatch.setenv("INVOCATION_ID", "abc")
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("0::/system.slice/hermes-gateway.service\n")
        monkeypatch.setattr(sf, "open", lambda *a, **k: open(cgroup_file, *a[1:], **k), raising=False)

        def fake_run(cmd, **kwargs):
            if "--user" in cmd:
                return _FakeCompletedProcess(
                    "TimeoutStopUSec=1min 30s\nLoadState=not-found\n"
                )
            return _FakeCompletedProcess(
                "TimeoutStopUSec=3min 30s\nLoadState=loaded\n"
            )

        monkeypatch.setattr(sf.subprocess, "run", fake_run)
        result = sf.check_systemd_timing_alignment(180.0, cron_drain_timeout=30.0)
        assert result is not None
        assert result["timeout_stop_sec"] == 210.0
        assert result["mismatch"] is False
