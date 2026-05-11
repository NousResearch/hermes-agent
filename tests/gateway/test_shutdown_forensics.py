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
    def test_known_signals_resolve_to_names(self):
        assert sf._signal_name(signal.SIGTERM) == "SIGTERM"
        assert sf._signal_name(signal.SIGINT) == "SIGINT"

    def test_unknown_int_returns_signal_num_token(self):
        # Pick an integer extremely unlikely to ever be a real signal alias
        assert sf._signal_name(9999) == "signal#9999"

    def test_none_returns_unknown(self):
        assert sf._signal_name(None) == "UNKNOWN"

    def test_non_integer_falls_back_to_str(self):
        assert sf._signal_name("SIGTERM") == "SIGTERM"


# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:
    def test_includes_self_pid_and_signal(self):
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["pid"] == os.getpid()
        assert ctx["signal"] == "SIGTERM"
        assert ctx["signal_num"] == int(signal.SIGTERM)

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

    @pytest.mark.skipif(sys.platform == "win32", reason="Linux /proc not present")
    def test_includes_parent_summary_on_linux(self):
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "parent" in ctx
        assert ctx["parent"]["pid"] == os.getppid()

    def test_under_systemd_flag_uses_invocation_id(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc123")
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["under_systemd"] is True
        assert ctx["systemd_invocation_id"] == "abc123"

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

    def test_completes_quickly(self):
        """Snapshot must NOT block — it runs inside the asyncio signal handler."""
        start = time.monotonic()
        sf.snapshot_shutdown_context(signal.SIGTERM)
        elapsed = time.monotonic() - start
        # Generous bound; the function should be sub-millisecond in practice.
        assert elapsed < 0.5, f"snapshot took {elapsed:.3f}s — too slow"

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

    def test_detects_takeover_marker_for_other(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            '{"target_pid": 1, "replacer_pid": 99999}', encoding="utf-8"
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert ctx["takeover_marker_for_self"] is False

    def test_detects_planned_stop_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-planned-stop.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}}}', encoding="utf-8"
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "planned_stop_marker" in ctx


# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_format_context_for_log_includes_signal_and_parent(self):
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        line = sf.format_context_for_log(ctx)
        assert "signal=SIGTERM" in line
        assert "parent_pid=" in line
        assert "parent_cmdline=" in line

    def test_context_as_json_round_trips(self):
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        payload = sf.context_as_json(ctx)
        decoded = json.loads(payload)
        assert decoded["pid"] == os.getpid()
        assert decoded["signal"] == "SIGTERM"

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
                time.sleep(0.5)
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

    def test_returns_none_on_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sf, "sys", type("M", (), {"platform": "win32"})())
        result = sf.spawn_async_diagnostic(
            tmp_path / "diag.log", "SIGTERM", timeout_seconds=1.0
        )
        assert result is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_handles_unwritable_log_path_gracefully(self, tmp_path):
        # Point at a nonexistent parent that we can't create
        log_path = Path("/proc/cant-write-here/diag.log")
        result = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=1.0)
        assert result is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only diagnostic")
    def test_does_not_block_caller(self, tmp_path):
        """The spawn must return immediately even if ``ps`` takes seconds."""
        log_path = tmp_path / "diag.log"
        start = time.monotonic()
        sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=10.0)
        elapsed = time.monotonic() - start
        # Spawning bash in detached mode takes a few ms; anything under 1s
        # is plenty of headroom and proves we're not waiting on it.
        assert elapsed < 1.0, f"spawn blocked for {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# _parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf._parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf._parse_systemd_duration_to_us("3min") == 180 * 1_000_000

    def test_combined_min_sec(self):
        assert sf._parse_systemd_duration_to_us("1min 30s") == 90 * 1_000_000

    def test_hours(self):
        assert sf._parse_systemd_duration_to_us("1h") == 3600 * 1_000_000

    def test_milliseconds(self):
        assert sf._parse_systemd_duration_to_us("500ms") == 500_000

    def test_empty_returns_none(self):
        assert sf._parse_systemd_duration_to_us("") is None

    def test_unknown_unit_returns_none(self):
        assert sf._parse_systemd_duration_to_us("90weeks") is None


# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:
    def test_returns_none_when_not_under_systemd(self, monkeypatch):
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        result = sf.check_systemd_timing_alignment(180.0)
        assert result is None

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)
