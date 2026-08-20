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
# _parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf._parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf._parse_systemd_duration_to_us("3min") == 180 * 1_000_000


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


class TestLabelMarkerDrivenShutdown:
    """#61596: Windows planned stops (marker watcher → signal=None) must not
    be reported as signal=UNKNOWN — the marker already proved intent."""

    def _ctx(self):
        from gateway.shutdown_forensics import snapshot_shutdown_context
        return snapshot_shutdown_context(None)

    def test_planned_stop_with_no_signal_is_labeled(self):
        from gateway.shutdown_forensics import label_marker_driven_shutdown

        ctx = self._ctx()
        assert ctx["signal"] == "UNKNOWN"
        label_marker_driven_shutdown(
            ctx, planned_takeover=False, planned_stop=True, received_signal=None
        )
        assert ctx["signal"] == "PLANNED_STOP"

    def test_planned_takeover_wins_over_planned_stop(self):
        from gateway.shutdown_forensics import label_marker_driven_shutdown

        ctx = self._ctx()
        label_marker_driven_shutdown(
            ctx, planned_takeover=True, planned_stop=True, received_signal=None
        )
        assert ctx["signal"] == "PLANNED_TAKEOVER"

    def test_real_signal_keeps_its_true_name(self):
        """POSIX behavior unchanged: a real SIGTERM that happens to carry a
        planned-stop marker keeps the actual signal name."""
        import signal as _signal

        from gateway.shutdown_forensics import (
            label_marker_driven_shutdown,
            snapshot_shutdown_context,
        )

        ctx = snapshot_shutdown_context(_signal.SIGTERM)
        before = ctx["signal"]
        label_marker_driven_shutdown(
            ctx, planned_takeover=False, planned_stop=True,
            received_signal=_signal.SIGTERM,
        )
        assert ctx["signal"] == before
        assert "TERM" in ctx["signal"]

    def test_unplanned_none_signal_stays_unknown(self):
        """A genuinely unexplained handler invocation keeps UNKNOWN."""
        from gateway.shutdown_forensics import label_marker_driven_shutdown

        ctx = self._ctx()
        label_marker_driven_shutdown(
            ctx, planned_takeover=False, planned_stop=False, received_signal=None
        )
        assert ctx["signal"] == "UNKNOWN"

    def test_never_raises_on_non_dict_ctx(self):
        from gateway.shutdown_forensics import label_marker_driven_shutdown

        label_marker_driven_shutdown(
            None, planned_takeover=False, planned_stop=True, received_signal=None
        )
