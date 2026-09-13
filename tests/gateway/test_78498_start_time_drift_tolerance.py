"""Regression tests for #78498.

The gateway liveness PID-reuse guard compared start-time fingerprints with
strict equality.  On macOS/Windows the fingerprint comes from
``psutil.create_time()`` quantized to centiseconds, and that value can shift by
~1s for the *same* live process across a psutil upgrade in the venv or an NTP
step adjustment.  Every liveness probe then rejected the live PID as recycled
and the dashboard reported a healthy gateway as ``gateway_running: false``.

The unified guard (``_start_times_conflict``) now tolerates a small (~2s) drift
while still catching a genuine PID reuse (fingerprints far apart), with the
cmdline/profile identity checks gating on top.  The deliberate strict-identity
force-kill path (``_start_times_agree``, 1ms) is intentionally left untouched.
"""

from gateway import status


class TestStartTimesConflict:
    def test_tolerates_drift_but_catches_reuse(self):
        """No conflict within the drift tolerance or when a fingerprint is
        unknown (``None`` -> never a positive mismatch); a gap beyond the
        tolerance (a recycled PID) is still a conflict."""
        tol = status._START_TIME_DRIFT_TOLERANCE
        assert status._start_times_conflict(1000, 1000) is False
        assert status._start_times_conflict(1000, 1000 - tol) is False
        assert status._start_times_conflict(1000, 1000 + tol) is False
        assert status._start_times_conflict(1000, 1000 - (tol + 1)) is True
        assert status._start_times_conflict(1000, 1000 + (tol + 1)) is True
        # Unknown on either side must not be treated as a conflict.
        assert status._start_times_conflict(None, 1000) is False
        assert status._start_times_conflict(1000, None) is False
        assert status._start_times_conflict(None, None) is False


class TestLivePidFromRecordDrift:
    """``_live_pid_from_record`` is the chokepoint feeding both liveness read
    paths (``runtime_status_pid_is_live`` and ``get_running_pid``)."""

    def _record(self, recorded: int) -> dict:
        return {"pid": 4242, "start_time": recorded}

    def test_tolerates_subsecond_start_time_drift(self, monkeypatch):
        """A sub-2s drift for the SAME live process must still resolve the PID.
        ``psutil.create_time()`` shifted ~1s (100 centiseconds) after a psutil
        upgrade / NTP step; strict equality wrongly rejected the PID."""
        recorded = 178526682901  # observed at spawn (issue #78498)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        # Same live process reads back exactly 1.00s (100 centiseconds) lower.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: recorded - 100)

        assert status._live_pid_from_record(self._record(recorded)) == 4242
        assert status.runtime_status_pid_is_live(self._record(recorded)) is True

    def test_still_rejects_large_start_time_gap(self, monkeypatch):
        """A start-time gap beyond the drift tolerance is a genuinely different
        process on a recycled PID and must remain rejected."""
        recorded = 178526682901
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        # 5s apart -> a different process; the guard must still reject it.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: recorded - 500)

        assert status._live_pid_from_record(self._record(recorded)) is None
        assert status.runtime_status_pid_is_live(self._record(recorded)) is False
