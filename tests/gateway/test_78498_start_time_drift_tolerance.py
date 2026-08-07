"""Regression tests for #78498.

The gateway liveness PID-reuse guard compared start-time fingerprints with
strict equality.  On macOS/Windows the fingerprint comes from
``psutil.create_time()`` quantized to centiseconds, and that value can shift by
~1s for the *same* live process across a psutil upgrade in the venv or an NTP
step adjustment.  Every liveness probe then rejected the live PID as recycled
and the dashboard reported a healthy gateway as ``gateway_running: false``.

The guard now tolerates a small (~2s) drift while still catching a genuine PID
reuse (fingerprints far apart), with the cmdline/profile identity checks gating
on top.
"""

from gateway import status


class TestStartTimesMatch:
    def test_tolerates_drift_but_catches_reuse(self):
        """Matches within the drift tolerance and on a ``None`` on either side
        (unknown fingerprint → never a positive mismatch), but rejects a gap
        beyond the tolerance (a recycled PID)."""
        tol = status._START_TIME_DRIFT_TOLERANCE
        assert status._start_times_match(1000, 1000) is True
        assert status._start_times_match(1000, 1000 - tol) is True
        assert status._start_times_match(1000, 1000 + tol) is True
        assert status._start_times_match(1000, 1000 - (tol + 1)) is False
        assert status._start_times_match(1000, 1000 + (tol + 1)) is False
        # Unknown on either side must not be treated as a mismatch.
        assert status._start_times_match(None, 1000) is True
        assert status._start_times_match(1000, None) is True
        assert status._start_times_match(None, None) is True


class TestRuntimeStatusRunningPidDrift:
    def _payload(self, recorded: int) -> dict:
        return {
            "pid": 4242,
            "gateway_state": "running",
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
            "start_time": recorded,
        }

    def test_tolerates_subsecond_start_time_drift(self, monkeypatch):
        """A sub-2s drift for the SAME live process must still be reported
        running.  ``psutil.create_time()`` shifted ~1s (100 centiseconds) after
        a psutil upgrade / NTP step; strict equality wrongly rejected the PID.
        """
        recorded = 178526682901  # observed at spawn (issue #78498)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "hermes gateway run"
        )
        # Same live process reads back exactly 1.00s (100 centiseconds) lower.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: recorded - 100)

        assert status.get_runtime_status_running_pid(self._payload(recorded)) == 4242

    def test_still_rejects_large_start_time_gap(self, monkeypatch):
        """A start-time gap beyond the drift tolerance is a genuinely different
        process on a recycled PID and must remain rejected."""
        recorded = 178526682901
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "hermes gateway run"
        )
        # 5s apart → a different process; the guard must still reject it.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: recorded - 500)

        assert status.get_runtime_status_running_pid(self._payload(recorded)) is None
