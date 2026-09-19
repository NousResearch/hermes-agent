"""Per-step timing records for guarded desktop runs — RFC #112639.

Behavior contracts: every confirmed mid-run step records one timing entry;
durations are non-negative and start offsets non-decreasing; timings exist on
both clean completion and early stop; records carry metadata only (action
names, never args, screenshots, or secrets).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.guarded_run_executor import guarded_run_readiness_stop
from agent.guarded_run_timing import CONFIRM_STATUSES, RunTimings
from tools.computer_use.readiness import ReadinessResult


def _tc(action=None, tool="computer_use", call_id=None, **args):
    payload = dict(args)
    if action is not None:
        payload["action"] = action
    return SimpleNamespace(
        id=call_id or f"call-{tool}-{action or 'x'}",
        function=SimpleNamespace(name=tool, arguments=json.dumps(payload)),
    )


def _run_calls():
    return [
        _tc("click", coordinate=[260, 116], call_id="a"),
        _tc("type", text="hello", call_id="b"),
        _tc("click", coordinate=[260, 168], call_id="c"),
    ]


class _Backend:
    def __init__(self, status="satisfied", detail="driver status"):
        self._active_pid = 1234
        self._active_window_id = 5678
        self._last_target = {"pid": 1234, "window_id": 5678}
        self._result = ReadinessResult(status=status, detail=detail, duration_ms=9.1)

    def verify_readiness(self, **kwargs):
        return self._result


def _confirm_two_clean_steps():
    timings = RunTimings()
    calls = _run_calls()
    backend = _Backend(status="satisfied")
    assert guarded_run_readiness_stop(calls, 0, backend, timings=timings) is None
    assert guarded_run_readiness_stop(calls, 1, backend, timings=timings) is None
    return timings


class TestTimingRecords:
    def test_one_record_per_executed_step(self):
        timings = _confirm_two_clean_steps()
        assert [s.step_index for s in timings.steps] == [0, 1]
        assert [s.action for s in timings.steps] == ["click", "type"]

    def test_durations_non_negative_and_offsets_monotonic(self):
        timings = _confirm_two_clean_steps()
        offsets = [s.started_offset_ms for s in timings.steps]
        assert all(s.confirm_duration_ms is not None and s.confirm_duration_ms >= 0
                   for s in timings.steps)
        assert offsets == sorted(offsets)
        assert all(o >= 0 for o in offsets)

    def test_timings_present_on_early_stop(self):
        timings = RunTimings()
        calls = _run_calls()
        backend = _Backend(status="unsatisfied", detail="driver status: unsatisfied")
        stopped = guarded_run_readiness_stop(calls, 0, backend, timings=timings,
                                             tool_duration_ms=120.0)
        assert stopped is not None
        assert len(timings.steps) == 1
        step = timings.steps[0]
        assert step.confirm_status == "unsatisfied"
        assert step.verdict == "stopped"
        assert step.tool_duration_ms == 120.0

    def test_no_check_records_fail_open_status(self):
        timings = RunTimings()
        calls = _run_calls()
        backend = _Backend()
        backend._active_pid = backend._active_window_id = None
        backend._last_target = {}
        assert guarded_run_readiness_stop(calls, 0, backend, timings=timings) is None
        assert len(timings.steps) == 1
        assert timings.steps[0].confirm_status == "no_check"
        assert timings.steps[0].verdict == "continue"


class TestEvidenceShape:
    def test_evidence_carries_metadata_only(self):
        timings = _confirm_two_clean_steps()
        evidence = timings.as_evidence()
        assert evidence["step_count"] == 2
        assert evidence["capture_duration_ms"] is None
        assert evidence["capture_mode"] is None
        blob = json.dumps(evidence)
        assert "hello" not in blob  # typed text never lands in records
        assert "260" not in blob  # coordinates never land in records
        for step in evidence["steps"]:
            assert step["confirm_status"] in CONFIRM_STATUSES

    def test_capture_hook_is_opt_in_not_fabricated(self):
        timings = RunTimings()
        assert timings.as_evidence()["capture_duration_ms"] is None
        timings.record_capture(95.0, "vision")
        evidence = timings.as_evidence()
        assert evidence["capture_duration_ms"] == 95.0
        assert evidence["capture_mode"] == "vision"

    def test_summary_names_actions_not_args(self):
        timings = _confirm_two_clean_steps()
        line = timings.summary()
        assert "click" in line and "type" in line
        assert "hello" not in line
