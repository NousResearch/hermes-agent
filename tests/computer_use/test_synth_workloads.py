"""Tests for the replayable synthetic computer-use workloads (#112639 P0)."""

import time

from tools.computer_use.synth_workloads import (
    PHASES,
    dialog_dismiss_workload,
    form_fill_workload,
    long_stall_workload,
    run_workload,
)


def _span_key(s):
    return (s.phase, round(s.start_ms, 6), round(s.end_ms, 6),
            s.tool_call_id, tuple(sorted(s.attrs.items())))


def test_same_seed_identical_trace():
    # Same seed -> identical action sequence and timing trace.
    a = run_workload(long_stall_workload, seed=7)
    b = run_workload(long_stall_workload, seed=7)
    assert a.actions == b.actions
    assert [_span_key(s) for s in a.spans] == [_span_key(s) for s in b.spans]
    assert a.stall_windows == b.stall_windows
    assert a.wall_ms == b.wall_ms


def test_different_seed_different_trace():
    # Different seed -> at least one span duration differs (latencies jitter).
    a = run_workload(form_fill_workload, seed=1)
    b = run_workload(form_fill_workload, seed=2)
    assert a.actions == b.actions  # scripted sequence is seed-independent
    assert [_span_key(s) for s in a.spans] != [_span_key(s) for s in b.spans]


def test_long_stall_fixture_window():
    # The Blender-like fixture exposes one deterministic multi-second stall.
    r = run_workload(long_stall_workload, seed=42)
    assert len(r.stall_windows) == 1
    start, end = r.stall_windows[0]
    assert end - start >= 4000.0  # deterministic long stall, virtual ms
    assert "start_render" in r.actions
    assert "render_complete" in r.actions
    # Stall dominates the workload: this is what P3 will try to fill.
    assert r.avoidable_stall_ms == end - start
    assert r.avoidable_stall_ms > 0.5 * r.wall_ms


def test_form_fill_end_to_end():
    r = run_workload(form_fill_workload, seed=3)
    assert r.actions == ["capture:form", "type:name", "type:email",
                        "type:notes", "capture:form_filled", "click:submit",
                        "capture:done"]
    assert r.stall_windows == []
    assert r.avoidable_stall_ms == 0.0


def test_dialog_dismiss_end_to_end():
    r = run_workload(dialog_dismiss_workload, seed=5)
    assert r.actions == ["capture:modal", "click:dismiss", "capture:main"]


def test_spans_explain_wall_time():
    # P0 exit gate shape: named spans explain >90% of benchmark wall time.
    for workload in (form_fill_workload, dialog_dismiss_workload,
                     long_stall_workload):
        r = run_workload(workload, seed=11)
        explained = sum(s.duration_ms for s in r.spans)
        assert explained >= 0.9 * r.wall_ms
        assert all(s.phase in PHASES for s in r.spans)
        assert all(s.end_ms >= s.start_ms for s in r.spans)


def test_virtual_clock_no_real_sleep():
    # Replays run on the virtual clock: even the long stall finishes fast.
    t0 = time.monotonic()
    r = run_workload(long_stall_workload, seed=9)
    elapsed = time.monotonic() - t0
    assert r.wall_ms >= 4000.0  # virtual stall happened
    assert elapsed < 2.0  # but no real sleeping


def test_span_correlation_ids():
    r = run_workload(form_fill_workload, seed=13, task_id="task-xyz")
    assert all(s.task_id == "task-xyz" for s in r.spans)
    assert all(s.session_id == "session-13" for s in r.spans)
    assert all(s.tool_call_id for s in r.spans)
