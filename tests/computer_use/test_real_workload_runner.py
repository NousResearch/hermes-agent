"""Tests for the real-workload measurement runner (#112639 P0).

The headline real runs (Xvfb + cua-driver + LibreOffice) are executed by the
maintainer and reported in the PR body; this file keeps the deterministic CI
regression checks: the recorder math, the long-stall fixture report, the
hardware sampler's never-raise contract, and the timing lock.
"""

import json
import threading
import time

import pytest

from tools.computer_use.real_workload_runner import (
    Span,
    SpanRecorder,
    _RecorderRef,
    _TimedRLock,
    _wrap_backend_class,
    sample_hardware,
)


def _recorder_with(spans):
    rec = SpanRecorder()
    rec.spans.extend(spans)
    return rec


class TestSpanRecorder:
    def test_span_records_duration_and_dims(self):
        rec = SpanRecorder()
        with rec.span("capture", mode="ax"):
            time.sleep(0.01)
        assert len(rec.spans) == 1
        span = rec.spans[0]
        assert span.name == "capture"
        assert span.duration_ms >= 5.0
        assert span.dims == {"mode": "ax"}

    def test_totals_sum_same_phase(self):
        rec = _recorder_with([Span("input", 100.0), Span("input", 50.0), Span("capture", 25.0)])
        assert rec.totals() == {"input": 150.0, "capture": 25.0}

    def test_cold_starts_counts_only_true_dims(self):
        rec = _recorder_with([
            Span("backend_start", 10.0, {"cold_start": True}),
            Span("backend_start", 5.0, {"cold_start": False}),
            Span("capture", 5.0),
        ])
        assert rec.cold_starts() == 1

    def test_queue_time_sums_dispatch_lock_wait(self):
        rec = _recorder_with([
            Span("dispatch_lock_wait", 3.0, {"queue_time_ms": 3.0}),
            Span("dispatch_lock_wait", 7.0, {"queue_time_ms": 7.0}),
            Span("input", 5.0),
        ])
        assert rec.queue_time_ms() == pytest.approx(10.0)

    def test_report_explains_long_stall_fixture(self):
        """CI regression check: a known trace with a 5s stall must be explained
        by the report and name the stalled phase as critical."""
        rec = _recorder_with([
            Span("backend_start", 200.0, {"cold_start": True}),
            Span("dispatch_lock_wait", 5.0, {"queue_time_ms": 5.0}),
            Span("capture", 300.0),
            Span("input", 100.0),
            Span("backend_call", 5000.0),  # the stall: a hung driver round-trip
            Span("response_shape", 100.0),
        ])
        report = rec.report(wall_ms=6000.0)
        assert report["explained_pct"] >= 90.0
        assert report["critical_phase"] == "backend_call"
        assert report["cold_starts"] == 1
        assert report["dispatch_queue_ms"] == pytest.approx(5.0)
        assert report["phases"]["backend_call"] == 5000.0

    def test_report_empty_is_zeroed(self):
        report = SpanRecorder().report(wall_ms=100.0)
        assert report["explained_pct"] == 0.0
        assert report["critical_phase"] == "none"


class TestHardwareSampler:
    def test_never_raises_and_returns_dict(self):
        snap = sample_hardware()
        assert isinstance(snap, dict)

    def test_values_are_sane_when_present(self):
        snap = sample_hardware()
        if "mem_total_mb" in snap:
            assert snap["mem_total_mb"] > 0
        if "mem_used_mb" in snap:
            assert 0 <= snap["mem_used_mb"] <= snap["mem_total_mb"]
        if "vram_used_mb" in snap:
            assert 0 <= snap["vram_used_mb"] <= snap["vram_total_mb"]

    def test_gpu_probe_is_cached(self):
        first = sample_hardware()
        second = sample_hardware()
        assert first.get("gpu_name") == second.get("gpu_name")


class TestTimedRLock:
    def test_records_queue_wait_as_dispatch_lock_wait(self):
        ref = _RecorderRef()
        lock = _TimedRLock(ref)
        with lock:
            pass
        waits = [s for s in ref.recorder.spans if s.name == "dispatch_lock_wait"]
        assert len(waits) == 1
        assert waits[0].dims["queue_time_ms"] >= 0.0

    def test_records_contended_wait(self):
        ref = _RecorderRef()
        lock = _TimedRLock(ref)
        lock.acquire()
        done = threading.Event()

        def contender():
            with lock:
                done.set()

        t = threading.Thread(target=contender)
        t.start()
        time.sleep(0.05)
        lock.release()
        t.join(timeout=5.0)
        assert done.is_set()
        waits = [s for s in ref.recorder.spans if s.name == "dispatch_lock_wait"]
        assert len(waits) == 2
        assert max(w.dims["queue_time_ms"] for w in waits) >= 20.0


class TestWrapBackendClass:
    def test_wrap_is_idempotent(self):
        ref = _RecorderRef()
        first = _wrap_backend_class(ref)
        second = _wrap_backend_class(ref)
        assert second == []
        assert set(first) >= {"start", "capture"}

    def test_wrapped_start_records_cold_start_dim(self, monkeypatch):
        from tools.computer_use.cua_backend import CuaDriverBackend

        calls = []
        monkeypatch.setattr(CuaDriverBackend, "start",
                            lambda self: calls.append("start"))
        ref = _RecorderRef()
        wrapped = _wrap_backend_class(ref)
        assert "start" in wrapped

        backend = object.__new__(CuaDriverBackend)
        backend.start()
        backend.start()
        assert calls == ["start", "start"]
        starts = [s for s in ref.recorder.spans if s.name == "backend_start"]
        assert len(starts) == 2
        assert starts[0].dims["cold_start"] is True
        assert starts[1].dims["cold_start"] is False


class TestTaskContracts:
    def test_task_registry_covers_three_real_tasks(self):
        from tools.computer_use.real_workload_runner import TASKS
        assert set(TASKS) == {"perceive", "act", "dialog"}

    def test_preflight_rejects_missing_display(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        from tools.computer_use import real_workload_runner as runner
        with pytest.raises(SystemExit, match="No DISPLAY"):
            runner._preflight()
