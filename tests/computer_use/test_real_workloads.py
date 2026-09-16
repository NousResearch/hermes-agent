"""Tests for the real computer-use workload runner (RFC #112639, P0 slice).

The live X path needs a display, so these tests drive the task functions with a
fake backend double at the capture/input boundary and assert the measurement
contract: spans recorded per phase in order, trace JSON schema, and success
detection. Real runs happen on Xvfb (see the PR body for measured numbers).
"""

import json

from tools.computer_use import real_workloads
from tools.computer_use import synth_workloads


class _FakeImage:
    size = (1280, 800)

    def load(self):
        return {}


class _FakeBackend:
    """Double for LiveBackend: deterministic, no X server needed."""

    def __init__(self, rec):
        self.rec = rec
        self.clicks = []
        self.typed = []

    def capture(self):
        return self.rec.span("capture", lambda: _FakeImage())

    def persist(self, img, path):
        self.rec.span("capture_persist", lambda: None)

    def locate_color(self, img, rgb, tol=40):
        return self.rec.span("element_processing", lambda: (640, 400),
                             attrs={"method": "color-blob"})

    def ax_rect(self, widget):
        return self.rec.span("element_processing", lambda: (100, 100, 200, 40),
                             attrs={"method": "ax-geom"})

    def click(self, x, y):
        self.clicks.append((x, y))
        self.rec.span("input", lambda: None, attrs={"x": str(x), "y": str(y)})

    def type_text(self, text):
        self.typed.append(text)
        self.rec.span("input", lambda: None)

    def backend_call(self, name, fn):
        return self.rec.span("backend_call", fn, attrs={"action": name})

    def close(self):
        pass


class TestSpanContract:
    def test_recorder_orders_spans_and_totals(self):
        rec = real_workloads.SpanRecorder("t1", session_id="s1")
        be = _FakeBackend(rec)
        be.capture()
        be.click(10, 20)
        be.type_text("hi")
        phases = [s.phase for s in rec.spans]
        assert phases == ["capture", "input", "input"]
        totals = rec.phase_ms()
        assert set(totals) == {"capture", "input"}
        assert all(v >= 0 for v in totals.values())

    def test_span_ids_correlate(self):
        rec = real_workloads.SpanRecorder("task-9", session_id="sess-9")
        _FakeBackend(rec).capture()
        s = rec.spans[0]
        assert s.task_id == "task-9"
        assert s.session_id == "sess-9"
        assert s.tool_call_id.startswith("tc-")
        assert s.duration_ms >= 0

    def test_span_shape_matches_report_vocabulary(self):
        # The critical-path report consumes these spans; the dataclass shape
        # must stay identical to the report's Span.
        assert set(real_workloads.PHASES) <= set(synth_workloads.PHASES)
        s = synth_workloads.Span(task_id="t", phase="capture",
                                 start_ms=0.0, end_ms=1.0)
        assert s.duration_ms == 1.0


class TestTraceSchema:
    def _trace(self):
        rec = real_workloads.SpanRecorder("real-form-fill")
        be = _FakeBackend(rec)
        be.capture()
        be.click(1, 2)
        return real_workloads.Trace("real-form-fill", "form_fill", True,
                                    12.5, rec.spans, rec.phase_ms())

    def test_json_round_trip(self):
        d = json.loads(self._trace().to_json())
        assert d["task_id"] == "real-form-fill"
        assert d["task"] == "form_fill"
        assert d["success"] is True
        assert d["wall_ms"] == 12.5
        assert len(d["spans"]) == 2
        assert d["spans"][0]["phase"] == "capture"
        assert "capture" in d["phase_ms"]

    def test_phase_totals_cover_all_spans(self):
        t = self._trace()
        measured = sum(t.phase_ms.values())
        span_sum = sum(s.duration_ms for s in t.spans)
        assert measured == span_sum
        assert measured >= 0


class TestScriptedPolicy:
    def test_form_fill_plan_shape(self):
        obs = {"name": (1, 1), "email": (2, 2), "submit": (3, 3)}
        plan = real_workloads._scripted_policy("form_fill", obs)
        kinds = [k for k, _ in plan]
        assert kinds == ["click", "type", "click", "type", "click"]
        assert plan[0][1] == (1, 1)
        assert plan[-1][1] == (3, 3)

    def test_dialog_dismiss_single_click(self):
        plan = real_workloads._scripted_policy("dialog_dismiss", {"ok": (9, 9)})
        assert plan == [("click", (9, 9))]

    def test_task_registry_covers_three_tasks(self):
        assert set(real_workloads.TASKS) == {
            "form_fill", "dialog_dismiss", "text_entry"}
