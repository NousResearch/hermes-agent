"""Phase 0A exit-gate tests for #112734: phase spans are recorded in-process and the
critical-path report explains the call from them.

Covers: spans recorded for a capture call with >90% of wall time attributed,
revision invalidations recorded with their reason, the recorder staying a no-op
when instrumentation is off, and spans staying content-free.
"""
import json
from types import SimpleNamespace

import pytest

from tools.computer_use import phase_spans as spans_mod
from tools.computer_use import tool as tool_mod
from tools.computer_use.backend import CaptureResult, UIElement


@pytest.fixture
def clean_tool_state():
    tool_mod.reset_backend_for_tests()
    yield
    tool_mod.reset_backend_for_tests()


@pytest.fixture
def instrumented(monkeypatch):
    monkeypatch.setattr("agent.relay_runtime.relay_instrumentation_enabled", lambda: True)


def _stub_backend(**kw):
    elements = [UIElement(index=1, role="button", label="SecretLabel-unique-xyz",
                          bounds=(10, 10, 80, 30), app="kate")]
    return SimpleNamespace(
        _last_app="kate", _last_target={"pid": 1, "window_id": "w1"},
        capture=lambda **ckw: CaptureResult(mode="ax", width=800, height=600,
                                            png_b64="ZmFrZXBuZy1ieXRlcy11bmlxdWU=",
                                            png_bytes_len=24,
                                            elements=elements, **kw),
        **kw)


def _span_map(rec):
    return {s.phase: s for s in rec.spans}


def test_capture_call_records_phases_and_explains_itself(clean_tool_state, instrumented):
    with spans_mod.record_call("capture", session_id="s1", task_id="t1") as rec:
        out = json.loads(tool_mod._guarded_capture(_stub_backend(), "s1", mode="ax"))
    assert out["mode"] == "ax"  # behavior unchanged by instrumentation
    phases = _span_map(rec)
    for expected in ("admission", "capture", "validate", "element_processing", "response_shape"):
        assert expected in phases, f"missing phase span: {expected}"
    report = rec.task_report("t1")
    assert report.explained_pct > 90  # Phase 0A exit gate: >90% of wall time named


def test_revision_invalidation_is_recorded_with_reason(clean_tool_state, instrumented, monkeypatch):
    calls = {"n": 0}
    real = tool_mod._current_revision_facts

    def flipping(sid):
        calls["n"] += 1
        facts = dict(real(sid))
        facts["backend_generation"] = 1 if calls["n"] == 1 else 2
        return facts

    monkeypatch.setattr(tool_mod, "_current_revision_facts", flipping)
    with spans_mod.record_call("capture", session_id="s1", task_id="t1") as rec:
        out = json.loads(tool_mod._guarded_capture(_stub_backend(), "s1", mode="ax"))
    assert out["code"] == "revision_invalidated"
    validate_span = _span_map(rec)["validate"]
    assert validate_span.attrs.get("revision_invalidated") is True
    assert validate_span.attrs.get("invalidation_reason") == "backend_replaced"


def test_recorder_is_noop_when_instrumentation_off(clean_tool_state, monkeypatch):
    monkeypatch.setattr("agent.relay_runtime.relay_instrumentation_enabled", lambda: False)
    with spans_mod.record_call("capture", session_id="s1", task_id="t1") as rec:
        out = json.loads(tool_mod._guarded_capture(_stub_backend(), "s1", mode="ax"))
    assert out["mode"] == "ax"  # tool works the same with recording off
    assert rec.spans == []


def test_spans_are_content_free(clean_tool_state, instrumented):
    with spans_mod.record_call("capture", session_id="s1", task_id="t1") as rec:
        tool_mod._guarded_capture(_stub_backend(), "s1", mode="ax")
    for span in rec.spans:
        for value in span.attrs.values():
            text = str(value)
            assert "SecretLabel-unique-xyz" not in text  # element text never enters telemetry
            assert "ZmFrZXBuZy1ieXRlcy11bmlxdWU=" not in text  # screenshot bytes never enter telemetry
