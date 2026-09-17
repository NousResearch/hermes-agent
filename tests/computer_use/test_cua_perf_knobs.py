"""Behavior contracts for computer_use latency knobs and the P0 critical-path report."""

from unittest.mock import patch

from tools.computer_use import critical_path as cp
from tools.computer_use import cua_backend
from tools.computer_use import tool as cu_tool


def test_max_image_dimension_default():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert cua_backend._computer_use_max_image_dimension() == 1456




def test_capture_after_mode_default_som():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert cu_tool._capture_after_mode() == "som"






def test_aux_vision_route_caches_per_provider_model(monkeypatch):
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    calls = {"n": 0}

    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: "openai"
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: "gpt-test"
    )

    def fake_load():
        calls["n"] += 1
        return {"auxiliary": {"vision": {}}}

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
    monkeypatch.setattr(
        "tools.computer_use.vision_routing.should_route_capture_to_aux_vision",
        lambda *a, **k: True,
    )

    assert cu_tool._should_route_through_aux_vision() is True
    assert cu_tool._should_route_through_aux_vision() is True
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# P0 slice: deterministic critical-path fixtures (RFC #112639).
#
# Each fixture is one task's phase spans with fixed timestamps, correlated the
# way the lifecycle hooks carry them (session_id / task_id / tool_call_id).
# They stand in for the spans the recording half (#112778) emits; the report
# must reconstruct the right critical path from them and nothing else.
# ---------------------------------------------------------------------------

def _span(task_id, phase, start, end, tool_call_id="", **attrs):
    return cp.Span(
        task_id=task_id, phase=phase, start_ms=start, end_ms=end,
        tool_call_id=tool_call_id, session_id="sess-1", attrs=attrs,
    )


def fixture_spans_ax_only(task_id="t-ax"):
    """AX walk, no screenshot: element_processing dominates the tool turn."""
    return [
        _span(task_id, "model", 0, 1200),
        _span(task_id, "admission", 1225, 1230, "tc-1"),
        _span(task_id, "backend_resolve", 1230, 1260, "tc-1", backend="cua"),
        _span(task_id, "dispatch_lock_wait", 1260, 1275, "tc-1"),
        _span(task_id, "capture", 1275, 1575, "tc-1", capture_mode="ax"),
        _span(task_id, "element_processing", 1575, 1725, "tc-1"),
        _span(task_id, "response_shape", 1725, 1735, "tc-1"),
        _span(task_id, "model", 1760, 2960),
        _span(task_id, "admission", 2960, 2965, "tc-2"),
        _span(task_id, "input", 2965, 3015, "tc-2", action="click"),
        _span(task_id, "response_shape", 3015, 3020, "tc-2"),
    ]


def fixture_spans_som(task_id="t-som"):
    """SOM flow: screenshot captured, persisted, then element-processed."""
    return [
        _span(task_id, "model", 0, 1500),
        _span(task_id, "admission", 1520, 1526, "tc-1"),
        _span(task_id, "backend_resolve", 1526, 1540, "tc-1", backend="cua"),
        _span(task_id, "backend_start", 1540, 1600, "tc-1"),
        _span(task_id, "capture", 1600, 2100, "tc-1", capture_mode="som"),
        _span(task_id, "capture_persist", 2100, 2130, "tc-1"),
        _span(task_id, "element_processing", 2130, 2580, "tc-1"),
        _span(task_id, "response_shape", 2580, 2590, "tc-1"),
        _span(task_id, "model", 2610, 4110),
        _span(task_id, "admission", 4110, 4115, "tc-2"),
        _span(task_id, "input", 4115, 4165, "tc-2", action="click"),
        _span(task_id, "response_shape", 4165, 4170, "tc-2"),
    ]


def fixture_spans_vision(task_id="t-vision"):
    """Pure vision flow: approval_wait is the avoidable idle on the path."""
    return [
        _span(task_id, "model", 0, 2400),
        _span(task_id, "approval_wait", 2420, 3420, "tc-1"),
        _span(task_id, "admission", 3420, 3425, "tc-1"),
        _span(task_id, "backend_resolve", 3425, 3440, "tc-1", backend="cua"),
        _span(task_id, "capture", 3440, 4240, "tc-1", capture_mode="vision"),
        _span(task_id, "response_shape", 4240, 4250, "tc-1"),
    ]


def fixture_spans_aux_vision(task_id="t-aux"):
    """Aux-vision flow: the screenshot is pre-analysed off the main model."""
    return [
        _span(task_id, "model", 0, 900),
        _span(task_id, "admission", 920, 925, "tc-1"),
        _span(task_id, "capture", 925, 1425, "tc-1", capture_mode="vision"),
        _span(task_id, "aux_vision", 1425, 2925, "tc-1"),
        _span(task_id, "element_processing", 2925, 2975, "tc-1"),
        _span(task_id, "response_shape", 2975, 2980, "tc-1"),
    ]


def fixture_spans_capture_after(task_id="t-capafter"):
    """Capture-after flow: a verification capture follows the input turn."""
    return [
        _span(task_id, "model", 0, 1100),
        _span(task_id, "admission", 1120, 1125, "tc-1"),
        _span(task_id, "capture", 1125, 1625, "tc-1", capture_mode="som"),
        _span(task_id, "element_processing", 1625, 1725, "tc-1"),
        _span(task_id, "response_shape", 1725, 1730, "tc-1"),
        _span(task_id, "model", 1750, 2850),
        _span(task_id, "admission", 2850, 2855, "tc-2"),
        _span(task_id, "input", 2855, 2905, "tc-2", action="type"),
        _span(task_id, "capture", 2905, 3405, "tc-2", capture_mode="som", capture_after="true"),
        _span(task_id, "response_shape", 3405, 3410, "tc-2"),
    ]


_FIXTURES = (
    fixture_spans_ax_only,
    fixture_spans_som,
    fixture_spans_vision,
    fixture_spans_aux_vision,
    fixture_spans_capture_after,
)


def test_critical_path_fixture_phases_use_shared_vocabulary():
    for make in _FIXTURES:
        for span in make():
            assert span.phase in cp.PHASES, span.phase


def test_critical_path_reconstructs_serial_phases_in_order():
    report = cp.build_task_report(fixture_spans_ax_only(), "t-ax")
    phases = [s.phase for s in report.segments if s.kind == "phase"]
    assert phases == [
        "model", "admission", "backend_resolve", "dispatch_lock_wait",
        "capture", "element_processing", "response_shape",
        "model", "admission", "input", "response_shape",
    ]
    assert report.e2e_ms == 3020
    assert report.measured_ms == 2970
    assert report.explained_pct == 100.0 * 2970 / 3020
    assert report.phase_totals_ms["model"] == 2400


def test_critical_path_flags_avoidable_idle():
    ax = cp.build_task_report(fixture_spans_ax_only(), "t-ax")
    assert ax.idle_ms == 50  # two 25ms scheduling gaps
    assert ax.avoidable_idle_ms == 65  # gaps + 15ms dispatch_lock_wait
    vision = cp.build_task_report(fixture_spans_vision(), "t-vision")
    assert vision.avoidable_idle_ms == 1020  # 20ms gap + 1000ms approval_wait


def test_critical_path_never_double_counts_envelope_spans():
    spans = [
        _span("t-x", "model", 0, 100),
        _span("t-x", "total", 0, 300, "tc-1"),  # envelope over its sub-phases
        _span("t-x", "capture", 100, 300, "tc-1"),
    ]
    report = cp.build_task_report(spans, "t-x")
    assert report.measured_ms == 300
    assert report.idle_ms == 0
    assert report.explained_pct == 100.0


def test_critical_path_aggregate_explains_p50_p95():
    reports = [cp.build_task_report(make(), make()[0].task_id) for make in _FIXTURES]
    agg = cp.aggregate(reports)
    assert agg.tasks == 5
    e2es = sorted(r.e2e_ms for r in reports)
    assert agg.e2e_p50_ms == e2es[2]  # nearest-rank p50 of 5
    assert agg.e2e_p95_ms == e2es[4]  # nearest-rank p95 of 5
    assert agg.explained_p50_pct > 98.0
    assert agg.avoidable_idle_total_ms == 1185
    text = cp.render(agg)
    assert "e2e p50=" in text and "avoidable idle total=" in text

