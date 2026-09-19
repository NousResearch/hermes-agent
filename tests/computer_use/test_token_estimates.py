"""Token estimates in computer_use reports (gate from #112734).

The reports carried observation/delta bytes only; they now carry token estimates
too — per action (shadow-state metrics), per phase and as totals (critical-path
report). The estimator is documented in ``tools/computer_use/token_estimates.py``
and is deterministic: identical input always yields the identical estimate.
"""

import pytest

from tools.computer_use import critical_path as cp
from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.token_estimates import (
    METHOD,
    estimate_image_tokens,
    estimate_text_tokens,
)

W, H = 800, 600


def _el(index, role, label, bounds):
    return UIElement(index=index, role=role, label=label, bounds=bounds,
                     app="TestApp", pid=4242, window_id=7)


def _elements():
    return [
        _el(1, "AXWindow", "Settings", (0, 0, 800, 600)),
        _el(2, "AXTextField", "Search settings", (100, 100, 300, 30)),
        _el(3, "AXButton", "Clear data", (100, 150, 120, 30)),
    ]


@pytest.fixture(autouse=True)
def _reset_shadow():
    cu_tool.reset_shadow_state_for_tests()
    yield
    cu_tool.reset_shadow_state_for_tests()


def test_text_estimate_contract():
    assert estimate_text_tokens(0) == 0
    assert estimate_text_tokens(1) == 1
    assert estimate_text_tokens(4) == 1  # ceiling of bytes/4
    assert estimate_text_tokens(5) == 2
    assert estimate_text_tokens(800) == 200
    assert estimate_text_tokens(800) == estimate_text_tokens(800)  # deterministic


def test_image_estimate_contract():
    # 800x600 scales to 1024x768 (768 short side) = 2x2 tiles -> 85 + 4*170.
    assert estimate_image_tokens(800, 600) == 765
    # 512x512 scales to 768x768 -> same 4-tile grid.
    assert estimate_image_tokens(512, 512) == 765
    assert estimate_image_tokens(0, 600) == 0  # missing dimension: no image to bill
    assert estimate_image_tokens(800, 600) == estimate_image_tokens(800, 600)


def test_metrics_carry_token_fields_and_method():
    cap = CaptureResult(mode="som", width=W, height=H, png_b64="ZmFrZXBuZy1ieXRlcw==",
                        png_bytes_len=16, elements=_elements(),
                        app="TestApp", window_title="Settings")
    cu_tool._shadow_state_observe(cap, "sess-tok")
    m = cu_tool.get_shadow_state_metrics("sess-tok")
    for key in ("delta_tokens_est", "full_observation_tokens_est",
                "image_tokens_est", "token_estimate_method"):
        assert key in m, key
    assert m["token_estimate_method"] == METHOD
    assert m["full_observation_tokens_est"] > 0
    assert m["delta_tokens_est"] <= m["full_observation_tokens_est"]
    assert m["image_tokens_est"] == estimate_image_tokens(W, H)  # real screenshot: tile grid
    # AX-only capture: no image, so no image estimate — text estimate still present.
    cu_tool._shadow_state_observe(
        CaptureResult(mode="ax", width=W, height=H, elements=_elements(),
                      app="TestApp", window_title="Settings"), "sess-ax")
    m2 = cu_tool.get_shadow_state_metrics("sess-ax")
    assert m2["image_tokens_est"] == 0
    assert m2["full_observation_tokens_est"] > 0


def test_metrics_token_fields_deterministic_for_identical_input():
    def run(session):
        cu_tool._shadow_state_observe(
            CaptureResult(mode="som", width=W, height=H, png_b64="ZmFrZXBuZy1ieXRlcw==",
                          png_bytes_len=16, elements=_elements(),
                          app="TestApp", window_title="Settings"), session)
        cu_tool._shadow_state_observe(
            CaptureResult(mode="som", width=W, height=H, png_b64="ZmFrZXBuZy1ieXRlcw==",
                          png_bytes_len=16, elements=_elements(),
                          app="TestApp", window_title="Settings"), session)
        return cu_tool.get_shadow_state_metrics(session)

    first = run("sess-a")
    cu_tool.reset_shadow_state_for_tests()
    second = run("sess-b")
    token_keys = ("delta_tokens_est", "full_observation_tokens_est",
                  "image_tokens_est", "token_estimate_method")
    assert all(first[k] == second[k] for k in token_keys)


def _span(phase, start, end, **attrs):
    return cp.Span(task_id="t-tok", phase=phase, start_ms=start, end_ms=end,
                   tool_call_id="tc-1", session_id="sess-1", attrs=attrs)


def test_task_report_aggregates_per_phase_tokens():
    spans = [
        _span("capture", 0, 100, image_token_est=765),
        _span("total", 0, 300, token_est=10**9),  # envelope: never on-path
        _span("element_processing", 100, 200, token_est=120, image_token_est=765),
        _span("response_shape", 200, 210),  # no token attrs: contributes nothing
    ]
    report = cp.build_task_report(spans, "t-tok")
    assert report.phase_tokens_est == {"capture": 765, "element_processing": 885}
    assert report.total_tokens_est == 1650
    assert "response_shape" not in report.phase_tokens_est


def test_phase_token_aggregation_deterministic_for_identical_spans():
    def build():
        return cp.build_task_report([
            _span("capture", 0, 100, image_token_est=765),
            _span("element_processing", 100, 200, token_est=120),
        ], "t-tok")
    assert build().phase_tokens_est == build().phase_tokens_est
    assert build().total_tokens_est == build().total_tokens_est


def test_report_render_shows_per_phase_and_total_tokens():
    spans = [
        _span("capture", 0, 100, image_token_est=765),
        _span("element_processing", 100, 200, token_est=120),
    ]
    text = cp.render(cp.build_task_report(spans, "t-tok"))
    assert "tokens_est=885" in text  # task total
    assert "capture: 100ms tokens_est=765" in text
    assert "element_processing: 100ms tokens_est=120" in text


def test_aggregate_render_shows_token_total():
    reports = [cp.build_task_report([_span("capture", 0, 100, image_token_est=765)], "t-tok")]
    agg = cp.aggregate(reports)
    assert agg.total_tokens_est == 765
    assert "tokens_est total=765" in cp.render(agg)
