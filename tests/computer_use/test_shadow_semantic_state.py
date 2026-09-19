"""Shadow semantic state + delta contracts (Phase 1A of #112734).

Deterministic fixtures only — synthetic AX trees, no Bot Screen dependency. Each test
asserts a behavior contract between two pieces of data (build vs diff), never a snapshot.
"""

import pytest

from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.semantic_state import build_state
from tools.computer_use.state_diff import delta_bytes, diff_states, full_observation_bytes

W, H = 800, 600


def _el(index, role, label, bounds, token=None, app="TestApp", window_id=7, **flags):
    return UIElement(index=index, role=role, label=label, bounds=bounds, app=app,
                     pid=4242, window_id=window_id, attributes=dict(flags),
                     element_token=token)


def _form_elements(**overrides):
    focused = overrides.get("focused", False)
    return [
        _el(1, "AXWindow", "Settings", (0, 0, 800, 600), token=overrides.get("token")),
        _el(2, "AXTextField", "Search settings", (100, 100, 300, 30), focused=focused),
        _el(3, "AXButton", "Clear data", (100, 150, 120, 30)),
        _el(4, "AXCheckBox", "Sync", (100, 200, 120, 20)),
    ]


def _modal_elements():
    return _form_elements() + [
        _el(5, "AXDialog", "Clear browsing data", (200, 150, 400, 200)),
        _el(6, "AXButton", "Cancel", (220, 300, 100, 30)),
        _el(7, "AXButton", "Clear data", (340, 300, 100, 30)),
    ]


def _list_elements(order):
    return [_el(i + 1, "AXStaticText", name, (50, 100 + i * 40, 200, 24)) for i, name in enumerate(order)]


def _state(elements, revision=1):
    return build_state(elements, revision=revision, target="TestApp/Settings",
                       width=W, height=H)


@pytest.fixture(autouse=True)
def _reset_shadow():
    cu_tool.reset_shadow_state_for_tests()
    yield
    cu_tool.reset_shadow_state_for_tests()


def test_build_is_deterministic():
    assert _state(_form_elements()).elements == _state(_form_elements()).elements


def test_form_focus_transition_represented_correctly():
    delta = diff_states(_state(_form_elements(), 1), _state(_form_elements(focused=True), 2))
    assert not delta.added and not delta.removed and not delta.ambiguous
    assert delta.matched == 4 and delta.identity_retention == 1.0
    assert len(delta.changed) == 1
    change = delta.changed[0]
    assert change.element.name == "Search settings"
    flag_change = next(v for f, v in change.changes if f == "flags")
    assert "False" in flag_change[0] and "True" in flag_change[1]


def test_driver_token_rebind_is_not_a_change():
    old = _state(_form_elements(token="s00000183:17"), 1)
    new = _state(_form_elements(token="s00000184:21"), 2)
    delta = diff_states(old, new)
    assert not delta.changed and delta.matched == 4  # tokens stay snapshot-scoped (issue section E)


def test_modal_appear_and_dismiss():
    opened = diff_states(_state(_form_elements(), 1), _state(_modal_elements(), 2))
    assert len(opened.added) == 3
    assert {e.name for e in opened.added} == {"Clear browsing data", "Cancel", "Clear data"}
    assert not opened.removed
    closed = diff_states(_state(_modal_elements(), 2), _state(_form_elements(), 3))
    assert len(closed.removed) == 3 and not closed.added


def test_reordered_list_keeps_identity_through_geometry_change():
    delta = diff_states(_state(_list_elements(["Alpha", "Beta", "Gamma"]), 1),
                        _state(_list_elements(["Gamma", "Alpha", "Beta"]), 2))
    assert delta.matched == 3 and not delta.added and not delta.removed
    assert delta.identity_retention == 1.0
    assert len(delta.changed) == 3  # every item moved: geometry diffs, identity kept
    assert all(any(f == "rel_geom" for f, _ in c.changes) for c in delta.changed)


def test_twin_unnamed_buttons_surface_ambiguity_instead_of_misbinding():
    # Same evidence key twice (duplicated/ambiguous driver report); the new capture
    # carries one of them.
    old_els = [_el(1, "AXButton", "", (10, 10, 50, 20)), _el(2, "AXButton", "", (10, 10, 50, 20))]
    new_els = [_el(1, "AXButton", "", (10, 10, 50, 20))]
    delta = diff_states(_state(old_els, 1), _state(new_els, 2))
    assert len(delta.ambiguous) == 1  # surfaced, not silently mis-bound
    assert delta.matched == 1 and len(delta.removed) == 1


def test_delta_projection_smaller_than_full_observation():
    delta = diff_states(_state(_form_elements(), 1), _state(_form_elements(focused=True), 2))
    assert delta_bytes(delta) < full_observation_bytes(_state(_form_elements(focused=True), 2))


def test_shadow_observe_records_section_j_metrics():
    cap = lambda els: CaptureResult(mode="ax", width=W, height=H, elements=els,
                                    app="TestApp", window_title="Settings")
    cu_tool._shadow_state_observe(cap(_form_elements()), "sess-1")
    cu_tool._shadow_state_observe(cap(_form_elements(focused=True)), "sess-1")
    m = cu_tool.get_shadow_state_metrics("sess-1")
    assert m["revision"] == 2 and m["elements"] == 4
    assert m["changed_element_ratio"] == pytest.approx(0.25)
    assert m["identity_retention"] == pytest.approx(1.0)
    assert m["ambiguous"] == 0
    assert m["delta_bytes"] <= m["full_observation_bytes"]
    assert 0 <= m["reconciliation_ms"] < 1000  # µs-scale work, far below a fresh capture round-trip


def test_shadow_observe_is_measurement_only():
    cap = CaptureResult(mode="ax", width=W, height=H, elements=_form_elements(),
                        app="TestApp", window_title="Settings")
    assert cu_tool._shadow_state_observe(cap, "sess-9") is None
    assert cu_tool.get_shadow_state_metrics("no-such-session") == {}
    # Vision-only captures carry no semantic content: nothing recorded, nothing raised.
    cu_tool._shadow_state_observe(CaptureResult(mode="vision", width=W, height=H), "sess-9")
    assert cu_tool.get_shadow_state_metrics("sess-9")["revision"] == 1
