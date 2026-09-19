"""Behavior contracts for per-action readiness predicates (RFC #112639)."""

import pytest

from tools.computer_use.readiness_predicates import (
    predicates_for_action,
    predicates_for_click,
    predicates_for_key,
    predicates_for_set_value,
    predicates_for_type,
    verify_action,
)


class _StubSession:
    def _has_tool(self, name):
        return name == "verify_state"


class _StubBackend:
    def __init__(self):
        self._session = _StubSession()
        self.calls = []

    def call_tool(self, name, args=None, timeout=30.0):
        self.calls.append({"name": name, "args": dict(args or {}), "timeout": timeout})
        return {"structuredContent": {"status": "satisfied"}}


def _assert_well_formed(expect):
    assert isinstance(expect, list) and 1 <= len(expect) <= 8
    for p in expect:
        assert isinstance(p, dict)
        assert set(p) <= {"element", "window"}
        if "element" in p:
            el = p["element"]
            assert "selector" in el and el["selector"]  # driver requires a selector
            assert el.get("exists") is True  # driver rejects exists:false


def test_type_with_field_label_asserts_value():
    expect = predicates_for_type("hello", field_label="Name")
    _assert_well_formed(expect)
    el = expect[0]["element"]
    assert el["selector"] == {"label_contains": "Name"}
    assert el["value_equals"] == "hello"


def test_type_without_field_falls_back_to_window_exists():
    assert predicates_for_type("hello") == [{"window": {"exists": True}}]


def test_type_rejects_empty_text():
    with pytest.raises(ValueError):
        predicates_for_type("")


def test_set_value_asserts_value():
    expect = predicates_for_set_value("opt2", field_role="combobox")
    _assert_well_formed(expect)
    el = expect[0]["element"]
    assert el["selector"] == {"role": "combobox"}
    assert el["value_equals"] == "opt2"


def test_click_with_expected_element():
    expect = predicates_for_click(expect_label="Submit", expect_enabled=True)
    _assert_well_formed(expect)
    el = expect[0]["element"]
    assert el["selector"] == {"label_contains": "Submit"}
    assert el["enabled"] is True


def test_click_selected_postcondition():
    expect = predicates_for_click(expect_label="Remember me", expect_selected=True)
    _assert_well_formed(expect)
    assert expect[0]["element"]["selected"] is True


def test_click_without_expectation_falls_back():
    assert predicates_for_click() == [{"window": {"exists": True}}]


def test_key_with_expected_element():
    expect = predicates_for_key(expect_label="Done")
    _assert_well_formed(expect)
    assert expect[0]["element"]["selector"] == {"label_contains": "Done"}


def test_empty_selector_rejected():
    with pytest.raises(ValueError):
        predicates_for_click(expect_label="", expect_role="")


@pytest.mark.parametrize("action", ["drag", "scroll", "focus_app", "wait", "capture", "bogus"])
def test_unverifiable_actions_fall_back(action):
    assert predicates_for_action(action) == [{"window": {"exists": True}}]


@pytest.mark.parametrize("action", ["click", "double_click", "right_click", "middle_click"])
def test_click_variants_share_builder(action):
    expect = predicates_for_action(action, expect_label="OK")
    _assert_well_formed(expect)
    assert expect[0]["element"]["selector"] == {"label_contains": "OK"}


def test_action_dispatch_type():
    expect = predicates_for_action("type", "abc", field_label="Name")
    assert expect[0]["element"]["value_equals"] == "abc"


def test_verify_action_builds_and_calls():
    backend = _StubBackend()
    out = verify_action(backend, action="type", pid=7, window_id=9,
                        text="hi", field_label="Name")
    assert out.status == "satisfied"
    assert len(backend.calls) == 1
    payload = backend.calls[0]["args"]
    assert payload["pid"] == 7 and payload["window_id"] == 9
    expect = payload["expect"]
    _assert_well_formed(expect)
    assert expect[0]["element"]["value_equals"] == "hi"


def test_verify_action_fallback_reaches_driver():
    backend = _StubBackend()
    verify_action(backend, action="scroll", pid=1, window_id=2)
    assert backend.calls[0]["args"]["expect"] == [{"window": {"exists": True}}]
