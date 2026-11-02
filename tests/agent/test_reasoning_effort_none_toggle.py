"""Regression tests for #118871: a bare ``{"effort": "none"}`` must disable
thinking on the toggle wires (Moonshot/DeepSeek) instead of falling through
to the toggle-on default."""

from agent.reasoning_effort import thinking_toggle_extras

EFFORTS = ["low", "medium", "high"]


def test_bare_effort_none_emits_disabled_toggle():
    extras, top = thinking_toggle_extras({"effort": "none"}, EFFORTS)
    assert extras == {"thinking": {"type": "disabled"}}
    assert top == {}


def test_bare_effort_none_emits_disabled_toggle_with_always_emit():
    extras, top = thinking_toggle_extras({"effort": "none"}, EFFORTS, always_emit_toggle=True)
    assert extras == {"thinking": {"type": "disabled"}}
    assert top == {}


def test_effort_none_is_case_insensitive():
    extras, top = thinking_toggle_extras({"effort": "None"}, EFFORTS)
    assert extras == {"thinking": {"type": "disabled"}}
    assert top == {}


def test_enabled_false_still_emits_disabled_toggle():
    extras, top = thinking_toggle_extras({"enabled": False, "effort": "none"}, EFFORTS)
    assert extras == {"thinking": {"type": "disabled"}}
    assert top == {}


def test_valid_effort_still_maps_to_wire_effort():
    extras, top = thinking_toggle_extras({"effort": "medium"}, EFFORTS)
    assert extras == {}
    assert top == {"reasoning_effort": "medium"}


def test_valid_effort_with_always_emit_keeps_toggle_on():
    extras, top = thinking_toggle_extras({"effort": "high"}, EFFORTS, always_emit_toggle=True)
    assert extras == {"thinking": {"type": "enabled"}}
    assert top == {"reasoning_effort": "high"}
