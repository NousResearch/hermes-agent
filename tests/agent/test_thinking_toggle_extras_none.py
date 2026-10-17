"""Regression test for #118871: effort=none must disable thinking on the provider-profile wire."""

from agent.reasoning_effort import (
    DEEPSEEK_V4_EFFORTS,
    DEEPSEEK_V4_OVERRIDES,
    thinking_toggle_extras,
)


def test_thinking_toggle_extras_effort_none_disables_thinking() -> None:
    """A bare {"effort": "none"} must produce thinking disabled — not enabled."""
    extra_body, top_level = thinking_toggle_extras(
        {"effort": "none"}, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES,
        always_emit_toggle=True,
    )
    assert extra_body == {"thinking": {"type": "disabled"}}
    assert top_level == {}


def test_thinking_toggle_extras_effort_none_without_toggle() -> None:
    """Same behavior when always_emit_toggle=False."""
    extra_body, top_level = thinking_toggle_extras(
        {"effort": "none"}, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES,
        always_emit_toggle=False,
    )
    assert extra_body == {"thinking": {"type": "disabled"}}
    assert top_level == {}


def test_thinking_toggle_extras_enabled_false_still_works() -> None:
    """The existing enabled=False path is preserved."""
    extra_body, top_level = thinking_toggle_extras(
        {"enabled": False}, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES,
        always_emit_toggle=True,
    )
    assert extra_body == {"thinking": {"type": "disabled"}}
    assert top_level == {}


def test_thinking_toggle_extras_effort_low_still_enables() -> None:
    """A valid effort level still enables thinking."""
    extra_body, top_level = thinking_toggle_extras(
        {"effort": "low"}, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES,
        always_emit_toggle=True,
    )
    assert extra_body == {"thinking": {"type": "enabled"}}
    assert top_level == {"reasoning_effort": "low"}


def test_thinking_toggle_extras_none_effort_value_in_config() -> None:
    """reasoning_config with effort="none" (no enabled key) disables thinking."""
    extra_body, top_level = thinking_toggle_extras(
        {"enabled": True, "effort": "none"}, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES,
        always_emit_toggle=True,
    )
    assert extra_body == {"thinking": {"type": "disabled"}}
    assert top_level == {}
