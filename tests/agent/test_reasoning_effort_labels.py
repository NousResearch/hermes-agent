"""#61634: ``ultra`` is Hermes-internal and every wire clamps it; the display label used by the
effort pickers and ``/reasoning`` status must say what the route really sends."""
from agent.reasoning_effort import effort_display_label




def test_supported_level_label_is_the_level_itself():
    assert effort_display_label("max", "openai-codex", "gpt-5.6-sol") == "max"
    assert effort_display_label("high", None, None) == "high"
    assert effort_display_label("", None, None) == ""
