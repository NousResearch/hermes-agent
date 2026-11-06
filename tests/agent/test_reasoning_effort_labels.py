"""#61634: ``ultra`` is Hermes-internal and every wire clamps it; the display label used by the
effort pickers and ``/reasoning`` status must say what the route really sends."""
from agent.reasoning_effort import effort_display_label




def test_supported_level_label_is_the_level_itself():
    assert effort_display_label("max", "openai-codex", "gpt-5.6-sol") == "max"
    assert effort_display_label("high", None, None) == "high"
    assert effort_display_label("", None, None) == ""


def test_codex_app_server_sends_ultra_verbatim_only_where_the_model_reaches_max():
    assert effort_display_label("ultra", "openai-codex", "gpt-5.6-sol", "codex_app_server") == "ultra"
    assert effort_display_label("ultra", "openai-codex", "gpt-5.6-sol").startswith("ultra (sends max")
    assert effort_display_label("ultra", "openai-codex", "gpt-5.5", "codex_app_server").startswith("ultra (sends xhigh")


def test_compact_label_hides_defaults_and_keeps_non_default_levels():
    """The status bar shows nothing for what every route does when nobody chose a level."""
    from agent.reasoning_effort import effort_compact_label

    assert effort_compact_label("") == ""
    assert effort_compact_label(None) == ""
    for level in ("medium", "normal", "default", "MEDIUM"):
        assert effort_compact_label(level) == ""
    assert effort_compact_label("high") == "high"
    assert effort_compact_label("none") == "none"


def test_compact_label_states_the_wire_level_when_the_route_clamps():
    from agent.reasoning_effort import effort_compact_label

    assert effort_compact_label("ultra", None, None) == "ultra→max"
    assert effort_compact_label("ultra", "openai-codex", "gpt-5.6-sol") == "ultra→max"
    assert effort_compact_label("ultra", "openai-codex", "gpt-5.6-sol", "codex_app_server") == "ultra"
    assert effort_compact_label("high", "openai-codex", "gpt-5.6-sol") == "high"
