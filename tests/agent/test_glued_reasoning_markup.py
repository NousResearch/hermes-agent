"""Reasoning-prefix XML names must survive malformed opener stripping (#136557)."""

import pytest

from agent.agent_runtime_helpers import strip_think_blocks


@pytest.mark.parametrize(
    "text",
    (
        "Here is the layout:\n<thoughts>\n  <item>a</item>\n</thoughts>\nDone.",
        "Config:\n<reasoningEffort>high</reasoningEffort>\nThat's it.",
        "Answer:\n<thinkpad-model>X1</thinkpad-model>\nend",
    ),
)
def test_reasoning_prefix_element_names_are_not_stripped(text):
    """An unrelated markup tag must not erase the remainder of a real answer."""
    assert strip_think_blocks(None, text) == text


def test_glued_opener_without_angle_bracket_is_still_stripped():
    """Keep the malformed opener reported by Code0020161314 suppressed."""
    assert strip_think_blocks(None, '"<thinkingBackup done. Now let me run the script') == ""
