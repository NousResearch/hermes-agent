"""The unterminated-reasoning strip must open a block on a payload glued to the tag name.

A degenerate relay/model output can emit the reasoning opener with no ``>`` of its own —
``<thinkingBackup done. Now let me run …`` (#136557) — where ``\b`` alone cannot fire (both
sides of the boundary are word characters). The strip still has to stay boundary-gated:
mid-line prose mentions of a think tag must survive, and a glued opener with no ``>`` anywhere
later still opens nothing (the pass needs a closing ``>`` for the opener, e.g. from a
tool-call tail).
"""

from __future__ import annotations

from agent.agent_runtime_helpers import strip_think_blocks


def test_glued_open_tag_with_later_close_bracket_strips_to_end():
    # Micro form of the #136557 field report: 139k chars of reasoning whose only `>`s belong
    # to the degenerate <invoke></parameter> repetition tail.
    text = "<thinkingBackup done. Now let me run the maintenance script in dry-run mode <invoke>x"
    assert strip_think_blocks(None, text) == ""


def test_glued_open_tag_on_later_line_strips_from_that_line():
    text = "Visible intro.\n<thinkingI should check the config first. Value > 1 applies."
    assert strip_think_blocks(None, text) == "Visible intro."


def test_mid_line_prose_mention_survives():
    text = "Use the <thinking> tag carefully in prompts"
    out = strip_think_blocks(None, text)
    assert "Use the" in out and "tag carefully in prompts" in out


def test_glued_without_any_close_bracket_stays():
    # No `>` anywhere later: the opener never completes, so the pass leaves the text alone.
    text = "<thinkingless rambling with no close"
    assert strip_think_blocks(None, text) == text


def test_line_start_complete_opener_still_strips_to_end():
    # Regression anchor for the pre-existing shape: a complete opener at a block boundary.
    text = "<think> actual reasoning tail with a > inside"
    assert strip_think_blocks(None, text) == ""


def test_cjk_tag_glued_continuation_stays():
    # CJK tag names keep the strict \b: a glued CJK continuation is indistinguishable from
    # ordinary prose starting with the tag word.
    text = "<思考的结论是这样的> 有些内容"
    assert strip_think_blocks(None, text) == text


def test_element_name_extending_a_tag_stays():
    # A legitimate XML/markup element whose name merely starts with a think-tag name is not a
    # degenerate opener: the glued token is followed by `>` or `-`, not whitespace. Regression
    # rows from the #136562 review.
    assert strip_think_blocks(
        None, "Here is the layout:\n<thoughts>\n  <item>a</item>\n</thoughts>\nDone."
    ) == "Here is the layout:\n<thoughts>\n  <item>a</item>\n</thoughts>\nDone."
    assert strip_think_blocks(
        None, "Config:\n<reasoningEffort>high</reasoningEffort>\nThat's it."
    ) == "Config:\n<reasoningEffort>high</reasoningEffort>\nThat's it."
    assert strip_think_blocks(
        None, "Answer:\n<thinkpad-model>X1</thinkpad-model>\nend"
    ) == "Answer:\n<thinkpad-model>X1</thinkpad-model>\nend"
