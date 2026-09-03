"""Gateway chat-surface sanitizer must strip tool-call render lines (#95705).

The agent session for an inbound chat surfaces its working transcript —
status chrome like ``🔍 Searching past sessions (×2)`` — alongside the final
assistant reply.  ``_sanitize_gateway_final_response`` is the last line of
defense before that text reaches the chat.  These tests pin the strip
behavior: standalone tool-render header lines are dropped, the assistant's
own prose survives, and the raw-text/programmatic surfaces (CLI ``local``,
API JSON, webhook payloads) keep the full transcript unchanged.
"""

from __future__ import annotations

import pytest

from gateway.run import (
    _GATEWAY_RAW_TEXT_PLATFORMS,
    _sanitize_gateway_final_response,
    _strip_gateway_tool_render_lines,
)


CHAT_PLATFORM = "whatsapp"  # representative; also covers telegram/discord/etc.


# ---------------------------------------------------------------------------
# Unit-level: _strip_gateway_tool_render_lines
# ---------------------------------------------------------------------------


def test_strips_session_search_render_with_count_suffix():
    """The exact example from the live incident report (#95705)."""
    text = "🔍 Searching past sessions (×2)\n\nI found two relevant ones."
    out = _strip_gateway_tool_render_lines(text)
    assert "🔍 Searching past sessions (×2)" not in out
    assert "I found two relevant ones." in out


def test_strips_terminal_command_render():
    """`⚙️ Running <cmd>` lines must not leak through."""
    text = "⚙️ Running npm test\n\nTests passed."
    out = _strip_gateway_tool_render_lines(text)
    assert "⚙️ Running npm test" not in out
    assert "Tests passed." in out


def test_strips_edit_render():
    text = "✏️ Editing src/foo.py\n\nPatch applied."
    out = _strip_gateway_tool_render_lines(text)
    assert "✏️ Editing" not in out
    assert "Patch applied." in out


def test_strips_memory_render():
    text = "💾 Updating memory\n\nSaved the user profile."
    out = _strip_gateway_tool_render_lines(text)
    assert "💾 Updating memory" not in out
    assert "Saved the user profile." in out


def test_strips_skill_listing_render():
    text = "⚡ Listing skills\n\nThree skills matched."
    out = _strip_gateway_tool_render_lines(text)
    assert "⚡ Listing skills" not in out
    assert "Three skills matched." in out


def test_preserves_assistant_prose_with_emoji_inline():
    """A normal sentence that mentions an emoji mid-paragraph must survive.

    Only whole-line tool-render chrome is stripped; inline emoji usage in
    assistant prose stays untouched so we don't over-strip.
    """
    text = "I did 🔍 research and found the answer. The answer is: 42."
    out = _strip_gateway_tool_render_lines(text)
    assert out == text


def test_preserves_assistant_prose_with_unknown_verb_after_emoji():
    """A real assistant reply like '🔍 research project' must be kept.

    The strip regex anchors on the (emoji)(known-verb) pattern; ``research``
    isn't a curated tool verb, so the line stays.
    """
    text = "🔍 research project A\n\nDetails follow."
    out = _strip_gateway_tool_render_lines(text)
    assert "🔍 research project A" in out
    assert "Details follow." in out


def test_collapses_excess_blank_lines_after_strip():
    text = "🔍 Searching past sessions\n\n\n\nThe answer is 42."
    out = _strip_gateway_tool_render_lines(text)
    # Three or more newlines should compress to a single blank between paragraphs.
    assert "\n\n\n" not in out
    assert "The answer is 42." in out
    assert out.startswith("The answer is 42.")


def test_empty_input_returns_empty():
    assert _strip_gateway_tool_render_lines("") == ""


def test_single_line_tool_render_only_collapses_to_empty():
    """A standalone tool-render line with no prose collapses to empty.

    The sanitizer at run.py surfaces empty for safety, but the helper itself
    keeps the input unchanged here — the empty-collapse is the wrapper's job.
    """
    out = _strip_gateway_tool_render_lines("🔍 Searching past sessions (×2)")
    assert out == ""


def test_single_line_assistant_prose_kept():
    out = _strip_gateway_tool_render_lines("Hello, this is the answer.")
    assert out == "Hello, this is the answer."


# ---------------------------------------------------------------------------
# Integration: _sanitize_gateway_final_response on chat surfaces
# ---------------------------------------------------------------------------


CHAT_PLATFORMS = ["whatsapp", "telegram", "discord", "slack", "signal"]


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_sanitizer_strips_tool_render_on_chat_platform(platform):
    """The full final-response sanitizer must drop tool render on every chat."""
    raw = "🔍 Searching past sessions (×2)\n\nFinal answer."
    out = _sanitize_gateway_final_response(platform, raw)
    assert "🔍" not in out
    assert "Searching" not in out
    assert "Final answer." in out


@pytest.mark.parametrize("platform", sorted(_GATEWAY_RAW_TEXT_PLATFORMS))
def test_sanitizer_keeps_tool_render_on_raw_text_platform(platform):
    """Raw-text/programmatic surfaces must keep the full transcript.

    CLI ``local`` diagnostics, API JSON, webhook payloads all deliberately
    pass through unchanged so an operator inspecting logs / piping JSON
    gets the unmodified response.
    """
    raw = "🔍 Searching past sessions (×2)\n\nFinal answer."
    out = _sanitize_gateway_final_response(platform, raw)
    assert out == raw


def test_sanitizer_keeps_empty_input():
    """Empty final_response must stay empty (no surprise '\n')."""
    assert _sanitize_gateway_final_response("whatsapp", "") == ""
    assert _sanitize_gateway_final_response("local", "") == ""


def test_sanitizer_keeps_normal_response_intact():
    """A real assistant reply with no tool chrome must round-trip unchanged."""
    raw = (
        "Hermes is a personal AI agent that runs the same core across "
        "CLI, gateway, and desktop. Let me know which surface you'd like."
    )
    out = _sanitize_gateway_final_response("whatsapp", raw)
    assert out == raw


def test_sanitizer_strips_multiple_render_lines_and_keeps_prose():
    """A multi-render final_response must drop ALL the chrome, keep ALL prose."""
    raw = (
        "🔍 Searching past sessions\n"
        "✏️ Editing memory.md\n"
        "⚙️ Running grep\n"
        "\n"
        "Here is the answer the user asked for."
    )
    out = _sanitize_gateway_final_response("whatsapp", raw)
    assert "🔍" not in out
    assert "✏️" not in out
    assert "⚙️" not in out
    assert "Here is the answer the user asked for." in out
