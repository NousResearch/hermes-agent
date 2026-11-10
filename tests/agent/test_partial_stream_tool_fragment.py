"""Tests ensuring partial_stream_recovery rejects raw/unparsed tool-call fragments."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from agent.turn_empty_response import recover_empty_response


@pytest.mark.parametrize(
    "fragment",
    [
        "mes/cache/terminal-output/out-1.log | sort -u | head -12\n</parameter>",
        "<parameter name=\"command\">ls -la</parameter>",
        "<tool_call>\n<function=terminal>",
        "sort -u | head -12</parameter>",
        "<function=web_search>",
    ],
)
def test_partial_stream_recovery_rejects_tool_markup_fragments(fragment: str):
    agent = MagicMock()
    agent._current_streamed_assistant_text = fragment
    agent._has_content_after_think_block = lambda t: bool(t)
    agent._strip_think_blocks = lambda t: t
    assistant_message = MagicMock(content="", tool_calls=None)

    verdict = recover_empty_response(
        agent,
        assistant_message,
        MagicMock(),
        "stop",
        final_response="",
        messages=[],
        api_messages=[],
        conversation_history=None,
        active_system_prompt=None,
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )
    assert verdict.turn_exit_reason != "partial_stream_recovery"


def test_partial_stream_recovery_accepts_clean_prose():
    agent = MagicMock()
    clean_text = "Here is the summary of the previous session."
    agent._current_streamed_assistant_text = clean_text
    agent._has_content_after_think_block = lambda t: bool(t)
    agent._strip_think_blocks = lambda t: t
    assistant_message = MagicMock(content="", tool_calls=None)

    verdict = recover_empty_response(
        agent,
        assistant_message,
        MagicMock(),
        "stop",
        final_response="",
        messages=[],
        api_messages=[],
        conversation_history=None,
        active_system_prompt=None,
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )
    assert verdict.turn_exit_reason == "partial_stream_recovery"
    assert verdict.action == "break"
    assert verdict.final_response == clean_text
