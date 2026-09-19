"""Regression tests for DeepSeek DSML tool-call markup stripping (#115475).

DSML is DeepSeek's line-based tool-call serialization. When a model emits it in the
text channel (e.g. deepseek models served through OpenAI-compatible gateways), the
markup can survive into the visible answer and reach chat surfaces raw:

    DSML | tool_calls
    DSML | invoke name="terminal"
    DSML | parameter name="command" string="true"
    python3 /path/to/script.py --flag
    DSML | /parameter
    DSML | /invoke
    DSML | /tool_calls

These tests lock the contract for ``strip_dsml_blocks`` and its two consumers that
strip tool-call markup today (``strip_think_blocks`` for final responses, and the CLI
display stripper ``_strip_reasoning_tags``).
"""

from __future__ import annotations

import pytest

from agent.agent_runtime_helpers import strip_dsml_blocks, strip_think_blocks
from cli import _strip_reasoning_tags


DSML_TOOL_CALLS_BARE = (
    'DSML | tool_calls\n'
    'DSML | invoke name="terminal"\n'
    'DSML | parameter name="command" string="true"\n'
    'python3 /path/to/script.py --flag\n'
    'DSML | parameter name="timeout" string="false"\n'
    '120\n'
    'DSML | /parameter\n'
    'DSML | /parameter\n'
    'DSML | /invoke\n'
    'DSML | /tool_calls\n'
)

DSML_TOOL_CALLS_ANGLED = (
    '<DSML | tool_calls>\n'
    '<DSML | invoke name="get_weather">\n'
    '<DSML | parameter name="city" string="true">\n'
    'Hangzhou\n'
    '<DSML | /parameter>\n'
    '<DSML | /invoke>\n'
    '<DSML | /tool_calls>\n'
)

DSML_FUNCTION_RESULTS = (
    'DSML | function_results\n'
    'DSML | result_item name="terminal"\n'
    '{"status":"success","success":true,"result":{"stdout":"hi"}}\n'
    'DSML | /result_item\n'
    'DSML | /function_results\n'
)


class TestStripDsmlBlocks:
    def test_closed_tool_calls_block_bare_removed(self) -> None:
        text = "Sure, running it now.\n" + DSML_TOOL_CALLS_BARE + "Done — exit code 0."
        out = strip_dsml_blocks(text)
        assert "DSML" not in out
        assert "python3 /path/to/script.py --flag" not in out
        assert "Sure, running it now." in out
        assert "Done — exit code 0." in out

    def test_closed_tool_calls_block_angled_removed(self) -> None:
        text = "Checking the weather.\n" + DSML_TOOL_CALLS_ANGLED + "26°C and sunny."
        out = strip_dsml_blocks(text)
        assert "DSML" not in out
        assert "Hangzhou" not in out
        assert "Checking the weather." in out
        assert "26°C and sunny." in out

    def test_function_results_block_removed(self) -> None:
        text = DSML_FUNCTION_RESULTS + "All good."
        out = strip_dsml_blocks(text)
        assert "DSML" not in out
        assert '"stdout"' not in out
        assert "All good." in out

    def test_unterminated_block_stripped_to_eof(self) -> None:
        # Stream cut mid-block: no /tool_calls closer. Everything from the opener on goes.
        text = (
            "Working…\n"
            'DSML | tool_calls\n'
            'DSML | invoke name="terminal"\n'
            'DSML | parameter name="command" string="true"\n'
            'python3 /path/to/script.py --flag\n'
        )
        out = strip_dsml_blocks(text)
        assert "Working…" in out
        assert "DSML" not in out
        assert "python3 /path/to/script.py" not in out

    def test_case_insensitive_and_indented(self) -> None:
        text = "ok\n" + "  dsml | tool_calls\n  dsml | /tool_calls\n" + "tail"
        out = strip_dsml_blocks(text)
        assert "DSML" not in out and "dsml" not in out
        assert "ok" in out and "tail" in out

    def test_prose_mentioning_dsml_mid_line_preserved(self) -> None:
        text = "The DSML | tool_calls syntax is how DeepSeek serializes calls.\nReal answer."
        out = strip_dsml_blocks(text)
        assert "DSML | tool_calls syntax" in out
        assert "Real answer." in out

    def test_prose_line_starting_with_dsml_word_preserved(self) -> None:
        # A line starting with "DSML" but no pipe marker is prose, not a directive.
        text = "DSML stands for DeepSeek Markup Language.\nAnswer."
        out = strip_dsml_blocks(text)
        assert out == text

    def test_stray_directive_line_without_block_removed(self) -> None:
        # A lone directive line (split-off markup) is still markup noise.
        text = 'Here you go:\nDSML | invoke name="terminal"\nresult'
        out = strip_dsml_blocks(text)
        assert "DSML" not in out
        assert "Here you go:" in out
        assert "result" in out

    def test_empty_and_non_string_safe(self) -> None:
        assert strip_dsml_blocks("") == ""
        assert strip_dsml_blocks(None) is None
        assert strip_dsml_blocks("plain text") == "plain text"


class TestStripThinkBlocksDsmlIntegration:
    """The canonical stripper (final_response, stream delivery, compaction) must drop DSML too."""

    def test_final_response_path_strips_dsml(self) -> None:
        final = "Summary:\n" + DSML_TOOL_CALLS_BARE + "The task is complete."
        out = strip_think_blocks(None, final)
        assert "DSML" not in out
        assert "Summary:" in out
        assert "The task is complete." in out

    def test_final_response_of_only_dsml_becomes_empty(self) -> None:
        out = strip_think_blocks(None, DSML_TOOL_CALLS_BARE)
        assert out.strip() == ""


class TestCliStripReasoningTagsDsml:
    """The CLI display stripper must stay in sync with the canonical stripper (#115475)."""

    def test_cli_display_strips_dsml(self) -> None:
        text = "Intro\n" + DSML_TOOL_CALLS_BARE + "Outro"
        out = _strip_reasoning_tags(text)
        assert "DSML" not in out
        assert "Intro" in out
        assert "Outro" in out

    def test_cli_display_unterminated_dsml(self) -> None:
        text = "Intro\nDSML | tool_calls\nDSML | invoke name=\"terminal\"\n"
        out = _strip_reasoning_tags(text)
        assert "DSML" not in out
        assert "Intro" in out
