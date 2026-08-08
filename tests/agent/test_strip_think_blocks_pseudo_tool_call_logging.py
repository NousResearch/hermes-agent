"""Tests for agent.agent_runtime_helpers.strip_think_blocks -- specifically
the pseudo-tool-call logging added for #56461.

A model that doesn't reliably support native function calling (seen with
some local Ollama models) can emit its tool call as plain text
(``<tool_call>...</tool_call>``) instead of the structured ``tool_calls``
API field. Hermes strips that pseudo-call so it doesn't pollute the visible
response (ported from openclaw/openclaw#67318, originally for Gemma), but
stripping it silently left no trace that the call never actually executed --
the model's surrounding narration ("Got it, saved that!") survived
untouched, making a real failure look like a success. This must stay in
sync with cli.py::_strip_reasoning_tags for tool-call tag coverage.
"""

import logging
from types import SimpleNamespace

from agent.agent_runtime_helpers import strip_think_blocks


def _agent():
    return SimpleNamespace()


class TestPseudoToolCallLogging:
    def test_stripped_tool_call_logs_warning(self, caplog):
        content = 'Got it, saved that!\n<tool_call>{"name": "memory"}</tool_call>'
        with caplog.at_level(logging.WARNING):
            result = strip_think_blocks(_agent(), content)
        assert "Got it, saved that!" in result
        assert "<tool_call>" not in result
        assert any("Stripped a <tool_call>" in r.message for r in caplog.records)

    def test_stripped_function_calls_block_logs_warning(self, caplog):
        content = '<function_calls>[{}]</function_calls>\nanswer'
        with caplog.at_level(logging.WARNING):
            result = strip_think_blocks(_agent(), content)
        assert "answer" in result
        assert any("Stripped a <function_calls>" in r.message for r in caplog.records)

    def test_stripped_gemma_function_block_logs_warning(self, caplog):
        content = (
            'Reading.\n'
            '<function name="r"><parameter name="p">/tmp/x</parameter></function>\n'
            'Done.'
        )
        with caplog.at_level(logging.WARNING):
            result = strip_think_blocks(_agent(), content)
        assert "Reading." in result and "Done." in result
        assert any("Stripped a <function>" in r.message for r in caplog.records)

    def test_reasoning_tag_strip_does_not_log(self, caplog):
        """Reasoning-tag stripping is normal, expected behavior for every
        turn -- only tool-call-shaped blocks represent an undiagnosable
        failure worth logging."""
        with caplog.at_level(logging.WARNING):
            strip_think_blocks(_agent(), "<think>plan</think> answer")
        assert not caplog.records

    def test_plain_text_does_not_log(self, caplog):
        with caplog.at_level(logging.WARNING):
            strip_think_blocks(_agent(), "just a normal reply")
        assert not caplog.records

    def test_prose_mention_of_function_not_logged(self, caplog):
        """Boundary-gated <function> stripping must not fire (or log) for
        ordinary prose mentions."""
        with caplog.at_level(logging.WARNING):
            result = strip_think_blocks(_agent(), "Use <function> declarations in JavaScript.")
        assert "JavaScript" in result
        assert not caplog.records
