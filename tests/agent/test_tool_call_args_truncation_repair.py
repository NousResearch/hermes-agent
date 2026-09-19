"""Truncated tool-call argument JSON must be repaired — not silently discarded (#115061).

The shared repair pass (``agent/message_sanitization._repair_tool_call_arguments``) is
reached from the streaming assembler (``_StreamingCall._assemble_tool_calls``), the
send-path canonicalizer (``_canonicalize_api_tool_calls``) and the history sanitizer, so a
hole here hands *every* provider's broken arguments to plugins as ``{}``.

Contract pinned here:

* a payload truncated inside an array (or inside an object nested in one) still closes
  cleanly and every parameter survives to the assembled tool call;
* a payload truncated *inside* a string value has lost data that cannot be invented, so it
  falls back to ``{}`` — loudly, keeping the original bytes in the warning (never silently).
"""

from __future__ import annotations

import json
import logging

from agent.message_sanitization import _repair_tool_call_arguments

# The exact shape reported in #115061 / observed on a real host: a truncated argument list
# of object elements (``{"operations": [{"action": ...``), cut before any closer.
_TRUNCATED_NESTED_ARRAY = (
    '{"operations": [{"action": "patch", "file_path": "references/timestamps-vacances.md"'
)


class TestTruncatedNestedArgumentsRepair:
    """A tail cut inside an array is recoverable: close innermost first."""

    def test_truncated_array_element_list_is_repaired(self):
        repaired = _repair_tool_call_arguments(_TRUNCATED_NESTED_ARRAY, "skill_manage")
        assert json.loads(repaired) == {
            "operations": [{"action": "patch", "file_path": "references/timestamps-vacances.md"}]
        }

    def test_truncated_array_inside_object_keeps_its_parameters(self):
        repaired = _repair_tool_call_arguments('{"path": "/tmp/a.txt", "edits": [{"old": "x"', "patch")
        parsed = json.loads(repaired)
        assert parsed["path"] == "/tmp/a.txt"
        assert parsed["edits"] == [{"old": "x"}]

    def test_truncated_bare_array_tail_is_closed(self):
        assert json.loads(_repair_tool_call_arguments('{"a": [1, 2', "t")) == {"a": [1, 2]}

    def test_truncated_right_after_a_comma_is_repaired(self):
        """A stream cut at a token boundary lands here: comma, then EOF."""
        assert json.loads(_repair_tool_call_arguments('{"a": [1, 2,', "t")) == {"a": [1, 2]}
        assert json.loads(_repair_tool_call_arguments('{"path": "/tmp/a.txt",', "t")) == {"path": "/tmp/a.txt"}

    def test_mismatched_closer_does_not_invent_content(self):
        """A closer the nesting cannot justify is left alone, so garbage stays garbage
        instead of being re-parsed into a plausible-but-wrong tool call."""
        assert _repair_tool_call_arguments('{"a": [1} , "b": 2', "t") == "{}"

    def test_brackets_inside_string_values_do_not_shift_the_close_order(self):
        raw = '{"code": "if x: { y }", "args": [1'
        assert json.loads(_repair_tool_call_arguments(raw, "t")) == {"code": "if x: { y }", "args": [1]}


class TestArgsReachTheAssembledToolCall:
    """The streaming assembler every provider funnels through hands over the repaired args."""

    def test_streamed_truncated_args_reach_the_tool_call(self):
        from agent.chat_completion_helpers import _StreamingCall

        acc = {0: {"id": "call_1", "type": "function",
                   "function": {"name": "skill_manage", "arguments": _TRUNCATED_NESTED_ARRAY}}}
        mock_tool_calls, truncated = _StreamingCall._assemble_tool_calls(acc, None)

        assert truncated is False, "a repairable truncation must not be flagged as a drop"
        parsed = json.loads(mock_tool_calls[0].function.arguments)
        assert parsed["operations"][0]["file_path"] == "references/timestamps-vacances.md"


class TestUnrepairableTruncationFailsLoudly:
    """Data cut mid-string cannot be invented; the fallback must be visible, not silent."""

    def test_unrepairable_arguments_fall_back_to_empty_object(self):
        assert _repair_tool_call_arguments('{"path": "/tmp/a.t', "t") == "{}"

    def test_unrepairable_fallback_logs_the_original_arguments(self, caplog):
        raw = '{"path": "/tmp/a.txt", "content": "hello wor'
        with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
            assert _repair_tool_call_arguments(raw, "write_file") == "{}"

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "dropping arguments must never be silent"
        assert any("write_file" in m for m in warnings), warnings
        assert any(raw in m for m in warnings), "the original bytes are the last copy of the data"
