"""Tests for tool call argument repair in the streaming assembly path.

The streaming path (run_agent._call_chat_completions) assembles tool call
deltas into full arguments.  When a model truncates or malforms the JSON
(e.g. GLM-5.1 via Ollama), the assembly path used to pass the broken JSON
straight through — setting has_truncated_tool_args but NOT repairing it.
That triggered the truncation handler to kill the session with /new required.

The fix: repair arguments in the streaming assembly path using
_repair_tool_call_arguments() so repairable malformations (trailing commas,
unclosed brackets, Python None) don't kill the session.
"""

import json

from agent.chat_completion_helpers import _StreamingCall
from agent.message_sanitization import _repair_tool_call_arguments


def _accumulated(arguments: str) -> dict:
    return {0: {"id": "call_1", "type": "function", "function": {"name": "edit", "arguments": arguments}}}


def test_streamed_misnested_arguments_are_repaired_not_flagged_truncated():
    # deepseek-v4-flash via a portal closes an array of objects with "}}" where "}]}" is
    # required (#115061); the assembler must hand the tool a repaired call, not drop it
    # as an unrepairable "{}" flagged as truncated args.
    raw = '{"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}}'
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated(raw), "tool_calls")
    assert truncated is False
    assert json.loads(calls[0].function.arguments) == {
        "edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}],
    }


def test_streamed_mid_string_cut_stays_unrepairable():
    # Control: content cut inside a string is unrecoverable and must still be flagged,
    # never guessed into a wrong tool call.
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated('{"q": "unterminated string'), None)
    assert truncated is True
    assert calls[0].function.arguments == '{"q": "unterminated string'


def _accumulated_for(name: str, arguments: str) -> dict:
    return {0: {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}}


def test_cut_argument_bag_never_executes_an_effect_capable_tool():
    # Cut right after content's closing quote: closing the brace yields a complete-looking
    # write_file that silently lacks everything after the cut (qwen-code#12970 class). Only
    # a no-effect tool may run the closed repair; an effect-capable tool keeps the raw bag
    # and takes the truncation path (provider finish_reason trusted, usage not disproving).
    cut = '{"path": "a.py", "content": "line one\\n"'
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated_for("write_file", cut), "stop")
    assert truncated is True
    assert calls[0].function.arguments == cut
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated_for("read_file", cut), "stop")
    assert truncated is False
    assert json.loads(calls[0].function.arguments) == {"path": "a.py", "content": "line one\n"}


def test_disproved_truncation_keeps_raw_args_and_provider_finish_reason():
    # Usage well under budget on a normal stop: not a cut, so the bag is neither repaired nor
    # stamped "length" (which would spend four boosted max_tokens retries and blame the
    # output cap); raw args + provider finish_reason route it to the malformed-JSON recovery.
    fused = '{"path": "a.py"}{"path": "b.py", "content": "x"'
    calls, truncated = _StreamingCall._assemble_tool_calls(
        _accumulated_for("write_file", fused), "stop", truncation_disproved=True)
    assert truncated is False
    assert calls[0].function.arguments == fused


class TestStreamingAssemblyRepair:
    """Verify that _repair_tool_call_arguments is applied to streaming tool
    call arguments before they're assembled into mock_tool_calls.

    These tests verify the REPAIR FUNCTION itself works correctly for the
    cases that arise during streaming assembly.  Integration tests that
    exercise the full streaming path are in run_agent.py's streaming tests.
    """

    # -- Truncation cases (most common streaming failure) --

    def test_truncated_object_no_close_brace(self):
        """Model stops mid-JSON, common with output length limits."""
        raw = '{"command": "ls -la", "timeout": 30'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed["command"] == "ls -la"
        assert parsed["timeout"] == 30



    # -- Trailing comma cases (Ollama/GLM common) --



    # -- Python None from model output --


    # -- Empty arguments (some models emit empty string) --

    def test_empty_string(self):
        assert _repair_tool_call_arguments("", "test") == "{}"


    # -- Already-valid JSON passes through unchanged --


    # -- Extra closing brackets (rare but happens) --


    # -- Real-world GLM-5.1 truncation pattern --


