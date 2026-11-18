"""Responses-wire parity with Chat Completions for truncated tool-call arguments.

A ``function_call`` item whose ``arguments`` were cut mid-JSON (output cap reached while
the model was still writing them) must not be executed. Chat Completions drops such a
batch in ``_StreamingCall._assemble_tool_calls``; the Responses wire used to hand the
broken call straight to the tool executor, which answered with an "Invalid tool
arguments" tool RESULT and let the turn continue — and because the send-path
canonicalizer rewrites an unrepairable payload to ``"{}"``, the next request showed the
model a valid empty object plus an error about it, so it re-emitted the same call.
"""

import json
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _normalize_codex_response

# Cut mid-string, exactly how an output cap lands inside a large argument payload.
_TRUNCATED_ARGS = '{"path": "notes.md", "content": "# Heading\\n\\nA long body that never fini'
_REPAIRABLE_ARGS = '{"path": "notes.md", "content": "ok",}'  # trailing comma
_VALID_ARGS = '{"path": "notes.md", "content": "ok"}'


def _function_call(name="file_write", arguments=_VALID_ARGS, call_id="call_1"):
    return SimpleNamespace(
        type="function_call", id="fc_1", call_id=call_id, name=name,
        arguments=arguments, status="completed",
    )


def _response(*output_items, status="completed"):
    return SimpleNamespace(status=status, output=list(output_items))


def test_truncated_tool_call_arguments_are_not_executed():
    """The contract: a call the tool executor could only reject never reaches it."""
    message, finish_reason = _normalize_codex_response(
        _response(_function_call(arguments=_TRUNCATED_ARGS))
    )

    assert finish_reason == "incomplete"
    assert not message.tool_calls


def test_repairable_arguments_are_repaired_and_still_executed():
    """Only unrepairable payloads are dropped; a trailing comma is fixed in place."""
    message, finish_reason = _normalize_codex_response(
        _response(_function_call(arguments=_REPAIRABLE_ARGS))
    )

    assert finish_reason == "tool_calls"
    assert len(message.tool_calls) == 1
    assert json.loads(message.tool_calls[0].function.arguments) == {
        "path": "notes.md", "content": "ok",
    }


def test_valid_arguments_pass_through_byte_for_byte():
    message, finish_reason = _normalize_codex_response(
        _response(_function_call(arguments=_VALID_ARGS))
    )

    assert finish_reason == "tool_calls"
    assert message.tool_calls[0].function.arguments == _VALID_ARGS


@pytest.mark.parametrize("empty_args", ["", "   ", "{}"])
def test_zero_argument_calls_stay_executable(empty_args):
    """Backends that send no argument deltas for a zero-arg tool must keep working."""
    message, finish_reason = _normalize_codex_response(
        _response(_function_call(name="get_time", arguments=empty_args))
    )

    assert finish_reason == "tool_calls"
    assert len(message.tool_calls) == 1


def test_custom_tool_call_input_is_never_json_validated():
    """``custom_tool_call.input`` is free-form text (code, prose) — not JSON, and not subject
    to the truncation guard."""
    message, finish_reason = _normalize_codex_response(
        _response(SimpleNamespace(
            type="custom_tool_call", id="ctc_1", call_id="call_1", name="python_exec",
            input="print('hello')  # not JSON, and that is fine", status="completed",
        ))
    )

    assert finish_reason == "tool_calls"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.arguments == "print('hello')  # not JSON, and that is fine"


def test_one_truncated_call_drops_the_whole_batch():
    """Chat Completions drops the batch, not the single bad call — the surviving calls of a
    partial batch may depend on the one that was cut."""
    message, finish_reason = _normalize_codex_response(
        _response(
            _function_call(name="read_file", arguments=_VALID_ARGS, call_id="call_1"),
            _function_call(name="file_write", arguments=_TRUNCATED_ARGS, call_id="call_2"),
        )
    )

    assert finish_reason == "incomplete"
    assert not message.tool_calls


def test_assistant_text_alongside_a_truncated_call_is_kept():
    """Dropping the call must not discard what the model already said."""
    message, finish_reason = _normalize_codex_response(
        _response(
            SimpleNamespace(
                type="message", role="assistant", status="completed",
                content=[SimpleNamespace(type="output_text", text="Writing the file now.")],
            ),
            _function_call(arguments=_TRUNCATED_ARGS),
        )
    )

    assert finish_reason == "incomplete"
    assert message.content == "Writing the file now."
    assert not message.tool_calls


def test_both_wires_agree_on_the_same_truncated_payload():
    """Same argument string, both transports: neither may present it as an executable call."""
    from agent.chat_completion_helpers import _StreamingCall

    chat_calls, chat_truncated = _StreamingCall._assemble_tool_calls(
        {0: {"id": "call_1", "type": "function",
             "function": {"name": "file_write", "arguments": _TRUNCATED_ARGS}}},
        "length",
    )
    _, responses_finish_reason = _normalize_codex_response(
        _response(_function_call(arguments=_TRUNCATED_ARGS))
    )

    assert chat_truncated is True
    assert chat_calls  # chat keeps the items but stamps the turn "length"
    assert responses_finish_reason == "incomplete"
