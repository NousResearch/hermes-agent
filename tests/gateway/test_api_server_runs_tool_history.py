"""Caller-supplied ``conversation_history`` keeps its tool scaffolding on ``/v1/runs``.

A client that replays its own transcript (the WebUI does, so edit/retry/rewind stay authoritative)
used to have every entry flattened to ``{"role", "content"}``: assistant ``tool_calls`` and tool
``tool_call_id`` were dropped, so the model lost sight of what its tools did between turns.  These
tests pin the projection contract, not the run machinery behind it.
"""

import pytest

from gateway.platforms.api_server_runs import _resolve_conversation_history

CALL = {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}


def _openai_error(message: str, **_kwargs):
    return {"error": {"message": message}}


def _resolve(history):
    return _resolve_conversation_history(None, {"conversation_history": history}, "next", _openai_error=_openai_error)


def test_tool_rows_survive_and_plain_rows_are_unchanged():
    history, _, _, error = _resolve([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "content": "out", "tool_call_id": "call_1", "tool_name": "terminal"},
        {"role": "assistant", "content": "answer"},
    ])

    assert error is None
    assert history == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "content": "out", "tool_call_id": "call_1", "tool_name": "terminal"},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.parametrize("bad_row", [
    {"role": "tool", "content": "x"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
    {"role": "assistant", "content": "", "tool_calls": "nope"},
])
def test_malformed_tool_scaffolding_is_rejected(bad_row):
    _, _, _, error = _resolve([bad_row])

    assert error is not None
    assert error.status == 400
