"""Responses function-call identity across streamed item aliases.

Synthetic protocol witnesses, not captured production SSE.
"""
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream


def _call(kind, item_id, call_id, arguments="", index=None):
    event = SimpleNamespace(
        type=f"response.output_item.{kind}",
        item=SimpleNamespace(
            type="function_call", id=item_id, call_id=call_id,
            name="read_file", arguments=arguments,
        ),
    )
    if index is not None:
        event.output_index = index
    return event


def _terminal(status="completed"):
    return SimpleNamespace(
        type=f"response.{status}",
        response=SimpleNamespace(status=status, output=None),
    )


def _normalized_calls(events):
    response = _consume_codex_event_stream(events, model="gpt-test")
    message, _ = _normalize_codex_response(response)
    return [(call.id, call.function.arguments) for call in message.tool_calls]


@pytest.mark.parametrize("index", [None, 0])
def test_done_item_alias_confirms_semantic_call_once(index):
    events = [
        _call("added", "call_probe", "call_probe", index=index),
        _call("done", "fc_probe", "call_probe", '{"path":"/tmp/example"}', index),
        _terminal(),
    ]
    assert _normalized_calls(events) == [("call_probe", '{"path":"/tmp/example"}')]


@pytest.mark.parametrize("added_call_id", [None, "call_probe"])
def test_output_index_correlates_alias_when_call_id_is_missing(added_call_id):
    events = [
        _call("added", "opaque-added", added_call_id, index=0),
        _call("done", "opaque-done", "call_probe", '{"path":"/tmp/example"}', 0),
        _terminal(),
    ]
    assert _normalized_calls(events) == [("call_probe", '{"path":"/tmp/example"}')]


@pytest.mark.parametrize("indexes", [(None, None), (0, 1), (0, 0)])
def test_same_name_distinct_call_ids_remain_separate(indexes):
    first_index, second_index = indexes
    events = [
        _call("added", "added-a", "call_a", index=first_index),
        _call("added", "added-b", "call_b", index=second_index),
        _call("done", "done-a", "call_a", '{"path":"same"}', first_index),
        _call("done", "done-b", "call_b", '{"path":"same"}', second_index),
        _terminal(),
    ]
    assert _normalized_calls(events) == [
        ("call_a", '{"path":"same"}'), ("call_b", '{"path":"same"}'),
    ]


@pytest.mark.parametrize("indexes", [(None, None), (0, 1), (0, 0)])
@pytest.mark.parametrize("first_done", [False, True])
def test_alias_confirmation_keeps_announced_order(indexes, first_done):
    first_index, second_index = indexes
    events = [
        _call("added", "added-a", "call_a", '{"path":"a"}', first_index),
        _call("added", "added-b", "call_b", '{"path":"b"}', second_index),
        _call("done", "done-b", "call_b", '{"path":"b"}', second_index),
    ]
    if first_done:
        events.append(_call("done", "done-a", "call_a", '{"path":"a"}', first_index))
    assert _normalized_calls(events + [_terminal()]) == [
        ("call_a", '{"path":"a"}'), ("call_b", '{"path":"b"}'),
    ]


@pytest.mark.parametrize("kind", ["delta", "done"])
@pytest.mark.parametrize("identity", [{"output_index": 0}, {"call_id": "call_probe"}])
def test_argument_event_alias_updates_pending_call(kind, identity):
    argument_field = "delta" if kind == "delta" else "arguments"
    events = [
        _call("added", "added-probe", "call_probe", index=0),
        SimpleNamespace(
            type=f"response.function_call_arguments.{kind}", item_id="other-item-id",
            **identity, **{argument_field: '{"path":"/tmp/example"}'},
        ),
        _terminal(),
    ]
    assert _normalized_calls(events) == [("call_probe", '{"path":"/tmp/example"}')]


@pytest.mark.parametrize("done", [False, True])
def test_legitimate_zero_argument_call_is_preserved(done):
    events = [_call("added", "zero-added", "call_zero")]
    if done:
        events.append(_call("done", "zero-done", "call_zero", "{}"))
    assert _normalized_calls(events + [_terminal()]) == [("call_zero", "{}")]


@pytest.mark.parametrize("terminal_status", ["incomplete", "failed", None])
def test_unsuccessful_stream_does_not_settle_unsafe_pending_calls(terminal_status):
    events = [
        _call("added", "safe-added", "call_safe", index=0),
        _call("added", "unsafe-added", "call_unsafe", index=1),
        SimpleNamespace(
            type="response.function_call_arguments.delta", item_id="unsafe-alias",
            output_index=1, delta='{"path":',
        ),
        _call("done", "safe-done", "call_safe", "{}", 0),
    ]
    if terminal_status is not None:
        events.append(_terminal(terminal_status))
    response = _consume_codex_event_stream(events, model="gpt-test")
    assert [item.call_id for item in response.output] == ["call_safe"]
    if terminal_status == "failed":
        with pytest.raises(RuntimeError, match="status 'failed'"):
            _normalize_codex_response(response)
    else:
        assert _normalized_calls(events) == [("call_safe", "{}")]


def test_dict_events_confirm_alias_without_duplicate_output():
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": {
            "type": "function_call", "id": "opaque-a", "call_id": "call_probe",
            "name": "read_file", "arguments": "",
        }},
        {"type": "response.output_item.done", "output_index": 0, "item": {
            "type": "function_call", "id": "opaque-b", "call_id": "call_probe",
            "name": "read_file", "arguments": '{"path":"/tmp/example"}',
        }},
        {"type": "response.completed", "response": {"status": "completed", "output": None}},
    ]
    response = _consume_codex_event_stream(events, model="gpt-test")
    assert response.output == [events[1]["item"]]


@pytest.mark.parametrize("repeated_id", ["fc_probe", "opaque-alias"])
def test_repeated_done_emits_one_call(repeated_id):
    events = [
        _call("done", "fc_probe", "call_probe", '{"path":"/tmp/example"}'),
        _call("done", repeated_id, "call_probe", '{"path":"/tmp/example"}'),
        _terminal(),
    ]
    assert _normalized_calls(events) == [("call_probe", '{"path":"/tmp/example"}')]
