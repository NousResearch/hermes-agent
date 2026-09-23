"""Responses identity continuity: rotating wire IDs must not duplicate tool calls.

Reduced from a real Copilot stream: item IDs changed at every event while
output_index and added/done call_id stayed stable. No live API or tools run.
"""

import json

import httpx
import pytest
from openai import OpenAI

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream


def _item_event(phase, item_id, call_id, arguments, index: int | None = 0):
    event = {
        "type": f"response.output_item.{phase}",
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": "diagnostic_echo",
            "arguments": arguments,
            "status": "completed" if phase == "done" else "in_progress",
        },
    }
    if index is not None:
        event["output_index"] = index
    return event


def _argument_event(phase, item_id, value, index: int | None = 0):
    event = {
        "type": f"response.function_call_arguments.{phase}",
        "item_id": item_id,
        "delta" if phase == "delta" else "arguments": value,
    }
    if index is not None:
        event["output_index"] = index
    return event


def _replay(events):
    events = [
        *events,
        {
            "type": "response.completed",
            "response": {"id": "resp_test", "status": "completed", "output": None},
        },
    ]
    wire = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()
    # Exercise the installed SDK's actual SSE decoder, not stand-in model calls.
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=wire,
        )
    )
    with OpenAI(
        api_key="inert-test",
        base_url="https://offline.invalid",
        http_client=httpx.Client(transport=transport),
    ) as client:
        with client.responses.create(model="test", input="test", stream=True) as stream:
            final = _consume_codex_event_stream(stream, model="test")
    message, _ = _normalize_codex_response(final)
    return [
        (call.id, call.function.name, json.loads(call.function.arguments))
        for call in message.tool_calls
    ]


@pytest.mark.parametrize(
    "identity",
    [
        "stable",
        "rotating",
        "call-id-only",
        "pending-deltas",
        "pending-arguments-done",
        "pending-alias",
        "parallel",
    ],
)
def test_logical_calls_are_emitted_once(identity):
    index = None if identity == "call-id-only" else 0
    done_id = "announced" if identity == "stable" else "completed-alias"
    expected = [("call_one", "diagnostic_echo", {"value": "hello"})]
    streams = {
        "pending-deltas": [
            _argument_event("delta", "delta-alias-one", '{"value":'),
            _argument_event("delta", "delta-alias-two", '"hello"}'),
        ],
        "pending-arguments-done": [
            _argument_event("delta", "delta-alias", '{"value":"partial"}'),
            _argument_event("done", "arguments-done-alias", '{"value":"hello"}'),
        ],
        # A learned item alias must work when a later event omits output_index.
        "pending-alias": [
            _argument_event("delta", "delta-alias", '{"value":'),
            _argument_event("delta", "delta-alias", '"hello"}', None),
        ],
        "parallel": [
            _item_event("added", "announced-two", "call_two", "", 1),
            _item_event("added", "announced-zero", "call_zero", "", 2),
            _argument_event("delta", "one-alias", '{"value":', 0),
            _argument_event("delta", "two-alias", '{"value":', 1),
            _argument_event("delta", "one-next-alias", '"hello"}', 0),
            _argument_event("delta", "two-next-alias", '"partial"}', 1),
            # A done-confirmed sibling stays authoritative; the other two settle.
            _item_event("done", "two-done-alias", "call_two", '{"value":"second"}', 1),
        ],
    }
    default = [
        _argument_event("delta", "delta-alias", '{"value":"partial"}', index),
        _item_event("done", done_id, "call_one", '{"value":"hello"}', index),
    ]
    events = [
        _item_event("added", "announced", "call_one", "", index),
        *streams.get(identity, default),
    ]
    if identity == "parallel":
        expected += [
            ("call_two", "diagnostic_echo", {"value": "second"}),
            ("call_zero", "diagnostic_echo", {}),
        ]
    assert _replay(events) == expected


@pytest.mark.parametrize(
    "conflict", ["call-id", "output-index", "crossed-argument", "crossed-done"]
)
def test_conflicting_call_identity_is_rejected(conflict):
    events = [
        _item_event("added", "announced", "call_one", "", 0),
        _item_event("added", "announced-two", "call_two", "", 1),
    ]
    contradictions = {
        "call-id": _item_event("done", "new-id", "call_other", "{}", 0),
        "output-index": _item_event("done", "announced", "call_one", "{}", 2),
        "crossed-argument": _argument_event("delta", "announced", "{}", 1),
        "crossed-done": _item_event("done", "announced", "call_two", "{}", 1),
    }
    with pytest.raises(
        ValueError, match="Conflicting Responses function call identity"
    ):
        _replay([*events, contradictions[conflict]])


@pytest.mark.parametrize("aliases", [False, True], ids=["stable-ids", "rotating-ids"])
@pytest.mark.parametrize("index", [0, None], ids=["indexed", "no-index"])
def test_all_completed_calls_keep_announced_order(aliases, index):
    second_index = None if index is None else 1
    first_done = "done_first" if aliases else "start_first"
    second_done = "done_second" if aliases else "start_second"
    # Same tool name, different logical calls; completion order is reversed.
    events = [
        _item_event("added", "start_first", "call_first", "", index),
        _item_event("added", "start_second", "call_second", "", second_index),
        _item_event("done", second_done, "call_second", '{"step":2}', second_index),
        _item_event("done", first_done, "call_first", '{"step":1}', index),
    ]
    assert _replay(events) == [
        ("call_first", "diagnostic_echo", {"step": 1}),
        ("call_second", "diagnostic_echo", {"step": 2}),
    ]


@pytest.mark.parametrize("completed", [False, True], ids=["settled", "done"])
def test_legitimate_empty_argument_call_survives(completed):
    events = [_item_event("added", "empty_start", "call_empty", "", 0)]
    if completed:
        events.append(_item_event("done", "empty_start", "call_empty", "{}", 0))
    assert _replay(events) == [("call_empty", "diagnostic_echo", {})]
