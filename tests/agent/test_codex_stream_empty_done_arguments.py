"""A Responses ``output_item.done`` for a function call can carry ``arguments: ""`` after the arguments
were streamed as deltas; the call must keep the streamed arguments, not run with ``{}``."""

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _events(done_arguments, *, as_objects=False):
    item = lambda args: {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
                         "arguments": args, "status": "completed"}
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": item("")},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": '{"city":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": '"Paris"}'},
        {"type": "response.output_item.done", "output_index": 0, "item": item(done_arguments)},
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "usage": None}},
    ]
    if as_objects:
        for event in events:
            if "item" in event:
                event["item"] = SimpleNamespace(**event["item"])
    return events


def _arguments(output_item):
    return output_item["arguments"] if isinstance(output_item, dict) else output_item.arguments


def test_empty_done_arguments_keep_the_streamed_arguments():
    for as_objects in (False, True):
        response = _consume_codex_event_stream(iter(_events("", as_objects=as_objects)), model="gpt-5.5")
        assert [_arguments(o) for o in response.output] == ['{"city":"Paris"}']


def test_nonempty_done_arguments_still_win():
    response = _consume_codex_event_stream(iter(_events('{"city":"Rome"}')), model="gpt-5.5")
    assert [_arguments(o) for o in response.output] == ['{"city":"Rome"}']
