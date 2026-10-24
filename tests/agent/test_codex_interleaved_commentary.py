"""Responses item correlation keeps commentary/private text off the final-answer rail (#69260)."""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream, run_codex_stream


def _message(item_id, phase=None, text=""):
    return {"type": "message", "id": item_id, "phase": phase,
            "content": [{"type": "output_text", "text": text}]}


def _added(item, index=None):
    event = {"type": "response.output_item.added", "item": item}
    if index is not None:
        event["output_index"] = index
    return event


def _delta(text, **aliases):
    return {"type": "response.output_text.delta", "delta": text, **aliases}


def _done(item, index=None):
    return {**_added(item, index), "type": "response.output_item.done"}


def _interleaved_events(other_type, alias):
    commentary = _message("commentary", "commentary")
    other = (_message("final", "final_answer", "Final answer.") if other_type == "message" else
             {"type": "function_call", "id": "announced-call", "call_id": "call", "name": "terminal",
              "arguments": ""})
    # The completed message omits text: delivery must use its own full buffer, not a final item's deltas.
    events = [_added(commentary, 0), _delta("First ", **alias), _added(other, 1)]
    if other_type == "message":
        events.append(_delta("Final answer.", item_id="final", output_index=1))
    else:
        events.append({"type": "response.function_call_arguments.delta", "item_id": "rotated-call",
                       "output_index": 1, "delta": '{"command":"pwd"}'})
    events.append(_delta("second", **alias))
    events.append(_done(commentary, 0))
    if other_type == "function_call":
        other = {**other, "id": "completed-call", "arguments": '{"command":"pwd"}'}
    events.append(_done(other, 1))
    return events


@pytest.mark.parametrize("event_shape", [dict, SimpleNamespace])
@pytest.mark.parametrize(("events", "commentary", "visible", "reasoning", "tool"), [
    *[
        pytest.param(_interleaved_events(other_type, alias), ["First second"],
                     ["Final answer."] if other_type == "message" else [], [],
                     other_type == "function_call", id=f"interleaved-{other_type}-{name}")
        for other_type in ("message", "function_call")
        for name, alias in (("id", {"item_id": "commentary"}), ("index", {"output_index": 0}),
                            ("both", {"item_id": "commentary", "output_index": 0}),
                            ("rotated-id", {"item_id": "rotated-message", "output_index": 0}))
    ],
    pytest.param([
        _added(_message("first", "commentary"), 0), _delta("First ", item_id="first"),
        _added(_message("second", "commentary"), 1), _delta("Other", output_index=1),
        _delta("second", output_index=0), _done(_message("second", "commentary"), 1),
        _done(_message("first", "commentary"), 0),
    ], ["Other", "First second"], [], [], False, id="two-commentary-buffers"),
    pytest.param([
        _added({"type": "message", "phase": "commentary"}), _delta("Public preface."),
        _added({"type": "message", "phase": "analysis"}), _added({"type": "function_call"}),
        _delta("PRIVATE"),
    ], ["Public preface."], [], ["PRIVATE"], False, id="unaliased-private-after-tool"),
    pytest.param([
        _added(_message("analysis", "analysis")), _delta("PRIVATE", output_index=0),
    ], [], [], ["PRIVATE"], False, id="unmatched-index"),
    pytest.param([
        _added({"type": "message", "phase": "analysis"}, 0), _delta("PRIVATE", item_id="analysis"),
    ], [], [], ["PRIVATE"], False, id="unmatched-id"),
    pytest.param([
        _added(_message("analysis", "analysis"), 0), _added({"type": "function_call", "id": "tool"}, 1),
        _delta("PRIVATE", item_id="unknown", output_index=2),
    ], [], [], ["PRIVATE"], False, id="unmatched-private-after-tool"),
    pytest.param([
        _added(_message("commentary", "commentary"), 0), _delta("Public preface.", item_id="commentary"),
        _delta("PRIVATE", item_id="unknown", output_index=2), _done(_message("commentary", "commentary"), 0),
    ], ["Public preface."], [], ["PRIVATE"], False, id="unmatched-does-not-contaminate-commentary"),
    *[
        pytest.param([_delta("Standalone final.", **alias)], [], ["Standalone final."], [], False,
                     id=f"standalone-unphased-{name}")
        for name, alias in (("none", {}), ("id", {"item_id": "final"}), ("index", {"output_index": 0}))
    ],
])
def test_message_rails_follow_item_identity(events, commentary, visible, reasoning, tool, event_shape):
    received_commentary, received_visible, received_reasoning = [], [], []
    first_deltas = []
    events = [*events, {"type": "response.completed", "response": {"status": "completed"}}]
    if event_shape is SimpleNamespace:
        events = [SimpleNamespace(**event) for event in events]
    response = _consume_codex_event_stream(
        events, model="diagnostic", on_text_delta=received_visible.append,
        on_reasoning_delta=received_reasoning.append, on_commentary_message=received_commentary.append,
        on_first_delta=lambda: first_deltas.append(True),
    )
    assert received_commentary == commentary
    assert received_visible == visible
    assert received_reasoning == reasoning
    assert response.output_text == "".join(visible)
    assert first_deltas == ([True] if visible else [])
    if tool:
        assert response.output[-1] == {
            "type": "function_call", "id": "completed-call", "call_id": "call", "name": "terminal",
            "arguments": '{"command":"pwd"}',
        }
        assert [item["type"] for item in response.output] == ["message", "function_call"]


@pytest.mark.parametrize(("show_commentary", "has_callback"), [(True, True), (False, True), (True, False)])
def test_concrete_response_delivers_only_commentary_through_real_relay(show_commentary, has_callback):
    commentary, visible, reasoning = [], [], []
    response = SimpleNamespace(
        output=[_message("commentary", "commentary", "Inspecting the logs."),
                _message("analysis", "analysis", "PRIVATE"),
                _message("final", "final_answer", "Final answer.")],
        output_text="Final answer.", status="completed",
    )
    # Real run_codex_stream -> relay_llm.stream -> ManagedLlmStream concrete-response path;
    # only the provider edge is substituted, so no network/provider request is made.
    agent = SimpleNamespace(
        session_id="", provider="openai-codex", model="diagnostic", show_commentary=show_commentary,
        interim_assistant_callback=commentary.append if has_callback else None,
        _fire_streamed_codex_commentary=commentary.append, _fire_stream_delta=visible.append,
        _fire_reasoning_delta=reasoning.append, _touch_activity=lambda *_: None,
        _client_log_context=lambda: "", _interrupt_requested=False,
    )
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return response

    returned = run_codex_stream(agent, {"model": "diagnostic", "input": "inspect"},
                                client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert returned is response
    assert commentary == (["Inspecting the logs."] if show_commentary and has_callback else [])
    assert visible == reasoning == []
    assert requests == [{"model": "diagnostic", "input": "inspect", "stream": True}]
