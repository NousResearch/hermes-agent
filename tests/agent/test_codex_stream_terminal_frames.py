"""Which terminal frame ended a Codex Responses stream decides the assembled response.

``response.completed`` settles announced-but-unconfirmed function calls;
``response.incomplete`` and ``response.failed`` keep only confirmed items but
still let an announced answer message be recovered from its text deltas; a
stream with no terminal frame recovers nothing beside a pending call.
"""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


def _answer_then_pending_call():
    return [
        SimpleNamespace(type="response.output_item.added", output_index=0,
                        item=SimpleNamespace(type="message", id="msg_1", role="assistant")),
        SimpleNamespace(type="response.output_text.delta", item_id="msg_1", delta="Checking the weather."),
        SimpleNamespace(type="response.output_item.added", output_index=1,
                        item=SimpleNamespace(type="function_call", id="fc_1", call_id="call_1",
                                             name="get_weather", arguments="")),
        SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_1", output_index=1,
                        delta='{"city": "SF"}'),
    ]


def _types(final):
    return [getattr(item, "type", "") for item in final.output]


def test_completed_frame_settles_the_pending_call_after_the_answer():
    events = _answer_then_pending_call() + [SimpleNamespace(
        type="response.completed", response=SimpleNamespace(id="resp_1", status="completed"))]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert _types(final) == ["message", "function_call"]
    assert final.output[1].arguments == '{"city": "SF"}'
    assert (final.status, final.id) == ("completed", "resp_1")


def test_incomplete_frame_recovers_the_answer_but_not_the_pending_call():
    details = SimpleNamespace(reason="max_output_tokens")
    events = _answer_then_pending_call() + [SimpleNamespace(
        type="response.incomplete",
        response=SimpleNamespace(status="incomplete", incomplete_details=details))]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert _types(final) == ["message"]
    assert final.output[0].content[0].text == "Checking the weather."
    assert final.status == "incomplete"
    assert final.incomplete_details is details


def test_failed_frame_reports_status_and_error_without_settling():
    error = SimpleNamespace(code="server_error", message="boom")
    events = _answer_then_pending_call() + [SimpleNamespace(
        type="response.failed", response=SimpleNamespace(status="failed", error=error))]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert _types(final) == ["message"]
    assert final.status == "failed"
    assert final.error is error


@pytest.mark.parametrize("event_type", ["response.incomplete", "response.failed", "response.completed"])
def test_empty_status_falls_back_to_the_frame_type(event_type):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="partial"),
        SimpleNamespace(type=event_type, response=SimpleNamespace(status="")),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert final.status == event_type.removeprefix("response.")


def test_no_terminal_frame_with_a_pending_call_raises():
    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        _consume_codex_event_stream(_answer_then_pending_call(), model="gpt-test")


def test_no_terminal_frame_with_plain_text_reports_completed():
    final = _consume_codex_event_stream(
        [SimpleNamespace(type="response.output_text.delta", delta="cut off")], model="gpt-test")
    assert _types(final) == ["message"]
    assert final.status == "completed"
