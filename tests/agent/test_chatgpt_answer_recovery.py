"""ChatGPT plan answer recovery preserves explicit stream completion and tool ownership."""

from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream


def _answer_and_tool_events(terminal, *, call_done=True):
    events = [
        NS(type="response.output_item.added", output_index=0, item=NS(
            id="msg_answer", type="message", role="assistant", status="in_progress", content=[])),
        NS(type="response.output_text.delta", item_id="msg_answer", delta="Reading the note."),
        NS(type="response.output_item.added", output_index=1, item=NS(
            id="msg_commentary", type="message", role="assistant", phase="commentary", content=[])),
        NS(type="response.output_item.added", output_index=2, item=NS(
            id="fc_read", type="function_call", call_id="call_read", name="read_file",
            arguments="", status="in_progress")),
        NS(type="response.output_text.delta", item_id="msg_commentary", delta="Private planning."),
        NS(type="response.function_call_arguments.delta", item_id="fc_read", delta='{"path":"note.txt"}'),
    ]
    if call_done:
        events.append(NS(type="response.output_item.done", output_index=2, item=NS(
            id="fc_read", type="function_call", call_id="call_read", name="read_file",
            arguments='{"path":"note.txt"}', status="completed")))
    if terminal:
        events.append(NS(type=f"response.{terminal}", response=NS(
            id="resp_plan", status=terminal, incomplete_details={"reason": "max_output_tokens"})))
    return events


@pytest.mark.parametrize("terminal", ["completed", "incomplete"])
@pytest.mark.parametrize("call_done", [True, False])
def test_plan_recovers_announced_answer_beside_tool_without_promoting_incomplete(terminal, call_done):
    response = _consume_codex_event_stream(
        _answer_and_tool_events(terminal, call_done=call_done), model="m", require_completed=True)
    assistant, _finish = _normalize_codex_response(response)
    assert assistant.content == "Reading the note."
    expected_calls = ["read_file"] if call_done or terminal == "completed" else []
    assert [call.function.name for call in (assistant.tool_calls or [])] == expected_calls
    assert response.status == terminal
    if terminal == "incomplete":
        assert response.incomplete_details == {"reason": "max_output_tokens"}


def test_plan_rejects_missing_terminal_even_with_completed_tool_and_answer_text():
    with pytest.raises(RuntimeError, match="response.completed"):
        _consume_codex_event_stream(_answer_and_tool_events(None), model="m", require_completed=True)
