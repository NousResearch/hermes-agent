"""Answer text a Responses stream sends beside a tool call, without that message item's done event,
must reach the assistant turn (#143893).

The assembler only synthesized a message from buffered text deltas when the stream carried no tool
call. With a commentary item done and a call pending, the normalizer's ``output_text`` fallback is
skipped too, so the narration vanished from the turn: no bubble, an empty stored assistant row.
"""

from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream

NARRATION = "Let me read the file first."


def _stream(with_commentary: bool):
    msg = NS(id="msg_1", type="message", role="assistant", status="in_progress", content=[])
    call = NS(id="fc_1", type="function_call", call_id="call_1", name="read_file", arguments="",
              status="in_progress")
    done_call = NS(id="fc_1", type="function_call", call_id="call_1", name="read_file",
                   arguments='{"path": "a.txt"}', status="completed")
    events = [NS(type="response.created", response=NS(id="resp_1"))]
    if with_commentary:
        commentary = NS(id="msg_c", type="message", role="assistant", status="completed", phase="commentary",
                        content=[NS(type="output_text", text="Planning.")])
        events += [NS(type="response.output_item.added", output_index=0, item=NS(**{**vars(commentary), "content": []})),
                   NS(type="response.output_text.delta", item_id="msg_c", output_index=0, delta="Planning."),
                   NS(type="response.output_item.done", output_index=0, item=commentary)]
    events += [
        NS(type="response.output_item.added", output_index=1, item=call),
        NS(type="response.output_item.added", output_index=2, item=msg),
        NS(type="response.output_text.delta", item_id="msg_1", output_index=2, delta=NARRATION),
        NS(type="response.function_call_arguments.delta", item_id="fc_1", output_index=1, delta='{"path": "a.txt"}'),
        NS(type="response.output_item.done", output_index=1, item=done_call),
        NS(type="response.completed", response=NS(id="resp_1", status="completed", output=None,
                                                  usage=NS(input_tokens=1, output_tokens=1, total_tokens=2))),
    ]
    return events


def test_narration_beside_a_commentary_item_and_a_call_reaches_the_turn():
    response = _consume_codex_event_stream(iter(_stream(with_commentary=True)), model="m")
    assistant, _finish = _normalize_codex_response(response)
    assert NARRATION in (assistant.content or "")
    assert [c.function.name for c in assistant.tool_calls] == ["read_file"]


def test_narration_beside_a_call_is_carried_once():
    response = _consume_codex_event_stream(iter(_stream(with_commentary=False)), model="m")
    assistant, _finish = _normalize_codex_response(response)
    assert (assistant.content or "").count(NARRATION) == 1
    assert [c.function.name for c in assistant.tool_calls] == ["read_file"]


def test_a_stream_cut_before_its_terminal_frame_still_fails_for_a_retry():
    """Answer announced, partial text, a call announced, then the connection ends with no terminal frame: the
    turn must fail and retry, never complete on the cut-off sentence with the announced call dropped. Later
    commentary named by its item id stays narration, never fused into the recovered answer."""
    cut = _stream(with_commentary=False)[:4]  # created, call added, answer added, partial answer delta
    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        _consume_codex_event_stream(iter(cut), model="m")

    commentary = NS(id="msg_c", type="message", role="assistant", status="in_progress", phase="commentary", content=[])
    events = _stream(with_commentary=False)
    events[4:4] = [NS(type="response.output_item.added", output_index=3, item=commentary),
                   NS(type="response.output_item.added", output_index=4,
                      item=NS(id="fc_2", type="function_call", call_id="call_2", name="read_file", arguments="",
                              status="in_progress")),
                   NS(type="response.output_text.delta", item_id="msg_c", output_index=3, delta="more planning")]
    assistant, _finish = _normalize_codex_response(_consume_codex_event_stream(iter(events), model="m"))
    assert "more planning" not in (assistant.content or "")
    assert NARRATION in (assistant.content or "")
