"""Regression: settle pending function calls when a Responses stream
completes successfully without ``response.output_item.done``.

Some OpenAI-compatible backends (see anomalyco/opencode#37159, fixed in
anomalyco/opencode#43575) omit per-item ``response.output_item.done``
events on a successful completion. ``_consume_codex_event_stream`` assembles
its final ``output`` purely from ``.done`` items, so a function call that
was announced via ``response.output_item.added`` and streamed argument
deltas is silently dropped: the turn ends with an empty output and the tool
never executes.

These tests pin the desired behavior (settle the pending call from the
accumulated stream state at the terminal event, mirroring the opencode fix
semantics). They fail against current ``main`` — that red state is the
reproduction for the linked issue.

Review hardening (PR #92767): settlement additionally requires an observed
``response.completed`` frame (not the ``terminal_status`` default), empty
arguments canonicalize to ``{}``, and settled calls merge with ``.done``
items in ``output_index`` order.
"""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


def _stream_completed_without_done():
    """Successful Responses stream whose only function call never receives
    ``response.output_item.done`` (backend omits it; terminal response
    carries ``output=None`` so there is nothing to reconstruct from)."""
    return [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="get_weather",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta='{"city"',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta=': "SF"}',
        ),
        # NOTE: no response.output_item.done for fc_1.
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_1",
                status="completed",
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
                output=None,
            ),
        ),
    ]


def test_completed_without_done_settles_pending_function_call():
    final = _consume_codex_event_stream(_stream_completed_without_done(), model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls, (
        "function_call announced via output_item.added (+ argument deltas) "
        "was silently dropped on successful completion without "
        "output_item.done; the tool never executes"
    )
    settled = calls[0]
    assert getattr(settled, "name", None) == "get_weather"
    assert getattr(settled, "arguments", None) == '{"city": "SF"}'
    assert final.status == "completed"


def test_control_stream_with_done_still_authoritative():
    """Sanity: the same stream WITH output_item.done must keep working
    (the fix must not regress the normal path or override .done data)."""
    events = _stream_completed_without_done()
    done = SimpleNamespace(
        type="response.output_item.done",
        output_index=0,
        item=SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"city": "SF"}',
        ),
    )
    events.insert(-1, done)
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls and calls[0].arguments == '{"city": "SF"}'
    assert final.status == "completed"


def test_no_terminal_frame_does_not_settle_pending_function_call():
    """EOF/interruption before any terminal frame must NOT settle the pending
    call: unconfirmed stream state must not become executable authority.
    With no ``.done`` items and no text, the existing truncation guard fires
    instead of normalizing the partial stream into tool calls."""
    events = _stream_completed_without_done()[:-1]  # drop response.completed
    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        _consume_codex_event_stream(events, model="gpt-test")


def test_zero_argument_call_settles_with_canonical_empty_object():
    """A call announced with zero argument deltas settles with ``{}`` instead
    of ``""`` — the argument parser rejects ``json.loads("")``, which would
    keep zero-argument tools unexecutable (review P2)."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="list_tools",
                arguments="",
            ),
        ),
        # NOTE: zero argument deltas and no output_item.done for fc_1.
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls, "zero-argument pending call was dropped instead of settled"
    assert calls[0].arguments == "{}"


def test_mixed_pending_and_done_calls_preserve_output_index_order():
    """A pending call at output_index 0 that never receives ``.done`` must
    still come before a completed call at output_index 1: appending settled
    calls at the tail inverts dependent side effects (review P1)."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="first_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_a",
            output_index=0,
            delta='{"step": 1}',
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_b",
                call_id="call_b",
                name="second_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_b",
            output_index=1,
            delta='{"step": 2}',
        ),
        # fc_b is confirmed by .done; fc_a never is.
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_b",
                call_id="call_b",
                name="second_tool",
                arguments='{"step": 2}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert [getattr(call, "name", None) for call in calls] == [
        "first_tool",
        "second_tool",
    ]
    assert calls[0].arguments == '{"step": 1}'
    assert calls[1].arguments == '{"step": 2}'


def test_missing_done_output_index_preserves_observed_order():
    """A missing output_index must use event order as a fallback, not sort
    the item after every indexed pending call."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="first_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                role="assistant",
                status="completed",
                content=[],
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert [getattr(item, "type", None) for item in final.output] == [
        "function_call",
        "message",
    ]
