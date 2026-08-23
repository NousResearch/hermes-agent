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
"""

from types import SimpleNamespace

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
