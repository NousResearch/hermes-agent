"""Regression: a second phase-less ``message`` output item must start a new
text segment (issue #121797).

Grok (xAI Responses) often emits a short status ``message`` item, then
reasons, then emits the answer as another ``message`` item in the same
response. Neither item carries a ``phase``, so every ``output_text.delta``
streamed into the same text channel with no separator: the gateway shows
the status line glued to the answer, and ``result().output_text`` joins the
two segments the same way. A new final-answer message item after text was
already streamed should open a new paragraph — mirroring the blank line the
assembler already inserts when a reasoning summary's ``summary_index``
changes.
"""

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _message_item(item_id, *, phase=None):
    item = SimpleNamespace(type="message", id=item_id, role="assistant")
    if phase is not None:
        item.phase = phase
    return item


def _two_segment_stream():
    """Status message → reasoning item → answer message, all in one response."""
    return [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        SimpleNamespace(
            type="response.output_item.added", output_index=0, item=_message_item("msg_1"),
        ),
        SimpleNamespace(type="response.output_text.delta", item_id="msg_1", output_index=0, delta="Ik zoek een goed recept."),
        SimpleNamespace(type="response.output_text.delta", item_id="msg_1", output_index=0, delta=" Voor 6 personen."),
        SimpleNamespace(
            type="response.output_item.done", output_index=0, item=_message_item("msg_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added", output_index=1,
            item=SimpleNamespace(type="reasoning", id="rs_1"),
        ),
        SimpleNamespace(
            type="response.output_item.done", output_index=1,
            item=SimpleNamespace(type="reasoning", id="rs_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added", output_index=2, item=_message_item("msg_2"),
        ),
        SimpleNamespace(
            type="response.output_text.delta", item_id="msg_2", output_index=2,
            delta="Klassiek pannenkoekenrecept voor 6 personen.",
        ),
        SimpleNamespace(
            type="response.output_item.done", output_index=2, item=_message_item("msg_2"),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_1", status="completed", output=None,
                usage=SimpleNamespace(input_tokens=10, output_tokens=9, total_tokens=19),
            ),
        ),
    ]


def test_second_message_item_opens_a_new_streamed_segment():
    streamed = []
    final = _consume_codex_event_stream(
        _two_segment_stream(), model="grok-4.7",
        on_text_delta=streamed.append, on_first_delta=lambda: None,
    )

    # The answer segment's first delta carries the paragraph break; the first
    # segment streams untouched.
    assert "".join(streamed) == (
        "Ik zoek een goed recept. Voor 6 personen."
        "\n\nKlassiek pannenkoekenrecept voor 6 personen."
    )
    # The assembled final text is separated too, not glued.
    assert final.output_text == (
        "Ik zoek een goed recept. Voor 6 personen."
        "\n\nKlassiek pannenkoekenrecept voor 6 personen."
    )


def test_single_message_item_streams_without_separator():
    events = [
        ev for ev in _two_segment_stream()
        if not (getattr(getattr(ev, "item", None), "id", "") in ("msg_2", "rs_1")
                or getattr(ev, "item_id", "") == "msg_2")
    ]
    streamed = []
    final = _consume_codex_event_stream(
        events, model="grok-4.7",
        on_text_delta=streamed.append, on_first_delta=lambda: None,
    )

    assert "".join(streamed) == "Ik zoek een goed recept. Voor 6 personen."
    assert final.output_text == "Ik zoek een goed recept. Voor 6 personen."


def test_commentary_then_final_message_streams_without_separator():
    """Harmony commentary never reaches ``text_deltas``; the first final
    message after it is segment one, not segment two."""
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        SimpleNamespace(
            type="response.output_item.added", output_index=0, item=_message_item("msg_c", phase="commentary"),
        ),
        SimpleNamespace(
            type="response.output_text.delta", item_id="msg_c", output_index=0, delta="Even nadenken…",
        ),
        SimpleNamespace(
            type="response.output_item.done", output_index=0, item=_message_item("msg_c", phase="commentary"),
        ),
        SimpleNamespace(
            type="response.output_item.added", output_index=1, item=_message_item("msg_1"),
        ),
        SimpleNamespace(
            type="response.output_text.delta", item_id="msg_1", output_index=1, delta="Het antwoord.",
        ),
        SimpleNamespace(
            type="response.output_item.done", output_index=1, item=_message_item("msg_1"),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    streamed, commentary = [], []
    final = _consume_codex_event_stream(
        events, model="grok-4.7",
        on_text_delta=streamed.append, on_first_delta=lambda: None,
        on_commentary_message=commentary.append,
    )

    assert streamed == ["Het antwoord."]
    assert final.output_text == "Het antwoord."
    assert commentary == ["Even nadenken…"]


def test_tool_call_turn_separates_segments_in_assembled_text():
    """With ``has_tool_calls`` the text callback is suppressed, but the
    assembled ``output_text`` still must not glue two message items."""
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        SimpleNamespace(
            type="response.output_item.added", output_index=0,
            item=SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="get_weather", arguments=""),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta", item_id="fc_1", output_index=0, delta='{"city": "SF"}',
        ),
        SimpleNamespace(
            type="response.output_item.added", output_index=1, item=_message_item("msg_1"),
        ),
        SimpleNamespace(type="response.output_text.delta", item_id="msg_1", output_index=1, delta="Eerste deel."),
        SimpleNamespace(
            type="response.output_item.added", output_index=2, item=_message_item("msg_2"),
        ),
        SimpleNamespace(type="response.output_text.delta", item_id="msg_2", output_index=2, delta="Tweede deel."),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    streamed = []
    final = _consume_codex_event_stream(
        events, model="grok-4.7",
        on_text_delta=streamed.append, on_first_delta=lambda: None,
    )

    assert streamed == []  # tool-call turns suppress streaming
    assert final.output_text == "Eerste deel.\n\nTweede deel."
