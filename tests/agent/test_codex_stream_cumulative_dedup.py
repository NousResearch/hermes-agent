"""Cumulative ``message`` snapshot dedup in the Codex Responses assembler.

Bedrock mantle (see #41321) re-sends the same ``message`` output item as
``response.output_item.done`` snapshots where each is a prefix-superset of
the previous. Keeping them all duplicates the text once the items are
concatenated downstream (a 5957-char answer delivered as 9505+), and the
text deltas re-send each snapshot's full text too. These tests pin the
collapse (prefix chains only) and the guarded ``output_text`` derivation.
"""

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _msg(text: str, *, item_id: str = "msg_1") -> SimpleNamespace:
    return SimpleNamespace(
        type="message", id=item_id, role="assistant", status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _item_done(item: SimpleNamespace, output_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_item.done", output_index=output_index, item=item)


def _completed() -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(id="resp_1", status="completed", usage=None, output=None),
    )


def _texts(final) -> list:
    return [
        "".join(p.text for p in item.content if p.type == "output_text")
        for item in final.output if getattr(item, "type", None) == "message"
    ]


class TestCumulativeSnapshotCollapse:
    def test_single_message_item_passes_through(self):
        final = _consume_codex_event_stream(
            [_item_done(_msg("hello")), _delta("hello"), _completed()], model="gpt-5.2")
        assert _texts(final) == ["hello"]
        assert final.output_text == "hello"

    def test_cumulative_snapshots_collapse_to_the_last(self):
        final = _consume_codex_event_stream(
            [
                _item_done(_msg("Hel")), _delta("Hel"),
                _item_done(_msg("Hello")), _delta("Hello"),
                _item_done(_msg("Hello world")), _delta("Hello world"),
                _completed(),
            ],
            model="gpt-5.2")
        # One message item — the superset snapshot, not three concatenated.
        assert _texts(final) == ["Hello world"]
        # The delta join re-sent every snapshot ("HelHelloHello world"); the
        # collapsed superset text must win.
        assert final.output_text == "Hello world"

    def test_distinct_messages_are_both_kept(self):
        final = _consume_codex_event_stream(
            [
                _item_done(_msg("first answer", item_id="msg_1")),
                _delta("first answer"),
                _item_done(_msg("second answer", item_id="msg_2")),
                _delta("second answer"),
                _completed(),
            ],
            model="gpt-5.2")
        # Neither text is a prefix of the other: no collapse, no truncation.
        assert _texts(final) == ["first answer", "second answer"]
        assert final.output_text == "first answersecond answer"

    def test_snapshots_without_deltas_still_derive_output_text(self):
        final = _consume_codex_event_stream(
            [
                _item_done(_msg("Hel")),
                _item_done(_msg("Hello world")),
                _completed(),
            ],
            model="gpt-5.2")
        assert _texts(final) == ["Hello world"]
        # No deltas arrived at all — the surviving snapshot is the only text source.
        assert final.output_text == "Hello world"

    def test_cumulative_messages_followed_by_function_call(self):
        """#41321 review ask: snapshots then a tool call — output_text is the
        single final message text, and the function_call item survives."""
        call_done = _item_done(
            SimpleNamespace(
                type="function_call", id="fc_1", call_id="call_1",
                name="get_weather", arguments='{"city": "SF"}', status="completed"),
            output_index=1)
        final = _consume_codex_event_stream(
            [
                _item_done(_msg("Let me check")), _delta("Let me check"),
                _item_done(_msg("Let me check the weather")), _delta("Let me check the weather"),
                call_done,
                _completed(),
            ],
            model="gpt-5.2")
        assert _texts(final) == ["Let me check the weather"]
        types = [getattr(item, "type", None) for item in final.output]
        assert types.count("function_call") == 1
        # The delta join is the duplicated cumulative re-send; the final
        # message text must be delivered exactly once.
        assert final.output_text == "Let me check the weather"
