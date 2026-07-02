"""Tests for _consume_codex_event_stream cumulative message deduplication.

Bedrock mantle emits cumulative ``response.output_item.done`` events where
each ``message`` item is a prefix-superset of the prior.  Without
deduplication the downstream normalizer concatenates them → quadratic text
duplication.
"""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


def _make_event(event_type: str, **fields) -> SimpleNamespace:
    """Build a minimal SSE event-like object."""
    return SimpleNamespace(type=event_type, **fields)


def _make_message_item(text: str, *, item_type: str = "message") -> SimpleNamespace:
    """Build a message output item with output_text content."""
    return SimpleNamespace(
        type=item_type,
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


class TestCumulativeMessageDedup:
    """Bedrock mantle sends cumulative message snapshots — keep only the last."""

    def test_single_message_preserved(self):
        """Normal case: one message item passes through unchanged."""
        events = [
            _make_event("response.output_item.done", item=_make_message_item("hello")),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        assert len(result.output) == 1
        # output_text is from deltas (empty here); check output items directly.
        assert result.output[0].content[0].text == "hello"

    def test_cumulative_messages_collapsed_to_last(self):
        """Multiple cumulative message items → only the last (longest) survives."""
        events = [
            _make_event("response.output_item.done", item=_make_message_item("ab")),
            _make_event("response.output_item.done", item=_make_message_item("abcd")),
            _make_event("response.output_item.done", item=_make_message_item("abcdef")),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        assert len(result.output) == 1
        assert result.output[0].content[0].text == "abcdef"

    def test_function_call_items_preserved_alongside_message(self):
        """function_call items are kept even when message items are deduped."""
        fc_item = SimpleNamespace(type="function_call", name="tool", arguments="{}")
        events = [
            _make_event("response.output_item.done", item=_make_message_item("part1")),
            _make_event("response.output_item.done", item=_make_message_item("part1part2")),
            _make_event("response.output_item.done", item=fc_item),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        assert len(result.output) == 2
        # Message collapsed to last; function_call preserved.
        assert result.output[0].type == "message"
        assert result.output[0].content[0].text == "part1part2"
        assert result.output[1].type == "function_call"

    def test_message_after_function_call_kept_separately(self):
        """A message item after a function_call block starts a new slot."""
        fc_item = SimpleNamespace(type="function_call", name="tool", arguments="{}")
        events = [
            _make_event("response.output_item.done", item=_make_message_item("before")),
            _make_event("response.output_item.done", item=fc_item),
            _make_event("response.output_item.done", item=_make_message_item("after")),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        assert len(result.output) == 3
        assert result.output[0].content[0].text == "before"
        assert result.output[1].type == "function_call"
        assert result.output[2].content[0].text == "after"

    def test_no_output_items_uses_text_delta_fallback(self):
        """When no output_item.done arrives, text deltas are assembled."""
        events = [
            _make_event("response.output_text.delta", delta="hello "),
            _make_event("response.output_text.delta", delta="world"),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        assert result.output_text == "hello world"
        assert len(result.output) == 1
        assert result.output[0].content[0].text == "hello world"

    def test_output_text_not_duplicated_by_cumulative_items(self):
        """When cumulative message items are deduped, output_text must use
        the final item's text — not the joined deltas (which contain every
        prefix-superset, producing quadratic duplication)."""
        # Simulate Bedrock mantle: 3 cumulative message items + matching deltas
        msg_a = _make_message_item("Hello")
        msg_ab = _make_message_item("Hello, world")
        msg_abc = _make_message_item("Hello, world! How are you?")
        events = [
            _make_event("response.output_item.done", item=msg_a),
            _make_event("response.output_text.delta", delta="Hello"),
            _make_event("response.output_item.done", item=msg_ab),
            _make_event("response.output_text.delta", delta=", world"),
            _make_event("response.output_item.done", item=msg_abc),
            _make_event("response.output_text.delta", delta="! How are you?"),
            _make_event("response.completed", response=SimpleNamespace(
                status="completed", usage=None, id="r1",
            )),
        ]
        result = _consume_codex_event_stream(iter(events), model="test")
        # output_items: only the last message kept
        assert len(result.output) == 1
        assert result.output[0].content[0].text == "Hello, world! How are you?"
        # output_text: must match the deduped item, NOT "HelloHello, worldHello, world! How are you?"
        assert result.output_text == "Hello, world! How are you?"
