"""Unit tests for SSE [DONE] sentinel tracking (issue #90848)."""

from __future__ import annotations

import json

from hermes_cli.proxy.sse_done import DONE_SSE_FRAME, SseDoneTracker, content_type_is_sse


def _data_line(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return b"data: " + bytes(obj) + b"\n\n"
    if obj == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def test_complete_stream_without_done_appends():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {"content": "LONGCAT_OK"}}]}))
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    tracker.feed(
        _data_line(
            {
                "choices": [],
                "lastOne": True,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
    )
    assert tracker.should_append_done() is True
    assert tracker.saw_done is False
    assert DONE_SSE_FRAME.startswith(b"data: [DONE]")


def test_finish_reason_alone_is_enough_without_last_one():
    """Solar-shaped stream: finish_reason present, no lastOne, no [DONE]."""
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {"content": "SOLAR_OK"}}]}))
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    assert tracker.should_append_done() is True


def test_last_one_alone_is_enough_without_finish_reason():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {"content": "x"}}]}))
    tracker.feed(_data_line({"choices": [], "lastOne": True}))
    assert tracker.should_append_done() is True


def test_existing_done_is_not_duplicated():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    tracker.feed(_data_line("[DONE]"))
    assert tracker.should_append_done() is False
    assert tracker.saw_done is True


def test_error_event_blocks_done_synthesis():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {"content": "partial"}}]}))
    tracker.feed(_data_line({"error": {"message": "upstream failed", "type": "api_error"}}))
    assert tracker.should_append_done() is False


def test_error_finish_reason_blocks_done_synthesis():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "error"}]}))
    assert tracker.should_append_done() is False


def test_malformed_event_blocks_done_synthesis():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    tracker.feed(b'data: {"choices": [MALFORMED]}\n\n')
    assert tracker.should_append_done() is False


def test_truncated_trailing_event_blocks_done_synthesis():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    tracker.feed(b'data: {"choices": [{"delta": {"content": "tail"}}]')
    assert tracker.should_append_done() is False


def test_interrupted_stream_never_appends_done():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    tracker.mark_interrupted()
    assert tracker.should_append_done() is False


def test_incomplete_stream_without_terminal_marker_does_not_append():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [{"delta": {"content": "mid"}}]}))
    assert tracker.should_append_done() is False


def test_feed_preserves_chunk_boundaries_across_split_lines():
    """A finish_reason frame split across TCP chunks must still be detected."""
    tracker = SseDoneTracker()
    payload = _data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    mid = len(payload) // 2
    tracker.feed(payload[:mid])
    tracker.feed(payload[mid:])
    assert tracker.should_append_done() is True


def test_content_type_is_sse():
    assert content_type_is_sse({"Content-Type": "text/event-stream"}) is True
    assert content_type_is_sse({"content-type": "text/event-stream; charset=utf-8"}) is True
    assert content_type_is_sse({"Content-Type": "application/json"}) is False


def test_multiline_data_event_joins_per_sse_spec():
    """One JSON event split across consecutive data: lines (spec-legal) must
    parse as a single event, not two malformed fragments."""
    tracker = SseDoneTracker()
    tracker.feed(
        b'data: {"choices":[{"delta":{},\n'
        b'data: "finish_reason":"stop"}]}\n'
        b"\n"
    )
    assert tracker.saw_malformed_event is False
    assert tracker.should_append_done() is True


def test_multiline_malformed_event_still_disables_synthesis():
    tracker = SseDoneTracker()
    tracker.feed(b"data: {broken\ndata: json!\n\n")
    tracker.feed(_data_line({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    assert tracker.should_append_done() is False


def test_truthy_integer_last_one_counts_as_terminal():
    tracker = SseDoneTracker()
    tracker.feed(_data_line({"choices": [], "lastOne": 1, "usage": {}}))
    assert tracker.should_append_done() is True


def test_final_event_without_trailing_blank_line_still_dispatches():
    """Clean EOF right after the last data: line (no blank-line boundary)."""
    tracker = SseDoneTracker()
    tracker.feed(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n')
    assert tracker.should_append_done() is True
