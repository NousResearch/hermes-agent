"""Suppressed-content streaming path sanitizes lone surrogates.

Once tool-call deltas start accumulating, content deltas bypass
``_fire_stream_delta()`` and go straight through
``_StreamingCall._route_suppressed_text()`` (agent/chat_completion_helpers.py)
to the delta callback and the recorded partial-stream text. A lone surrogate
emitted by a byte-level tokenizer (mimo, kimi, glm) in that position must be
scrubbed, or a downstream UTF-8 consumer crashes.
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import _StreamingCall


def test_route_suppressed_text_replaces_surrogates_before_callback_and_record():
    observed = []
    recorded = []
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = SimpleNamespace(
        stream_delta_callback=observed.append,
        _record_streamed_assistant_text=recorded.append,
    )

    call._route_suppressed_text("bad \udce7 chunk")

    assert observed == ["bad \ufffd chunk"]
    assert recorded == ["bad \ufffd chunk"]
    observed[0].encode("utf-8")
    recorded[0].encode("utf-8")


def test_route_suppressed_text_noop_without_callback():
    """No ``stream_delta_callback`` → nothing recorded, nothing crashes."""
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = SimpleNamespace(
        stream_delta_callback=None,
        _record_streamed_assistant_text=None,
    )

    call._route_suppressed_text("bad \udce7 chunk")
