"""The Bedrock Converse and Codex Responses streams cut a repetition loop mid-stream, like the others.

#130551 wired ``RunawayStreamWatch`` into the chat_completions and Anthropic Messages streams. The
Bedrock and Codex consumers read to the end of a looping stream, and Converse sends no ``maxTokens``
by default, so a loop there runs to the model's output limit before the post-stream check sees it.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from agent.bedrock_adapter import stream_converse_with_callbacks
from agent.codex_runtime import _consume_codex_event_stream
from agent.repetition_guard import STOP_PATH_MIN_CHARS
from agent.turn_truncation import _CONTINUABLE_MODES

_LOOP_LINE = "I will now check the configuration file again.\n"
_LOOP = [_LOOP_LINE * 2] * 1500  # ~141K chars of one line
_LOOP_TOTAL = sum(map(len, _LOOP))
# Repetition the user asked for stays below the stop path's scale floor.
_ASKED_FOR_REPEAT = "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50
# Distinct rows sharing a long suffix: repeated windows dominate, yet every line differs.
_DISTINCT_ROWS = "".join(
    f"| {i:05d} | status: pending review by the on-call engineer before the next release window |\n"
    for i in range(500)
)
_LONG_PREFIX = "".join(hashlib.sha256(str(i).encode()).hexdigest() + "\n" for i in range(1540))
assert len(_ASKED_FOR_REPEAT) < STOP_PATH_MIN_CHARS <= len(_DISTINCT_ROWS)


def _chunks(text: str, size: int = 96) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


# ── Bedrock Converse ─────────────────────────────────────────────────────────

def _converse_events(chunks: list[str], *, reasoning: bool = False) -> list[dict]:
    key = "reasoningContent" if reasoning else "text"
    deltas = [{"contentBlockDelta": {"contentBlockIndex": 0,
                                     "delta": {key: {"text": c} if reasoning else c}}} for c in chunks]
    return deltas + [{"contentBlockStop": {"contentBlockIndex": 0}}, {"messageStop": {"stopReason": "end_turn"}}]


def _consume_converse(chunks, *, reasoning=False, watch=True):
    seen = [0]

    def counted():
        for event in _converse_events(chunks, reasoning=reasoning):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                seen[0] += len(delta["reasoningContent"]["text"] if reasoning else delta["text"])
            yield event

    response = stream_converse_with_callbacks({"stream": counted()}, watch_runaway=watch)
    return response, seen[0]


@pytest.mark.parametrize("reasoning", [False, True], ids=["text", "reasoning"])
def test_bedrock_looping_stream_is_cut(reasoning):
    response, consumed = _consume_converse(_LOOP, reasoning=reasoning)
    assert consumed < _LOOP_TOTAL // 4, "the loop was read far past the point it could be judged"
    assert response._runaway_repetition is True
    assert response.choices[0].finish_reason == "length"


def test_bedrock_loop_after_long_legitimate_prefix_is_cut_before_the_reply_doubles():
    chunks = _chunks(_LONG_PREFIX) + _LOOP
    response, consumed = _consume_converse(chunks)
    assert getattr(response, "_runaway_repetition", False)
    assert consumed < 2 * len(_LONG_PREFIX) + 100_000


@pytest.mark.parametrize("text", [_ASKED_FOR_REPEAT, _DISTINCT_ROWS], ids=["asked-for-repeat", "distinct-rows"])
def test_bedrock_repetitive_but_legitimate_stream_is_delivered(text):
    response, consumed = _consume_converse(_chunks(text))
    assert consumed == len(text)
    assert not getattr(response, "_runaway_repetition", False)
    assert response.choices[0].message.content == text
    assert response.choices[0].finish_reason == "stop"


def test_bedrock_without_watch_keeps_reading_to_the_end():
    """The adapter's default is unchanged; only the live stream path opts in."""
    response, consumed = _consume_converse(_LOOP, watch=False)
    assert consumed == _LOOP_TOTAL
    assert not getattr(response, "_runaway_repetition", False)


def test_bedrock_tool_call_cut_mid_arguments_is_dropped():
    events = [{"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "t1", "name": "write_file"}}}}]
    events += [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"content": "'}}}}]
    events += [{"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": c}}} for c in _LOOP]
    response = stream_converse_with_callbacks({"stream": iter(events)}, watch_runaway=True)
    assert response._runaway_repetition is True
    message = response.choices[0].message
    assert not message.tool_calls
    # No dangling toolUse block is left for replay: Converse rejects a toolUse without its toolResult.
    assert not any("toolUse" in block for block in (message.bedrock_content_blocks or []))


# ── Codex Responses ──────────────────────────────────────────────────────────

def _codex_events(chunks: list[str], kind: str = "response.output_text.delta"):
    for c in chunks:
        yield SimpleNamespace(type=kind, delta=c, output_index=0, item_id="msg_1", content_index=0)


def _consume_codex(chunks, kind="response.output_text.delta"):
    seen = [0]

    def counted():
        for event in _codex_events(chunks, kind):
            seen[0] += len(event.delta)
            yield event

    return _consume_codex_event_stream(counted(), model="fixture"), seen[0]


@pytest.mark.parametrize("kind", ["response.output_text.delta", "response.reasoning_text.delta",
                                  "response.reasoning_summary_text.delta"])
def test_codex_looping_stream_is_cut(kind):
    response, consumed = _consume_codex(_LOOP, kind)
    assert consumed < _LOOP_TOTAL // 4
    assert response._runaway_repetition is True


@pytest.mark.parametrize("text", [_ASKED_FOR_REPEAT, _DISTINCT_ROWS], ids=["asked-for-repeat", "distinct-rows"])
def test_codex_repetitive_but_legitimate_stream_is_delivered(text):
    response, consumed = _consume_codex(_chunks(text))
    assert consumed == len(text)
    assert not getattr(response, "_runaway_repetition", False)
    assert response.output_text == text


def test_codex_cut_routes_the_turn_to_the_length_path():
    from agent.turn_response_check import _derive_finish_reason
    response, _ = _consume_codex(_LOOP)
    agent = SimpleNamespace(api_mode="codex_responses")
    assert _derive_finish_reason(agent, response, []) == "length"


# ── Coverage: every continuable stream mode has a mid-stream cut ─────────────

# chat_completions and anthropic_messages are covered end to end in test_stream_runaway_repetition.py.
_RUNAWAY_CUT_COVERED = {
    "chat_completions": "test_stream_runaway_repetition.py",
    "anthropic_messages": "test_stream_runaway_repetition.py",
    "bedrock_converse": "test_bedrock_looping_stream_is_cut",
    "codex_responses": "test_codex_looping_stream_is_cut",
}


def test_every_continuable_stream_mode_has_a_runaway_cut():
    """A new streaming mode must wire RunawayStreamWatch and add its test here (two modes were missed once)."""
    assert set(_CONTINUABLE_MODES) <= set(_RUNAWAY_CUT_COVERED), (
        f"no mid-stream runaway cut covered for: {sorted(set(_CONTINUABLE_MODES) - set(_RUNAWAY_CUT_COVERED))}")


def test_bedrock_live_stream_path_hands_the_cut_to_the_turn():
    """Through the live Bedrock dispatch: the cut response comes back marked, and the provider stream closes."""
    from unittest.mock import patch

    from agent import chat_completion_helpers as cch

    class _Agent:
        api_mode, provider = "bedrock_converse", "bedrock"
        model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
        _interrupt_requested = _disable_streaming = False
        reasoning_callback = stream_delta_callback = None
        _consecutive_stale_streams = 0

        def _has_stream_consumers(self): return False
        def _buffer_status(self, *a, **k): pass
        def _claim_stream_writer(self): return 1
        def _fire_stream_delta(self, text): pass
        def _fire_tool_gen_started(self, name): pass
        def _fire_reasoning_delta(self, text): pass
        def _safe_print(self, *a, **k): pass

    class _LoopingProviderStream:
        def __init__(self):
            self.consumed, self.closed = 0, False

        def __iter__(self):
            for event in _converse_events(_LOOP):
                if "contentBlockDelta" in event:
                    self.consumed += len(event["contentBlockDelta"]["delta"]["text"])
                yield event

        def close(self):
            self.closed = True

    provider_stream = _LoopingProviderStream()
    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": provider_stream})
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: r), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        out = cch.interruptible_streaming_api_call(_Agent(), {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True})
    assert out._runaway_repetition is True
    assert out.choices[0].finish_reason == "length"
    assert provider_stream.consumed < _LOOP_TOTAL // 4
    assert provider_stream.closed is True
