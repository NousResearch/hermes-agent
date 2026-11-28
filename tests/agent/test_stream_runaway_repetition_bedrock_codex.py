"""The Bedrock Converse and Codex Responses streams cut a repetition loop mid-stream, like the others.

#130551 wired ``RunawayStreamWatch`` into the chat_completions and Anthropic Messages streams. The
Bedrock and Codex consumers read to the end of a looping stream, and Converse sends no ``maxTokens``
by default, so a loop there runs to the model's output limit before the post-stream check sees it.

One invariant per wire: a runaway loop stops early, and a repeat the user actually asked for is
still delivered whole. Both halves matter — a cut that fires on legitimate repetition is worse than
no cut at all.
"""

from __future__ import annotations

import pytest

from agent.bedrock_adapter import stream_converse_with_callbacks
from agent.codex_runtime import _consume_codex_event_stream
from agent.repetition_guard import STOP_PATH_MIN_CHARS

_LOOP_LINE = "I will now check the configuration file again.\n"
_LOOP = [_LOOP_LINE * 2] * 1500  # ~141K chars of one line
_LOOP_TOTAL = sum(map(len, _LOOP))
# Repetition the user asked for stays below the stop path's scale floor.
_ASKED_FOR_REPEAT = "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50
assert len(_ASKED_FOR_REPEAT) < STOP_PATH_MIN_CHARS


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
def test_bedrock_converse_cuts_a_runaway_loop_but_delivers_a_requested_repeat(reasoning):
    response, consumed = _consume_converse(_LOOP, reasoning=reasoning)
    assert consumed < _LOOP_TOTAL // 4, "the loop was read far past the point it could be judged"
    assert response._runaway_repetition is True
    assert response.choices[0].finish_reason == "length"

    response, consumed = _consume_converse(_chunks(_ASKED_FOR_REPEAT), reasoning=reasoning)
    assert consumed == len(_ASKED_FOR_REPEAT)
    assert not getattr(response, "_runaway_repetition", False)


# ── Codex Responses ──────────────────────────────────────────────────────────

def _codex_events(chunks: list[str], kind: str = "response.output_text.delta"):
    from types import SimpleNamespace
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
def test_codex_responses_cuts_a_runaway_loop_but_delivers_a_requested_repeat(kind):
    response, consumed = _consume_codex(_LOOP, kind)
    assert consumed < _LOOP_TOTAL // 4
    assert response._runaway_repetition is True

    if kind != "response.output_text.delta":
        # Reasoning deltas arrive without the terminal event a finished reply carries; delivery is
        # a property of the visible reply, so it is asserted on that wire only.
        return
    response, consumed = _consume_codex(_chunks(_ASKED_FOR_REPEAT), kind)
    assert consumed == len(_ASKED_FOR_REPEAT)
    assert not getattr(response, "_runaway_repetition", False)
