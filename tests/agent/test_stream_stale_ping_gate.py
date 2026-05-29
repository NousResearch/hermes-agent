"""Regression guard: content-less SSE keepalive pings must NOT reset the
stale-stream timer in the chat_completions streaming loop.

Root cause (gpt-5.5 / openai-codex stall investigation):
The ChatGPT codex backend (chatgpt.com/backend-api/codex) keeps a stalled
stream alive with SSE keepalive pings while delivering zero content tokens.
The old loop reset ``last_chunk_time`` on *every* chunk, so those pings
defeated both the stale-stream detector AND the httpx read timeout — a hung
request stayed alive until the backend gave up (~10 min) instead of being
killed and retried at the stale threshold (the immediate retry succeeds in
seconds). A trivial "pong" request was measured at 615s before the fix.

The fix gates the ``last_chunk_time`` reset on chunks that carry real
progress (content / reasoning / tool-call / function-call deltas).

This test mirrors the EXACT predicate shape used in
``agent/chat_completion_helpers.py``'s chunk loop so any future refactor
must preserve the invariant:

    keepalive ping (empty choices)      -> does NOT reset the stale timer
    role-only / empty delta             -> does NOT reset the stale timer
    content / reasoning / tool_calls    -> DOES reset the stale timer

If you change the predicate in chat_completion_helpers.py, change it here
too — or, better, refactor it into a shared helper that both sites import.
"""
from __future__ import annotations

import time


def _chunk_resets_stale_timer(chunk) -> bool:
    """Exact shape of the chat_completion_helpers stale-timer gate.

    Kept in lock-step with the source streaming loop.
    """
    _pg = chunk.choices[0].delta if chunk.choices else None
    return bool(
        _pg is not None
        and (
            getattr(_pg, "content", None)
            or getattr(_pg, "reasoning_content", None)
            or getattr(_pg, "reasoning", None)
            or getattr(_pg, "tool_calls", None)
            or getattr(_pg, "function_call", None)
        )
    )


class _Delta:
    def __init__(self, **kw):
        for attr in ("content", "reasoning_content", "reasoning", "tool_calls", "function_call"):
            setattr(self, attr, kw.get(attr))


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


class TestStreamStalePingGate:
    def test_keepalive_ping_does_not_reset(self):
        """A keepalive ping (no choices) must not count as progress."""
        assert _chunk_resets_stale_timer(_Chunk([])) is False

    def test_role_only_delta_does_not_reset(self):
        """A role-only / empty delta carries no tokens — not progress."""
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta())])) is False

    def test_content_token_resets(self):
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta(content="po"))])) is True

    def test_reasoning_content_resets(self):
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta(reasoning_content="think"))])) is True

    def test_reasoning_alt_field_resets(self):
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta(reasoning="think"))])) is True

    def test_tool_call_delta_resets(self):
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta(tool_calls=[object()]))])) is True

    def test_legacy_function_call_resets(self):
        assert _chunk_resets_stale_timer(_Chunk([_Choice(_Delta(function_call=object()))])) is True

    def test_ping_only_stall_is_detected(self):
        """A long run of pings must let ``stale_elapsed`` grow past the
        threshold so the detector fires (kill + reconnect)."""
        threshold = 90.0
        last = {"t": time.time() - 200.0}  # last real content 200s ago
        for _ in range(200):
            ping = _Chunk([])
            if _chunk_resets_stale_timer(ping):
                last["t"] = time.time()
        stale_elapsed = time.time() - last["t"]
        assert stale_elapsed > threshold, (
            f"ping-only stall not detected: stale_elapsed={stale_elapsed:.0f}s "
            f"<= threshold={threshold:.0f}s"
        )
