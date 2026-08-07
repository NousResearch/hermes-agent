"""Regression: /stop must not be swallowed on the Codex Responses non-streaming
poll loop (interruptible_api_call).

Companion to the Bedrock/main-streaming post-worker guards
(tests/agent/test_bedrock_interrupt_post_worker.py). Codex Responses turns
route through interruptible_api_call() -> agent._run_codex_stream() ->
_consume_codex_event_stream(), whose own interrupt_check() breaks the SSE
loop and returns a PARTIAL response WITHOUT raising. The worker thread then
sets result["response"] and exits cleanly with agent._interrupt_requested
still True. Without a post-worker re-check, interruptible_api_call would
return that partial response and silently swallow the /stop signal — the
same bug class the Bedrock streaming loop had, just reached through the
non-streaming poll loop that codex_responses turns (both "streaming" and
"non-streaming") are dispatched into.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import chat_completion_helpers as cch


class _FakeAgent:
    api_mode = "codex_responses"
    provider = ""
    _interrupt_requested = False  # not interrupted at entry

    def _create_request_openai_client(self, *, reason, api_kwargs):
        return SimpleNamespace()

    def _close_request_openai_client(self, client, *, reason):
        pass

    def _abort_request_openai_client(self, client, *, reason):
        pass

    def _compute_non_stream_stale_timeout(self, api_kwargs):
        return 300.0

    def _touch_activity(self, message):
        pass

    def _buffer_status(self, message):
        pass


def test_codex_stream_interrupt_not_swallowed_post_worker():
    """A /stop arriving mid-stream: _run_codex_stream's own interrupt_check
    breaks the SSE loop and returns a partial response WITHOUT raising,
    leaving _interrupt_requested True. The post-worker re-check in
    interruptible_api_call must raise InterruptedError instead of returning
    the partial."""
    agent = _FakeAgent()

    partial = SimpleNamespace(output=[], output_text="partial", status="incomplete")

    # Simulate the real _consume_codex_event_stream: on interrupt it breaks
    # out and returns a partial response WITHOUT raising.
    def _fake_run_codex_stream(api_kwargs, client=None, on_first_delta=None):
        agent._interrupt_requested = True
        return partial

    with patch.object(agent, "_run_codex_stream", side_effect=_fake_run_codex_stream, create=True):
        with pytest.raises(InterruptedError):
            cch.interruptible_api_call(agent, {"model": "gpt-5-codex"})


def test_codex_stream_returns_normally_when_not_interrupted():
    """Sanity: with no interrupt, the same path returns the response (guard
    must not fire spuriously)."""
    agent = _FakeAgent()

    resp = SimpleNamespace(output=[], output_text="done", status="completed")

    with patch.object(agent, "_run_codex_stream", return_value=resp, create=True):
        out = cch.interruptible_api_call(agent, {"model": "gpt-5-codex"})
        assert out is resp
