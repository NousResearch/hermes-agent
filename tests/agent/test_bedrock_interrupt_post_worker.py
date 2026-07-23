"""Regression: /stop must not be swallowed on the Bedrock streaming path.

Companion to the OpenAI/Anthropic streaming post-worker guard. The Bedrock
Converse stream callback (bedrock_adapter.stream_converse_with_callbacks) breaks
out of its event loop on interrupt and returns a PARTIAL response WITHOUT
raising. The worker thread then sets result["response"] and exits cleanly with
agent._interrupt_requested still True. Without a post-worker re-check in the
poll loop, interruptible_streaming_api_call would return that partial response
and silently swallow the /stop signal.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import chat_completion_helpers as cch


class _FakeAgent:
    api_mode = "bedrock_converse"
    _interrupt_requested = False  # not interrupted at entry (passes pre-flight)
    _disable_streaming = False
    reasoning_callback = None
    stream_delta_callback = None
    # Real AIAgent always carries these; the streaming stale-timeout derivation
    # (chat_completion_helpers._derive_stream_stale_timeout) reads them.
    provider = "bedrock"
    model = "anthropic.claude-3-sonnet-20240229-v1:0"
    _consecutive_stale_streams = 0

    def _has_stream_consumers(self):
        return False

    def _buffer_status(self, *a, **k):
        pass

    def _claim_stream_writer(self):
        return 1

    def _fire_stream_delta(self, text):
        pass

    def _fire_tool_gen_started(self, name):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _safe_print(self, *a, **k):
        pass


def test_bedrock_stream_interrupt_not_swallowed_post_worker():
    """A /stop arriving MID-stream: the pre-flight check (top of function) has
    already passed, the worker's stream callback breaks and returns a partial
    response WITHOUT raising, leaving _interrupt_requested True. The post-worker
    re-check must raise InterruptedError instead of returning the partial."""
    agent = _FakeAgent()

    partial = SimpleNamespace(choices=[], usage=None, stop_reason="interrupted")

    # Simulate the real adapter: on interrupt it breaks out and returns a
    # partial response WITHOUT raising. Flip the interrupt flag here to model
    # /stop arriving mid-stream (after the pre-flight check, during the worker).
    def _fake_stream(*args, **kwargs):
        agent._interrupt_requested = True
        return partial

    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": []})

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", side_effect=_fake_stream), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: r), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        api_kwargs = {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True}
        with pytest.raises(InterruptedError):
            cch.interruptible_streaming_api_call(agent, api_kwargs)


def test_bedrock_stream_returns_normally_when_not_interrupted():
    """Sanity: with no interrupt, the same path returns the response (guard
    must not fire spuriously)."""
    agent = _FakeAgent()
    agent._interrupt_requested = False

    resp = SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")
    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": []})

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", return_value=resp), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: r), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        api_kwargs = {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True}
        out = cch.interruptible_streaming_api_call(agent, api_kwargs)
        assert out is resp
