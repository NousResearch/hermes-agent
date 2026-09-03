"""Regression tests for #97184 — stale pooled connection retry classification.

On unmodified main (pre-fix):
  - httpx.RemoteProtocolError("Server disconnected…") was retried (correct)
  - httpx.ReadError("SSL: UNEXPECTED_EOF_WHILE_READING") was NOT retried (bug)
    → Streaming failed before delivery, no fresh client, hard failure.

After fix (agent/chat_completion_helpers.py):
  - ReadError is in _is_conn_err → retried with fresh httpx.Client/pool
  - APIError wrapper with "unexpected_eof" is in _SSE_CONN_PHRASES → retried

Reuse helpers/patterns from tests/run_agent/test_streaming.py:
  - _make_stream_chunk, MagicMock client, patch _create/_close_request_openai_client
  - HERMES_STREAM_RETRIES env gating
  - AIAgent._interruptible_streaming_api_call as entry point
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _make_chunk(content="hello", finish_reason="stop"):
    delta = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test", usage=None)


def _make_agent(**kw):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key=kw.get("api_key", "test-key"),
        base_url=kw.get("base_url", "https://openrouter.ai/api/v1"),
        model=kw.get("model", "test/model"),
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestStaleConnectionRetry97184:
    """Pre-delivery transient errors must retry with a fresh client/pool."""

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_pre_delivery_remote_protocol_error_retries_with_fresh_client(
        self, mock_create, mock_close
    ):
        """Baseline: RemoteProtocolError already retried on main — keep covered."""
        from run_agent import AIAgent

        # First attempt raises, second succeeds
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        created_ids: list[int] = []

        def fake_create(*a, **kw):
            m = MagicMock()
            # Track identity for fresh-client assertion
            created_ids.append(id(m))

            def _create_side_effect(**kwargs):
                # Use call count to decide: first client raises, second succeeds
                if len(created_ids) == 1:
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response.", request=req
                    )
                # Second client returns a successful stream
                return iter([_make_chunk("hello"), _make_chunk(" world")])

            m.chat.completions.create.side_effect = _create_side_effect
            return m

        # Need to return different mock per call to prove fresh instance
        def create_effect(*a, **kw):
            return fake_create(*a, **kw)

        mock_create.side_effect = create_effect

        agent = _make_agent()

        prev = os.environ.get("HERMES_STREAM_RETRIES")
        os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            resp = agent._interruptible_streaming_api_call({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        finally:
            if prev is None:
                os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                os.environ["HERMES_STREAM_RETRIES"] = prev

        # Must have retried (2 creates) and succeeded
        assert mock_create.call_count == 2, (
            f"Expected 2 creates (retry), got {mock_create.call_count}"
        )
        assert created_ids[0] != created_ids[1], (
            "Retry must use fresh httpx.Client/pool"
        )
        # Verify close was called with stream_retry_cleanup then stream_request_complete or stream_error_cleanup
        assert any(
            "stream_retry_cleanup" in str(c) for c in mock_close.call_args_list
        ), f"Expected stream_retry_cleanup close, got {mock_close.call_args_list}"
        assert resp.choices[0].message.content == "hello world"

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_pre_delivery_read_error_unexpected_eof_retries_with_fresh_client(
        self, mock_create, mock_close
    ):
        """BUG TEST: ReadError(UNEXPECTED_EOF) must retry (was hard failure on unmodified main)."""
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        created_ids: list[int] = []

        def fake_create(*a, **kw):
            m = MagicMock()
            created_ids.append(id(m))

            def _side(**kwargs):
                if len(created_ids) == 1:
                    raise httpx.ReadError(
                        "SSL: UNEXPECTED_EOF_WHILE_READING", request=req
                    )
                return iter([_make_chunk("hello"), _make_chunk(" world")])

            m.chat.completions.create.side_effect = _side
            return m

        mock_create.side_effect = lambda *a, **kw: fake_create(*a, **kw)
        agent = _make_agent()

        prev = os.environ.get("HERMES_STREAM_RETRIES")
        os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            resp = agent._interruptible_streaming_api_call({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        finally:
            if prev is None:
                os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                os.environ["HERMES_STREAM_RETRIES"] = prev

        # Must have retried — this FAILS on unmodified main (call_count==1, raises)
        assert mock_create.call_count == 2, (
            f"ReadError UNEXPECTED_EOF should retry with fresh client (call_count==2), got {mock_create.call_count}. "
            "On unmodified main this fails because ReadError not in _is_conn_err."
        )
        assert created_ids[0] != created_ids[1], "Retry must use fresh client/pool"
        assert any("stream_retry_cleanup" in str(c) for c in mock_close.call_args_list)
        assert resp.choices[0].message.content == "hello world"

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_pre_delivery_wrapped_api_error_unexpected_eof_retries(
        self, mock_create, mock_close
    ):
        """APIError wrapper (OpenAI SDK) with no status_code and 'unexpected_eof' must retry."""
        from openai import APIError

        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        created_ids: list[int] = []

        def fake_create(*a, **kw):
            m = MagicMock()
            created_ids.append(id(m))

            def _side(**kwargs):
                if len(created_ids) == 1:
                    # OpenAI SDK wraps transport errors as APIError with no status_code
                    raise APIError(
                        message="SSL: UNEXPECTED_EOF_WHILE_READING",
                        request=req,
                        body=None,
                    )
                return iter([_make_chunk("hello")])

            m.chat.completions.create.side_effect = _side
            return m

        mock_create.side_effect = lambda *a, **kw: fake_create(*a, **kw)
        agent = _make_agent()

        prev = os.environ.get("HERMES_STREAM_RETRIES")
        os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            resp = agent._interruptible_streaming_api_call({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        finally:
            if prev is None:
                os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                os.environ["HERMES_STREAM_RETRIES"] = prev

        assert mock_create.call_count == 2, (
            f"Wrapped APIError(unexpected_eof) should retry, got {mock_create.call_count}"
        )
        assert created_ids[0] != created_ids[1]
        assert resp.choices[0].message.content == "hello"

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_non_transient_error_does_not_retry(self, mock_create, mock_close):
        """Sanity: non-transient ValueError must NOT retry (still hard failure)."""
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        m = MagicMock()
        m.chat.completions.create.side_effect = ValueError(
            "not a transient network error"
        )
        mock_create.return_value = m

        agent = _make_agent()
        prev = os.environ.get("HERMES_STREAM_RETRIES")
        os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            with pytest.raises(ValueError):
                agent._interruptible_streaming_api_call({
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "hi"}],
                })
        finally:
            if prev is None:
                os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                os.environ["HERMES_STREAM_RETRIES"] = prev

        # Should NOT have retried via stream_retry_cleanup; only one create
        assert mock_create.call_count == 1
