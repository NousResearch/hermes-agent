"""Tests for #7100 — transient failures (429/timeout) must not drop the
user message from the transcript.

The #1630 fix introduced a blanket skip of transcript writes on any
``failed`` agent result.  That was correct for context-overflow failures
(which would otherwise cause a session-growth loop), but it also caused
transient provider failures (rate limits, read timeouts, connection
resets) to silently drop the user's message — so the agent had no memory
of the last turn on the next attempt.

The gateway classifier must distinguish:

* ``compression_exhausted=True`` OR context-keyword errors OR a generic
  ``400`` on a long history  → context-overflow → skip transcript
* everything else that fails → transient → persist the user message
"""

import pytest


def _classify(agent_result: dict, history_len: int) -> tuple[bool, bool]:
    """Replicate the gateway classifier from GatewayRunner._run_agent.

    Returns ``(agent_failed_early, is_context_overflow_failure)``.
    """
    agent_failed_early = bool(agent_result.get("failed"))
    err = str(agent_result.get("error", "")).lower()
    is_context_overflow_failure = agent_failed_early and (
        bool(agent_result.get("compression_exhausted"))
        or any(p in err for p in (
            "context length", "context size", "context window",
            "maximum context", "token limit", "too many tokens",
            "reduce the length", "exceeds the limit",
            "request entity too large", "prompt is too long",
            "payload too large", "input is too long",
        ))
        or ("400" in err and history_len > 50)
    )
    return agent_failed_early, is_context_overflow_failure


class TestContextOverflowStillSkipsTranscript:
    """#1630 behavior must be preserved for real context-overflow cases."""

    def test_compression_exhausted_is_context_overflow(self):
        agent_result = {
            "failed": True,
            "compression_exhausted": True,
            "error": "Request payload too large: max compression attempts reached.",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=100)
        assert failed
        assert ctx_overflow

    def test_explicit_context_length_error_is_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "prompt is too long: 250000 tokens > 200000 maximum",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert failed
        assert ctx_overflow

    def test_generic_400_on_large_session_is_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "error code: 400 - {'type': 'error', 'message': 'Error'}",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=100)
        assert failed
        assert ctx_overflow


class TestTransientFailureKeepsUserMessage:
    """Transient provider failures must NOT skip the transcript — doing so
    drops the user message and the agent forgets the turn. (#7100)"""

    def test_rate_limit_429_is_not_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": (
                "API call failed after 3 retries: 429 Too Many Requests "
                "— rate limit exceeded"
            ),
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert failed
        assert not ctx_overflow

    def test_read_timeout_is_not_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "ReadTimeout: HTTPSConnectionPool(host='api.z.ai'): Read timed out.",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert failed
        assert not ctx_overflow

    def test_connection_reset_is_not_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "ConnectionError: [Errno 54] Connection reset by peer",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert failed
        assert not ctx_overflow

    def test_provider_500_is_not_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "API call failed after 3 retries: 500 Internal Server Error",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert failed
        assert not ctx_overflow

    def test_generic_400_on_short_session_is_not_context_overflow(self):
        """A 400 on a short session is a real client error, not context
        overflow — still not a reason to drop the user turn."""
        agent_result = {
            "failed": True,
            "error": "error code: 400 - invalid model",
        }
        failed, ctx_overflow = _classify(agent_result, history_len=5)
        assert failed
        assert not ctx_overflow


class TestSuccessfulResultUnaffected:
    def test_successful_result_neither_failed_nor_overflow(self):
        agent_result = {
            "final_response": "Hello!",
            "messages": [{"role": "assistant", "content": "Hello!"}],
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert not failed
        assert not ctx_overflow
