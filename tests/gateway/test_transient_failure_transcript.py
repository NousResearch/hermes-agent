"""Tests for #7100 — transient failures (429/timeout) must not drop the
user message from the transcript.

The #1630 fix introduced a blanket skip of transcript writes on any
``failed`` agent result.  That was correct for context-overflow failures
(which would otherwise cause a session-growth loop), but it also caused
transient provider failures (rate limits, read timeouts, connection
resets) to silently drop the user's message — so the agent had no memory
of the last turn on the next attempt.

The gateway classifier must distinguish:

* ``compression_exhausted=True`` OR context-keyword errors OR an unclassified
  generic ``400`` on a long history  → context-overflow → skip transcript
* everything else that fails → transient → persist the user message
"""


from gateway.run_turn_context_overflow import is_context_overflow_exception, is_context_overflow_failure_result


def _classify(agent_result: dict, history_len: int) -> tuple[bool, bool]:
    """``(agent_failed_early, is_context_overflow_failure)`` as the gateway computes them."""
    return bool(agent_result.get("failed")), is_context_overflow_failure_result(agent_result, history_len)


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

    def test_explicit_context_overflow_verdict_is_preserved(self):
        result = {"failed": True, "error": "HTTP 400: request rejected", "failure_reason": "context_overflow"}
        assert is_context_overflow_failure_result(result, history_len=407)

    def test_stamped_payload_too_large_is_context_pressure(self):
        result = {"failed": True, "error": "payload too large", "failure_reason": "payload_too_large"}
        assert is_context_overflow_failure_result(result, history_len=407)


class TestClassifiedBadRequestIsNotContextOverflow:
    """A provider's named 400 verdict must beat the old long-history status guess."""

    def test_content_policy_block_is_not_context_overflow(self):
        result = {
            "failed": True,
            "error": "content_policy_blocked: HTTP 400: Content Exists Risk",
            "failure_reason": "content_policy_blocked",
            "failure_retryable": False,
        }
        failed, ctx_overflow = _classify(result, history_len=407)
        assert failed
        assert not ctx_overflow

    def test_format_error_400_is_not_context_overflow(self):
        result = {
            "failed": True,
            "error": "HTTP 400: Unsupported parameter: max_tokens",
            "failure_reason": "format_error",
        }
        assert not is_context_overflow_failure_result(result, history_len=407)

    def test_reasoning_effort_400_is_not_context_overflow(self):
        result = {
            "failed": True,
            "error": "HTTP 400: invalid_reasoning_effort",
            "failure_reason": "reasoning_mandatory",
        }
        assert not is_context_overflow_failure_result(result, history_len=407)

    def test_classified_content_policy_exception_is_not_context_overflow(self):
        error = RuntimeError("HTTP 400: Content Exists Risk")
        error.status_code = 400
        assert not is_context_overflow_exception(error, history_len=407)

    def test_classified_max_tokens_exception_is_not_context_overflow(self):
        error = RuntimeError("HTTP 400: Unsupported parameter: max_tokens")
        error.status_code = 400
        assert not is_context_overflow_exception(
            error, history_len=407, approx_tokens=190_000, context_length=200_000,
        )

    def test_classified_reasoning_effort_exception_is_not_context_overflow(self):
        error = RuntimeError("HTTP 400: invalid_reasoning_effort")
        error.status_code = 400
        assert not is_context_overflow_exception(error, history_len=407)

    def test_bare_400_uses_session_token_estimate_and_context_length(self):
        error = RuntimeError("HTTP 400: Bad Request")
        error.status_code = 400
        assert is_context_overflow_exception(
            error, history_len=60, approx_tokens=160_000, context_length=200_000,
        )

    def test_local_memory_ceiling_with_context_phrase_is_not_overflow(self):
        error = RuntimeError(
            "predicted peak would exceed prefill safety cap 77.8GB. Reduce context length."
        )
        assert not is_context_overflow_exception(error, history_len=407)

    def test_payload_too_large_exception_is_context_pressure(self):
        error = RuntimeError("payload too large")
        error.status_code = 400
        assert is_context_overflow_exception(error, history_len=407)


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


class TestSuccessfulResultUnaffected:
    def test_successful_result_neither_failed_nor_overflow(self):
        agent_result = {
            "final_response": "Hello!",
            "messages": [{"role": "assistant", "content": "Hello!"}],
        }
        failed, ctx_overflow = _classify(agent_result, history_len=10)
        assert not failed
        assert not ctx_overflow
