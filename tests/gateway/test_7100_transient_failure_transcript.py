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


from gateway.run_turn import is_context_overflow_failure_result


class TestContextOverflowStillSkipsTranscript:
    """#1630 behavior must be preserved for real context-overflow cases."""

    def test_compression_exhausted_is_context_overflow(self):
        agent_result = {
            "failed": True,
            "compression_exhausted": True,
            "error": "Request payload too large: max compression attempts reached.",
        }
        assert is_context_overflow_failure_result(agent_result, history_len=100)

    def test_bare_400_status_on_long_session_is_context_overflow(self):
        agent_result = {"failed": True, "error": 'HTTP 400: {"object":"error","model":"deepseek-v4-flash"}'}
        assert is_context_overflow_failure_result(agent_result, history_len=138)

    def test_digits_400_inside_a_larger_number_are_not_a_status(self):
        agent_result = {
            "failed": True,
            "error": "API call failed after 3 retries: HTTP 429 rate limit exceeded. Limit 40000, Used 39990",
        }
        assert not is_context_overflow_failure_result(agent_result, history_len=138)


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
        assert not is_context_overflow_failure_result(agent_result, history_len=10)

    def test_read_timeout_is_not_context_overflow(self):
        agent_result = {
            "failed": True,
            "error": "ReadTimeout: HTTPSConnectionPool(host='api.z.ai'): Read timed out.",
        }
        assert not is_context_overflow_failure_result(agent_result, history_len=10)


class TestSuccessfulResultUnaffected:
    def test_successful_result_neither_failed_nor_overflow(self):
        agent_result = {
            "final_response": "Hello!",
            "messages": [{"role": "assistant", "content": "Hello!"}],
        }
        assert not is_context_overflow_failure_result(agent_result, history_len=10)
