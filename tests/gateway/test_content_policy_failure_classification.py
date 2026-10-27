"""Tests for #134261 — a provider content-moderation block must not be
classified as a context overflow by the gateway.

The agent-side classifier (``agent/error_classifier.py``) already routes the
provider's safety-filter rejection (e.g. DeepSeek's HTTP 400 ``Content Exists
Risk``) to ``failure_reason="content_policy_blocked"``, and the turn result
carries the polite refusal copy in ``final_response``.  But the gateway's
``is_context_overflow_failure_result`` only looked at the error *string*:
a bare ``400`` on a session longer than 50 messages counted as an overflow.
The user's turn was then never persisted (anti-growth rule #1630 applied to a
session at 39% of its context) and the user-facing reply was forced to the
overflow copy (``/compress`` on a just-compressed session is a no-op), so the
session re-tripped the same provider filter on every retry until ``/reset``.

The fix: a turn the agent already classified as ``content_policy_blocked`` is
never an overflow — session size is irrelevant to a deterministic per-prompt
refusal.  The generic ``400``-on-long-history heuristic is unchanged for
failures that carry no structured reason.
"""

from gateway.run_turn import is_context_overflow_failure_result


def _deepseek_moderation_result() -> dict:
    """The turn-result shape the agent produces for DeepSeek's HTTP 400
    ``{"error": {"message": "Content Exists Risk", ...}}`` — classifier verdict
    ``content_policy_blocked`` with the detail in the error string (#134261's
    reporter hit this at 407 messages / 39% of the detected context)."""
    return {
        "failed": True,
        "completed": False,
        "error": (
            'content_policy_blocked: API call failed after 1 retries: HTTP 400 '
            '{"error": {"message": "Content Exists Risk", "type": "content_filter", '
            '"code": 20086}}'
        ),
        "failure_reason": "content_policy_blocked",
        "failure_retryable": False,
        "final_response": "⚠️ The AI model service's safety filter rejected the request.",
    }


class TestContentPolicyBlockIsNotOverflow:
    """#134261: moderation blocks on long sessions must reach the user and the
    transcript as refusals, not disappear behind the overflow remedy."""

    def test_moderation_400_on_long_history_is_not_overflow(self):
        # RED on main: "400" in the error string + history_len > 50 → True.
        assert not is_context_overflow_failure_result(_deepseek_moderation_result(), history_len=407)

    def test_moderation_block_without_status_text_is_not_overflow(self):
        # HTTP-200 refusal path (finish_reason=content_filter): no "400" in the
        # string, still carries the structured reason.
        result = {
            "failed": True,
            "completed": False,
            "error": "content_policy_blocked: model declined to respond (content_filter)",
            "failure_reason": "content_policy_blocked",
            "failure_retryable": False,
        }
        assert not is_context_overflow_failure_result(result, history_len=407)

    def test_moderation_block_on_short_history_is_not_overflow(self):
        # Control: already False on main (no "400" in a short-history win), kept
        # so the exemption can never flip the short-session answer either.
        assert not is_context_overflow_failure_result(_deepseek_moderation_result(), history_len=10)


class TestOverflowHeuristicsUnchanged:
    """The #1630 anti-growth behavior must survive: the exemption is keyed on
    the agent's structured verdict, not on the mere presence of "400"."""

    def test_generic_400_on_long_history_is_still_overflow(self):
        agent_result = {"failed": True, "error": "API call failed after 3 retries: HTTP 400 Bad Request"}
        assert is_context_overflow_failure_result(agent_result, history_len=407)

    def test_context_keyword_is_still_overflow(self):
        agent_result = {
            "failed": True,
            "error": "Error code: 400 — this model's maximum context length is exceeded",
        }
        assert is_context_overflow_failure_result(agent_result, history_len=20)

    def test_compression_exhausted_is_still_overflow(self):
        agent_result = {"failed": True, "compression_exhausted": True, "error": "max compression attempts reached"}
        assert is_context_overflow_failure_result(agent_result, history_len=407)

    def test_transient_429_is_still_not_overflow(self):
        agent_result = {"failed": True, "error": "API call failed after 3 retries: 429 Too Many Requests"}
        assert not is_context_overflow_failure_result(agent_result, history_len=407)
