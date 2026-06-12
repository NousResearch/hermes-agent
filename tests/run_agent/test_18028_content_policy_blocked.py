"""Regression guard for #18028: provider content-policy / safety-filter
blocks must classify as ``content_policy_blocked``, be non-retryable, and
trigger the ``is_client_error`` abort path so the loop jumps straight to a
configured fallback or surfaces a clear policy-block message — instead of
burning ``api_max_retries`` paid attempts on a deterministic refusal and
delivering "API failed after 3 retries" to Telegram/cron with no provider
context.

Real-world symptom from the issue:
    ``API call failed after 3 retries — This content was flagged for
    possible cybersecurity risk... | provider=openai-codex model=gpt-5.5``
repeating across cron jobs and gateway sessions, with the user unable to
tell whether the gateway was broken, the model was down, or their prompt
was the problem.
"""
from __future__ import annotations

from agent.conversation_loop import _is_client_error_from_policy
from agent.failure_policy import decide_api_recovery

class TestContentPolicyBlockedClassification:
    """Verify classify_api_error returns the right shape so downstream
    recovery (fallback activation, final_response wording) fires correctly.
    """

    def test_openai_codex_cybersecurity_no_status(self):
        """The reported #18028 case — SDK raises without a status code."""
        from agent.error_classifier import classify_api_error, FailoverReason

        e = Exception(
            "This content was flagged for possible cybersecurity risk. "
            "If this seems wrong, try rephrasing your request. To get "
            "authorized for security work, join the Trusted Access for "
            "Cyber program."
        )
        result = classify_api_error(e, provider="openai-codex", model="gpt-5.5")
        # Must NOT fall into the retryable ``unknown`` bucket — that's what
        # caused the 3x retry burn.
        assert result.reason == FailoverReason.content_policy_blocked
        assert result.retryable is False
        # Recovery is fallback model, not credential rotation or compression.
        assert result.should_fallback is True
        assert result.should_compress is False
        assert result.should_rotate_credential is False


class TestContentPolicyTriggersClientErrorAbort:
    """Verify the live recovery-policy wiring takes the abort path."""

    def test_content_policy_blocked_triggers_abort(self):
        """Safety-filter block must reach is_client_error → fallback/abort."""
        from agent.error_classifier import ClassifiedError, FailoverReason

        classified = ClassifiedError(
            reason=FailoverReason.content_policy_blocked,
            retryable=False,
            should_compress=False,
        )
        recovery_decision = decide_api_recovery(classified)

        assert _is_client_error_from_policy(
            classified,
            recovery_decision,
        ), (
            "FailoverReason.content_policy_blocked must trigger the "
            "is_client_error path so fallback fires immediately instead of "
            "burning api_max_retries paid attempts on a deterministic "
            "safety refusal — see #18028."
        )


class TestContentPolicyPatternsAreNarrow:
    """Defensive guard: the safety-filter patterns must not collide with
    benign error wording from billing / format / generic 400 errors. If
    these regress to ``content_policy_blocked``, recovery will route to
    the wrong code path (fallback model instead of credential rotation).
    """

    def test_generic_400_format_error_not_misclassified(self):
        from agent.error_classifier import classify_api_error, FailoverReason

        class _Err(Exception):
            def __init__(self, msg, status_code):
                super().__init__(msg)
                self.status_code = status_code

        e = _Err("Invalid request: messages must be a non-empty list", status_code=400)
        result = classify_api_error(e, provider="openai", model="gpt-4o")
        assert result.reason != FailoverReason.content_policy_blocked

    def test_billing_402_not_misclassified(self):
        from agent.error_classifier import classify_api_error, FailoverReason

        class _Err(Exception):
            def __init__(self, msg, status_code):
                super().__init__(msg)
                self.status_code = status_code

        e = _Err("Insufficient credits. Top up your balance.", status_code=402)
        result = classify_api_error(e, provider="openrouter", model="anthropic/claude-opus")
        assert result.reason == FailoverReason.billing

    def test_openrouter_account_policy_block_stays_distinct(self):
        """``provider_policy_blocked`` (OpenRouter account-level data
        policy) must remain a separate classification from
        ``content_policy_blocked`` (upstream model safety filter) — they
        have different recovery strategies.
        """
        from agent.error_classifier import classify_api_error, FailoverReason

        class _Err(Exception):
            def __init__(self, msg, status_code):
                super().__init__(msg)
                self.status_code = status_code

        e = _Err(
            "No endpoints available matching your guardrail restrictions "
            "and data policy",
            status_code=404,
        )
        result = classify_api_error(e, provider="openrouter", model="anthropic/claude-opus")
        assert result.reason == FailoverReason.provider_policy_blocked
        assert result.reason != FailoverReason.content_policy_blocked
