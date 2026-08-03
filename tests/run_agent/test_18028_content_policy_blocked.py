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

    def test_minimax_output_safety_filter(self):
        """#32421 — MiniMax output-layer safety filter (e.g. ``output
        new_sensitive (1027)``) trips mid-stream when the model emits a
        large tool-call argument block. Must classify as
        ``content_policy_blocked`` so the loop aborts the 3x retry burn and
        routes to a configured fallback model.
        """
        from agent.error_classifier import classify_api_error, FailoverReason

        e = Exception(
            "Stream stalled mid tool-call: output new_sensitive (1027) "
            "[MiniMax-M2.7] — request was rejected by upstream safety "
            "filter, see provider response for details."
        )
        result = classify_api_error(e, provider="MiniMax", model="MiniMax-M2.7")
        assert result.reason == FailoverReason.content_policy_blocked
        assert result.retryable is False
        assert result.should_fallback is True
        assert result.should_compress is False
        assert result.should_rotate_credential is False


class TestContentPolicyTriggersClientErrorAbort:
    """Mirror the ``is_client_error`` predicate in
    ``agent/conversation_loop.py`` and verify
    ``FailoverReason.content_policy_blocked`` resolves to True so the loop
    aborts (after attempting fallback) instead of falling into the
    retry-backoff path.
    """

    def _mirror_is_client_error(
        self,
        *,
        classified_retryable: bool,
        classified_reason,
        classified_should_compress: bool = False,
        is_local_validation_error: bool = False,
        is_context_length_error: bool = False,
    ) -> bool:
        """Exact shape of conversation_loop.py's is_client_error check.

        Kept in lock-step with the source. If you change one, change both.
        """
        from agent.error_classifier import FailoverReason

        return (
            is_local_validation_error
            or (
                not classified_retryable
                and not classified_should_compress
                and classified_reason not in {
                    FailoverReason.rate_limit,
                    FailoverReason.overloaded,
                    FailoverReason.context_overflow,
                    FailoverReason.payload_too_large,
                    FailoverReason.long_context_tier,
                    FailoverReason.thinking_signature,
                }
            )
        ) and not is_context_length_error

    def test_content_policy_blocked_triggers_abort(self):
        """Safety-filter block must reach is_client_error → fallback/abort."""
        from agent.error_classifier import FailoverReason

        # What classify_api_error returns for a content-policy block:
        #   reason=content_policy_blocked, retryable=False, should_compress=False
        assert self._mirror_is_client_error(
            classified_retryable=False,
            classified_reason=FailoverReason.content_policy_blocked,
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
