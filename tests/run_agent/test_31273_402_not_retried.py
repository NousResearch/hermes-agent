"""Regression guard for #31273: HTTP 402 (billing exhaustion) must abort
after credential-pool rotation and provider fallback have failed.

Before the fix, ``FailoverReason.billing`` was in the exclusion set that
prevents the loop's ``is_client_error`` branch from firing.  When a user
ran a pay-per-token provider (OpenRouter, etc.) with no credential pool
and no fallback configured, a single 402 cascaded into
``agent.api_max_retries`` paid requests against an exhausted balance.
Real-world impact: ~$40 burned in 48h on a 24/7 gateway routing Telegram
+ Discord traffic.

The fix removes ``FailoverReason.billing`` from the exclusion set.  By
the time control reaches the ``is_client_error`` check:
  * credential-pool rotation has already run (and either ``continue``d
    on rotation, or returned False because the pool is exhausted/absent).
  * the eager-fallback branch for billing has also run (and either
    ``continue``d on fallback activation, or fell through because no
    fallback is configured).
Falling through to the retry-backoff path from here just burns paid
requests with no recovery mechanism left.  Aborting mirrors how 401/403
(also ``should_fallback=True``) already behave once their recovery paths
have failed.
"""
from __future__ import annotations


class TestBillingTriggersClientErrorAbort:
    """Mirror the ``is_client_error`` predicate shape used in
    ``agent/conversation_loop.py`` and verify ``FailoverReason.billing``
    now resolves to True (i.e. aborts the loop).
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

        Kept in lock-step with the source.  If you change one, change
        both — or, better, refactor the predicate into a shared helper
        and have both sites import it.
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

    def test_billing_now_aborts_the_loop(self):
        """402 with no fallback / no pool entry → ``is_client_error`` True."""
        from agent.error_classifier import FailoverReason

        # This is what classify_api_error() returns for a plain 402:
        #   reason=billing, retryable=False, should_compress=False
        assert self._mirror_is_client_error(
            classified_retryable=False,
            classified_reason=FailoverReason.billing,
        ), (
            "FailoverReason.billing must trigger is_client_error abort after "
            "credential-pool rotation and provider fallback have failed — see #31273."
        )

    def test_rate_limit_still_retries(self):
        """Sanity check: rate_limit must still fall through to backoff retry."""
        from agent.error_classifier import FailoverReason

        # 429 / transient 402 / rate-limited usage: must NOT abort,
        # because Retry-After backoff and pool rotation are the right
        # recovery paths.
        assert not self._mirror_is_client_error(
            classified_retryable=True,
            classified_reason=FailoverReason.rate_limit,
        )

    def test_local_validation_error_still_aborts(self):
        """Sanity check: bare ValueError/TypeError still abort."""
        from agent.error_classifier import FailoverReason

        assert self._mirror_is_client_error(
            classified_retryable=True,
            classified_reason=FailoverReason.unknown,
            is_local_validation_error=True,
        )

    def test_context_overflow_still_falls_through_to_compression(self):
        """Sanity check: context-overflow must NOT be classified as
        client error — compression is the recovery path."""
        from agent.error_classifier import FailoverReason

        assert not self._mirror_is_client_error(
            classified_retryable=True,
            classified_reason=FailoverReason.context_overflow,
            classified_should_compress=True,
        )


class TestSourceStillHasBillingExclusionRemoved:
    """Belt-and-suspenders: the production source must actually omit
    ``FailoverReason.billing`` from the ``is_client_error`` exclusion
    set.  Protects against an accidental re-introduction.
    """

    def test_conversation_loop_omits_billing_from_client_error_exclusion(self):
        import inspect
        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)

        # Locate the is_client_error block and inspect its exclusion set.
        marker = "is_client_error = ("
        assert marker in src, (
            "agent/conversation_loop.py must define is_client_error — "
            "the bug-fix anchor for #31273 has moved or been renamed."
        )
        idx = src.index(marker)
        # Window large enough to span the full predicate (~30 lines).
        window = src[idx:idx + 2000]

        assert "FailoverReason.rate_limit" in window, (
            "is_client_error exclusion set has changed shape — re-verify "
            "that FailoverReason.billing is still NOT in it (#31273)."
        )
        assert "FailoverReason.billing" not in window, (
            "FailoverReason.billing must NOT appear in the is_client_error "
            "exclusion set — see #31273.  Billing (HTTP 402) is non-retryable "
            "by the time control reaches this block: credential-pool rotation "
            "and provider fallback have both already had their chance to "
            "continue the loop.  Re-adding it causes runaway token spend on "
            "depleted balances."
        )
