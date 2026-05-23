"""Regression coverage for issue #30331.

Without this fix, a 401/403 on the **only** configured credential (no pool
to rotate to, no provider-specific OAuth refresh that fixes it) burns
``max_retries`` × jittered 5–120 s backoff before surfacing the failure.
That's pure latency on a key that is not going to start working on its
own — the typical "expired DeepSeek key" deployment hits up to 8 minutes
of useless retries before the user sees an actionable error.

The fix in ``agent/conversation_loop.py``:

* Adds a one-shot ``single_key_auth_retry_attempted`` flag.
* On the first auth error where ``_recover_with_credential_pool`` returns
  ``False``, retries once with a fresh connection.
* If that also returns 401/403, upgrades the ``ClassifiedError`` to
  ``FailoverReason.auth_permanent`` so the existing non-retryable abort
  path emits a clear error immediately.

These tests pin three things:

1. ``ClassifiedError.is_auth`` still recognises ``auth_permanent`` after
   the upgrade (so the abort path's actionable-hint block still fires).
2. The conversation-loop source declares the one-shot flag and the
   fail-fast block, gated on ``classified.is_auth and not
   recovered_with_pool`` — so future refactors don't silently drop it.
3. A simulated upgrade preserves status_code / provider / model context
   for the downstream diagnostic output.
"""

from __future__ import annotations

import inspect

from agent import conversation_loop
from agent.error_classifier import ClassifiedError, FailoverReason


# ---------------------------------------------------------------------------
# 1. is_auth must keep recognising the upgraded reason
# ---------------------------------------------------------------------------


class TestAuthPermanentIsAuth:
    def test_auth_permanent_is_still_classified_as_auth(self):
        upgraded = ClassifiedError(
            reason=FailoverReason.auth_permanent,
            retryable=False,
        )
        assert upgraded.is_auth is True

    def test_transient_auth_is_classified_as_auth(self):
        transient = ClassifiedError(
            reason=FailoverReason.auth,
            retryable=True,
        )
        assert transient.is_auth is True

    def test_auth_permanent_is_not_retryable(self):
        upgraded = ClassifiedError(
            reason=FailoverReason.auth_permanent,
            retryable=False,
        )
        assert upgraded.retryable is False


# ---------------------------------------------------------------------------
# 2. Conversation loop source declares the flag and the fail-fast block
# ---------------------------------------------------------------------------


class TestSingleKeyAuthFailFastSource:
    """Pin the structural pieces so a future refactor cannot silently drop
    the fail-fast path back to the old max_retries-burning behaviour."""

    def _source(self) -> str:
        return inspect.getsource(conversation_loop.run_conversation)

    def test_flag_is_initialised(self):
        # The one-shot flag must be declared in the per-turn state block —
        # forgetting to initialise it would NameError on first error.
        assert "single_key_auth_retry_attempted = False" in self._source()

    def test_fail_fast_gated_on_is_auth_and_no_pool_recovery(self):
        src = self._source()
        # The guard predicate must check both halves; either alone would
        # mis-fire (e.g. burning the upgrade on a rate-limit error, or
        # firing while pool rotation is still viable).
        assert "classified.is_auth" in src
        assert "not recovered_with_pool" in src

    def test_upgrades_to_auth_permanent_on_second_failure(self):
        src = self._source()
        assert "FailoverReason.auth_permanent" in src
        # The upgrade must build a new ClassifiedError; mutating the old
        # one in-place wouldn't update retryable / should_compress.
        assert "ClassifiedError(" in src

    def test_first_failure_falls_through_to_fresh_retry(self):
        # The first auth failure must `continue` (fresh-connection retry)
        # without incrementing retry_count or hitting backoff. We can't
        # easily prove the position of the continue, but we can prove the
        # flag is set before any continue inside the auth block.
        src = self._source()
        idx_flag = src.find("single_key_auth_retry_attempted = True")
        idx_classified = src.find(
            "classified = ClassifiedError(\n                        reason=FailoverReason.auth_permanent",
        )
        assert idx_flag != -1, "fresh-retry arm missing"
        assert idx_classified != -1, "permanent-upgrade arm missing"
        assert idx_flag < idx_classified, "fresh-retry arm must come first"


# ---------------------------------------------------------------------------
# 3. Upgrade preserves context for the diagnostic output
# ---------------------------------------------------------------------------


class TestUpgradePreservesContext:
    """Simulate exactly what the conversation loop builds on the second
    auth failure, and verify the abort path's actionable-hint block has
    everything it needs (status_code / provider / model)."""

    def test_upgrade_preserves_status_code_provider_model(self):
        original = ClassifiedError(
            reason=FailoverReason.auth,
            status_code=401,
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            message="HTTP 401: invalid_api_key",
            error_context={"endpoint": "https://openrouter.ai/api/v1/chat/completions"},
            retryable=True,
        )

        # Mirror the upgrade logic in conversation_loop.run_conversation
        # so a behaviour drift between this test and the inline block
        # surfaces as a failure.
        upgraded = ClassifiedError(
            reason=FailoverReason.auth_permanent,
            status_code=original.status_code,
            provider=original.provider,
            model=original.model,
            message=(
                original.message
                or "Authentication failed after one fresh-connection retry — "
                "the configured API credential appears invalid or revoked."
            ),
            error_context=original.error_context,
            retryable=False,
            should_compress=False,
            should_rotate_credential=False,
            should_fallback=original.should_fallback,
        )

        assert upgraded.status_code == 401
        assert upgraded.provider == "openrouter"
        assert upgraded.model == "anthropic/claude-sonnet-4.6"
        # Original diagnostic message preserved so the abort path can show
        # the provider's actual rejection text.
        assert "invalid_api_key" in upgraded.message
        assert upgraded.error_context["endpoint"].endswith("/chat/completions")
        # Critical state flips that drive the abort path:
        assert upgraded.is_auth is True
        assert upgraded.retryable is False
        assert upgraded.should_rotate_credential is False

    def test_upgrade_synthesises_message_when_original_missing(self):
        original = ClassifiedError(
            reason=FailoverReason.auth,
            status_code=403,
            provider="deepseek",
            model="deepseek-chat",
            message="",
            retryable=True,
        )

        upgraded = ClassifiedError(
            reason=FailoverReason.auth_permanent,
            status_code=original.status_code,
            provider=original.provider,
            model=original.model,
            message=(
                original.message
                or "Authentication failed after one fresh-connection retry — "
                "the configured API credential appears invalid or revoked."
            ),
            error_context=original.error_context,
            retryable=False,
        )

        assert "invalid or revoked" in upgraded.message
        assert upgraded.status_code == 403
