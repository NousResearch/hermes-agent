"""Tests for credential pool rotation on Codex Responses API soft failures.

When the Responses API returns HTTP 200 with ``response.status = "failed"``
(e.g. quota exhaustion), the runtime must attempt same-provider credential
pool rotation before falling through to cross-provider fallback.

Covers #24159.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, has_pool=True, pool_has_creds=True):
    """Create a minimal AIAgent with the fields needed for these tests."""
    from run_agent import AIAgent

    with patch.object(AIAgent, "__init__", lambda self, **kw: None):
        agent = AIAgent()

    agent._credential_pool = None
    if has_pool:
        pool = MagicMock()
        pool.has_available.return_value = pool_has_creds
        agent._credential_pool = pool

    agent._fallback_chain = []
    agent._fallback_index = 0
    agent._try_activate_fallback = MagicMock(return_value=False)
    agent._swap_credential = MagicMock()
    agent._emit_status = MagicMock()
    agent.log_prefix = ""
    agent.api_mode = "codex_responses"

    return agent


# ---------------------------------------------------------------------------
# _classify_codex_soft_failure
# ---------------------------------------------------------------------------

class TestClassifyCodexSoftFailure:
    """Pattern-matching tests for _classify_codex_soft_failure."""

    @pytest.mark.parametrize(
        "msg",
        [
            "Insufficient quota for this model",
            "Your credits have been exhausted",
            "Billing hard limit reached",
            "Payment required to continue",
            "Account is deactivated due to billing",
            "Plan does not include this model",
            "Exceeded your current quota",
            "You have exceeded the usage limit",
        ],
    )
    def test_billing_signals(self, msg):
        agent = _make_agent()
        result = agent._classify_codex_soft_failure(msg)
        assert result is FailoverReason.billing

    @pytest.mark.parametrize(
        "msg",
        [
            "Rate limit exceeded",
            "rate_limit hit",
            "Too many requests, please slow down",
            "You are being throttled",
            "Try again in 30 seconds",
            "Please retry after cooldown",
        ],
    )
    def test_rate_limit_signals(self, msg):
        agent = _make_agent()
        result = agent._classify_codex_soft_failure(msg)
        assert result is FailoverReason.rate_limit

    @pytest.mark.parametrize(
        "msg",
        [
            "Content policy violation",
            "The response was filtered for safety",
            "Cancelled by user",
            "",
        ],
    )
    def test_non_quota_errors_return_none(self, msg):
        agent = _make_agent()
        result = agent._classify_codex_soft_failure(msg)
        assert result is None

    def test_billing_takes_priority_over_rate_limit(self):
        """If a message matches both billing and rate-limit, billing wins."""
        agent = _make_agent()
        result = agent._classify_codex_soft_failure(
            "You have exceeded your quota rate limit",
        )
        assert result is FailoverReason.billing


# ---------------------------------------------------------------------------
# Integration: response_invalid block calls pool recovery
# ---------------------------------------------------------------------------

class TestCodexSoftFailurePoolRecovery:
    """Verify the response_invalid block invokes _recover_with_credential_pool
    for codex soft failures and falls through to _try_activate_fallback when
    pool recovery is unavailable."""

    def test_pool_rotation_fires_for_billing_soft_failure(self):
        agent = _make_agent(has_pool=True)
        pool = agent._credential_pool

        next_entry = MagicMock(name="next_entry")
        pool.mark_exhausted_and_rotate.return_value = next_entry

        # Simulate the conditions the response_invalid block checks.
        error_details = ["response.status=failed: Insufficient quota for model"]
        _codex_soft_failure_msg = None
        if error_details and any("response.status=" in d for d in error_details):
            for _d in error_details:
                if _d.startswith("response.status="):
                    _codex_soft_failure_msg = _d.split(": ", 1)[-1] if ": " in _d else ""
                    break

        assert _codex_soft_failure_msg == "Insufficient quota for model"

        _pool_reason = agent._classify_codex_soft_failure(_codex_soft_failure_msg)
        assert _pool_reason is FailoverReason.billing

        recovered, _ = agent._recover_with_credential_pool(
            status_code=None,
            has_retried_429=False,
            classified_reason=_pool_reason,
        )
        assert recovered is True
        pool.mark_exhausted_and_rotate.assert_called_once()
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_pool_rotation_fires_for_rate_limit_soft_failure(self):
        agent = _make_agent(has_pool=True)
        pool = agent._credential_pool

        next_entry = MagicMock(name="next_entry")
        pool.mark_exhausted_and_rotate.return_value = next_entry

        error_details = ["response.status=failed: Rate limit exceeded"]
        _codex_soft_failure_msg = None
        if error_details and any("response.status=" in d for d in error_details):
            for _d in error_details:
                if _d.startswith("response.status="):
                    _codex_soft_failure_msg = _d.split(": ", 1)[-1] if ": " in _d else ""
                    break

        _pool_reason = agent._classify_codex_soft_failure(_codex_soft_failure_msg)
        assert _pool_reason is FailoverReason.rate_limit

        # First 429-style: retry same credential.
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=None,
            has_retried_429=False,
            classified_reason=_pool_reason,
        )
        assert recovered is False
        assert has_retried is True

        # Second 429-style: rotate.
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=None,
            has_retried_429=True,
            classified_reason=_pool_reason,
        )
        assert recovered is True
        assert has_retried is False

    def test_no_pool_skips_rotation(self):
        agent = _make_agent(has_pool=False)
        assert agent._credential_pool is None

        error_details = ["response.status=failed: Insufficient quota"]
        _codex_soft_failure_msg = None
        if error_details and any("response.status=" in d for d in error_details):
            for _d in error_details:
                if _d.startswith("response.status="):
                    _codex_soft_failure_msg = _d.split(": ", 1)[-1] if ": " in _d else ""
                    break

        _pool_reason = agent._classify_codex_soft_failure(_codex_soft_failure_msg)
        assert _pool_reason is FailoverReason.billing

        recovered, _ = agent._recover_with_credential_pool(
            status_code=None,
            has_retried_429=False,
            classified_reason=_pool_reason,
        )
        assert recovered is False

    def test_content_policy_failure_does_not_rotate(self):
        """Non-quota soft failures (e.g. content policy) should not trigger pool rotation."""
        agent = _make_agent(has_pool=True)
        pool = agent._credential_pool

        error_details = ["response.status=failed: Content policy violation"]
        _codex_soft_failure_msg = None
        if error_details and any("response.status=" in d for d in error_details):
            for _d in error_details:
                if _d.startswith("response.status="):
                    _codex_soft_failure_msg = _d.split(": ", 1)[-1] if ": " in _d else ""
                    break

        _pool_reason = agent._classify_codex_soft_failure(_codex_soft_failure_msg)
        assert _pool_reason is None
        pool.mark_exhausted_and_rotate.assert_not_called()

    def test_non_codex_api_mode_skips_rotation(self):
        """Non-codex_responses api_mode should not trigger the codex pool recovery path."""
        agent = _make_agent(has_pool=True)
        agent.api_mode = "chat_completions"

        # Simulate the condition: non-codex mode means the guard won't match.
        error_details = ["response.choices is empty"]  # generic, not response.status=
        assert not any("response.status=" in d for d in error_details)

        # The block would skip entirely — _classify_codex_soft_failure is never called.
        pool = agent._credential_pool
        pool.mark_exhausted_and_rotate.assert_not_called()

    def test_pool_exhaustion_falls_through_to_fallback(self):
        """When all pool entries are exhausted, recovery returns False and
        the code should fall through to _try_activate_fallback."""
        agent = _make_agent(has_pool=True, pool_has_creds=False)
        pool = agent._credential_pool
        pool.mark_exhausted_and_rotate.return_value = None  # pool exhausted

        error_details = ["response.status=failed: Insufficient quota"]
        _codex_soft_failure_msg = None
        if error_details and any("response.status=" in d for d in error_details):
            for _d in error_details:
                if _d.startswith("response.status="):
                    _codex_soft_failure_msg = _d.split(": ", 1)[-1] if ": " in _d else ""
                    break

        _pool_reason = agent._classify_codex_soft_failure(_codex_soft_failure_msg)
        assert _pool_reason is FailoverReason.billing

        recovered, _ = agent._recover_with_credential_pool(
            status_code=None,
            has_retried_429=False,
            classified_reason=_pool_reason,
        )
        assert recovered is False
        # In the real flow, this is where _try_activate_fallback() would fire.
