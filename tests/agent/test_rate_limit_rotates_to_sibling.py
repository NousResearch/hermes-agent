"""A rate limit with a healthy sibling credential rotates instead of waiting.

Regression for the case where a provider's per-account 429 carries none of the
usage-limit tokens the recovery path matches on. Anthropic returns "This
request would exceed your account's rate limit. Please try again later." — no
"usage limit reached", no "usage_limit_reached" — so the first 429 fell through
to the generic backoff and slept out the provider's Retry-After, up to a 600s
cap, while a second authenticated credential sat in the pool unused.

These assert the behaviour contract, not the message text: what matters is that
the decision follows *whether another credential could serve the call*, so the
next provider wording change cannot reintroduce the wait.
"""

import time

import pytest

from agent.agent_runtime_helpers import _another_credential_is_usable, _recover_rate_limit
from agent.credential_pool import (
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)


def _credential(cid, *, status=STATUS_OK, status_at=None, reset_at=None):
    entry = PooledCredential(
        provider="anthropic",
        id=cid,
        label=cid,
        auth_type="oauth",
        priority=0,
        source="auth.json",
        access_token=f"token-{cid}",
    )
    entry.last_status = status
    if status_at is not None:
        entry.last_status_at = status_at
    if reset_at is not None:
        entry.last_error_reset_at = reset_at
    return entry


def _pool(*entries):
    return CredentialPool("anthropic", list(entries))


# The wording the provider actually sent when this was measured. It is here as
# a fixture, not as an assertion target — no test below matches on it.
ACCOUNT_RATE_LIMIT = {
    "reason": "rate_limit_error",
    "message": "This request would exceed your account's rate limit. Please try again later.",
}


class TestAnotherCredentialIsUsable:
    def test_a_healthy_sibling_counts_as_usable(self):
        first, second = _credential("one"), _credential("two")
        assert _another_credential_is_usable(_pool(first, second), first) is True

    def test_the_failing_credential_does_not_count_as_its_own_sibling(self):
        only = _credential("one")
        assert _another_credential_is_usable(_pool(only), only) is False

    def test_a_sibling_still_in_cooldown_does_not_count(self):
        first = _credential("one")
        second = _credential(
            "two",
            status=STATUS_EXHAUSTED,
            reset_at=time.time() + 3600,
        )
        assert _another_credential_is_usable(_pool(first, second), first) is False

    def test_a_sibling_whose_cooldown_has_passed_counts_again(self):
        first = _credential("one")
        second = _credential(
            "two",
            status=STATUS_EXHAUSTED,
            reset_at=time.time() - 60,
        )
        assert _another_credential_is_usable(_pool(first, second), first) is True


class TestRecoverRateLimit:
    def _run(self, pool, current_id, *, has_retried=False):
        rotated = []

        def rotate_and_swap(code, reason):
            rotated.append((code, reason))
            return True

        result = _recover_rate_limit(
            pool,
            has_retried_429=has_retried,
            error_context=ACCOUNT_RATE_LIMIT,
            api_key_hint=None,
            credential_id=current_id,
            rotate_and_swap=rotate_and_swap,
        )
        return result, rotated

    def test_rotates_on_the_first_429_when_a_sibling_can_answer(self):
        """The defect: this used to return (False, True) and sleep instead."""
        first, second = _credential("one"), _credential("two")
        (retry_now, _), rotated = self._run(_pool(first, second), "one")

        assert retry_now is True, "should retry immediately on the rotated credential"
        assert rotated, "a usable sibling must be rotated to, not waited out"

    def test_waits_when_the_failing_credential_is_the_only_one(self):
        """A pool of one has nothing to rotate to; backing off is correct."""
        only = _credential("one")
        (retry_now, has_retried), rotated = self._run(_pool(only), "one")

        assert retry_now is False
        assert has_retried is True
        assert rotated == [], "nothing to rotate to — the wait is the right answer"

    def test_waits_when_every_sibling_is_still_in_cooldown(self):
        first = _credential("one")
        second = _credential(
            "two", status=STATUS_EXHAUSTED, reset_at=time.time() + 3600
        )
        (retry_now, _), rotated = self._run(_pool(first, second), "one")

        assert retry_now is False
        assert rotated == []

    def test_an_already_exhausted_credential_still_rotates_first(self):
        """The pre-existing fast path must keep working."""
        first = _credential(
            "one", status=STATUS_EXHAUSTED, reset_at=time.time() + 3600
        )
        second = _credential("two")
        (retry_now, _), rotated = self._run(_pool(first, second), "one")

        assert retry_now is True
        assert rotated and "pre-exhausted" in rotated[0][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
