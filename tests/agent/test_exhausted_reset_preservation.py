"""Exhausted-entry reset preservation on re-mark.

Cluster: the pool gate fails a turn LOCALLY (``select()`` → "no available entries" →
synthesized RateLimitError) when the sole credential is already exhausted. That local
error carries no provider payload, so the rotate path re-marks the entry with
``error_context=None`` — and ``_mark_exhausted`` overwrote the provider-declared
``last_error_reset_at`` (Codex ``resets_at``) with ``None``. The next ``next_available_at()``
fell back to the 60s sole-credential TTL, the cooldown sizing saw no wait information,
and long-lived gateway sessions re-probed the proven-empty quota window every turn.

Invariant: a re-mark without fresh wait information never erases a provider-declared reset.
"""

import time

from agent.credential_pool import (
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)


def _exhausted_entry(reset_at):
    return PooledCredential(
        id="test-entry", provider="openai-codex", label="codex", source="device_code",
        auth_type="oauth", priority=0, access_token="tok",
        last_status=STATUS_EXHAUSTED, last_status_at=time.time(),
        last_error_code=429, last_error_reset_at=reset_at,
    )


class TestResetPreservation:
    def test_remark_without_context_preserves_reset(self):
        """A local-fail re-mark (error_context=None) keeps the stored provider reset."""
        reset_at = time.time() + 100_000
        entry = _exhausted_entry(reset_at)
        pool = CredentialPool("openai-codex", [entry])

        marked = pool._mark_exhausted(entry, 429, None, persist=False)
        assert marked.last_error_reset_at == reset_at

    def test_remark_with_fresh_reset_wins(self):
        """A wire 429 with a fresh provider payload overwrites the stored reset."""
        old_reset = time.time() + 100_000
        entry = _exhausted_entry(old_reset)
        pool = CredentialPool("openai-codex", [entry])

        fresh_reset = time.time() + 200_000
        marked = pool._mark_exhausted(
            entry, 429, {"reset_at": fresh_reset, "reason": "usage_limit_reached"}, persist=False,
        )
        assert marked.last_error_reset_at == fresh_reset
