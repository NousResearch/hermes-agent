"""Tests for STATUS_DEAD: permanent auth failure handling in credential pool.

Verifies that token_invalidated/token_revoked errors cause credentials to be
permanently removed from rotation (STATUS_DEAD) instead of getting a temporary
exhausted cooldown (STATUS_EXHAUSTED) that lets them re-enter after 5 minutes.

Issue: https://github.com/NousResearch/hermes-agent/issues/32849
"""

import json
import time
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import (
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
    _is_permanent_auth_failure,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_entry(
    entry_id: str = "test-1",
    label: str = "Test Key",
    last_status: str | None = None,
) -> PooledCredential:
    return PooledCredential(
        provider="openrouter",
        id=entry_id,
        label=label,
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-test-fake-key",
        last_status=last_status,
    )


def _make_pool(n_entries: int = 3) -> CredentialPool:
    """Create a pool with n_entries fake credentials (no file I/O)."""
    entries = [
        _make_entry(entry_id=f"key-{i}", label=f"Key {i}")
        for i in range(n_entries)
    ]
    with patch.object(CredentialPool, "_persist"):
        pool = CredentialPool.__new__(CredentialPool)
        pool.provider = "openrouter"
        pool._entries = entries
        pool._current_id = entries[0].id
        pool._strategy = "fill_first"
        pool._lock = MagicMock()
        pool._lock.__enter__ = MagicMock(return_value=None)
        pool._lock.__exit__ = MagicMock(return_value=False)
        pool._active_leases = {}
        pool._max_concurrent = 1
        pool._persist = MagicMock()
    return pool


# ── _is_permanent_auth_failure ──────────────────────────────────────────


class TestIsPermanentAuthFailure:
    """Test the pattern-matching function that distinguishes permanent vs transient."""

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("token_invalidated", True),
            ("Error: token_invalidated for key", True),
            ("token_revoked", True),
            ("The token_revoked error occurred", True),
            ("invalid_api_key", True),
            ("API key has been revoked", True),
            ("API key is invalid", True),
            ("API key not found", True),
            ("key has been deactivated", True),
            # Transient patterns — should NOT be permanent
            ("token expired", False),
            ("Token has been invalidated", False),  # spaces, not underscore — not our pattern
            ("authentication required", False),
            ("unauthorized access", False),
            ("rate limit exceeded", False),
            ("", False),
        ],
    )
    def test_message_patterns(self, message: str, expected: bool):
        ctx = {"message": message, "reason": "", "code": "", "error": ""}
        assert _is_permanent_auth_failure(401, ctx) is expected

    def test_reason_field_also_checked(self):
        ctx = {"message": "", "reason": "token_invalidated", "code": "", "error": ""}
        assert _is_permanent_auth_failure(401, ctx) is True

    def test_error_field_also_checked(self):
        ctx = {"message": "", "reason": "", "code": "", "error": "token_revoked"}
        assert _is_permanent_auth_failure(401, ctx) is True

    def test_non_401_status_not_permanent(self):
        ctx = {"message": "token_revoked", "reason": "", "code": "", "error": ""}
        # 429 with "token_revoked" in message — not an auth status code
        assert _is_permanent_auth_failure(429, ctx) is False

    def test_403_can_be_permanent(self):
        ctx = {"message": "account is deactivated", "reason": "", "code": "", "error": ""}
        assert _is_permanent_auth_failure(403, ctx) is True

    def test_none_context_not_permanent(self):
        assert _is_permanent_auth_failure(401, None) is False

    def test_empty_context_not_permanent(self):
        assert _is_permanent_auth_failure(401, {}) is False


# ── _mark_exhausted → STATUS_DEAD ──────────────────────────────────────


class TestMarkExhaustedDead:
    """Test that _mark_exhausted uses STATUS_DEAD for permanent failures."""

    def test_permanent_401_gets_dead_status(self):
        pool = _make_pool(2)
        entry = pool._entries[0]

        error_ctx = {
            "message": "token_invalidated: the API key has been revoked",
            "reason": "token_invalidated",
        }
        updated = pool._mark_exhausted(entry, status_code=401, error_context=error_ctx)

        assert updated.last_status == STATUS_DEAD
        assert updated.last_error_reset_at is None  # No cooldown
        assert updated.last_error_code == 401

    def test_transient_401_gets_exhausted_status(self):
        pool = _make_pool(2)
        entry = pool._entries[0]

        error_ctx = {
            "message": "Unauthorized: token expired",
            "reason": "auth",
        }
        updated = pool._mark_exhausted(entry, status_code=401, error_context=error_ctx)

        assert updated.last_status == STATUS_EXHAUSTED
        # reset_at may be None if no provider-supplied timestamp;
        # _exhausted_until() falls back to last_status_at + TTL
        from agent.credential_pool import _exhausted_until
        cooldown = _exhausted_until(updated)
        assert cooldown is not None and cooldown > time.time()  # Has cooldown window

    def test_429_stays_exhausted(self):
        pool = _make_pool(2)
        entry = pool._entries[0]

        error_ctx = {"message": "rate limit exceeded"}
        updated = pool._mark_exhausted(entry, status_code=429, error_context=error_ctx)

        assert updated.last_status == STATUS_EXHAUSTED


# ── _available_entries skips DEAD ──────────────────────────────────────


class TestAvailableEntriesSkipsDead:
    """Test that STATUS_DEAD entries are never returned as available."""

    def test_dead_entry_excluded_from_available(self):
        pool = _make_pool(3)
        # Mark first entry as dead
        dead_entry = replace(pool._entries[0], last_status=STATUS_DEAD)
        pool._entries[0] = dead_entry

        available = pool._available_entries(clear_expired=True)
        assert len(available) == 2
        assert all(e.last_status != STATUS_DEAD for e in available)

    def test_dead_entry_never_resurrects_after_time(self):
        pool = _make_pool(3)
        # Mark first entry as dead (with timestamp in the past)
        dead_entry = replace(
            pool._entries[0],
            last_status=STATUS_DEAD,
            last_status_at=time.time() - 3600,  # 1 hour ago
        )
        pool._entries[0] = dead_entry

        # Even with clear_expired=True, dead entries stay dead
        available = pool._available_entries(clear_expired=True)
        assert len(available) == 2

    def test_all_dead_returns_empty(self):
        pool = _make_pool(2)
        pool._entries = [
            replace(e, last_status=STATUS_DEAD)
            for e in pool._entries
        ]

        available = pool._available_entries(clear_expired=True)
        assert available == []


# ── mark_exhausted_and_rotate with dead ─────────────────────────────────


class TestMarkExhaustedAndRotateDead:
    """Integration test: mark_exhausted_and_rotate correctly handles permanent failures."""

    def test_permanent_failure_rotates_past_dead(self):
        pool = _make_pool(3)

        error_ctx = {
            "message": "invalid_api_key: the key has been deactivated",
            "reason": "invalid_api_key",
        }
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context=error_ctx,
            api_key_hint="sk-test-fake-key",
        )

        # Should rotate to a different entry
        assert next_entry is not None
        assert next_entry.id != "key-0"

        # Original entry should be dead
        original = pool._entries[0]
        assert original.last_status == STATUS_DEAD

    def test_permanent_failure_then_rotation_excludes_dead(self):
        """After marking dead, subsequent rotations skip the dead entry."""
        pool = _make_pool(3)

        # Kill key-0
        error_ctx = {"message": "token_revoked"}
        pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context=error_ctx,
            api_key_hint="sk-test-fake-key",
        )

        # Now kill key-1 with a transient error
        pool._current_id = pool._entries[1].id
        pool.mark_exhausted_and_rotate(status_code=429)

        # Next available should be key-2, not key-0 (dead)
        available = pool._available_entries(clear_expired=False)
        ids = [e.id for e in available]
        assert "key-0" not in ids  # Dead — excluded
        assert "key-2" in ids      # Still available


# ── error_classifier integration ────────────────────────────────────────


class TestErrorClassifierPermanentAuth:
    """Verify error_classifier returns auth_permanent for permanent 401 errors."""

    def test_token_revoked_classified_permanent(self):
        from agent.error_classifier import FailoverReason, classify_api_error

        exc = Exception("401 token_revoked: The API key has been invalidated")
        exc.status_code = 401  # type: ignore[attr-defined]
        result = classify_api_error(exc)
        assert result.reason == FailoverReason.auth_permanent
        assert result.should_rotate_credential is True
        assert result.retryable is False

    def test_invalid_api_key_classified_permanent(self):
        from agent.error_classifier import FailoverReason, classify_api_error

        exc = Exception('{"error": {"message": "invalid_api_key", "code": 401}}')
        exc.status_code = 401  # type: ignore[attr-defined]
        result = classify_api_error(exc)
        assert result.reason == FailoverReason.auth_permanent

    def test_transient_401_stays_auth(self):
        from agent.error_classifier import FailoverReason, classify_api_error

        exc = Exception("Unauthorized: token expired")
        exc.status_code = 401  # type: ignore[attr-defined]
        result = classify_api_error(exc)
        assert result.reason == FailoverReason.auth
        assert result.reason != FailoverReason.auth_permanent


# ── reset_statuses can revive dead entries ──────────────────────────────


class TestResetStatusesRevivesDead:
    """Manual reset (user action) should clear STATUS_DEAD too."""

    def test_reset_clears_dead(self):
        pool = _make_pool(2)
        pool._entries[0] = replace(
            pool._entries[0],
            last_status=STATUS_DEAD,
            last_error_code=401,
            last_error_message="token_revoked",
        )

        count = pool.reset_statuses()
        assert count == 1
        assert pool._entries[0].last_status is None
