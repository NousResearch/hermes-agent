"""Tests for gateway/pairing.py — DM pairing security system."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from gateway.pairing import (
    PairingStore,
    ALPHABET,
    CODE_LENGTH,
    CODE_TTL_SECONDS,
    RATE_LIMIT_SECONDS,
    MAX_PENDING_PER_PLATFORM,
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_SECONDS,
    _secure_write,
)


def _make_store(tmp_path):
    """Create a PairingStore with PAIRING_DIR pointed to tmp_path."""
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        return PairingStore()


# ---------------------------------------------------------------------------
# _secure_write
# ---------------------------------------------------------------------------


class TestSecureWrite:
    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "sub" / "dir" / "file.json"
        _secure_write(target, '{"hello": "world"}')
        assert target.exists()
        assert json.loads(target.read_text()) == {"hello": "world"}

    def test_sets_file_permissions(self, tmp_path):
        target = tmp_path / "secret.json"
        _secure_write(target, "data")
        mode = oct(target.stat().st_mode & 0o777)
        assert mode == "0o600"


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


class TestCodeGeneration:
    def test_code_format(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
        assert code is not None
        assert len(code) == CODE_LENGTH
        assert all(c in ALPHABET for c in code)

    def test_code_uniqueness(self, tmp_path):
        """Multiple codes for different users should be distinct."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            codes = set()
            for i in range(3):
                code = store.generate_code("telegram", f"user{i}")
                assert code is not None
                codes.add(code)
        assert len(codes) == 3

    def test_stores_pending_entry(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            pending = store.list_pending("telegram")
        assert len(pending) == 1
        assert pending[0]["code"] == code
        assert pending[0]["user_id"] == "user1"
        assert pending[0]["user_name"] == "Alice"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_same_user_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user1")
        assert code1 is not None
        assert code2 is None  # rate limited

    def test_different_users_not_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user2")
        assert code1 is not None
        assert code2 is not None

    def test_rate_limit_expires(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            assert code1 is not None

            # Simulate rate limit expiry
            limits = store._load_json(store._rate_limit_path())
            limits["telegram:user1"] = time.time() - RATE_LIMIT_SECONDS - 1
            store._save_json(store._rate_limit_path(), limits)

            code2 = store.generate_code("telegram", "user1")
        assert code2 is not None


# ---------------------------------------------------------------------------
# Max pending limit
# ---------------------------------------------------------------------------


class TestMaxPending:
    def test_max_pending_per_platform(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            codes = []
            for i in range(MAX_PENDING_PER_PLATFORM + 1):
                code = store.generate_code("telegram", f"user{i}")
                codes.append(code)

        # First MAX_PENDING_PER_PLATFORM should succeed
        assert all(c is not None for c in codes[:MAX_PENDING_PER_PLATFORM])
        # Next one should be blocked
        assert codes[MAX_PENDING_PER_PLATFORM] is None

    def test_different_platforms_independent(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for i in range(MAX_PENDING_PER_PLATFORM):
                store.generate_code("telegram", f"user{i}")
            # Different platform should still work
            code = store.generate_code("discord", "user0")
        assert code is not None


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


class TestApprovalFlow:
    def test_approve_valid_code(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", code)

        assert result is not None
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_approved_user_is_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            assert store.is_approved("telegram", "user1") is True

    def test_unapproved_user_not_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            assert store.is_approved("telegram", "nonexistent") is False

    def test_approve_removes_from_pending(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")
            store.approve_code("telegram", code)
            pending = store.list_pending("telegram")
        assert len(pending) == 0

    def test_approve_case_insensitive(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", code.lower())
        assert result is not None

    def test_approve_strips_whitespace(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", f"  {code}  ")
        assert result is not None

    def test_invalid_code_returns_none(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            result = store.approve_code("telegram", "INVALIDCODE")
        assert result is None


# ---------------------------------------------------------------------------
# Lockout after failed attempts
# ---------------------------------------------------------------------------


class TestLockout:
    def test_lockout_after_max_failures(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            # Generate a valid code so platform has data
            store.generate_code("telegram", "user1")

            # Exhaust failed attempts
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONGCODE")

            # Platform should now be locked out — can't generate new codes
            assert store._is_locked_out("telegram") is True

    def test_lockout_blocks_code_generation(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONG")

            code = store.generate_code("telegram", "newuser")
        assert code is None

    def test_lockout_expires(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONG")

            # Simulate lockout expiry
            limits = store._load_json(store._rate_limit_path())
            lockout_key = "_lockout:telegram"
            limits[lockout_key] = time.time() - 1  # expired
            store._save_json(store._rate_limit_path(), limits)

            assert store._is_locked_out("telegram") is False


# ---------------------------------------------------------------------------
# Code expiry
# ---------------------------------------------------------------------------


class TestCodeExpiry:
    def test_expired_codes_cleaned_up(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")

            # Manually expire the code
            pending = store._load_json(store._pending_path("telegram"))
            pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
            store._save_json(store._pending_path("telegram"), pending)

            # Cleanup happens on next operation
            remaining = store.list_pending("telegram")
        assert len(remaining) == 0

    def test_expired_code_cannot_be_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")

            # Expire it
            pending = store._load_json(store._pending_path("telegram"))
            pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
            store._save_json(store._pending_path("telegram"), pending)

            result = store.approve_code("telegram", code)
        assert result is None


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_approved_user(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            assert store.is_approved("telegram", "user1") is True

            revoked = store.revoke("telegram", "user1")
        assert revoked is True
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            assert store.is_approved("telegram", "user1") is False

    def test_revoke_nonexistent_returns_false(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            assert store.revoke("telegram", "nobody") is False


# ---------------------------------------------------------------------------
# List & clear
# ---------------------------------------------------------------------------


class TestListAndClear:
    def test_list_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            approved = store.list_approved("telegram")
        assert len(approved) == 1
        assert approved[0]["user_id"] == "user1"
        assert approved[0]["platform"] == "telegram"

    def test_list_approved_all_platforms(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            c1 = store.generate_code("telegram", "user1")
            store.approve_code("telegram", c1)
            c2 = store.generate_code("discord", "user2")
            store.approve_code("discord", c2)
            approved = store.list_approved()
        assert len(approved) == 2

    def test_clear_pending(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user1")
            store.generate_code("telegram", "user2")
            count = store.clear_pending("telegram")
            remaining = store.list_pending("telegram")
        assert count == 2
        assert len(remaining) == 0

    def test_clear_pending_all_platforms(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user1")
            store.generate_code("discord", "user2")
            count = store.clear_pending()
        assert count == 2
