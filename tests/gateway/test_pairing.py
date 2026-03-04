"""Tests for gateway/pairing.py — DM pairing system.

Covers the security-critical PairingStore: code generation, approve/revoke
flow, rate limiting, lockout after failed attempts, code expiry, max pending
limits, and platform isolation.
"""

import sys
import time
import pytest
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


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """PairingStore rooted in a temp directory instead of ~/.hermes/pairing."""
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path)
    return PairingStore()


# =========================================================================
# _secure_write helper
# =========================================================================

class TestSecureWrite:
    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "file.json"
        _secure_write(target, '{"ok": true}')
        assert target.exists()
        assert target.read_text(encoding="utf-8") == '{"ok": true}'

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not supported on Windows")
    def test_file_permissions(self, tmp_path):
        target = tmp_path / "secret.json"
        _secure_write(target, "data")
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"


# =========================================================================
# Code generation
# =========================================================================

class TestCodeGeneration:
    def test_generates_valid_code(self, store):
        code = store.generate_code("telegram", "user1")
        assert code is not None
        assert len(code) == CODE_LENGTH

    def test_code_uses_correct_alphabet(self, store):
        """All characters must be from the unambiguous alphabet (no 0/O/1/I)."""
        for _ in range(20):
            code = store.generate_code("telegram", f"user_{_}")
            # Bypass rate limit by clearing it
            limits = store._load_json(store._rate_limit_path())
            limits.clear()
            store._save_json(store._rate_limit_path(), limits)
            # Clear pending to avoid hitting max
            store._save_json(store._pending_path("telegram"), {})

            if code is not None:
                for char in code:
                    assert char in ALPHABET, (
                        f"Character '{char}' not in allowed alphabet. "
                        f"Code: {code}"
                    )

    def test_ambiguous_chars_excluded(self, store):
        """0, O, 1, I must never appear in generated codes."""
        for forbidden in "0O1I":
            assert forbidden not in ALPHABET

    def test_generates_unique_codes(self, store):
        """Successive codes should be different (cryptographic randomness)."""
        # Clear rate limits between generations
        codes = set()
        for i in range(3):
            code = store.generate_code("telegram", f"unique_user_{i}")
            assert code is not None
            codes.add(code)
        # All 3 codes should be distinct (probability of collision is ~0)
        assert len(codes) == 3


# =========================================================================
# Approve / Revoke flow
# =========================================================================

class TestApproveRevokeFlow:
    def test_full_pairing_flow(self, store):
        """generate_code → approve_code → is_approved end-to-end."""
        assert store.is_approved("telegram", "u100") is False

        code = store.generate_code("telegram", "u100", "Alice")
        assert code is not None

        result = store.approve_code("telegram", code)
        assert result is not None
        assert result["user_id"] == "u100"
        assert result["user_name"] == "Alice"

        assert store.is_approved("telegram", "u100") is True

    def test_approve_case_insensitive(self, store):
        """Codes should work regardless of case (user might type lowercase)."""
        code = store.generate_code("telegram", "u200")
        assert code is not None

        result = store.approve_code("telegram", code.lower())
        assert result is not None
        assert result["user_id"] == "u200"

    def test_approve_invalid_code_returns_none(self, store):
        result = store.approve_code("telegram", "FAKECODE")
        assert result is None

    def test_code_single_use(self, store):
        """A code can only be approved once."""
        code = store.generate_code("telegram", "u300")
        store.approve_code("telegram", code)
        # Second approval of same code should fail
        result = store.approve_code("telegram", code)
        assert result is None

    def test_revoke_removes_approval(self, store):
        code = store.generate_code("telegram", "u400")
        store.approve_code("telegram", code)
        assert store.is_approved("telegram", "u400") is True

        revoked = store.revoke("telegram", "u400")
        assert revoked is True
        assert store.is_approved("telegram", "u400") is False

    def test_revoke_nonexistent_user(self, store):
        assert store.revoke("telegram", "ghost") is False

    def test_list_approved(self, store):
        code = store.generate_code("telegram", "u500", "Bob")
        store.approve_code("telegram", code)

        approved = store.list_approved("telegram")
        assert len(approved) == 1
        assert approved[0]["user_id"] == "u500"
        assert approved[0]["platform"] == "telegram"

    def test_list_approved_all_platforms(self, store):
        c1 = store.generate_code("telegram", "u600")
        store.approve_code("telegram", c1)
        c2 = store.generate_code("discord", "u700")
        store.approve_code("discord", c2)

        all_approved = store.list_approved()
        assert len(all_approved) == 2
        platforms = {a["platform"] for a in all_approved}
        assert platforms == {"telegram", "discord"}


# =========================================================================
# Rate limiting
# =========================================================================

class TestRateLimiting:
    def test_rate_limited_after_first_request(self, store):
        """Same user can't request a second code within RATE_LIMIT_SECONDS."""
        code1 = store.generate_code("telegram", "u_rl")
        assert code1 is not None

        code2 = store.generate_code("telegram", "u_rl")
        assert code2 is None  # Rate limited

    def test_different_users_not_rate_limited(self, store):
        """Rate limit is per-user, not per-platform."""
        code1 = store.generate_code("telegram", "userA")
        code2 = store.generate_code("telegram", "userB")
        assert code1 is not None
        assert code2 is not None

    def test_rate_limit_expires(self, store):
        """After RATE_LIMIT_SECONDS, user can request again."""
        store.generate_code("telegram", "u_expire")

        # Fast-forward past rate limit
        past = time.time() - RATE_LIMIT_SECONDS - 1
        limits = store._load_json(store._rate_limit_path())
        limits["telegram:u_expire"] = past
        store._save_json(store._rate_limit_path(), limits)

        code2 = store.generate_code("telegram", "u_expire")
        assert code2 is not None


# =========================================================================
# Lockout after failed attempts
# =========================================================================

class TestLockout:
    def test_lockout_after_max_failures(self, store):
        """Platform locks out after MAX_FAILED_ATTEMPTS wrong codes."""
        for i in range(MAX_FAILED_ATTEMPTS):
            store.approve_code("telegram", f"WRONG{i:03d}")

        # Platform should now be locked out — can't generate new codes
        code = store.generate_code("telegram", "innocent_user")
        assert code is None

    def test_lockout_does_not_affect_other_platforms(self, store):
        """Lockout is per-platform."""
        for i in range(MAX_FAILED_ATTEMPTS):
            store.approve_code("telegram", f"WRONG{i:03d}")

        # Discord should still work
        code = store.generate_code("discord", "d_user")
        assert code is not None

    def test_lockout_expires(self, store):
        """After LOCKOUT_SECONDS, platform is unlocked."""
        for i in range(MAX_FAILED_ATTEMPTS):
            store.approve_code("telegram", f"WRONG{i:03d}")

        # Fast-forward past lockout
        limits = store._load_json(store._rate_limit_path())
        limits["_lockout:telegram"] = time.time() - 1
        store._save_json(store._rate_limit_path(), limits)

        code = store.generate_code("telegram", "after_lockout")
        assert code is not None


# =========================================================================
# Code expiry
# =========================================================================

class TestCodeExpiry:
    def test_expired_code_rejected(self, store):
        """Codes older than CODE_TTL_SECONDS cannot be approved."""
        code = store.generate_code("telegram", "u_exp")
        assert code is not None

        # Backdate the pending entry
        pending = store._load_json(store._pending_path("telegram"))
        pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
        store._save_json(store._pending_path("telegram"), pending)

        result = store.approve_code("telegram", code)
        assert result is None

    def test_expired_codes_cleaned_up(self, store):
        """_cleanup_expired removes old entries from pending list."""
        code = store.generate_code("telegram", "u_cleanup")

        # Backdate
        pending = store._load_json(store._pending_path("telegram"))
        pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
        store._save_json(store._pending_path("telegram"), pending)

        # Trigger cleanup via list_pending
        result = store.list_pending("telegram")
        assert len(result) == 0  # Expired code should be gone


# =========================================================================
# Max pending limit
# =========================================================================

class TestMaxPending:
    def test_max_pending_per_platform(self, store):
        """Cannot exceed MAX_PENDING_PER_PLATFORM pending codes."""
        codes = []
        for i in range(MAX_PENDING_PER_PLATFORM):
            code = store.generate_code("telegram", f"pending_user_{i}")
            assert code is not None
            codes.append(code)
            # Clear rate limit so next user can request
            limits = store._load_json(store._rate_limit_path())
            for k in list(limits.keys()):
                if not k.startswith("_"):
                    limits[k] = 0
            store._save_json(store._rate_limit_path(), limits)

        # One more should be rejected
        extra = store.generate_code("telegram", "one_too_many")
        assert extra is None

    def test_max_pending_per_platform_independent(self, store):
        """Each platform has its own pending limit."""
        for i in range(MAX_PENDING_PER_PLATFORM):
            store.generate_code("telegram", f"tg_user_{i}")
            limits = store._load_json(store._rate_limit_path())
            for k in list(limits.keys()):
                if not k.startswith("_"):
                    limits[k] = 0
            store._save_json(store._rate_limit_path(), limits)

        # Discord should still accept codes
        code = store.generate_code("discord", "dc_user")
        assert code is not None


# =========================================================================
# Platform isolation
# =========================================================================

class TestPlatformIsolation:
    def test_approval_scoped_to_platform(self, store):
        """User approved on telegram is NOT approved on discord."""
        code = store.generate_code("telegram", "cross_user")
        store.approve_code("telegram", code)

        assert store.is_approved("telegram", "cross_user") is True
        assert store.is_approved("discord", "cross_user") is False

    def test_pending_scoped_to_platform(self, store):
        store.generate_code("telegram", "tg_only")
        store.generate_code("discord", "dc_only")

        tg_pending = store.list_pending("telegram")
        dc_pending = store.list_pending("discord")

        tg_ids = {p["user_id"] for p in tg_pending}
        dc_ids = {p["user_id"] for p in dc_pending}

        assert "tg_only" in tg_ids
        assert "dc_only" not in tg_ids
        assert "dc_only" in dc_ids
        assert "tg_only" not in dc_ids


# =========================================================================
# Pending list management
# =========================================================================

class TestPendingManagement:
    def test_list_pending(self, store):
        code = store.generate_code("telegram", "u_list", "TestUser")
        pending = store.list_pending("telegram")
        assert len(pending) == 1
        assert pending[0]["code"] == code
        assert pending[0]["user_id"] == "u_list"
        assert pending[0]["user_name"] == "TestUser"
        assert pending[0]["platform"] == "telegram"
        assert "age_minutes" in pending[0]

    def test_clear_pending_single_platform(self, store):
        store.generate_code("telegram", "u_clear1")
        store.generate_code("discord", "u_clear2")

        count = store.clear_pending("telegram")
        assert count == 1

        assert len(store.list_pending("telegram")) == 0
        assert len(store.list_pending("discord")) == 1

    def test_clear_pending_all_platforms(self, store):
        store.generate_code("telegram", "u_ca1")
        store.generate_code("discord", "u_ca2")

        count = store.clear_pending()
        assert count == 2
        assert len(store.list_pending()) == 0


# =========================================================================
# File persistence
# =========================================================================

class TestFilePersistence:
    def test_data_survives_new_store_instance(self, store, tmp_path, monkeypatch):
        """Data persists across PairingStore instances (file-backed)."""
        code = store.generate_code("telegram", "persist_user")
        store.approve_code("telegram", code)

        # monkeypatch is still active, so this new instance uses the same
        # tmp_path directory — simulating a process restart with the same data dir.
        store2 = PairingStore()
        assert store2.is_approved("telegram", "persist_user") is True

    def test_corrupt_json_handled(self, store, tmp_path, monkeypatch):
        """Corrupt JSON files should not crash, just return empty dict."""
        path = tmp_path / "telegram-approved.json"
        path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        assert store.is_approved("telegram", "anyone") is False
