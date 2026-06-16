"""Tests for gateway/pairing.py — DM pairing security system."""

import json
import os
import sys
import time
from unittest.mock import patch

import pytest

from gateway.pairing import (
    PairingStore,
    ALPHABET,
    CODE_LENGTH,
    CODE_TTL_SECONDS,
    RATE_LIMIT_SECONDS,
    MAX_PENDING_PER_PLATFORM,
    MAX_FAILED_ATTEMPTS,
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

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="POSIX file modes are not enforced on Windows",
    )
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
        assert isinstance(code, str) and len(code) == CODE_LENGTH
        assert len(code) == CODE_LENGTH
        assert all(c in ALPHABET for c in code)

    def test_code_uniqueness(self, tmp_path):
        """Multiple codes for different users should be distinct."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            codes = set()
            for i in range(3):
                code = store.generate_code("telegram", f"user{i}")
                assert isinstance(code, str) and len(code) == CODE_LENGTH
                codes.add(code)
        assert len(codes) == 3

    def test_stores_pending_entry(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            pending = store.list_pending("telegram")
        assert len(pending) == 1
        # list_pending no longer returns the original code — it returns a
        # truncated hash prefix.  Verify the metadata is correct instead.
        assert pending[0]["user_id"] == "user1"
        assert pending[0]["user_name"] == "Alice"
        # The code field is now a hash prefix, not the original plaintext code
        assert pending[0]["code"] != code


# ---------------------------------------------------------------------------
# Hashed storage
# ---------------------------------------------------------------------------


class TestHashedStorage:
    def test_pending_file_contains_hash_and_salt(self, tmp_path):
        """Stored entries must have 'hash' and 'salt', never the plaintext code."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            raw = json.loads(
                (tmp_path / "telegram-pending.json").read_text(encoding="utf-8")
            )

        assert len(raw) == 1
        entry = next(iter(raw.values()))
        # Must have hash and salt fields
        assert "hash" in entry
        assert "salt" in entry
        # Hash must be a valid hex SHA-256 digest (64 hex chars)
        assert len(entry["hash"]) == 64
        assert all(c in "0123456789abcdef" for c in entry["hash"])
        # Salt must be a valid hex string (32 hex chars for 16 bytes)
        assert len(entry["salt"]) == 32
        assert all(c in "0123456789abcdef" for c in entry["salt"])
        # The plaintext code must NOT appear as a key or value anywhere
        assert code not in raw  # not a key
        for key, val in raw.items():
            assert code != key
            for field_val in val.values():
                if isinstance(field_val, str):
                    assert field_val != code

    def test_plaintext_code_not_stored(self, tmp_path):
        """The raw JSON file must not contain the plaintext code anywhere."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")
            raw_text = (tmp_path / "telegram-pending.json").read_text(encoding="utf-8")
        assert code not in raw_text

    def test_valid_code_verifies_against_hash(self, tmp_path):
        """approve_code with the correct code should succeed."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Bob")
            result = store.approve_code("telegram", code)
        assert result is not None
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Bob"

    def test_invalid_code_rejected(self, tmp_path):
        """approve_code with a wrong code should fail."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user1")
            result = store.approve_code("telegram", "ZZZZZZZZ")
        assert result is None

    def test_different_salts_per_entry(self, tmp_path):
        """Each pending entry should have a unique salt."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user0")
            store.generate_code("telegram", "user1")
            store.generate_code("telegram", "user2")
            raw = json.loads(
                (tmp_path / "telegram-pending.json").read_text(encoding="utf-8")
            )
        salts = [entry["salt"] for entry in raw.values()]
        assert len(set(salts)) == 3  # all unique

    def test_hash_code_static_method(self, tmp_path):
        """_hash_code should be deterministic for the same code+salt."""
        salt = os.urandom(16)
        h1 = PairingStore._hash_code("ABCD1234", salt)
        h2 = PairingStore._hash_code("ABCD1234", salt)
        assert h1 == h2
        # Different salt should produce a different hash
        salt2 = os.urandom(16)
        h3 = PairingStore._hash_code("ABCD1234", salt2)
        assert h3 != h1


class TestLegacyPendingFileCompat:
    """Defensive coverage for pre-hash pending.json on upgraded installs.

    Existing user installs may have a pending.json written by the old
    code (plaintext code as key, no hash/salt fields). The new
    approve_code / list_pending / _cleanup_expired must not crash on
    those entries — they should be ignored and aged out at TTL.
    """

    @staticmethod
    def _write_legacy(tmp_path, code="ABCD1234", created_at=None):
        """Write a pre-hash pending.json with plaintext code as the key."""
        import time as _time
        if created_at is None:
            created_at = _time.time()
        legacy = {
            code: {
                "user_id": "legacy-user",
                "user_name": "Legacy",
                "created_at": created_at,
            }
        }
        (tmp_path / "telegram-pending.json").write_text(
            json.dumps(legacy), encoding="utf-8"
        )

    def test_approve_code_ignores_legacy_entries(self, tmp_path):
        """A valid old-format code must NOT silently approve under the new schema."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            self._write_legacy(tmp_path, code="LEGACY01")
            store = PairingStore()
            # The plaintext "code" used to be the key — under the new schema
            # it's not even looked at, and there's no hash/salt to verify.
            # Result: approve_code returns None, the legacy entry is left
            # alone (gets pruned by _cleanup_expired at TTL).
            result = store.approve_code("telegram", "LEGACY01")
            assert result is None
            # Approved list must be empty
            assert store.is_approved("telegram", "legacy-user") is False

    def test_list_pending_handles_legacy_entries(self, tmp_path):
        """list_pending must not KeyError on a missing 'hash' field."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            self._write_legacy(tmp_path)
            store = PairingStore()
            pending = store.list_pending("telegram")
        assert len(pending) == 1
        assert pending[0]["user_id"] == "legacy-user"
        assert pending[0]["code"] == "legacy"  # placeholder

    def test_cleanup_expired_removes_legacy_at_ttl(self, tmp_path):
        """Legacy entries past CODE_TTL must still get pruned."""
        import time as _time
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            self._write_legacy(
                tmp_path,
                code="LEGACY99",
                created_at=_time.time() - CODE_TTL_SECONDS - 1,
            )
            store = PairingStore()
            store._cleanup_expired("telegram")
            raw = json.loads(
                (tmp_path / "telegram-pending.json").read_text(encoding="utf-8")
            )
        assert raw == {}

    def test_cleanup_expired_handles_malformed_entries(self, tmp_path):
        """Non-dict / missing-created_at entries get evicted, not crashed on."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            (tmp_path / "telegram-pending.json").write_text(
                json.dumps({
                    "broken1": "not a dict",
                    "broken2": {"user_id": "x"},  # no created_at
                    "broken3": {"created_at": "not a number"},
                }),
                encoding="utf-8",
            )
            store = PairingStore()
            store._cleanup_expired("telegram")
            raw = json.loads(
                (tmp_path / "telegram-pending.json").read_text(encoding="utf-8")
            )
        assert raw == {}

    def test_approve_code_skips_malformed_entries(self, tmp_path):
        """Malformed entries must not crash approve_code's hash loop."""
        import time as _time
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            (tmp_path / "telegram-pending.json").write_text(
                json.dumps({
                    "broken": {"user_id": "x", "created_at": _time.time(),
                               "salt": "not-hex", "hash": "doesntmatter"},
                }),
                encoding="utf-8",
            )
            store = PairingStore()
            # Approving with any code must just return None, not crash.
            assert store.approve_code("telegram", "ABCD1234") is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_same_user_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user1")
        assert isinstance(code1, str) and len(code1) == CODE_LENGTH
        assert code2 is None  # rate limited

    def test_different_users_not_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user2")
        assert isinstance(code1, str) and len(code1) == CODE_LENGTH
        assert isinstance(code2, str) and len(code2) == CODE_LENGTH

    def test_rate_limit_expires(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            assert isinstance(code1, str) and len(code1) == CODE_LENGTH

            # Simulate rate limit expiry
            limits = store._load_json(store._rate_limit_path())
            limits["telegram:user1"] = time.time() - RATE_LIMIT_SECONDS - 1
            store._save_json(store._rate_limit_path(), limits)

            code2 = store.generate_code("telegram", "user1")
        assert isinstance(code2, str) and len(code2) == CODE_LENGTH
        assert code2 != code1

    def test_whatsapp_alias_flip_hits_same_rate_limit(self, tmp_path, monkeypatch):
        mapping_dir = tmp_path / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("whatsapp", "15551234567@s.whatsapp.net")
            code2 = store.generate_code("whatsapp", "999999999999999@lid")

        assert isinstance(code1, str) and len(code1) == CODE_LENGTH
        assert code2 is None


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
        assert all(isinstance(c, str) and len(c) == CODE_LENGTH for c in codes[:MAX_PENDING_PER_PLATFORM])
        # Next one should be blocked
        assert codes[MAX_PENDING_PER_PLATFORM] is None

    def test_different_platforms_independent(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for i in range(MAX_PENDING_PER_PLATFORM):
                store.generate_code("telegram", f"user{i}")
            # Different platform should still work
            code = store.generate_code("discord", "user0")
        assert isinstance(code, str) and len(code) == CODE_LENGTH


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


class TestApprovalFlow:
    def test_approve_valid_code(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", code)

        assert isinstance(result, dict)
        assert "user_id" in result
        assert "user_name" in result
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
        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_approve_strips_whitespace(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", f"  {code}  ")
        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_invalid_code_returns_none(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            result = store.approve_code("telegram", "INVALIDCODE")
        assert result is None

    def test_whatsapp_approved_user_survives_alias_flip(self, tmp_path, monkeypatch):
        mapping_dir = tmp_path / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("whatsapp", "15551234567@s.whatsapp.net", "Alice")
            store.approve_code("whatsapp", code)

            assert store.is_approved("whatsapp", "15551234567@s.whatsapp.net") is True
            assert store.is_approved("whatsapp", "999999999999999@lid") is True

            approved = store.list_approved("whatsapp")

        assert len(approved) == 1
        assert approved[0]["user_id"] == "15551234567"

    def test_whatsapp_legacy_raw_jid_approval_survives_alias_flip(self, tmp_path, monkeypatch):
        mapping_dir = tmp_path / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        approved_path = tmp_path / "whatsapp-approved.json"
        approved_path.write_text(
            json.dumps(
                {
                    "15551234567@s.whatsapp.net": {
                        "user_name": "Legacy Alice",
                        "approved_at": time.time(),
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            assert store.is_approved("whatsapp", "999999999999999@lid") is True


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

    def test_lockout_blocks_code_approval(self, tmp_path):
        """Regression guard for #10195: lockout must also gate approve_code.

        Prior to the fix, 5 failed approvals set the lockout flag but
        approve_code() never consulted it — so any valid code already
        in `pending` (or a later lucky guess) still got accepted,
        nullifying the brute-force protection.
        """
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            # Generate a valid code before triggering the lockout.
            valid_code = store.generate_code("telegram", "attacker", "Attacker")
            assert valid_code is not None

            # Trigger the lockout with wrong codes.
            for _ in range(MAX_FAILED_ATTEMPTS):
                assert store.approve_code("telegram", "WRONGCODE") is None
            assert store._is_locked_out("telegram") is True

            # The valid code must be rejected while the lockout is active,
            # and the user must NOT land in the approved list.
            result = store.approve_code("telegram", valid_code)
            assert result is None
            assert store.is_approved("telegram", "attacker") is False

            # Simulate lockout expiry — the valid code is still in pending
            # (we didn't pop it) and must now approve normally.
            limits = store._load_json(store._rate_limit_path())
            limits["_lockout:telegram"] = time.time() - 1
            store._save_json(store._rate_limit_path(), limits)

            result = store.approve_code("telegram", valid_code)
            assert result is not None
            assert result["user_id"] == "attacker"
            assert store.is_approved("telegram", "attacker") is True

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

            # Manually expire all pending entries
            pending = store._load_json(store._pending_path("telegram"))
            for entry_id in pending:
                pending[entry_id]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
            store._save_json(store._pending_path("telegram"), pending)

            # Cleanup happens on next operation
            remaining = store.list_pending("telegram")
        assert len(remaining) == 0

    def test_expired_code_cannot_be_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")

            # Expire all entries
            pending = store._load_json(store._pending_path("telegram"))
            for entry_id in pending:
                pending[entry_id]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
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
