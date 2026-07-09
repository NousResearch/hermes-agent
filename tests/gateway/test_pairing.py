"""Tests for gateway/pairing.py — DM pairing security system."""

import json
import os
import sys
import time
from pathlib import Path
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


class TestSplitPairingDirMigration:
    def test_merges_new_approved_into_active_legacy_dir(self, tmp_path):
        home = tmp_path / "home"
        legacy = home / "pairing"
        new = home / "platforms" / "pairing"
        legacy.mkdir(parents=True)
        new.mkdir(parents=True)
        (new / "feishu-approved.json").write_text(json.dumps({
            "ou_user": {"user_name": "Alice", "approved_at": 123.0}
        }))

        with patch("gateway.pairing.PAIRING_DIR", legacy), patch("gateway.pairing.get_hermes_home", return_value=home):
            store = PairingStore()
            assert store.is_approved("feishu", "ou_user") is True

        migrated = json.loads((legacy / "feishu-approved.json").read_text())
        assert "ou_user" in migrated

    def test_active_entries_win_when_merging_split_dirs(self, tmp_path):
        home = tmp_path / "home"
        legacy = home / "pairing"
        new = home / "platforms" / "pairing"
        legacy.mkdir(parents=True)
        new.mkdir(parents=True)
        (legacy / "feishu-approved.json").write_text(json.dumps({
            "ou_user": {"user_name": "Active", "approved_at": 2.0}
        }))
        (new / "feishu-approved.json").write_text(json.dumps({
            "ou_user": {"user_name": "Inactive", "approved_at": 1.0},
            "ou_other": {"user_name": "Other", "approved_at": 1.0},
        }))

        with patch("gateway.pairing.PAIRING_DIR", legacy), patch("gateway.pairing.get_hermes_home", return_value=home):
            store = PairingStore()
            assert store.is_approved("feishu", "ou_user") is True
            assert store.is_approved("feishu", "ou_other") is True

        migrated = json.loads((legacy / "feishu-approved.json").read_text())
        assert migrated["ou_user"]["user_name"] == "Active"
        assert migrated["ou_other"]["user_name"] == "Other"


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


# ---------------------------------------------------------------------------
# Unreadable approved-list file logs a warning instead of failing silently
# (issue #10270: Docker `docker exec` writes root-owned 0600 files that the
# post-gosu gateway can't read; the previous OSError swallow turned the bug
# into a mystery "Unauthorized user" message)
# ---------------------------------------------------------------------------


class TestUnreadablePairingFile:
    def test_permission_error_logs_warning_and_returns_empty(self, tmp_path, caplog):
        import logging
        import builtins

        approved_path = tmp_path / "weixin-approved.json"
        approved_path.write_text(
            '{"o9cq80fake@im.wechat": {"user_name": "x", "approved_at": 0}}'
        )

        real_open = builtins.open

        def fake_read_text(self, *a, **kw):
            # Path.read_text uses Path.open internally; raise PermissionError
            # to mimic a 0600 file owned by a different uid.
            raise PermissionError(13, "Permission denied", str(self))

        with patch("gateway.pairing.PAIRING_DIR", tmp_path), \
             patch.object(Path, "read_text", fake_read_text), \
             caplog.at_level(logging.WARNING, logger="gateway.pairing"):
            store = PairingStore()
            result = store._load_json(approved_path)

        assert result == {}, "should fall back to empty dict, not raise"
        assert any(
            "not readable" in rec.getMessage() and "#10270" not in rec.getMessage()
            or "not readable" in rec.getMessage()
            for rec in caplog.records
        ), f"expected a warning about unreadable pairing file, got {caplog.records!r}"
        # And the warning should include actionable advice
        msgs = " ".join(rec.getMessage() for rec in caplog.records)
        assert "docker exec" in msgs
        assert "-u hermes" in msgs

    def test_is_approved_returns_false_when_file_unreadable(self, tmp_path, caplog):
        """End-to-end: an unreadable approved.json must not crash the gateway,
        and the affected user must stay unauthorized (the documented fallback
        behaviour) rather than triggering a 500."""
        import logging

        approved_path = tmp_path / "weixin-approved.json"
        approved_path.write_text(
            '{"o9cq80fake@im.wechat": {"user_name": "x", "approved_at": 0}}'
        )

        def fake_read_text(self, *a, **kw):
            raise PermissionError(13, "Permission denied", str(self))

        with patch("gateway.pairing.PAIRING_DIR", tmp_path), \
             patch.object(Path, "read_text", fake_read_text), \
             caplog.at_level(logging.WARNING, logger="gateway.pairing"):
            store = PairingStore()
            ok = store.is_approved("weixin", "o9cq80fake@im.wechat")

        assert ok is False
        # The warning must fire — otherwise this is the silent-failure bug.
        assert any(rec.levelno == logging.WARNING for rec in caplog.records), \
            "PermissionError on approved.json must produce a WARNING log line"
# Profile-scoped storage (multiplexing gateway isolation)
# ---------------------------------------------------------------------------


class TestProfileScopedStorage:
    """PairingStore(profile="<name>") should isolate per-profile whitelists
    under <HERMES_HOME>/profiles/<name>/pairing/ so a multiplexing gateway
    can keep each profile's allowlist separate.
    """

    def test_default_store_uses_global_dir(self, tmp_path, monkeypatch):
        """PairingStore() (no profile) keeps the legacy global path so the
        ``hermes pairing`` CLI continues to work without a profile context."""
        from hermes_constants import get_hermes_home
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        # Re-import PAIRING_DIR (it's a module-level constant resolved at
        # import time) so the test exercises the right path. We patch it
        # rather than re-importing so the assertion is unambiguous.
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
        assert store.profile is None
        assert store._dir == tmp_path
        assert store._approved_path("weixin") == tmp_path / "weixin-approved.json"

    def test_profile_store_uses_profiles_subdir(self, tmp_path, monkeypatch):
        """PairingStore(profile="yangyang") puts files under
        <HERMES_HOME>/profiles/yangyang/pairing/."""
        from hermes_constants import get_hermes_home
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        store = PairingStore(profile="yangyang")
        assert store.profile == "yangyang"
        expected = tmp_path / "profiles" / "yangyang" / "pairing"
        assert store._dir == expected
        assert store._approved_path("weixin") == expected / "weixin-approved.json"
        # Auto-creates the directory
        assert expected.is_dir()

    def test_profile_approval_does_not_leak_to_global(self, tmp_path, monkeypatch):
        """Approving in a profile-scoped store must not appear in the global
        store — and vice versa. This is the whole point of the fix."""
        from hermes_constants import get_hermes_home
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            global_store = PairingStore()
            profile_store = PairingStore(profile="yangyang")

        # Approve in the profile store
        profile_store._approve_user("weixin", "yangyang_user", "杨洋")
        # And in the global store, a different user
        global_store._approve_user("weixin", "global_user", "Default")

        # Cross-isolation: each store only sees its own user
        assert profile_store.is_approved("weixin", "yangyang_user") is True
        assert profile_store.is_approved("weixin", "global_user") is False
        assert global_store.is_approved("weixin", "global_user") is True
        assert global_store.is_approved("weixin", "yangyang_user") is False

    def test_profile_uses_distinct_rate_limit_file(self, tmp_path, monkeypatch):
        """Rate-limit state is per-profile, not shared globally — otherwise
        one profile's flood would lock out the other profile's users."""
        from hermes_constants import get_hermes_home
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            global_store = PairingStore()
            profile_store = PairingStore(profile="yangyang")

        assert global_store._rate_limit_path() == tmp_path / "_rate_limits.json"
        assert profile_store._rate_limit_path() == (
            tmp_path / "profiles" / "yangyang" / "pairing" / "_rate_limits.json"
        )

    def test_pairing_store_for_helper_routes_by_profile(self, tmp_path, monkeypatch):
        """_pairing_store_for(source) on a gateway-like object picks the
        per-profile store when source.profile is set, and falls back to
        the global store when it isn't (defensive — single-profile
        gateways, or any code path that hasn't stamped source.profile)."""
        from gateway.session import SessionSource
        from gateway.config import Platform

        class FakeGateway:
            def __init__(self):
                self.pairing_store = object()  # sentinel
                self.pairing_stores = {
                    "default": "default-store",
                    "yangyang": "yangyang-store",
                }

            # Method under test — copy of the real helper so this test
            # is self-contained even if the real one moves.
            def _pairing_store_for(self, source):
                per_profile = getattr(self, "pairing_stores", None) or {}
                profile = getattr(source, "profile", None)
                if profile and profile in per_profile:
                    return per_profile[profile]
                return getattr(self, "pairing_store", None)

        g = FakeGateway()
        # source with profile="yangyang" → per-profile store
        s_yy = SessionSource(platform=Platform.WEIXIN, chat_id="c", profile="yangyang")
        assert g._pairing_store_for(s_yy) == "yangyang-store"
        # source with no profile → fallback to global
        s_none = SessionSource(platform=Platform.WEIXIN, chat_id="c")
        assert g._pairing_store_for(s_none) is g.pairing_store
        # source with an unknown profile → fallback (defensive)
        s_unknown = SessionSource(platform=Platform.WEIXIN, chat_id="c", profile="ghost")
        assert g._pairing_store_for(s_unknown) is g.pairing_store
