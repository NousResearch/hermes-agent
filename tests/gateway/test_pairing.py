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


# ---------------------------------------------------------------------------
# _secure_write
# ---------------------------------------------------------------------------


class TestSecureWrite:

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

# Profile-scoped storage (multiplexing gateway isolation)
# ---------------------------------------------------------------------------


class TestProfileScopedStorage:
    """PairingStore(profile="<name>") should isolate per-profile whitelists
    under <HERMES_HOME>/profiles/<name>/pairing/ so a multiplexing gateway
    can keep each profile's allowlist separate.
    """


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


