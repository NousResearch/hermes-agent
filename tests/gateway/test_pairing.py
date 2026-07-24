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


class TestProfileScopedDiscovery:
    def test_list_approved_scopes_platform_discovery_to_profile_dir(self, tmp_path):
        # A profile-scoped store must enumerate platforms from its own
        # per-profile directory (self._dir), not the module-global PAIRING_DIR.
        # Regression: _all_platforms iterated PAIRING_DIR while every per-file
        # path helper routed through self._dir, so a profile store confirmed a
        # user via is_approved() (reads self._dir) yet returned [] from
        # list_approved() (scanned the empty global dir).
        home = tmp_path / "home"
        global_dir = tmp_path / "global-pairing"
        global_dir.mkdir(parents=True)

        # PairingStore.__init__ resolves the profile dir via a *function-local*
        # ``from hermes_constants import get_hermes_home``, so the patch must
        # target ``hermes_constants.get_hermes_home`` (the source module) — not
        # ``gateway.pairing.get_hermes_home`` (the module-global binding, which
        # the local re-import bypasses). Patching the wrong target would leave
        # ``self._dir`` rooted at the real HERMES_HOME.
        with patch("gateway.pairing.PAIRING_DIR", global_dir), patch(
            "hermes_constants.get_hermes_home", return_value=home
        ):
            store = PairingStore(profile="alice")
            # Store lives under the mocked home's profile dir — provably scoped
            # there, and distinct from the module-global PAIRING_DIR.
            assert store._dir == home / "profiles" / "alice" / "pairing"
            assert store._dir != global_dir
            with store._lock:
                store._approve_user("telegram", "tg-456", "Bob")

            assert store.is_approved("telegram", "tg-456") is True
            approved = store.list_approved()

        assert [r["user_id"] for r in approved] == ["tg-456"]
        assert approved[0]["platform"] == "telegram"


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

    def test_approve_request_id_from_pending_list(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            bot_code = store.generate_code("telegram", "user1", "Alice")
            pending = store.list_pending("telegram")
            request_id = pending[0]["request_id"]

            assert request_id
            assert request_id != bot_code

            result = store.approve_request("telegram", request_id.upper())
            remaining = store.list_pending("telegram")

        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"
        assert remaining == []

    def test_approve_request_never_reveals_or_accepts_the_code_digest(self, tmp_path):
        """`list_pending` exposes an approvable id and nothing derived from the code.

        The pre-fix listing returned the code's hash prefix under a ``code``
        key, which admin GUIs posted straight back to approve — it could never
        match, because approval hashes its input and compares to that digest.
        """
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            bot_code = store.generate_code("telegram", "user1", "Alice")
            entry = store.list_pending("telegram")[0]

            digest = json.loads(
                (tmp_path / "telegram-pending.json").read_text()
            )[entry["request_id"]]["hash"]

            assert set(entry) == {
                "platform",
                "request_id",
                "user_id",
                "user_name",
                "age_minutes",
            }
            assert bot_code not in entry.values()
            assert entry["request_id"] not in (digest, digest[:8])
            # The digest prefix is not a credential on either grant path.
            assert store.approve_code("telegram", digest[:8]) is None
            assert store.approve_request("telegram", digest[:8]) is None

    def test_stale_request_id_never_locks_out_the_code_path(self, tmp_path):
        """Clicking Approve on an expired row is not a brute-force attempt.

        Request ids only reach an admin already authenticated to this store, so
        a miss means the row went stale — counting it toward the code lockout
        let a handful of GUI clicks lock the operator out of `pairing approve`.
        """
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            stale_id = store.list_pending("telegram")[0]["request_id"]
            assert store.approve_request("telegram", stale_id) is not None

            # Re-click the now-approved row well past the lockout threshold.
            for _ in range(MAX_FAILED_ATTEMPTS + 3):
                assert store.approve_request("telegram", stale_id) is None

            assert store._is_locked_out("telegram") is False
            # And the code path is still usable for the next real request.
            next_code = store.generate_code("telegram", "user2", "Bee")
            assert store.approve_code("telegram", next_code) is not None
            assert code != next_code


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


    def test_successful_approval_resets_failure_counter(self, tmp_path):
        """A successful approval clears the brute-force streak, so isolated
        typos across the gateway's lifetime don't accumulate into a spurious
        lockout that rejects a valid code.
        """
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()

            # One short of the lockout threshold — not locked out yet.
            for _ in range(MAX_FAILED_ATTEMPTS - 1):
                assert store.approve_code("telegram", "WRONGCODE") is None
            assert store._is_locked_out("telegram") is False

            # A legitimate approval must reset the accumulated failures.
            code = store.generate_code("telegram", "user1", "Alice")
            assert store.approve_code("telegram", code) is not None
            limits = store._load_json(store._rate_limit_path())
            assert limits.get("_failures:telegram", 0) == 0

            # Because the streak was cleared, a single fresh typo afterwards
            # must NOT trip the lockout (it would have with the stale count).
            assert store.approve_code("telegram", "WRONGCODE") is None
            assert store._is_locked_out("telegram") is False

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
    under each profile's own Hermes home so a multiplexing gateway can keep
    every profile's allowlist separate.
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
        """Explicit profile stores use that profile's normal Hermes layout."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = PairingStore(profile="yangyang")
        assert store.profile == "yangyang"
        expected = tmp_path / "profiles" / "yangyang" / "platforms" / "pairing"
        assert store._dir == expected
        assert store._approved_path("weixin") == expected / "weixin-approved.json"
        # Auto-creates the directory
        assert expected.is_dir()

    def test_profile_store_matches_profile_cli_home(self, tmp_path, monkeypatch):
        """Gateway and ``hermes -p`` must resolve the same pairing store."""
        from hermes_constants import get_hermes_dir

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        gateway_store = PairingStore(profile="coder")
        cli_dir = get_hermes_dir(
            "platforms/pairing",
            "pairing",
            home=profile_home,
        )

        assert gateway_store._dir == cli_dir

    def test_default_profile_store_is_global_store(self, tmp_path, monkeypatch):
        """Multiplexing must not invent a ``profiles/default`` store."""
        from hermes_constants import get_hermes_dir

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        expected = get_hermes_dir(
            "platforms/pairing",
            "pairing",
            home=tmp_path,
        )

        with patch("gateway.pairing.PAIRING_DIR", expected):
            assert PairingStore(profile="default")._dir == PairingStore()._dir

    def test_profile_store_merges_split_pairing_layouts(
        self, tmp_path, monkeypatch
    ):
        """Existing approvals survive either profile directory layout."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        profile_home = tmp_path / "profiles" / "coder"
        legacy_dir = profile_home / "pairing"
        consolidated_dir = profile_home / "platforms" / "pairing"
        legacy_dir.mkdir(parents=True)
        consolidated_dir.mkdir(parents=True)
        (legacy_dir / "telegram-approved.json").write_text(
            '{"legacy-user": {"user_name": "Legacy"}}',
            encoding="utf-8",
        )
        (consolidated_dir / "telegram-approved.json").write_text(
            '{"new-user": {"user_name": "New"}}',
            encoding="utf-8",
        )

        store = PairingStore(profile="coder")

        assert store.is_approved("telegram", "legacy-user")
        assert store.is_approved("telegram", "new-user")

    def test_profile_approval_does_not_leak_to_global(self, tmp_path, monkeypatch):
        """Approving in a profile-scoped store must not appear in the global
        store — and vice versa. This is the whole point of the fix."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            global_store = PairingStore()
            profile_store = PairingStore(profile="yangyang")

        assert global_store._rate_limit_path() == tmp_path / "_rate_limits.json"
        assert profile_store._rate_limit_path() == (
            tmp_path
            / "profiles"
            / "yangyang"
            / "platforms"
            / "pairing"
            / "_rate_limits.json"
        )


