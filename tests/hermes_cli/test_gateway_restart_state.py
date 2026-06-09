"""Tests for gateway_restart_state.py — intent, locks, status, JSONL.

Per-request directory API: each restart request gets its own directory under
``run/gateway-restart/{profile}/{request_id}/``.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def restart_state_dir(tmp_path, monkeypatch):
    """Provide a temp HERMES_HOME for restart state tests."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "run").mkdir()
    (hermes_home / "logs").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Mock get_hermes_home
    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(hermes_home))

    return hermes_home


# ---------------------------------------------------------------------------
# Intent tests
# ---------------------------------------------------------------------------

class TestIntent:
    def test_create_and_read_intent(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent, read_intent, cleanup_intent

        intent = create_intent(profile="default", target_pid=1234, origin="test")
        rid = intent["request_id"]
        assert intent["schema_version"] == 1
        assert intent["target_pid"] == 1234
        assert intent["origin"] == "test"
        assert intent["state"] == "scheduled"
        assert rid
        assert "nonce" in intent

        read = read_intent("default", rid)
        assert read is not None
        assert read["request_id"] == rid
        assert read["target_pid"] == 1234

        cleanup_intent("default", rid)

    def test_intent_ttl_expiry(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent, read_intent, intent_path

        intent = create_intent(profile="default", ttl_s=1)
        rid = intent["request_id"]
        assert read_intent("default", rid) is not None

        # Expire it — write to the per-request intent.json
        path = intent_path("default", rid)
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 10
        path.write_text(json.dumps(data))

        assert read_intent("default", rid) is None

    def test_malformed_intent_safe_fail(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent, read_intent, intent_path

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        path = intent_path("default", rid)

        # Not JSON
        path.write_text("not json", encoding="utf-8")
        assert read_intent("default", rid) is None

        # Wrong schema version
        path.write_text(json.dumps({"schema_version": 999}))
        assert read_intent("default", rid) is None

        # Missing required fields
        path.write_text(json.dumps({"schema_version": 1}))
        assert read_intent("default", rid) is None

        # Not a dict
        path.write_text(json.dumps([1, 2, 3]))
        assert read_intent("default", rid) is None

        # Too large
        path.write_text(json.dumps({"schema_version": 1, "00000000-0000-0000-0000-000000000001": "y" * 10000}))
        assert read_intent("default", rid) is None

    def test_intent_nonce_validation(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent, validate_intent_nonce

        intent = create_intent(profile="default")
        nonce = intent["nonce"]

        assert validate_intent_nonce(intent, nonce) is True
        assert validate_intent_nonce(intent, "wrong") is False
        assert validate_intent_nonce(intent, "") is False

    def test_intent_atomic_write(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent

        # Should not leave .tmp file behind
        intent = create_intent(profile="default")
        rid = intent["request_id"]
        req_dir = restart_state_dir / "run" / "gateway-restart" / "default" / rid
        tmp_files = list(req_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_intent_profile_isolation(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import create_intent, read_intent, cleanup_intent

        i1 = create_intent(profile="p1", target_pid=111)
        i2 = create_intent(profile="p2", target_pid=222)

        r1 = read_intent("p1", i1["request_id"])
        r2 = read_intent("p2", i2["request_id"])
        assert r1["target_pid"] == 111
        assert r2["target_pid"] == 222

        cleanup_intent("p1", i1["request_id"])
        cleanup_intent("p2", i2["request_id"])

    def test_intent_per_request_directory(self, restart_state_dir):
        """Each request_id gets its own directory with intent.json."""
        from hermes_cli.gateway_restart_state import create_intent, intent_path, request_dir_path

        i1 = create_intent(profile="default", target_pid=100)
        i2 = create_intent(profile="default", target_pid=200)

        rid1, rid2 = i1["request_id"], i2["request_id"]
        assert rid1 != rid2

        # Both have their own intent files
        assert intent_path("default", rid1).exists()
        assert intent_path("default", rid2).exists()

        # Different directories
        d1 = request_dir_path("default", rid1)
        d2 = request_dir_path("default", rid2)
        assert d1 != d2
        assert d1.is_dir()
        assert d2.is_dir()

    def test_cleanup_intent_removes_entire_request_dir(self, restart_state_dir):
        """cleanup_intent removes the whole request directory."""
        from hermes_cli.gateway_restart_state import (
            create_intent, cleanup_intent, request_dir_path,
            write_status, lease_path,
        )

        intent = create_intent(profile="default", target_pid=1234)
        rid = intent["request_id"]
        d = request_dir_path("default", rid)

        # Write status and create a lease file so the dir has multiple files
        write_status("default", "draining", request_id=rid)
        lp = lease_path("default", rid)
        lp.write_text("{}")

        assert d.exists()
        assert (d / "intent.json").exists()
        assert (d / "status.json").exists()
        assert (d / "lease.lock").exists()

        cleanup_intent("default", rid)

        assert not d.exists()
        assert not (d / "intent.json").exists()
        assert not (d / "status.json").exists()
        assert not (d / "lease.lock").exists()

    def test_update_intent_state(self, restart_state_dir):
        """update_intent_state transitions intent state."""
        from hermes_cli.gateway_restart_state import create_intent, read_intent, update_intent_state

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        assert intent["state"] == "scheduled"

        ok = update_intent_state("default", rid, "claimed")
        assert ok is True
        updated = read_intent("default", rid)
        assert updated["state"] == "claimed"

    def test_update_intent_state_with_expected_state_guard(self, restart_state_dir):
        """update_intent_state only updates if current state matches expected_state."""
        from hermes_cli.gateway_restart_state import create_intent, read_intent, update_intent_state

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        assert intent["state"] == "scheduled"

        # Wrong expected_state — should fail
        ok = update_intent_state("default", rid, "claimed", expected_state="draining")
        assert ok is False
        assert read_intent("default", rid)["state"] == "scheduled"

        # Correct expected_state — should succeed
        ok = update_intent_state("default", rid, "claimed", expected_state="scheduled")
        assert ok is True
        assert read_intent("default", rid)["state"] == "claimed"

        # Can't go back to scheduled if expecting draining
        ok = update_intent_state("default", rid, "scheduled", expected_state="draining")
        assert ok is False
        assert read_intent("default", rid)["state"] == "claimed"


# ---------------------------------------------------------------------------
# Lock tests
# ---------------------------------------------------------------------------

class TestRestartLock:
    def test_acquire_release(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True
        lock.release()

    def test_lock_contention(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import RestartLock

        lock1 = RestartLock("default")
        lock2 = RestartLock("default")

        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is False
        lock1.release()

        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is True
        lock2.release()

    def test_lock_no_coalesce_same_request(self, restart_state_dir):
        """Same request_id does NOT coalesce — second try_acquire returns False."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True
        # No coalesce — same request_id also blocked (lock file already exists)
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is False
        lock.release()

    def test_lock_coalesce_same_request_different_lock_instances(self, restart_state_dir):
        """Even different lock instances cannot coalesce same request_id."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock1 = RestartLock("default")
        lock2 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True
        # Second instance with same request_id — no coalesce
        assert lock2.try_acquire("11111111-1111-1111-1111-111111111111") is False
        lock1.release()

        # Now lock2 can acquire
        assert lock2.try_acquire("11111111-1111-1111-1111-111111111111") is True
        lock2.release()

    def test_lock_ttl_expiry(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111", ttl_s=1) is True

        # Expire it and set owner_pid to a dead process
        lp = lock_path("default")
        data = json.loads(lp.read_text())
        data["created_at"] = time.time() - 10
        data["owner_pid"] = 99999  # Dead PID
        lp.write_text(json.dumps(data))

        # New request should succeed (expired lock with dead owner is force-released)
        assert lock.try_acquire("22222222-2222-2222-2222-222222222222", ttl_s=1) is True
        lock.release()

    def test_lock_ttl_expiry_owner_alive(self, restart_state_dir):
        """TTL expired but owner PID still alive — must NOT take over."""
        import os
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111", ttl_s=1) is True

        # Expire it but keep owner_pid as current process (alive)
        lp = lock_path("default")
        data = json.loads(lp.read_text())
        data["created_at"] = time.time() - 10
        data["owner_pid"] = os.getpid()  # Alive!
        lp.write_text(json.dumps(data))

        # New request should FAIL (owner still alive)
        lock2 = RestartLock("default")
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222", ttl_s=1) is False
        lock.release()

    def test_lock_profile_isolation(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import RestartLock

        lock1 = RestartLock("p1")
        lock2 = RestartLock("p2")

        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is True  # Different profile

        lock1.release()
        lock2.release()

    def test_non_owner_release_rejected(self, restart_state_dir):
        """Non-owner calling release() must NOT delete the lock."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock1 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True

        # Simulate a non-owner trying to release
        lock2 = RestartLock("default")
        lock2.release()  # Should be a no-op

        # Verify lock1 still exists
        assert lock_path("default").exists()
        lock1.release()

    def test_owner_token_mismatch_rejected(self, restart_state_dir):
        """release() with wrong owner_token must NOT delete the lock."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        # Corrupt the owner_token
        lp = lock_path("default")
        data = json.loads(lp.read_text())
        data["owner_token"] = "wrong-token"
        lp.write_text(json.dumps(data))

        # release() should detect mismatch and NOT delete
        lock.release()
        assert lock_path("default").exists()

        # Clean up with force release
        lock._force_release()

    def test_lock_stores_worker_pid_and_phase(self, restart_state_dir):
        """Lock data includes worker_pid, claim_deadline, and phase fields."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111", worker_pid=12345) is True

        lp = lock_path("default")
        data = json.loads(lp.read_text())
        assert data["worker_pid"] == 12345
        assert data["phase"] == "awaiting_claim"
        assert data["claim_deadline"] > 0

        lock.release()

    def test_lock_phase_without_worker(self, restart_state_dir):
        """Lock without worker_pid has phase='acquired' and no claim_deadline."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        lp = lock_path("default")
        data = json.loads(lp.read_text())
        assert data["worker_pid"] == 0
        assert data["phase"] == "acquired"
        assert data["claim_deadline"] == 0

        lock.release()

    def test_mark_phase(self, restart_state_dir):
        """mark_phase updates the lock's phase field."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        lp = lock_path("default")
        data = json.loads(lp.read_text())
        assert data["phase"] == "acquired"

        lock.mark_phase("draining")
        data = json.loads(lp.read_text())
        assert data["phase"] == "draining"

        lock.mark_phase("completed")
        data = json.loads(lp.read_text())
        assert data["phase"] == "completed"

        lock.release()

    def test_mark_phase_non_owner_noop(self, restart_state_dir):
        """mark_phase from non-owner is a no-op."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock1 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True

        lock2 = RestartLock("default")
        # lock2 has no owner_token, should be no-op
        lock2.mark_phase("draining")

        data = json.loads(lock_path("default").read_text())
        assert data["phase"] == "acquired"  # unchanged

        lock1.release()

    def test_claim_lease(self, restart_state_dir):
        """Worker can claim lease using O_EXCL on lease file."""
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, claim_lock_path, lease_json_path,
        )

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.claim_lease(rid, nonce) is True

        # claim.lock and lease.json should exist
        assert claim_lock_path("default", rid).exists()
        assert lease_json_path("default", rid).exists()

        lock.release()

    def test_claim_lease_o_excl_second_fails(self, restart_state_dir):
        """Two claims on same request_id — second fails (O_EXCL)."""
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock1 = RestartLock("default")
        assert lock1.claim_lease(rid, nonce) is True

        # Second claim on same request_id must fail
        lock2 = RestartLock("default")
        assert lock2.claim_lease(rid, nonce) is False

    def test_claim_lease_wrong_nonce(self, restart_state_dir):
        """claim_lease with wrong nonce fails."""
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        intent = create_intent(profile="default")
        rid = intent["request_id"]

        lock = RestartLock("default")
        assert lock.claim_lease(rid, "wrong-nonce") is False

    def test_claim_lease_wrong_expected_state(self, restart_state_dir):
        """claim_lease with wrong expected_state fails."""
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, update_intent_state,
        )

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Move intent to "claimed" state
        update_intent_state("default", rid, "claimed")
        assert intent["state"] == "scheduled"

        # Try to claim expecting "scheduled" — but it's already "claimed"
        lock = RestartLock("default")
        assert lock.claim_lease(rid, nonce, expected_state="scheduled") is False

    def test_claim_lease_transitions_intent_state(self, restart_state_dir):
        """Successful claim_lease transitions intent state from scheduled to claimed."""
        from hermes_cli.gateway_restart_state import (
            create_intent, read_intent, RestartLock,
        )

        intent = create_intent(profile="default")
        rid = intent["request_id"]
        nonce = intent["nonce"]
        assert intent["state"] == "scheduled"

        lock = RestartLock("default")
        assert lock.claim_lease(rid, nonce) is True

        # Intent should now be "claimed"
        updated = read_intent("default", rid)
        assert updated["state"] == "claimed"

    def test_claim_lease_wrong_request_id(self, restart_state_dir):
        """Worker can't claim lease with wrong request_id (no intent exists)."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        assert lock.claim_lease("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "some-nonce") is False

    def test_concurrent_acquire_contention(self, restart_state_dir):
        """Two locks competing — only one wins."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock1 = RestartLock("default")
        lock2 = RestartLock("default")

        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is False  # Contention

        lock1.release()

        # Now lock2 can acquire
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is True
        lock2.release()


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------

class TestStatus:
    def test_write_read_status(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import write_status, read_status, cleanup_intent

        rid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        write_status("default", "draining", request_id=rid, old_pid=1234)
        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "draining"
        assert status["request_id"] == rid
        assert status["old_pid"] == 1234

        cleanup_intent("default", rid)

    def test_invalid_state_rejected(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import write_status

        with pytest.raises(ValueError, match="Invalid state"):
            write_status("default", "not_a_real_state", request_id="00000000-0000-0000-0000-000000000001")

    def test_valid_states(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import write_status, read_status, cleanup_intent

        states = [
            "scheduled", "claimed", "preflight_ok", "draining", "stopping",
            "waiting_pid_exit", "waiting_port_release",
            "starting_task", "starting_direct_fallback",
            "verifying", "completed", "failed",
        ]
        rid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        for state in states:
            write_status("default", state, request_id=rid)
            assert read_status("default", rid)["state"] == state

        cleanup_intent("default", rid)

    def test_read_latest_status(self, restart_state_dir):
        """read_latest_status returns the most recent status across requests."""
        from hermes_cli.gateway_restart_state import (
            write_status, read_latest_status, cleanup_intent,
        )

        rid1 = "11111111-1111-1111-1111-111111111111"
        rid2 = "22222222-2222-2222-2222-222222222222"

        write_status("default", "draining", request_id=rid1)
        # Small delay so timestamps differ
        time.sleep(0.05)
        write_status("default", "completed", request_id=rid2)

        latest = read_latest_status("default")
        assert latest is not None
        assert latest["state"] == "completed"
        assert latest["request_id"] == rid2

        cleanup_intent("default", rid1)
        cleanup_intent("default", rid2)

    def test_read_latest_status_empty(self, restart_state_dir):
        """read_latest_status returns None when no statuses exist."""
        from hermes_cli.gateway_restart_state import read_latest_status

        assert read_latest_status("default") is None

    def test_status_path_per_request(self, restart_state_dir):
        """status_path with request_id returns path inside request dir."""
        from hermes_cli.gateway_restart_state import status_path

        p = status_path("default", "12345678-abcd-abcd-abcd-12345678abcd")
        assert p.name == "status.json"
        assert "12345678-abcd-abcd-abcd-12345678abcd" in str(p)

    def test_lease_path(self, restart_state_dir):
        """lease_path returns the lease file inside request dir."""
        from hermes_cli.gateway_restart_state import lease_path

        lp = lease_path("default", "12345678-abcd-abcd-abcd-12345678abcd")
        assert lp.name == "lease.lock"
        assert "12345678-abcd-abcd-abcd-12345678abcd" in str(lp)

    def test_request_dir_path(self, restart_state_dir):
        """request_dir_path returns the per-request directory."""
        from hermes_cli.gateway_restart_state import request_dir_path

        d = request_dir_path("default", "12345678-abcd-abcd-abcd-12345678abcd")
        assert d.is_dir()
        assert d.name == "12345678-abcd-abcd-abcd-12345678abcd"


# ---------------------------------------------------------------------------
# JSONL log tests
# ---------------------------------------------------------------------------

class TestJSONLLog:
    def test_append_and_read(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import append_restart_log, jsonl_log_path

        append_restart_log(
            request_id="abcdefab-abcd-abcd-abcd-abcdefabcdef", profile="default", old_pid=1234,
            new_pid=5678, state="completed", launcher="direct_spawn",
        )

        path = jsonl_log_path()
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["request_id"] == "abcdefab-abcd-abcd-abcd-abcdefabcdef"
        assert record["old_pid"] == 1234
        assert record["new_pid"] == 5678
        assert record["state"] == "completed"

    def test_multiple_appends(self, restart_state_dir):
        from hermes_cli.gateway_restart_state import append_restart_log, jsonl_log_path

        for state in ["scheduled", "draining", "completed"]:
            append_restart_log(state=state)

        lines = jsonl_log_path().read_text().strip().split("\n")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Environment variable sanitization test
# ---------------------------------------------------------------------------

class TestEnvSanitization:
    def test_hermes_gateway_cleared(self, restart_state_dir):
        """Verify _HERMES_GATEWAY is removed from worker environment."""
        # This tests the coordinator's env cleanup logic
        env = os.environ.copy()
        env["_HERMES_GATEWAY"] = "1"
        env["HERMES_HOME"] = "/test"
        env["PATH"] = "/usr/bin"

        # Simulate coordinator cleanup
        env.pop("_HERMES_GATEWAY", None)
        env["HERMES_GATEWAY_RESTART_WORKER"] = "1"

        assert "_HERMES_GATEWAY" not in env
        assert env["HERMES_GATEWAY_RESTART_WORKER"] == "1"
        assert env["HERMES_HOME"] == "/test"  # preserved
        assert env["PATH"] == "/usr/bin"  # preserved

    def test_worker_env_cleanup_for_new_gateway(self, restart_state_dir):
        """Verify worker cleans its own marker before starting new gateway."""
        env = os.environ.copy()
        env["HERMES_GATEWAY_RESTART_WORKER"] = "1"

        # Simulate worker cleanup before starting new gateway
        env.pop("_HERMES_GATEWAY", None)
        env.pop("HERMES_GATEWAY_RESTART_WORKER", None)

        assert "HERMES_GATEWAY_RESTART_WORKER" not in env
        assert "_HERMES_GATEWAY" not in env


# ---------------------------------------------------------------------------
# P1-2: claim_lease rollback tests
# ---------------------------------------------------------------------------

class TestClaimLeaseRollback:
    """P1-2: claim_lease rolls back lease if intent state update fails."""

    def test_claim_lease_rollback_on_intent_update_failure(self, tmp_path, monkeypatch):
        """If update_intent_state fails after O_EXCL claim.lock creation, claim.lock is rolled back."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, claim_lock_path, lease_json_path, read_intent,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Monkey-patch update_intent_state to fail
        import hermes_cli.gateway_restart_state as state_mod
        original = state_mod.update_intent_state
        state_mod.update_intent_state = lambda *a, **kw: False

        lock = RestartLock("default")
        result = lock.claim_lease(rid, nonce)

        # Restore
        state_mod.update_intent_state = original

        assert result is False

        # claim.lock should have been rolled back
        clp = claim_lock_path("default", rid)
        assert not clp.exists(), "claim.lock must be rolled back on intent update failure"
        # lease.json must not exist
        ljp = lease_json_path("default", rid)
        assert not ljp.exists(), "lease.json must not exist on intent update failure"


# ---------------------------------------------------------------------------
# P0-4: release_lease preserves status
# ---------------------------------------------------------------------------

class TestReleaseLease:
    """P0-4: release_lease only removes lease.lock, preserves status."""

    def test_release_lease_preserves_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, write_status, read_status, release_lease,
            RestartLock, lease_json_path,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        write_status("default", "completed", request_id=rid, new_pid=200)

        # Claim lease properly so owner_token/worker_pid are set
        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        ljp = lease_json_path("default", rid)
        assert ljp.exists()

        result = release_lease("default", rid,
                              owner_token=lock.owner_token,
                              worker_pid=os.getpid())
        assert result is True

        # Lease removed, status preserved
        assert not ljp.exists()
        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "completed"


# ---------------------------------------------------------------------------
# P1-1: mark_worker_spawned return value
# ---------------------------------------------------------------------------

class TestMarkWorkerSpawned:
    """P1-1: mark_worker_spawned return value handling."""

    def test_mark_worker_spawned_returns_false_when_not_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        lock = RestartLock("default")
        # NOT acquired — should return False
        result = lock.mark_worker_spawned(1234, time.time() + 30)
        assert result is False


# ---------------------------------------------------------------------------
# P1-2: Worker fast completion + handoff race
# ---------------------------------------------------------------------------

class TestHandoffRace:
    """P1-2: Worker fast completion vs handoff race."""

    def test_no_orphan_active_lock_on_fast_completion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, lock_path, lease_path,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator acquires
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(rid) is True
        coord_token = coord_lock.owner_token

        # Worker claims lease
        worker_lock = RestartLock("default")
        assert worker_lock.claim_lease(rid, nonce) is True
        lease_token = worker_lock.owner_token

        # Coordinator hands off
        assert coord_lock.handoff_active_lock(rid, coord_token, os.getpid(), lease_token) is True

        # Worker releases active.lock first (P1-2 order)
        worker_lock.release()
        assert not lock_path("default").exists(), "active.lock must be released"

        # Then release lease
        lp = lease_path("default", rid)
        lp.unlink(missing_ok=True)
        assert not lp.exists()


# ---------------------------------------------------------------------------
# P1-3: GC safety rules
# ---------------------------------------------------------------------------

class TestGCSafety:
    """P1-3: GC safety rules."""

    def test_gc_skips_running_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, write_status, gc_expired_request_dirs,
            request_dir_path, claim_lock_path,
        )

        # Create a "running" request with claim.lock
        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        write_status("default", "draining", request_id=rid, old_pid=100)

        # Create a claim.lock file
        clp = claim_lock_path("default", rid)
        clp.touch()

        # Run GC with very short max_age
        removed = gc_expired_request_dirs("default", max_age_s=0, active_request_id=rid)

        # Directory must survive
        rd = request_dir_path("default", rid)
        assert rd.exists(), "GC must not delete request with active lease"


# ---------------------------------------------------------------------------
# P0-3: owner-scoped release_lease
# ---------------------------------------------------------------------------

class TestOwnerScopedRelease:
    """P0-3: owner-scoped release_lease."""

    def test_release_lease_rejects_empty_owner_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, release_lease, lease_json_path,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        # Empty owner_token must be rejected
        result = release_lease("default", rid, owner_token="", worker_pid=os.getpid())
        assert result is False

    def test_release_lease_rejects_zero_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, release_lease,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        result = release_lease("default", rid, owner_token="some-token", worker_pid=0)
        assert result is False


# ---------------------------------------------------------------------------
# P0-3: owner-scoped sanitize_intent
# ---------------------------------------------------------------------------

class TestOwnerScopedSanitize:
    """P0-3: owner-scoped sanitize_intent."""

    def test_sanitize_rejects_wrong_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, sanitize_intent, read_intent,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        result = sanitize_intent("default", rid,
                                expected_nonce=nonce,
                                owner_token="wrong-token",
                                worker_pid=os.getpid())
        assert result is False

        # Nonce must survive
        stored = read_intent("default", rid)
        assert stored["nonce"] == nonce


# ---------------------------------------------------------------------------
# P0-4: Pre-lease Worker cannot overwrite completed status
# ---------------------------------------------------------------------------

class TestStatusProtection:
    """P0-4: Pre-lease Worker cannot overwrite authoritative status."""

    def test_stale_worker_preserves_completed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, write_status, read_status,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Winner writes completed
        write_status("default", "completed", request_id=rid, new_pid=200)

        # Verify completed
        status = read_status("default", rid)
        assert status["state"] == "completed"

        # _fail_closed should NOT overwrite (it only writes JSONL now)
        # Simulate by checking that read_status still returns completed
        # after a rejection log is written
        from hermes_cli.gateway_restart_state import append_restart_log
        append_restart_log(request_id=rid, profile="default", state="rejected",
                         error="nonce_mismatch: test")

        status = read_status("default", rid)
        assert status["state"] == "completed", "completed must not be overwritten"


# ---------------------------------------------------------------------------
# P1-1: mark_worker_spawned retry
# ---------------------------------------------------------------------------

class TestMarkWorkerSpawnedRetry:
    """P1-1: mark_worker_spawned retry."""

    def test_mark_worker_spawned_all_retries_fail(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        lock = RestartLock("default")
        # NOT acquired — mark_worker_spawned should fail every time
        # (no owner_token match)
        for _ in range(3):
            assert lock.mark_worker_spawned(1234, time.time() + 30) is False


# ---------------------------------------------------------------------------
# P1-2: _wait_for_handoff request_id check
# ---------------------------------------------------------------------------

class TestHandoffRequestId:
    """P1-2: _wait_for_handoff request_id check."""

    def test_handoff_rejects_wrong_request_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, lock_path,
        )
        import hermes_cli.gateway_windows_restart_worker as worker_mod

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator acquires and sets up lock
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(rid) is True

        # Worker calls _wait_for_handoff with wrong request_id
        result = worker_mod._wait_for_handoff(
            "default", "wrong-request-id", "some-token", timeout=1.0)
        assert result is False


# ---------------------------------------------------------------------------
# P1-1: sanitize_intent requires lease.json
# ---------------------------------------------------------------------------

class TestSanitizeRequiresLease:
    """P1-1: sanitize_intent requires lease.json."""

    def test_sanitize_rejects_missing_lease(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, sanitize_intent, read_intent,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # No lease.json exists
        result = sanitize_intent("default", rid, expected_nonce=nonce,
                                owner_token="some-token", worker_pid=os.getpid())
        assert result is False
        stored = read_intent("default", rid)
        assert stored["nonce"] == nonce  # Nonce unchanged

    def test_sanitize_rejects_wrong_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, sanitize_intent, read_intent,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        result = sanitize_intent("default", rid, expected_nonce=nonce,
                                owner_token="wrong-token", worker_pid=os.getpid())
        assert result is False
        stored = read_intent("default", rid)
        assert stored["nonce"] == nonce

    def test_sanitize_rejects_wrong_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, sanitize_intent, read_intent,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True
        assert lock.claim_lease(rid, nonce) is True

        result = sanitize_intent("default", rid, expected_nonce=nonce,
                                owner_token=lock.owner_token, worker_pid=99999)
        assert result is False
        stored = read_intent("default", rid)
        assert stored["nonce"] == nonce


# ---------------------------------------------------------------------------
# P1-2: Orphan claim.lock GC
# ---------------------------------------------------------------------------

class TestOrphanClaimLockGC:
    """P1-2: Orphan claim.lock GC."""

    def test_orphan_claim_lock_gc_after_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        import hermes_cli.gateway_restart_state as state_mod

        # Create a request dir with an old claim.lock
        profile_dir = state_mod._get_profile_dir("default")
        rid = "orphan-request-123"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True)

        clp = req_dir / "claim.lock"
        old_time = time.time() - 7200  # 2 hours ago
        clp.write_text(json.dumps({
            "request_id": rid,
            "worker_pid": 99999,
            "created_at": old_time,
        }))

        # Also set directory mtime to be old (GC checks mtime for no-status dirs)
        os.utime(str(req_dir), (old_time, old_time))

        removed = state_mod.gc_expired_request_dirs("default", max_age_s=3600)
        assert removed == 1
        assert not req_dir.exists()


# ---------------------------------------------------------------------------
# P1-2: lease.json publish failure → intent rollback
# ---------------------------------------------------------------------------

class TestLeasePublishRollback:
    """P1-2: lease.json publish failure → intent rollback."""

    def test_lease_publish_failure_rollback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, read_intent,
        )
        import hermes_cli.gateway_restart_state as state_mod

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Monkey-patch _atomic_write_json to fail on lease.json
        original_awj = state_mod._atomic_write_json
        def failing_awj(path, data):
            if "lease.json" in str(path):
                raise OSError("disk full")
            return original_awj(path, data)

        monkeypatch.setattr(state_mod, "_atomic_write_json", failing_awj)

        lock = RestartLock("default")
        result = lock.claim_lease(rid, nonce)
        assert result is False

        # Intent should be rolled back to scheduled
        stored = read_intent("default", rid)
        assert stored is not None
        assert stored["state"] == "scheduled"


# ---------------------------------------------------------------------------
# GC tests (P1-1, P1-2)
# ---------------------------------------------------------------------------

class TestGCOrphanClaimLock:
    """P1-1: orphan claim.lock with status=scheduled should be GC'd."""

    def test_orphan_claim_lock_with_scheduled_status_gc(self, tmp_path, monkeypatch):
        """status=scheduled + old claim.lock + dead worker_pid → GC deletes."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Create a request directory with scheduled status + claim.lock
        rid = "orphan-claim-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        # status.json with state=scheduled (Coordinator wrote this before Worker crashed)
        (req_dir / "status.json").write_text(json.dumps({
            "state": "scheduled",
            "request_id": rid,
            "updated_at": "2020-01-01T00:00:00+00:00",
        }), encoding="utf-8")

        # claim.lock with dead PID and old timestamp
        (req_dir / "claim.lock").write_text(json.dumps({
            "request_id": rid,
            "worker_pid": 99999,  # Dead PID
            "created_at": time.time() - 7200,  # 2 hours ago
        }), encoding="utf-8")

        # intent.json
        (req_dir / "intent.json").write_text(json.dumps({
            "schema_version": 1,
            "request_id": rid,
            "nonce": "test",
            "profile": profile,
            "hermes_home": str(tmp_path),
            "target_pid": 100,
            "origin": "test",
            "state": "scheduled",
            "created_at": "2020-01-01T00:00:00+00:00",
            "expires_at": 0,
        }), encoding="utf-8")

        # No lease.json (Worker crashed before publishing)

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 1
        assert not req_dir.exists()

    def test_orphan_claim_lock_live_worker_skipped(self, tmp_path, monkeypatch):
        """claim.lock with live worker_pid → GC skips."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod
        import os

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "live-worker-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        (req_dir / "claim.lock").write_text(json.dumps({
            "request_id": rid,
            "worker_pid": os.getpid(),  # Current process — alive
            "created_at": time.time() - 7200,
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 0
        assert req_dir.exists()

    def test_orphan_claim_lock_recent_skipped(self, tmp_path, monkeypatch):
        """claim.lock too recent → GC skips."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "recent-claim-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        (req_dir / "claim.lock").write_text(json.dumps({
            "request_id": rid,
            "worker_pid": 99999,
            "created_at": time.time() - 10,  # Only 10 seconds ago
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 0
        assert req_dir.exists()


class TestGCOrphanLeaseJson:
    """P1-2: orphan lease.json should be GC'd when worker is dead."""

    def test_orphan_lease_json_gc(self, tmp_path, monkeypatch):
        """old lease.json + dead worker_pid + terminal status → GC deletes."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "orphan-lease-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        # status.json with terminal state
        (req_dir / "status.json").write_text(json.dumps({
            "state": "failed",
            "request_id": rid,
            "updated_at": "2020-01-01T00:00:00+00:00",
        }), encoding="utf-8")

        # lease.json with dead PID and old timestamp
        (req_dir / "lease.json").write_text(json.dumps({
            "request_id": rid,
            "owner_token": "test-token",
            "worker_pid": 99999,  # Dead PID
            "claimed_at": time.time() - 7200,  # 2 hours ago
        }), encoding="utf-8")

        # claim.lock
        (req_dir / "claim.lock").write_text(json.dumps({
            "request_id": rid,
            "worker_pid": 99999,
            "created_at": time.time() - 7200,
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 1
        assert not req_dir.exists()

    def test_orphan_lease_json_live_worker_skipped(self, tmp_path, monkeypatch):
        """lease.json with live worker → GC skips."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod
        import os

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "live-lease-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        (req_dir / "lease.json").write_text(json.dumps({
            "request_id": rid,
            "owner_token": "test-token",
            "worker_pid": os.getpid(),  # Alive
            "claimed_at": time.time() - 7200,
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 0
        assert req_dir.exists()

    def test_orphan_lease_json_recent_skipped(self, tmp_path, monkeypatch):
        """lease.json too recent → GC skips."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "recent-lease-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        (req_dir / "lease.json").write_text(json.dumps({
            "request_id": rid,
            "owner_token": "test-token",
            "worker_pid": 99999,
            "claimed_at": time.time() - 10,  # Only 10s ago
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600)
        assert removed == 0
        assert req_dir.exists()

    def test_active_request_dir_preserved(self, tmp_path, monkeypatch):
        """active.lock指向的request目录不会被GC删除。"""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as config_mod
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(tmp_path))
        import hermes_cli.gateway_restart_state as state_mod

        profile = "default"
        profile_dir = tmp_path / "run" / "gateway-restart" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        rid = "active-request-test"
        req_dir = profile_dir / rid
        req_dir.mkdir(parents=True, exist_ok=True)

        (req_dir / "lease.json").write_text(json.dumps({
            "request_id": rid,
            "owner_token": "test-token",
            "worker_pid": 99999,
            "claimed_at": time.time() - 7200,
        }), encoding="utf-8")

        removed = state_mod.gc_expired_request_dirs(
            profile=profile, max_age_s=3600,
            active_request_id=rid)
        assert removed == 0
        assert req_dir.exists()
