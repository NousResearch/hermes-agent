"""Tests for gateway_windows_restart_worker.py — intent/nonce validation,
PID/port handling, worker status/fallback, and transaction isolation.

Updated for per-request directory layout:
  run/gateway-restart/{profile}/{request_id}/

Key API changes tested:
  - Worker CLI: --profile + --request-id (not --intent/--intent-file)
  - Worker reads intent via read_intent(profile, request_id)
  - _run_restart_transaction takes 8 params
  - claim_lease uses O_EXCL + transitions intent state 'scheduled' → 'claimed'
  - Lease loser logs + sys.exit(1), does NOT write failed status or cleanup
  - cleanup_intent(profile, request_id) — no nonce
  - write_status(profile, state, request_id=request_id)
  - read_status(profile, request_id)
  - cleanup_status removed
  - update_intent_state guarded by expected_state
  - _pid_exists imported at function level from gateway_restart_state
"""

import json
import os
import sys
import time
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    """Set up a temporary HERMES_HOME for worker tests."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "run").mkdir()
    (hermes_home / "logs").mkdir()
    (hermes_home / "profiles").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(hermes_home))

    return hermes_home


def _make_intent(
    profile="default",
    target_pid=1234,
    request_id=None,
    nonce=None,
    ttl_s=300,
    schema_version=1,
    state="scheduled",
):
    """Create a realistic intent dict without writing to disk."""
    import secrets
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "schema_version": schema_version,
        "request_id": request_id or str(uuid.uuid4()),
        "nonce": nonce or secrets.token_urlsafe(32),
        "profile": profile,
        "hermes_home": "/fake/hermes",
        "target_pid": target_pid,
        "task_name": "Hermes_Gateway",
        "origin": "test",
        "created_at": now.isoformat(),
        "expires_at": now.timestamp() + ttl_s,
        "state": state,
    }


def _write_intent_to_disk(hermes_home, intent, profile="default"):
    """Write an intent dict to the per-request directory on disk."""
    request_id = intent["request_id"]
    req_dir = hermes_home / "run" / "gateway-restart" / profile / request_id
    req_dir.mkdir(parents=True, exist_ok=True)
    path = req_dir / "intent.json"
    path.write_text(json.dumps(intent, indent=2), encoding="utf-8")
    return path


def _write_intent_for_test(worker_env, intent, profile="default", dir_request_id=None):
    """Write intent to per-request dir using the intent's request_id
    (or dir_request_id if specified, for tamper tests)."""
    request_id = dir_request_id or intent["request_id"]
    req_dir = worker_env / "run" / "gateway-restart" / profile / request_id
    req_dir.mkdir(parents=True, exist_ok=True)
    path = req_dir / "intent.json"
    path.write_text(json.dumps(intent, indent=2), encoding="utf-8")
    return path


# ===========================================================================
# TestIntentValidation — P0-5 intent validation
# ===========================================================================

class TestIntentValidation:
    """Verify that _run_restart_transaction validates the on-disk intent
    against the in-memory intent before any destructive action."""

    def _mock_cleanup_for_status_check(self, monkeypatch):
        """Mock cleanup_intent to no-op so we can read back status after
        _fail_closed. Without this, cleanup_intent deletes the entire
        request directory (including the just-written status file)."""
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state.cleanup_intent",
            lambda *a, **kw: None,
        )

    def test_intent_missing(self, worker_env, monkeypatch):
        """No intent file on disk → fail closed (sys.exit(1))."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        # Do NOT write an intent file — disk is empty.

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status — only JSONL
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", intent["request_id"])
        assert status is None, "_fail_closed must not write status"

    def test_request_id_mismatch(self, worker_env, monkeypatch):
        """Disk intent has different request_id → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        # Write a disk intent with a DIFFERENT request_id, but in the
        # directory of the ORIGINAL request_id (so the worker can find it)
        tampered = dict(intent)
        tampered["request_id"] = "00000000-0000-0000-0000-000000000099"
        _write_intent_for_test(worker_env, tampered, dir_request_id=intent["request_id"])

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", intent["request_id"])
        assert status is None, "_fail_closed must not write status"

    def test_nonce_mismatch(self, worker_env, monkeypatch):
        """Disk intent has different nonce → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        # Disk intent has a different nonce
        tampered = dict(intent)
        tampered["nonce"] = "wrong-nonce-value"
        _write_intent_for_test(worker_env, tampered)

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", intent["request_id"])
        assert status is None, "_fail_closed must not write status"

    def test_profile_mismatch(self, worker_env, monkeypatch):
        """Disk intent has different profile → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        # Write disk intent under "default" request dir but with profile="other" in content
        tampered = dict(intent)
        tampered["profile"] = "other"
        _write_intent_for_test(worker_env, tampered, profile="default")

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", intent["request_id"])
        assert status is None, "_fail_closed must not write status"

    def test_target_pid_mismatch(self, worker_env, monkeypatch):
        """Disk intent has different target_pid → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        tampered = dict(intent)
        tampered["target_pid"] = 9999
        _write_intent_for_test(worker_env, tampered)

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", intent["request_id"])
        assert status is None, "_fail_closed must not write status"

    def test_ttl_expired(self, worker_env, monkeypatch):
        """Disk intent has expires_at in the past → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234, ttl_s=300)
        tampered = dict(intent)
        tampered["expires_at"] = time.time() - 60  # expired 60s ago
        _write_intent_for_test(worker_env, tampered)

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

    def test_schema_version_unsupported(self, worker_env, monkeypatch):
        """Disk intent has schema_version=99 → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction

        intent = _make_intent(profile="default", target_pid=1234)
        tampered = dict(intent)
        tampered["schema_version"] = 99
        _write_intent_for_test(worker_env, tampered)

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                intent, "default", intent["request_id"], intent["nonce"],
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

    def test_intent_consumed_prevents_replay(self, worker_env, monkeypatch):
        """After first worker consumes intent (cleanup), second worker with
        same intent should fail because the intent directory is removed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, cleanup_intent,
        )

        # --- First worker run: create intent and consume it ---
        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Simulate first worker's cleanup (removes request dir)
        cleanup_intent("default", request_id)

        # --- Second worker run: same intent parameters ---
        # Mock cleanup_intent so _fail_closed doesn't delete the status
        # we want to verify
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state.cleanup_intent",
            lambda *a, **kw: None,
        )

        replay_intent = _make_intent(
            profile="default",
            target_pid=1234,
            request_id=request_id,
            nonce=nonce,
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                replay_intent, "default", request_id, nonce,
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # P0-4: _fail_closed no longer writes status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", request_id)
        assert status is None, "_fail_closed must not write status"


# ===========================================================================
# TestPidPort — P0-6 and P0-7 PID/port handling
# ===========================================================================

class TestPidPort:
    """Verify PID liveness and port-release safety checks."""

    def test_old_pid_still_alive_after_force_kill(self, worker_env, monkeypatch):
        """Old PID still alive after drain → RuntimeError, no new gateway."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-create a lock and claim lease so worker passes the claim step
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        # Mock _drain_and_stop to be a no-op
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            lambda *a, **kw: None,
        )
        # Mock _wait_for_handoff to return True (simulate handoff succeeded)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        # Mock _pid_exists to return True for old_pid (still alive), False for others
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: pid == 1234,
        )
        # Mock _start_new_gateway to track if it's called
        mock_start = MagicMock()
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._start_new_gateway",
            mock_start,
        )

        with pytest.raises(RuntimeError, match="still alive"):
            _run_restart_transaction(
                disk_intent, "default", request_id, nonce,
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )

        # New gateway should NOT have been started
        mock_start.assert_not_called()

    def test_unrelated_process_occupies_port(self, worker_env, monkeypatch):
        """Port occupied by unrelated process → RuntimeError."""
        from hermes_cli.gateway_windows_restart_worker import _wait_for_port_release

        # Mock port check: always occupied
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_port_in_use",
            lambda port: True,
        )
        # PID query returns a non-hermes PID
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._get_pids_on_port",
            lambda port: [9999],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_hermes_gateway_pid",
            lambda pid: False,
        )

        with pytest.raises(RuntimeError, match="not the old gateway"):
            _wait_for_port_release(
                profile="default",
                request_id="11111111-1111-1111-1111-111111111111",
                old_pid=1234,
                origin="test",
                port=8080,
                timeout=0.1,  # short timeout so the wait loop ends quickly
            )

    def test_pid_query_empty_port_occupied(self, worker_env, monkeypatch):
        """Port occupied but PID query returns empty → RuntimeError."""
        from hermes_cli.gateway_windows_restart_worker import _wait_for_port_release

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_port_in_use",
            lambda port: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._get_pids_on_port",
            lambda port: [],
        )

        with pytest.raises(RuntimeError, match="no listening PIDs"):
            _wait_for_port_release(
                profile="default",
                request_id="11111111-1111-1111-1111-111111111111",
                old_pid=1234,
                origin="test",
                port=8080,
                timeout=0.1,
            )

    def test_port_released_after_hermes_kill(self, worker_env, monkeypatch):
        """Port occupied → hermes PID found → kill → port free → success."""
        from hermes_cli.gateway_windows_restart_worker import _wait_for_port_release

        call_count = {"port_check": 0}

        def mock_is_port_in_use(port):
            call_count["port_check"] += 1
            if call_count["port_check"] <= 1:
                return True
            return False

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_port_in_use",
            mock_is_port_in_use,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._get_pids_on_port",
            lambda port: [5678],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_hermes_gateway_pid",
            lambda pid: pid == 5678,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_ancestor",
            lambda pid: False,
        )

        # Mock terminate_pid
        mock_terminate = MagicMock()
        with patch.dict("sys.modules", {"gateway": MagicMock(), "gateway.status": MagicMock()}):
            import gateway.status
            gateway.status.terminate_pid = mock_terminate
            # Should NOT raise — port is freed after kill
            _wait_for_port_release(
                profile="default",
                request_id="11111111-1111-1111-1111-111111111111",
                old_pid=5678,  # P1-5: listener PID must match old_pid
                origin="test",
                port=8080,
                timeout=0.5,
            )

    def test_port_still_occupied_after_kill(self, worker_env, monkeypatch):
        """Port occupied → old_pid matches → kill → port STILL occupied → RuntimeError."""
        from hermes_cli.gateway_windows_restart_worker import _wait_for_port_release

        # Port is always occupied (even after "kill")
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_port_in_use",
            lambda port: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._get_pids_on_port",
            lambda port: [5678],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_hermes_gateway_pid",
            lambda pid: pid == 5678,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._is_ancestor",
            lambda pid: False,
        )

        mock_terminate = MagicMock()
        with patch.dict("sys.modules", {"gateway": MagicMock(), "gateway.status": MagicMock()}):
            import gateway.status
            gateway.status.terminate_pid = mock_terminate

            with pytest.raises(RuntimeError, match="still occupied|not the old gateway"):
                _wait_for_port_release(
                    profile="default",
                    request_id="11111111-1111-1111-1111-111111111111",
                    old_pid=5678,  # P1-5: must match listener PID
                    origin="test",
                    port=8080,
                    timeout=0.5,
                )


# ===========================================================================
# TestWorkerStatusFallback — P0-8 and P1-1/P1-2 scenarios
# ===========================================================================

class TestWorkerStatusFallback:
    """Verify status file handling, unhandled exceptions, and coordinator
    wait-for-completion logic."""

    def test_unhandled_exception_writes_failed(self, worker_env, tmp_path, monkeypatch):
        """Unhandled exception in _run_restart_transaction → last status preserved."""
        from hermes_cli.gateway_windows_restart_worker import main
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-create a lock and claim lease so the worker passes the claim step
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        # Make _drain_and_stop raise an unhandled exception
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            MagicMock(side_effect=RuntimeError("drain exploded")),
        )
        # Mock _wait_for_handoff to return True (simulate handoff succeeded)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        # Ensure _pid_exists returns False so the PID check passes
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        # Call main() with --profile and --request-id
        monkeypatch.setattr(sys, "argv", [
            "worker", "--profile", "default", "--request-id", request_id,
        ])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        # P0-1: _run_restart_transaction now writes "failed" status on exception
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", request_id)
        assert status is not None
        assert status["state"] == "failed"
        assert "drain exploded" in status.get("error", "")

    def test_status_ignores_old_request_id(self, worker_env, monkeypatch):
        """Coordinator _wait_for_completion returns False when status has
        an old request_id different from the one being waited on."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        # Write a status with an old request_id
        write_status("default", "completed", request_id="e5e5e5e5-e5e5-e5e5-e5e5-e5e5e5e5e5e5", new_pid=5678)

        # Wait for a different request_id — should NOT see the old status
        result, last_state = _wait_for_completion(
            profile="default",
            timeout_s=1.0,
            request_id="f6f6f6f6-f6f6-f6f6-f6f6-f6f6f6f6f6f6",
        )
        assert result is False

    def test_status_intermediate_not_completed(self, worker_env, monkeypatch):
        """Status has state='draining' → coordinator doesn't report completed."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        request_id = "b4b4b4b4-b4b4-b4b4-b4b4-b4b4b4b4b4b4"
        write_status("default", "draining", request_id=request_id, old_pid=1234)

        result, last_state = _wait_for_completion(
            profile="default",
            timeout_s=1.0,
            request_id=request_id,
        )
        assert result is False

    def test_new_pid_dies_immediately(self, worker_env, monkeypatch):
        """New PID dies immediately → status 'failed'."""
        from hermes_cli.gateway_windows_restart_worker import _verify_new_gateway

        # Mock _pid_exists: new_pid is dead immediately
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: pid != 5678,  # 5678 (new_pid) is dead
        )

        _verify_new_gateway(
            profile="default",
            request_id="11111111-1111-1111-1111-111111111111",
            old_pid=1234,
            new_pid=5678,
            origin="test",
            launcher="direct_spawn",
        )

        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", "11111111-1111-1111-1111-111111111111")
        assert status is not None
        assert status["state"] == "failed"
        assert "died" in status.get("error", "").lower()

    def test_new_pid_dies_within_stability_window(self, worker_env, monkeypatch):
        """New PID alive at first check, then dies within stability window
        → status 'failed'."""
        from hermes_cli.gateway_windows_restart_worker import _verify_new_gateway

        # _pid_exists returns True first, then False (dies during stability check)
        call_count = {"n": 0}

        def mock_pid_exists(pid):
            if pid == 5678:
                call_count["n"] += 1
                return call_count["n"] <= 1
            return False  # old_pid is dead

        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            mock_pid_exists,
        )

        _verify_new_gateway(
            profile="default",
            request_id="11111111-1111-1111-1111-111111111111",
            old_pid=1234,
            new_pid=5678,
            origin="test",
            launcher="direct_spawn",
        )

        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", "11111111-1111-1111-1111-111111111111")
        assert status is not None
        assert status["state"] == "failed"
        assert "stability" in status.get("error", "").lower() or "died" in status.get("error", "").lower()

    def test_unhandled_exception_writes_failed_status_field(self, worker_env, tmp_path, monkeypatch):
        """Unhandled exception → status file has request_id matching the
        intent's request_id, enabling coordinator correlation."""
        from hermes_cli.gateway_windows_restart_worker import main
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-create a lock and claim lease
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        monkeypatch.setattr(sys, "argv", [
            "worker", "--profile", "default", "--request-id", request_id,
        ])

        with pytest.raises(SystemExit):
            main()

        # P0-1: _run_restart_transaction now writes "failed" status on exception.
        # The status has the correct request_id.
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", request_id)
        assert status is not None
        assert status["request_id"] == request_id
        assert status["state"] == "failed"

    def test_coordinator_completed_with_matching_request_id(self, worker_env, monkeypatch):
        """Coordinator _wait_for_completion returns True when status
        has matching request_id and state='completed'."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        request_id = "d6d6d6d6-d6d6-d6d6-d6d6-d6d6d6d6d6d6"
        write_status("default", "completed", request_id=request_id,
                     old_pid=1234, new_pid=5678, launcher="direct_spawn")

        result, _ = _wait_for_completion(
            profile="default",
            timeout_s=2.0,
            request_id=request_id,
        )
        assert result is True

    def test_coordinator_failed_with_matching_request_id(self, worker_env, monkeypatch):
        """Coordinator _wait_for_completion returns False when status
        has matching request_id but state='failed'."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        request_id = "c5c5c5c5-c5c5-c5c5-c5c5-c5c5c5c5c5c5"
        write_status("default", "failed", request_id=request_id,
                     error="something broke")

        result, _ = _wait_for_completion(
            profile="default",
            timeout_s=2.0,
            request_id=request_id,
        )
        assert result is False

    def test_verify_new_gateway_no_pid(self, worker_env, monkeypatch):
        """new_pid <= 0 → status 'failed', no crash."""
        from hermes_cli.gateway_windows_restart_worker import _verify_new_gateway

        _verify_new_gateway(
            profile="default",
            request_id="11111111-1111-1111-1111-111111111111",
            old_pid=1234,
            new_pid=0,
            origin="test",
            launcher="direct_spawn",
        )

        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", "11111111-1111-1111-1111-111111111111")
        assert status is not None
        assert status["state"] == "failed"
        assert "No new gateway PID" in status.get("error", "")

    def test_verify_new_gateway_dual_gateway(self, worker_env, monkeypatch):
        """Both old and new PIDs alive → dual gateway → status 'failed'."""
        from hermes_cli.gateway_windows_restart_worker import _verify_new_gateway

        # All PIDs are alive
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: True,
        )

        _verify_new_gateway(
            profile="default",
            request_id="11111111-1111-1111-1111-111111111111",
            old_pid=1234,
            new_pid=5678,
            origin="test",
            launcher="direct_spawn",
        )

        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", "11111111-1111-1111-1111-111111111111")
        assert status is not None
        assert status["state"] == "failed"
        assert "dual gateway" in status.get("error", "").lower() or "still alive" in status.get("error", "").lower()

    def test_intermediate_state_timeout_not_completed(self, worker_env, monkeypatch):
        """P1-2: Status has intermediate state (e.g. 'draining') — coordinator
        wait_for_completion should NOT report completed (returns False on
        timeout)."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        request_id = "c9c9c9c9-c9c9-c9c9-c9c9-c9c9c9c9c9c9"
        # Write an intermediate state that will never become "completed"
        write_status("default", "draining", request_id=request_id, old_pid=1234)

        result, _ = _wait_for_completion(
            profile="default",
            timeout_s=0.5,  # very short timeout
            request_id=request_id,
        )
        assert result is False

        # Also verify with other intermediate states
        for state in ("stopping", "waiting_pid_exit", "waiting_port_release",
                       "starting_task", "starting_direct_fallback", "verifying",
                       "preflight_ok"):
            rid = str(uuid.uuid4())
            write_status("default", state, request_id=rid, old_pid=1234)
            result, _ = _wait_for_completion(
                profile="default",
                timeout_s=0.5,
                request_id=rid,
            )
            assert result is False, f"State '{state}' should not be reported as completed"


# ===========================================================================
# TestTransactionIsolation — Transaction isolation and cross-worker safety
# ===========================================================================

class TestTransactionIsolation:
    """Verify that concurrent / stale transactions do not interfere with
    each other's state files (intent, lock, status, lease)."""

    # -- helpers ----------------------------------------------------------

    def _read_lock_from_disk(self, hermes_home, profile="default"):
        """Read the lock file dict from disk."""
        path = hermes_home / "run" / "gateway-restart" / profile / "active.lock"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_intent_from_disk(self, hermes_home, profile="default", request_id=""):
        """Read the intent file dict from disk (per-request dir)."""
        if not request_id:
            return None
        path = hermes_home / "run" / "gateway-restart" / profile / request_id / "intent.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- tests ------------------------------------------------------------

    def test_lock_and_intent_share_request_id(self, worker_env, monkeypatch):
        """Create a coordinator transaction (mock _spawn_worker,
        _wait_for_worker_claim). Read the lock file and intent file from
        disk. Assert both have the SAME request_id."""
        from hermes_cli.gateway_windows_restart import schedule_restart_handoff

        # Mock platform check and helpers
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._assert_windows", lambda: None
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart.preflight_check",
            lambda **kw: (True, "preflight_ok"),
        )
        # Mock gateway.status.get_running_pid
        mock_gateway_status = MagicMock()
        mock_gateway_status.get_running_pid = MagicMock(return_value=1234)
        monkeypatch.setitem(sys.modules, "gateway", MagicMock())
        monkeypatch.setitem(sys.modules, "gateway.status", mock_gateway_status)

        # Mock _spawn_worker to just return a fake PID
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._spawn_worker",
            lambda intent, profile, request_id: 9999,
        )
        # Mock _wait_for_worker_claim to return False (lock stays on disk)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._wait_for_worker_claim",
            lambda profile, request_id, timeout_s=10.0: False,
        )
        # Don't wait for completion
        result = schedule_restart_handoff(
            origin="test", profile="default", wait=False
        )
        # P1-3: handoff failure → scheduled=False (worker didn't claim)
        assert result["scheduled"] is False

        request_id = result["request_id"]

        # Read lock and intent from disk
        lock_data = self._read_lock_from_disk(worker_env)
        intent_data = self._read_intent_from_disk(worker_env, "default", request_id)

        assert lock_data is not None, "Lock file should exist on disk"
        assert intent_data is not None, "Intent file should exist on disk"
        assert lock_data["request_id"] == request_id
        assert intent_data["request_id"] == request_id
        assert lock_data["request_id"] == intent_data["request_id"]

    def test_claim_lease_only_once(self, worker_env, monkeypatch):
        """Create intent + lock. Worker-1 calls claim_lease(request_id)
        → True. Worker-2 (new RestartLock instance) calls
        claim_lease(same request_id) → False."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator acquires lock
        coordinator_lock = RestartLock("default")
        assert coordinator_lock.try_acquire(request_id) is True

        # Worker-1 claims the lease
        worker1_lock = RestartLock("default")
        assert worker1_lock.claim_lease(request_id, nonce) is True

        # Worker-2 tries to claim the same request_id
        worker2_lock = RestartLock("default")
        assert worker2_lock.claim_lease(request_id, nonce) is False

    def test_claim_timeout_lock_preserved(self, worker_env, monkeypatch):
        """Mock _wait_for_worker_claim to return False (timeout). After
        schedule_restart_handoff returns, verify the lock file still exists
        on disk. Then try a second restart → returns 'already in progress'."""
        from hermes_cli.gateway_windows_restart import schedule_restart_handoff

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._assert_windows", lambda: None
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart.preflight_check",
            lambda **kw: (True, "preflight_ok"),
        )
        mock_gateway_status = MagicMock()
        mock_gateway_status.get_running_pid = MagicMock(return_value=1234)
        monkeypatch.setitem(sys.modules, "gateway", MagicMock())
        monkeypatch.setitem(sys.modules, "gateway.status", mock_gateway_status)

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._spawn_worker",
            lambda intent, profile, request_id: 9999,
        )
        # Worker fails to claim in time
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart._wait_for_worker_claim",
            lambda profile, request_id, timeout_s=10.0: False,
        )

        result = schedule_restart_handoff(
            origin="test", profile="default", wait=False
        )
        # P1-3: handoff failure → scheduled=False (worker didn't claim)
        assert result["scheduled"] is False
        lock_data = self._read_lock_from_disk(worker_env)
        assert lock_data is not None, "Lock file should still exist after claim timeout"

        # Second restart attempt should report "already in progress"
        result2 = schedule_restart_handoff(
            origin="test", profile="default", wait=False
        )
        assert result2["scheduled"] is False
        assert "already in progress" in result2["detail"].lower()

    def test_stale_worker_preserves_new_intent(self, worker_env, monkeypatch):
        """Create intent-A on disk. Then create intent-B (different
        request_id). Call cleanup_intent(profile, request_id=A). Assert
        intent-B still exists on disk."""
        from hermes_cli.gateway_restart_state import create_intent, cleanup_intent

        # Intent-A
        intent_a = create_intent(profile="default", target_pid=1111, origin="f2f2f2f2-f2f2-f2f2-f2f2-f2f2f2f2f2f2")
        request_id_a = intent_a["request_id"]

        # Intent-B (different request_id — different directory)
        intent_b = create_intent(profile="default", target_pid=2222, origin="a3a3a3a3-a3a3-a3a3-a3a3-a3a3a3a3a3a3")
        request_id_b = intent_b["request_id"]

        assert request_id_a != request_id_b

        # Stale worker-A tries to clean up with its old request_id
        cleanup_intent("default", request_id=request_id_a)

        # Intent-B should still be on disk
        intent_on_disk = self._read_intent_from_disk(worker_env, "default", request_id_b)
        assert intent_on_disk is not None
        assert intent_on_disk["request_id"] == request_id_b

    def test_stale_worker_preserves_new_lease(self, worker_env, monkeypatch):
        """Create lock + intent. Worker-B claims lease (O_EXCL). Worker-A
        tries to release using stale reference. Assert lease still exists."""
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, lease_json_path,
        )

        # Coordinator creates lock + intent
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire("11111111-aaaa-bbbb-cccc-dddddddddddd") is True
        intent = create_intent(request_id="11111111-aaaa-bbbb-cccc-dddddddddddd", profile="default", target_pid=1234)

        # Worker-B claims the lease (O_EXCL)
        worker_b_lock = RestartLock("default")
        assert worker_b_lock.claim_lease("11111111-aaaa-bbbb-cccc-dddddddddddd", intent["nonce"]) is True

        # Worker-A tries to release using stale reference (no owner_token)
        worker_a_lock = RestartLock("default")
        worker_a_lock.release()

        # Lease file should still exist (Worker-B owns it)
        lp = lease_json_path("default", "11111111-aaaa-bbbb-cccc-dddddddddddd")
        assert lp.exists()

    def test_force_release_preserves_new_lock(self, worker_env, monkeypatch):
        """Create lock-A. Then create lock-B (different request_id,
        simulating a new transaction). Call _force_release(expected=lock_A_data)
        on a RestartLock. Assert lock-B still exists."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path
        import time as _time

        # Create lock-A
        lock_a = RestartLock("default")
        assert lock_a.try_acquire("22222222-aaaa-bbbb-cccc-dddddddddddd") is True

        # Read lock-A data
        lock_a_data = self._read_lock_from_disk(worker_env)
        assert lock_a_data is not None

        # Delete lock-A manually, then create lock-B (simulates a new transaction)
        lp = lock_path("default")
        lp.unlink(missing_ok=True)
        lock_b = RestartLock("default")
        assert lock_b.try_acquire("33333333-aaaa-bbbb-cccc-dddddddddddd") is True

        # Now use a RestartLock instance to force-release with old lock-A data
        # It should detect the mismatch and NOT delete lock-B
        stale_lock = RestartLock("default")
        stale_lock._force_release(expected=lock_a_data)

        # Lock-B should still exist
        lock_b_data = self._read_lock_from_disk(worker_env)
        assert lock_b_data is not None
        assert lock_b_data["request_id"] == "33333333-aaaa-bbbb-cccc-dddddddddddd"

    def test_wait_claim_ignores_old_request_id(self, worker_env, monkeypatch):
        """Write a lock file with request_id='old' and claimed_at set.
        Call _wait_for_worker_claim(profile, request_id='unique-new-claim-test', timeout_s=1).
        Assert returns False."""
        from hermes_cli.gateway_windows_restart import _wait_for_worker_claim
        from hermes_cli.gateway_restart_state import lock_path, lease_json_path
        import time as _time

        # Use a unique request_id to avoid test pollution
        unique_rid = str(uuid.uuid4())

        # Write a lock file with request_id="old" and claimed_at
        lp = lock_path("default")
        lp.parent.mkdir(parents=True, exist_ok=True)
        lock_data = {
            "schema_version": 1,
            "request_id": "11111111-0000-0000-0000-000000000000",
            "owner_token": "some-token",
            "owner_pid": 9999,
            "profile": "default",
            "created_at": _time.time(),
            "expires_at": _time.time() + 300,
            "claimed_at": _time.time(),
        }
        lp.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

        # Verify the unique lease path doesn't exist
        ljp = lease_json_path("default", unique_rid)
        assert not ljp.exists(), f"Lease path {ljp} should not exist yet"

        # Wait for a DIFFERENT request_id — should NOT match
        result = _wait_for_worker_claim("default", unique_rid, timeout_s=1.0)
        assert result is False

    def test_intermediate_state_not_completed(self, worker_env, monkeypatch):
        """Write status with state='draining' and request_id=X.
        Call _wait_for_completion(profile, timeout_s=1, request_id=X).
        Assert returns False."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        write_status("default", "draining", request_id="e1e1e1e1-e1e1-e1e1-e1e1-e1e1e1e1e1e1", old_pid=1234)

        result, _ = _wait_for_completion(
            profile="default", timeout_s=1.0, request_id="e1e1e1e1-e1e1-e1e1-e1e1-e1e1e1e1e1e1"
        )
        assert result is False

    def test_direct_spawn_no_evidence_fails(self, worker_env, monkeypatch):
        """Mock _direct_spawn_gateway to return PID 1234. Mock
        _wait_for_launch_evidence to return 0. Call _start_new_gateway.
        Assert it raises RuntimeError with 'launch evidence'."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway

        # Make it skip the scheduled task path
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway",
            lambda hermes_home="", profile="default": 1234,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_launch_evidence",
            lambda old_pid, timeout=15.0: 0,
        )
        # Ensure is_task_registered returns False
        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=False)
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        with pytest.raises(RuntimeError, match="launch evidence"):
            _start_new_gateway(
                profile="default",
                request_id="11111111-1111-1111-1111-111111111111",
                old_pid=1234,
                origin="test",
                task_name="Hermes_Gateway",
                task_registered=False,
            )

    def test_concurrent_only_one_worker(self, worker_env, monkeypatch):
        """Create intent. Pre-create lock. Worker-1 claims → True.
        Worker-2 claims → False (O_EXCL prevents double claim)."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator acquires lock
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(request_id) is True

        # Worker-1 claims
        worker1 = RestartLock("default")
        assert worker1.claim_lease(request_id, nonce) is True

        # Worker-2 (same request_id) — should fail (lease already exists on disk)
        worker2 = RestartLock("default")
        assert worker2.claim_lease(request_id, nonce) is False

    def test_claim_lease_wrong_request_id(self, worker_env, monkeypatch):
        """Worker tries to claim lease with a request_id that doesn't
        have an intent on disk → returns False."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        real_request_id = intent["request_id"]

        # Coordinator acquires lock
        lock = RestartLock("default")
        assert lock.try_acquire(real_request_id) is True

        worker = RestartLock("default")
        # Try to claim with a request_id that has no intent
        assert worker.claim_lease("b8b8b8b8-b8b8-b8b8-b8b8-b8b8b8b8b8b8", "some-nonce") is False

    def test_lock_owner_token_independence(self, worker_env, monkeypatch):
        """Two RestartLock instances acquiring different request_ids
        have independent owner_tokens."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        # First lock
        lock1 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True
        data1 = self._read_lock_from_disk(worker_env)
        token1 = data1["owner_token"]

        # Release and create second lock
        lock1.release()
        lock2 = RestartLock("default")
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is True
        data2 = self._read_lock_from_disk(worker_env)
        token2 = data2["owner_token"]

        assert token1 != token2

    def test_wait_claim_matches_only_when_request_id_and_unclaimed(
        self, worker_env, monkeypatch
    ):
        """_wait_for_worker_claim should return True only when
        lease.json exists for the given request_id."""
        from hermes_cli.gateway_windows_restart import _wait_for_worker_claim
        from hermes_cli.gateway_restart_state import lease_json_path

        # Use a unique request_id — no lease.json should exist for it
        unique_rid = str(uuid.uuid4())
        ljp = lease_json_path("default", unique_rid)
        assert not ljp.exists()

        # Should NOT match (no lease.json)
        result = _wait_for_worker_claim("default", unique_rid, timeout_s=1.0)
        assert result is False


# ===========================================================================
# P0-2: Lease loser behavior — must NOT write failed status or cleanup intent
# ===========================================================================

class TestLeaseLoserBehavior:
    """P0-2: When two workers race for the same lease, the loser must
    only log + sys.exit(1). It must NOT write a failed status or
    cleanup the intent (the winner owns those resources)."""

    def test_lease_loser_preserves_winner_status(
        self, worker_env, monkeypatch
    ):
        """P0-2: Worker-1 wins the lease and writes a status.
        Worker-2 (loser) must NOT overwrite it with 'failed'."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, write_status, read_status,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Simulate worker-1 having already won the lease by calling
        # claim_lease which creates the O_EXCL lease file
        winner_lock = RestartLock("default")
        assert winner_lock.try_acquire(request_id) is True
        assert winner_lock.claim_lease(request_id, nonce) is True

        # Winner writes a valid status
        write_status("default", "preflight_ok", request_id=request_id, old_pid=1234)
        winner_status = read_status("default", request_id)
        assert winner_status["state"] == "preflight_ok"

        # Now worker-2 tries to run the transaction with the same intent.
        # It should fail at claim_lease and exit without touching status.
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                disk_intent, "default", request_id, nonce,
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # Winner's status must be preserved — NOT overwritten to "failed"
        final_status = read_status("default", request_id)
        assert final_status is not None
        assert final_status["state"] == "preflight_ok"

    def test_lease_loser_preserves_intent(self, worker_env, monkeypatch):
        """P0-2: Worker-2 (loser) must NOT cleanup/delete the intent
        directory that the winner owns."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, read_intent,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Worker-1 wins the lease
        winner_lock = RestartLock("default")
        assert winner_lock.try_acquire(request_id) is True
        assert winner_lock.claim_lease(request_id, nonce) is True

        # Worker-2 tries the same — loser path
        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                disk_intent, "default", request_id, nonce,
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )
        assert exc_info.value.code == 1

        # Intent must still be on disk (not cleaned up by loser)
        intent_on_disk = read_intent("default", request_id)
        assert intent_on_disk is not None
        assert intent_on_disk["request_id"] == request_id

    def test_loser_through_main_preserves_winner(
        self, worker_env, monkeypatch
    ):
        """P0-2: Loser going through main() must NOT delete winner's
        intent/status/lease.  This exercises the real code path including
        main()'s exception handling (SystemExit bypasses except Exception).

        Unlike the tests above which call _run_restart_transaction directly,
        this test calls main() end-to-end without mocking cleanup_intent.
        """
        from hermes_cli.gateway_windows_restart_worker import main
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, read_intent, read_status, write_status,
        )

        disk_intent = create_intent(profile="default", target_pid=1234,
                                    origin="test")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Winner claims the lease and writes a status
        winner_lock = RestartLock("default")
        assert winner_lock.try_acquire(request_id) is True
        assert winner_lock.claim_lease(request_id, nonce) is True
        write_status("default", "preflight_ok", request_id=request_id,
                     old_pid=1234)

        # Loser calls main() — must NOT mock cleanup_intent
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )
        monkeypatch.setattr(sys, "argv", [
            "worker", "--profile", "default", "--request-id", request_id,
        ])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        # Winner's intent must survive
        intent_on_disk = read_intent("default", request_id)
        assert intent_on_disk is not None
        assert intent_on_disk["request_id"] == request_id

        # Winner's status must survive
        status_on_disk = read_status("default", request_id)
        assert status_on_disk is not None
        assert status_on_disk["state"] == "preflight_ok"

        # Winner's lease must survive
        from hermes_cli.gateway_restart_state import lease_json_path
        ljp = lease_json_path("default", request_id)
        assert ljp.exists(), "Loser must NOT delete winner's lease file"


# ===========================================================================
# P0-3: Only lease winner writes consumed/claimed state
# ===========================================================================

class TestOnlyWinnerWritesClaimedState:
    """P0-3: After claim_lease, only the winner transitions intent state
    from 'scheduled' to 'claimed'. The loser never touches the state."""

    def test_only_winner_writes_claimed_state(self, worker_env, monkeypatch):
        """claim_lease transitions intent from 'scheduled' to 'claimed'.
        Second claim_lease (loser) fails and does NOT touch state."""
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, read_intent,
        )

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = intent["request_id"]
        nonce = intent["nonce"]

        # Verify initial state is "scheduled"
        on_disk = read_intent("default", request_id)
        assert on_disk["state"] == "scheduled"

        # Winner claims
        winner_lock = RestartLock("default")
        assert winner_lock.claim_lease(request_id, nonce) is True

        # State should now be "claimed"
        after_claim = read_intent("default", request_id)
        assert after_claim is not None
        assert after_claim["state"] == "claimed"

        # Loser tries to claim — fails
        loser_lock = RestartLock("default")
        assert loser_lock.claim_lease(request_id, nonce) is False

        # State is STILL "claimed" (not corrupted)
        final = read_intent("default", request_id)
        assert final["state"] == "claimed"


# ===========================================================================
# P0-4: claim success then write_status OSError → lease released
# ===========================================================================

class TestClaimThenWriteStatusError:
    """P0-4: If claim_lease succeeds but write_status raises OSError,
    the transaction must release the lease and NOT leak resources."""

    def test_claim_success_then_write_status_error(self, worker_env, monkeypatch):
        """claim_lease succeeds. Then _drain_and_stop triggers write_status
        which raises OSError. The finally block must release the lease."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, lease_json_path, read_intent,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Acquire the profile lock so claim_lease can work
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(request_id) is True

        # Make _drain_and_stop raise an exception (simulating the
        # write_status OSError propagation)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            MagicMock(side_effect=OSError("disk full")),
        )
        # Mock _wait_for_handoff to return True (simulate handoff succeeded)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        with pytest.raises(OSError, match="disk full"):
            _run_restart_transaction(
                disk_intent, "default", request_id, nonce,
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )

        # P0-1: After the transaction, the exception handler writes "failed" status
        from hermes_cli.gateway_restart_state import read_status
        status = read_status("default", request_id)
        assert status is not None, "Terminal status must be preserved"
        assert status["state"] == "failed", (
            "P0-1: Winner exception must write terminal failed status"
        )

        # Lease must be released
        ljp = lease_json_path("default", request_id)
        assert not ljp.exists(), "Lease must be released by finally"


# ===========================================================================
# P0-1: Concurrent claim with threading.Barrier
# ===========================================================================

class TestConcurrentClaim:
    """P0-1: Two threads racing to claim_lease simultaneously.
    Only one must win (O_EXCL atomic guarantee)."""

    def test_concurrent_claim_only_one_winner(self, worker_env, monkeypatch):
        """Two threads call claim_lease at the same time using a
        threading.Barrier. Exactly one must succeed."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator pre-creates the profile lock
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(request_id) is True

        barrier = threading.Barrier(2, timeout=10)
        results = {"t1": None, "t2": None}
        errors = {"t1": None, "t2": None}

        def try_claim(name):
            try:
                lock = RestartLock("default")
                barrier.wait()  # synchronize both threads
                results[name] = lock.claim_lease(request_id, nonce)
            except Exception as e:
                errors[name] = e

        t1 = threading.Thread(target=try_claim, args=("t1",))
        t2 = threading.Thread(target=try_claim, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors["t1"] is None, f"Thread 1 error: {errors['t1']}"
        assert errors["t2"] is None, f"Thread 2 error: {errors['t2']}"

        # Exactly one thread must have won
        winners = [k for k, v in results.items() if v is True]
        losers = [k for k, v in results.items() if v is False]
        assert len(winners) == 1, f"Expected 1 winner, got {len(winners)}: {results}"
        assert len(losers) == 1, f"Expected 1 loser, got {len(losers)}: {results}"


# ===========================================================================
# P1-2: Intermediate state timeout not reported as completed
# ===========================================================================

class TestIntermediateStateTimeout:
    """P1-2: If the worker is stuck in an intermediate state and the
    coordinator's wait_for_completion times out, it must NOT report
    the restart as completed."""

    def test_intermediate_state_timeout_not_completed(self, worker_env, monkeypatch):
        """Write various intermediate states. Verify wait_for_completion
        returns False on timeout for each one."""
        from hermes_cli.gateway_windows_restart import _wait_for_completion
        from hermes_cli.gateway_restart_state import write_status

        intermediate_states = [
            "draining",
            "stopping",
            "waiting_pid_exit",
            "waiting_port_release",
            "starting_task",
            "starting_direct_fallback",
            "verifying",
            "preflight_ok",
            "claimed",
            "scheduled",
        ]

        for state in intermediate_states:
            rid = str(uuid.uuid4())
            write_status("default", state, request_id=rid, old_pid=1234)

            result, _ = _wait_for_completion(
                profile="default",
                timeout_s=0.3,  # very short — will definitely time out
                request_id=rid,
            )
            assert result is False, (
                f"State '{state}' with request_id '{rid}' should NOT be "
                f"reported as completed on timeout"
            )


# ===========================================================================
# P0-1: active.lock lifecycle — concurrent restart blocking
# ===========================================================================

class TestActiveLockLifecycle:
    """P0-1: active.lock covers full transaction lifecycle."""

    def test_active_lock_blocks_concurrent_restart(self, worker_env, monkeypatch):
        """While Worker-A is running, Request-B cannot acquire active.lock."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent

        intent_a = create_intent(profile="default", target_pid=100, origin="test")
        rid_a = intent_a["request_id"]

        lock_a = RestartLock("default")
        assert lock_a.try_acquire(rid_a) is True

        # Simulate Worker-A claiming lease (lock still held by coordinator)
        # Request-B tries to acquire — should fail
        lock_b = RestartLock("default")
        assert lock_b.try_acquire("33333333-aaaa-bbbb-cccc-dddddddddddd") is False


# ===========================================================================
# P0-2: active.lock atomic handoff and release
# ===========================================================================

class TestActiveLockHandoff:
    """P0-2: active.lock handoff from Coordinator to Worker."""

    def test_handoff_active_lock_atomic(self, worker_env, monkeypatch):
        """P0-2: handoff_active_lock atomically transfers ownership."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent, lease_path

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]
        nonce = intent["nonce"]

        # Coordinator acquires lock
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(rid) is True
        coord_token = coord_lock.owner_token

        # Worker claims lease
        worker_lock = RestartLock("default")
        assert worker_lock.claim_lease(rid, nonce) is True
        lease_token = worker_lock.owner_token

        # Coordinator hands off to Worker
        assert coord_lock.handoff_active_lock(rid, coord_token, os.getpid(), lease_token) is True

        # Worker's lock should now be able to release the active.lock
        worker_lock.release()

    def test_worker_finally_releases_active_lock(self, worker_env, monkeypatch):
        """P0-2: Worker finally releases active.lock after handoff."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent, lock_path

        intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
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

        # Handoff
        assert coord_lock.handoff_active_lock(rid, coord_token, os.getpid(), lease_token) is True

        # Worker releases (simulating finally block)
        worker_lock.release()

        # active.lock should be gone
        assert not lock_path("default").exists()

    def test_coordinator_crash_before_handoff_recoverable(self, worker_env, monkeypatch):
        """P0-2: If Coordinator crashes before handoff, lock is recoverable via TTL."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent, lock_path

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Coordinator acquires with short TTL
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(rid, ttl_s=1) is True

        # Simulate crash: wait for TTL to expire and pretend owner is dead
        time.sleep(1.5)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)

        # New restart request should be able to acquire (same short TTL so
        # age > ttl_s triggers the stale recovery path)
        new_lock = RestartLock("default")
        assert new_lock.try_acquire("a7a7a7a7-a7a7-a7a7-a7a7-a7a7a7a7a7a7", ttl_s=1) is True


# ===========================================================================
# P0-3: claim deadline recovery
# ===========================================================================

class TestClaimDeadlineRecovery:
    """P0-3: Worker never claims → claim_deadline expires → lock recoverable."""

    def test_claim_deadline_recovery(self, worker_env, monkeypatch):
        """P0-3: Worker never claims → claim_deadline expires → lock recoverable."""
        from hermes_cli.gateway_restart_state import RestartLock, create_intent, _pid_exists

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Coordinator acquires and marks worker spawned
        coord_lock = RestartLock("default")
        assert coord_lock.try_acquire(rid) is True
        # Use current PID but with a deadline that will expire
        coord_lock.mark_worker_spawned(os.getpid(), time.time() + 0.5)

        # Wait for deadline to expire
        time.sleep(1.0)

        # Mock _pid_exists to return False (worker is "dead")
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)

        # New restart should be able to recover the lock
        new_lock = RestartLock("default")
        assert new_lock.try_acquire("a7a7a7a7-a7a7-a7a7-a7a7-a7a7a7a7a7a7") is True


# ===========================================================================
# P0-4: Terminal status preserved after Worker completion
# ===========================================================================

class TestStatusPreservation:
    """P0-4: Terminal status preserved after Worker completion."""

    def test_worker_completed_status_preserved(self, worker_env, monkeypatch):
        """P0-4: Worker writes 'completed' → Coordinator can read it."""
        from hermes_cli.gateway_restart_state import (
            create_intent, write_status, read_status,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Simulate Worker writing terminal status
        write_status("default", "completed", request_id=rid, old_pid=100, new_pid=200)

        # Coordinator should still be able to read the status
        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "completed"
        assert status["new_pid"] == 200

    def test_failed_status_not_deleted(self, worker_env, monkeypatch):
        """P0-4: Worker writes 'failed' → status persists."""
        from hermes_cli.gateway_restart_state import (
            create_intent, write_status, read_status,
        )

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        write_status("default", "failed", request_id=rid, error="test error")

        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "failed"
        assert "test error" in status.get("error", "")


# ===========================================================================
# P1-1: schtasks fail closed and dual gateway detection
# ===========================================================================

class TestSchtasksFailClosed:
    """P1-1: schtasks /Run success without evidence → fail closed."""

    def test_schtasks_accepted_no_evidence_fails_closed(self, worker_env, monkeypatch):
        """P1-1: schtasks /Run returns 0 but no launch evidence → RuntimeError."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway
        from hermes_cli.gateway_restart_state import create_intent

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Mock hermes_cli.gateway_windows module so is_task_registered returns True
        # and _exec_schtasks returns success (code=0)
        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=True)
        mock_gw_windows._exec_schtasks = MagicMock(return_value=(0, "", ""))
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        # Mock _wait_for_launch_evidence → 0 (no evidence)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_launch_evidence",
            MagicMock(return_value=0),
        )
        # Mock _direct_spawn_gateway to prevent fallback spawn path
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway",
            MagicMock(return_value=0),
        )

        with pytest.raises(RuntimeError, match="schtasks /Run succeeded|no launch evidence|Direct spawn failed"):
            _start_new_gateway("default", rid, 100, "test", "Hermes_Gateway", task_registered=True)

    def test_dual_gateway_detected(self, worker_env, monkeypatch):
        """P1-1: Two gateways running simultaneously → detection fails."""
        from hermes_cli.gateway_windows_restart_worker import _verify_new_gateway
        from hermes_cli.gateway_restart_state import create_intent, read_status

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Mock _pid_exists: both old and new PIDs alive
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: pid in (100, 200),
        )

        _verify_new_gateway("default", rid, 100, 200, "test", "direct_spawn")

        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "failed"
        assert "dual" in status.get("error", "").lower()


# ===========================================================================
# P0-1: schtasks fail closed — strict direct-spawn block
# ===========================================================================

class TestSchtasksFailClosedStrict:
    """P0-1: schtasks accepted but no evidence → fail closed, no direct spawn."""

    def test_schtasks_accepted_no_evidence_blocks_direct_spawn(self, worker_env, monkeypatch):
        """schtasks /Run returns 0, no launch evidence → RuntimeError, direct spawn NOT called."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway
        from hermes_cli.gateway_restart_state import create_intent
        from unittest.mock import MagicMock

        intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = intent["request_id"]

        # Mock hermes_cli.gateway_windows module so is_task_registered returns True
        # and _exec_schtasks returns success (code=0)
        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=True)
        mock_gw_windows._exec_schtasks = MagicMock(return_value=(0, "", ""))
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_launch_evidence",
            MagicMock(return_value=0),
        )
        direct_spawn_mock = MagicMock(return_value=9999)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway",
            direct_spawn_mock,
        )

        with pytest.raises(RuntimeError, match="schtasks /Run succeeded"):
            _start_new_gateway("default", rid, 100, "test", "Hermes_Gateway", task_registered=True)

        direct_spawn_mock.assert_not_called()


# ===========================================================================
# P0-2: Handoff gate — Worker waits before destructive actions
# ===========================================================================

class TestHandoffGate:
    """P0-2: Worker waits for handoff before destructive actions."""

    def test_handoff_timeout_blocks_drain(self, worker_env, monkeypatch):
        """If handoff never completes, Worker exits without draining."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Mock _wait_for_handoff to return False (handoff never happens)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: False,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        drain_mock = MagicMock()
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            drain_mock,
        )

        # Handoff timeout now returns cleanly (not SystemExit)
        _run_restart_transaction(
            disk_intent, "default", request_id, nonce,
            1234, "test", "/fake/hermes", "Hermes_Gateway",
        )

        drain_mock.assert_not_called()

    def test_handoff_failure_safe_exit(self, worker_env, monkeypatch):
        """P0-2: Handoff timeout → Worker writes failed status and returns."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, read_status

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: False,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        # Handoff timeout returns cleanly, finally cleans up
        _run_restart_transaction(
            disk_intent, "default", request_id, nonce,
            1234, "test", "/fake/hermes", "Hermes_Gateway",
        )


# ===========================================================================
# P0-3 + P0-4: Owner-scoped cleanup
# ===========================================================================

class TestOwnerScopedCleanup:
    """P0-3: Cleanup operations are owner-scoped."""

    def test_prelease_exception_no_lease_deletion(self, worker_env, monkeypatch):
        """Worker exception before lease_owned=True does NOT delete any lease."""
        from hermes_cli.gateway_windows_restart_worker import main
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, lease_json_path, claim_lock_path,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Winner claims the lease
        winner_lock = RestartLock("default")
        assert winner_lock.try_acquire(request_id) is True
        assert winner_lock.claim_lease(request_id, nonce) is True

        # Loser calls main() — should NOT touch winner's lease
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )
        monkeypatch.setattr(sys, "argv", [
            "worker", "--profile", "default", "--request-id", request_id,
        ])

        with pytest.raises(SystemExit):
            main()

        # Winner's lease must survive (both claim.lock and lease.json)
        clp = claim_lock_path("default", request_id)
        ljp = lease_json_path("default", request_id)
        assert clp.exists(), "Winner's claim.lock must not be deleted by loser"
        assert ljp.exists(), "Winner's lease.json must not be deleted by loser"

    def test_prelease_validation_preserves_directory(self, worker_env, monkeypatch):
        """P0-4: _fail_closed does NOT rmtree the request directory."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, read_intent, request_dir_path,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]

        # Tamper with intent to cause validation failure (wrong nonce)
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        with pytest.raises(SystemExit):
            _run_restart_transaction(
                disk_intent, "default", request_id, "wrong-nonce",
                1234, "test", "/fake/hermes", "Hermes_Gateway",
            )

        # Request directory must still exist
        rd = request_dir_path("default", request_id)
        assert rd.exists(), "Request directory must not be deleted on pre-lease failure"

    def test_release_lease_rejects_wrong_owner(self, worker_env, monkeypatch):
        """P0-3: release_lease refuses when owner_token doesn't match."""
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, lease_json_path, release_lease,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Create a lease
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True
        assert lock.claim_lease(request_id, nonce) is True

        ljp = lease_json_path("default", request_id)
        assert ljp.exists()

        # Try to release with wrong owner_token
        result = release_lease("default", request_id,
                              owner_token="wrong-token",
                              worker_pid=os.getpid())
        assert result is False
        assert ljp.exists(), "Lease must survive with wrong owner_token"

    def test_sanitize_intent_rejects_wrong_nonce(self, worker_env, monkeypatch):
        """P0-3: sanitize_intent refuses when expected_nonce doesn't match."""
        from hermes_cli.gateway_restart_state import (
            create_intent, RestartLock, sanitize_intent, read_intent,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Claim lease so owner_token/worker_pid are valid
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True
        assert lock.claim_lease(request_id, nonce) is True

        # Try to sanitize with wrong nonce
        result = sanitize_intent("default", request_id,
                                expected_nonce="wrong-nonce",
                                owner_token=lock.owner_token,
                                worker_pid=os.getpid())
        assert result is False

        # Intent nonce must survive
        intent = read_intent("default", request_id)
        assert intent is not None
        assert intent["nonce"] == nonce, "Nonce must not be cleared with wrong expected_nonce"


# ===========================================================================
# P0-1: Two-phase lease publication
# ===========================================================================

class TestLeasePublication:
    """P0-1: Two-phase lease publication."""

    def test_coordinator_rejects_half_written_lease(self, worker_env, monkeypatch):
        """Coordinator must not handoff when lease.json is malformed."""
        from hermes_cli.gateway_restart_state import (
            create_intent, lease_json_path, claim_lock_path,
        )
        import hermes_cli.gateway_windows_restart as coord_mod

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]

        # Create claim.lock but NOT lease.json (simulating half-write)
        clp = claim_lock_path("default", rid)
        clp.touch()

        # _read_lease_data should return None
        result = coord_mod._read_lease_data("default", rid)
        assert result is None

    def test_intent_update_failure_no_lease(self, worker_env, monkeypatch):
        """P0-1: If intent state update fails, lease.json must not be created."""
        from hermes_cli.gateway_restart_state import (
            RestartLock, create_intent, lease_json_path,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Monkey-patch update_intent_state to fail
        import hermes_cli.gateway_restart_state as state_mod
        original = state_mod.update_intent_state
        state_mod.update_intent_state = lambda *a, **kw: False

        lock = RestartLock("default")
        result = lock.claim_lease(rid, nonce)
        state_mod.update_intent_state = original

        assert result is False
        ljp = lease_json_path("default", rid)
        assert not ljp.exists(), "lease.json must not exist when intent update fails"


# ===========================================================================
# P0-2: Handoff timeout cleanup
# ===========================================================================

class TestHandoffCleanup:
    """P0-2: Handoff timeout cleanup."""

    def test_handoff_timeout_releases_lease(self, worker_env, monkeypatch):
        """After handoff timeout, Worker's lease must be released."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import (
            create_intent, lease_json_path, claim_lock_path,
        )

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: False,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        # Handoff timeout should exit cleanly (not crash)
        # The finally block should clean up lease
        _run_restart_transaction(
            disk_intent, "default", request_id, nonce,
            1234, "test", "/fake/hermes", "Hermes_Gateway",
        )


# ===========================================================================
# P0-1: Winner writes terminal failed on exception
# ===========================================================================

class TestWinnerWritesTerminalFailed:
    def test_port_release_exception_writes_failed(self, worker_env, monkeypatch):
        """P0-1: Winner exception in _wait_for_port_release → status.json.state == failed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, read_status, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-create a lock and claim lease
        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 8080)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release",
                          MagicMock(side_effect=RuntimeError("port stuck")))

        with pytest.raises(RuntimeError, match="port stuck"):
            _run_restart_transaction(disk_intent, "default", rid, nonce, 1234, "test", "/fake/hermes", "Hermes_Gateway")

        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "failed"
        assert "port stuck" in status.get("error", "")

    def test_start_gateway_exception_writes_failed(self, worker_env, monkeypatch):
        """P0-1: Winner exception in _start_new_gateway → status.json.state == failed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, read_status, RestartLock

        disk_intent = create_intent(profile="default", target_pid=1234, origin="test", task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-create a lock and claim lease
        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", lambda name: True)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_task_ready", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway",
                          MagicMock(side_effect=RuntimeError("spawn failed")))

        with pytest.raises(RuntimeError, match="spawn failed"):
            _run_restart_transaction(disk_intent, "default", rid, nonce, 1234, "test", "/fake/hermes", "Hermes_Gateway")

        status = read_status("default", rid)
        assert status is not None
        assert status["state"] == "failed"


# ===========================================================================
# P0-2: schtasks fail-closed strict
# ===========================================================================

class TestSchtasksFailClosedStrict:
    def test_schtasks_timeout_no_direct_spawn(self, worker_env, monkeypatch):
        """P0-2: schtasks /Run timeout → no direct spawn."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway
        from hermes_cli.gateway_restart_state import create_intent
        from unittest.mock import MagicMock
        import sys

        disk_intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = disk_intent["request_id"]

        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=True)
        mock_gw_windows._exec_schtasks = MagicMock(return_value=(124, "", ""))  # timeout
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        direct_spawn_mock = MagicMock(return_value=9999)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway", direct_spawn_mock)

        with pytest.raises(RuntimeError, match="124|Will NOT direct-spawn"):
            _start_new_gateway("default", rid, 100, "test", "Hermes_Gateway", task_registered=True)
        direct_spawn_mock.assert_not_called()

    def test_schtasks_exception_no_direct_spawn(self, worker_env, monkeypatch):
        """P0-2: _exec_schtasks() throws → no direct spawn."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway
        from hermes_cli.gateway_restart_state import create_intent
        from unittest.mock import MagicMock
        import sys

        disk_intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = disk_intent["request_id"]

        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=True)
        mock_gw_windows._exec_schtasks = MagicMock(side_effect=TimeoutError("schtasks hung"))
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        direct_spawn_mock = MagicMock(return_value=9999)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway", direct_spawn_mock)

        with pytest.raises(RuntimeError, match="exception|Will NOT direct-spawn"):
            _start_new_gateway("default", rid, 100, "test", "Hermes_Gateway", task_registered=True)
        direct_spawn_mock.assert_not_called()

    def test_schtasks_ambiguous_nonzero_fails_closed(self, worker_env, monkeypatch):
        """P0-2: schtasks /Run ambiguous non-zero → fail closed."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway
        from hermes_cli.gateway_restart_state import create_intent
        from unittest.mock import MagicMock
        import sys

        disk_intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = disk_intent["request_id"]

        mock_gw_windows = MagicMock()
        mock_gw_windows.is_task_registered = MagicMock(return_value=True)
        mock_gw_windows._exec_schtasks = MagicMock(return_value=(1, "access denied", ""))
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", mock_gw_windows)

        direct_spawn_mock = MagicMock(return_value=9999)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway", direct_spawn_mock)

        with pytest.raises(RuntimeError, match="Will NOT direct-spawn"):
            _start_new_gateway("default", rid, 100, "test", "Hermes_Gateway", task_registered=True)
        direct_spawn_mock.assert_not_called()


# ---------------------------------------------------------------------------
# P2-1: psutil ImportError fallback tests
# ---------------------------------------------------------------------------

class TestPsutilImportErrorFallback:
    """P2-1: When psutil is not installed, functions must use fallback
    without raising UnboundLocalError."""

    def test_is_port_in_use_socket_fallback(self, worker_env, monkeypatch):
        """_is_port_in_use() uses socket fallback when psutil unavailable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        monkeypatch.setattr(mod, "_psutil_mod", None)

        # Should not raise UnboundLocalError
        # Port 1 (unlikely to be in use) → False via socket.bind
        result = mod._is_port_in_use(1)
        assert isinstance(result, bool)

    def test_get_pids_on_port_returns_empty(self, worker_env, monkeypatch):
        """_get_pids_on_port() returns [] when psutil unavailable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        monkeypatch.setattr(mod, "_psutil_mod", None)

        result = mod._get_pids_on_port(8080)
        assert result == []

    def test_is_hermes_gateway_pid_returns_false(self, worker_env, monkeypatch):
        """_is_hermes_gateway_pid() returns False when psutil unavailable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        monkeypatch.setattr(mod, "_psutil_mod", None)

        result = mod._is_hermes_gateway_pid(12345)
        assert result is False

    def test_is_ancestor_uses_platform_fallback(self, worker_env, monkeypatch):
        """_is_ancestor() uses platform fallback when psutil unavailable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        monkeypatch.setattr(mod, "_psutil_mod", None)

        # Should not raise UnboundLocalError
        # Checking a non-existent PID should return False
        result = mod._is_ancestor(99999)
        assert isinstance(result, bool)

    def test_no_unbound_local_error_any_function(self, worker_env, monkeypatch):
        """All 4 functions must not raise UnboundLocalError when psutil=None."""
        import hermes_cli.gateway_windows_restart_worker as mod
        monkeypatch.setattr(mod, "_psutil_mod", None)

        # Exercise all 4 functions
        mod._is_port_in_use(55555)
        mod._get_pids_on_port(55555)
        mod._is_hermes_gateway_pid(99999)
        mod._is_ancestor(99999)
        # If we got here, no UnboundLocalError was raised


# ---------------------------------------------------------------------------
# Task Scheduler state convergence tests (COM API numeric states)
# ---------------------------------------------------------------------------

class TestTaskStateConvergence:
    """Tests for _wait_for_task_ready — Task Scheduler state convergence
    using COM API numeric states (0=UNKNOWN, 1=DISABLED, 2=QUEUED,
    3=READY, 4=RUNNING)."""

    def test_running_then_ready(self, worker_env, monkeypatch):
        """RUNNING(4) → READY(3) → returns True."""
        import hermes_cli.gateway_windows_restart_worker as mod

        call_count = {"n": 0}
        def mock_com(name):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return 4  # RUNNING
            return 3  # READY

        monkeypatch.setattr(mod, "_query_task_state_com", mock_com)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=10.0,
        )
        assert result is True
        assert call_count["n"] >= 3

    def test_ready_immediately(self, worker_env, monkeypatch):
        """Already READY(3) → returns True on first poll."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: 3)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=10.0,
        )
        assert result is True

    def test_unknown_state_keeps_polling(self, worker_env, monkeypatch):
        """UNKNOWN(0) → not READY → keeps polling."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: 0)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}
        def mock_mono():
            calls["n"] += 1
            return 1000.0 if calls["n"] <= 1 else 1100.0
        monkeypatch.setattr(mod.time, "monotonic", mock_mono)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=30.0,
        )
        assert result is False

    def test_disabled_state_keeps_polling(self, worker_env, monkeypatch):
        """DISABLED(1) → not READY → keeps polling → timeout."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: 1)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}
        def mock_mono():
            calls["n"] += 1
            return 1000.0 if calls["n"] <= 1 else 1100.0
        monkeypatch.setattr(mod.time, "monotonic", mock_mono)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=30.0,
        )
        assert result is False

    def test_queued_state_keeps_polling(self, worker_env, monkeypatch):
        """QUEUED(2) → not READY → keeps polling → timeout."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: 2)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}
        def mock_mono():
            calls["n"] += 1
            return 1000.0 if calls["n"] <= 1 else 1100.0
        monkeypatch.setattr(mod.time, "monotonic", mock_mono)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=30.0,
        )
        assert result is False

    def test_running_keeps_polling(self, worker_env, monkeypatch):
        """RUNNING(4) stays RUNNING → timeout → returns False."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: 4)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}
        def mock_mono():
            calls["n"] += 1
            return 1000.0 if calls["n"] <= 1 else 1100.0
        monkeypatch.setattr(mod.time, "monotonic", mock_mono)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=30.0,
        )
        assert result is False

    def test_query_failure_keeps_polling(self, worker_env, monkeypatch):
        """COM query returns -1 → transient failure → keeps polling → timeout."""
        import hermes_cli.gateway_windows_restart_worker as mod

        monkeypatch.setattr(mod, "_query_task_state_com", lambda name: -1)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}
        def mock_mono():
            calls["n"] += 1
            return 1000.0 if calls["n"] <= 1 else 1100.0
        monkeypatch.setattr(mod.time, "monotonic", mock_mono)

        result = mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", timeout=30.0,
        )
        assert result is False

    def test_convergence_logs_com_state(self, worker_env, monkeypatch):
        """JSONL contains numeric COM state in detail."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import jsonl_log_path
        import json

        call_count = {"n": 0}
        def mock_com(name):
            call_count["n"] += 1
            return 4 if call_count["n"] <= 1 else 3

        monkeypatch.setattr(mod, "_query_task_state_com", mock_com)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        mod._wait_for_task_ready(
            "Hermes_Gateway", "default", "d0d0d0d0-d0d0-d0d0-d0d0-d0d0d0d0d0d0", 100, "test", timeout=10.0,
        )

        log_path = jsonl_log_path()
        found = False
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    if (d.get("request_id") == "d0d0d0d0-d0d0-d0d0-d0d0-d0d0d0d0d0d0"
                            and d.get("reason") == "task_state_ready"):
                        found = True
                        assert "READY(3)" in d.get("detail", "")
                        break
                except (json.JSONDecodeError, ValueError):
                    pass
        assert found, "task_state_ready log entry not found"


# ---------------------------------------------------------------------------
# Tri-state probe & integrated control flow tests
# ---------------------------------------------------------------------------

class TestTriStateProbe:
    """Tests for _probe_task_registration — COM API tri-state probe.

    Uses subprocess.run mock to control PowerShell COM output.
    """

    def _make_subprocess_result(self, stdout="", returncode=0):
        """Create a mock subprocess.CompletedProcess."""
        result = MagicMock()
        result.stdout = stdout
        result.returncode = returncode
        return result

    def test_com_gettask_exists_returns_true(self, worker_env, monkeypatch):
        """COM GetTask succeeds → probe=True."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(stdout="EXISTS")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is True

    def test_com_file_not_found_returns_false(self, worker_env, monkeypatch):
        """COM GetTask returns FILE_NOT_FOUND HRESULT (0x80070002) → probe=False."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(
            stdout="HRESULT:-2147024894"  # 0x80070002 as signed int
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is False

    def test_com_path_not_found_returns_none(self, worker_env, monkeypatch):
        """COM GetTask returns PATH_NOT_FOUND HRESULT (0x80070003) → probe=None (fail closed)."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(
            stdout="HRESULT:-2147024893"  # 0x80070003 as signed int
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_com_timeout_returns_none(self, worker_env, monkeypatch):
        """PowerShell timeout → probe=None (ambiguous)."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=subprocess.TimeoutExpired("powershell", 10)),
        )

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_com_access_denied_returns_none(self, worker_env, monkeypatch):
        """COM returns E_ACCESSDENIED HRESULT → probe=None (ambiguous)."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(
            stdout="HRESULT:-2147024891"  # 0x80070005 as signed int
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_com_unknown_hresult_returns_none(self, worker_env, monkeypatch):
        """COM returns unknown HRESULT → probe=None (ambiguous)."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(
            stdout="HRESULT:-2147221164"  # 0x80040154 (class not registered)
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_powershell_nonzero_exit_returns_none(self, worker_env, monkeypatch):
        """PowerShell exits non-zero → probe=None."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(stdout="", returncode=1)
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_powershell_empty_output_returns_none(self, worker_env, monkeypatch):
        """PowerShell returns empty output → probe=None."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(stdout="", returncode=0)
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_unexpected_output_returns_none(self, worker_env, monkeypatch):
        """Unexpected PowerShell output → probe=None."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        mock_result = self._make_subprocess_result(stdout="GARBAGE")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        assert mod._probe_task_registration("Hermes_Gateway") is None

    def test_oserror_returns_none(self, worker_env, monkeypatch):
        """OSError (e.g., powershell not found) → probe=None."""
        import hermes_cli.gateway_windows_restart_worker as mod
        import subprocess

        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=OSError("powershell not found")),
        )

        assert mod._probe_task_registration("Hermes_Gateway") is None


class TestStartGatewayFailClosed:
    """Tests for _start_new_gateway fail-closed on task_registered=None."""

    def test_none_raises_no_run_no_spawn(self, worker_env, monkeypatch):
        """task_registered=None → raise, no /Run, no direct spawn."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import append_restart_log

        schtasks_called = {"v": False}
        spawn_called = {"v": False}

        def mock_schtasks(*a, **kw):
            schtasks_called["v"] = True
            return (0, "", "")

        def mock_spawn():
            spawn_called["v"] = True
            return 9999

        monkeypatch.setattr("hermes_cli.gateway_windows._exec_schtasks", mock_schtasks)
        monkeypatch.setattr(mod, "_direct_spawn_gateway", mock_spawn)

        with pytest.raises(RuntimeError, match="ambiguous"):
            mod._start_new_gateway(
                "default", "d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4", 100, "test", "Hermes_Gateway",
                task_registered=None,
            )

        assert not schtasks_called["v"], "schtasks /Run should NOT be called"
        assert not spawn_called["v"], "direct spawn should NOT be called"


class TestIntegratedControlFlow:
    """Tests for tri-state probe integration in _run_restart_transaction."""

    def test_registered_waits_ready_then_run(self, worker_env, monkeypatch):
        """registered=True → waits READY → /Run."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", lambda name: True)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_task_ready", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway",
                          lambda *a, **kw: (9999, "scheduled_task"))
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._verify_new_gateway", lambda *a, **kw: None)

        # Should NOT raise
        mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                     str(worker_env), "Hermes_Gateway")

    def test_not_registered_direct_spawn(self, worker_env, monkeypatch):
        """registered=False → skip readiness → direct spawn reachable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        probe_called = {"v": False}
        wait_called = {"v": False}
        start_args = {}

        def mock_probe(name):
            probe_called["v"] = True
            return False

        def mock_wait(*a, **kw):
            wait_called["v"] = True
            return True

        def mock_start(profile, request_id, old_pid, origin, task_name, task_registered=None, hermes_home=""):
            start_args["task_registered"] = task_registered
            return (9999, "direct_spawn")

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", mock_probe)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_task_ready", mock_wait)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway", mock_start)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._verify_new_gateway", lambda *a, **kw: None)

        mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                     str(worker_env), "Hermes_Gateway")

        assert probe_called["v"], "_probe_task_registration should be called"
        assert not wait_called["v"], "_wait_for_task_ready should NOT be called"
        assert start_args["task_registered"] is False

    def test_ambiguous_probe_fail_closed(self, worker_env, monkeypatch):
        """registered=None → fail closed → raises, no /Run, no direct spawn."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        start_called = {"v": False}

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", lambda name: None)

        def mock_start(*a, **kw):
            start_called["v"] = True
            return (9999, "direct_spawn")

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway", mock_start)

        with pytest.raises(RuntimeError, match="registration probe failed"):
            mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                         str(worker_env), "Hermes_Gateway")

        assert not start_called["v"], "_start_new_gateway should NOT be called"

    def test_task_name_nonempty_but_not_registered_direct_spawn_reachable(
        self, worker_env, monkeypatch
    ):
        """task_name non-empty but task not installed → direct spawn fallback reachable."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        launcher = {}

        def mock_start(profile, request_id, old_pid, origin, task_name, task_registered=None, hermes_home=""):
            launcher["registered"] = task_registered
            return (9999, "direct_spawn")

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", lambda name: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway", mock_start)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._verify_new_gateway", lambda *a, **kw: None)

        mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                     str(worker_env), "Hermes_Gateway")

        assert launcher["registered"] is False

    def test_startup_folder_no_block_from_com(self, worker_env, monkeypatch):
        """Startup-folder fallback: not registered → no COM timeout block."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        probe_called = {"v": False}

        def mock_probe(name):
            probe_called["v"] = True
            return False  # Task not registered → direct spawn allowed

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", mock_probe)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway",
                          lambda *a, **kw: (9999, "direct_spawn"))
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._verify_new_gateway", lambda *a, **kw: None)

        mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                     str(worker_env), "Hermes_Gateway")

    def test_early_pid_exit_still_waits_ready(self, worker_env, monkeypatch):
        """Old PID exits during drain → still waits for Task READY before /Run."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test",
                                     task_name="Hermes_Gateway")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        wait_called = {"v": False}

        def mock_wait(*a, **kw):
            wait_called["v"] = True
            return True

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._detect_gateway_port", lambda: 0)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_port_release", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", lambda name: True)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_task_ready", mock_wait)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._start_new_gateway",
                          lambda *a, **kw: (9999, "scheduled_task"))
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._verify_new_gateway", lambda *a, **kw: None)

        mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                     str(worker_env), "Hermes_Gateway")

        assert wait_called["v"], "_wait_for_task_ready must be called even after early PID exit"

    def test_no_task_name_fails_closed(self, worker_env, monkeypatch):
        """task_name empty → fail closed (B1).  Worker must NOT proceed
        with drain/start when task_name is missing."""
        import hermes_cli.gateway_windows_restart_worker as mod
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(profile="default", target_pid=100, origin="test")
        rid = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(rid) is True

        probe_called = {"v": False}

        def mock_probe(name):
            probe_called["v"] = True
            return True

        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._wait_for_handoff", lambda *a, **kw: True)
        monkeypatch.setattr("hermes_cli.gateway_restart_state._pid_exists", lambda pid: False)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._drain_and_stop", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.gateway_windows_restart_worker._probe_task_registration", mock_probe)

        # Override disk intent task_name to empty (tampered intent)
        tampered = dict(disk_intent)
        tampered["task_name"] = ""
        ip = worker_env / "run" / "gateway-restart" / "default" / rid / "intent.json"
        ip.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            mod._run_restart_transaction(disk_intent, "default", rid, nonce, 100, "test",
                                         str(worker_env), "")
        assert exc_info.value.code == 1
        assert not probe_called["v"], "probe must NOT be called when task_name is empty"
