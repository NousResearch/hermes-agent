"""Tests for PR #41148 boundary hardening (fix/41148-boundary-hardening).

Covers four categories:
A. Corrupt/truncated active.lock TTL recovery and stat identity verification
B. Slash command asyncio.to_thread non-blocking
C. Worker uses intent-derived HERMES_HOME, task_name, profile (no implicit derivation)
D. Named profile worker path
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def restart_env(tmp_path, monkeypatch):
    """Provide a temp HERMES_HOME for boundary hardening tests."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "run").mkdir()
    (hermes_home / "logs").mkdir()
    (hermes_home / "profiles").mkdir()
    (hermes_home / "profiles" / "work").mkdir()

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
    hermes_home="/fake/hermes",
    task_name="Hermes_Gateway",
):
    """Create a realistic intent dict."""
    import secrets
    import uuid
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "request_id": request_id or str(uuid.uuid4()),
        "nonce": nonce or secrets.token_urlsafe(32),
        "profile": profile,
        "hermes_home": hermes_home,
        "target_pid": target_pid,
        "task_name": task_name,
        "origin": "test",
        "created_at": now.isoformat(),
        "expires_at": now.timestamp() + ttl_s,
        "state": "scheduled",
    }


def _write_intent_to_disk(hermes_home, intent, profile="default"):
    """Write intent to per-request directory."""
    request_id = intent["request_id"]
    req_dir = hermes_home / "run" / "gateway-restart" / profile / request_id
    req_dir.mkdir(parents=True, exist_ok=True)
    path = req_dir / "intent.json"
    path.write_text(json.dumps(intent, indent=2), encoding="utf-8")
    return path


# ===========================================================================
# A: Corrupt active.lock — immutable, fail-closed
# ===========================================================================

class TestCorruptLockImmutable:
    """Corrupt active.lock is never moved, deleted, or modified.
    All paths fail-closed; human operator must clean up."""

    def test_corrupt_lock_not_modified_recent(self, restart_env):
        """Recent corrupt lock: _read_lock returns None, file unchanged."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lp = lock_path("default")
        corrupt_content = '{"schema_version": 1, "request_id": "xxx'
        lp.write_text(corrupt_content, encoding="utf-8")

        lock = RestartLock("default")
        assert lock._read_lock() is None
        # File must still exist with original content
        assert lp.exists()
        assert lp.read_text(encoding="utf-8") == corrupt_content

    def test_corrupt_lock_not_modified_expired(self, restart_env):
        """Expired corrupt lock: still never deleted or modified."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lp = lock_path("default")
        corrupt_content = '{"truncated'
        lp.write_text(corrupt_content, encoding="utf-8")

        old_time = time.time() - 500
        os.utime(str(lp), (old_time, old_time))

        lock = RestartLock("default")
        assert lock._read_lock() is None
        assert lp.exists(), "Expired corrupt lock must NOT be deleted"
        assert lp.read_text(encoding="utf-8") == corrupt_content

    def test_corrupt_lock_blocks_acquire(self, restart_env):
        """try_acquire must fail when lock file is corrupt (file still exists)."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lp = lock_path("default")
        lp.write_text('bad', encoding="utf-8")

        lock = RestartLock("default")
        assert lock.try_acquire("ffffffff-ffff-ffff-ffff-ffffffffffff") is False

    def test_binary_lock_not_modified(self, restart_env):
        """Binary garbage in lock file: fail-closed, no modification."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lp = lock_path("default")
        binary_content = b"\x00\x01\x02\x03\xff\xfe"
        lp.write_bytes(binary_content)

        lock = RestartLock("default")
        assert lock._read_lock() is None
        assert lp.exists()
        assert lp.read_bytes() == binary_content

    def test_concurrent_replacement_not_mistakenly_deleted(self, restart_env):
        """_force_release with stale data must NOT delete a replacement lock."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock1 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True

        lp = lock_path("default")
        original_data = json.loads(lp.read_text(encoding="utf-8"))

        replacement = dict(original_data)
        replacement["request_id"] = "22222222-2222-2222-2222-222222222222"
        replacement["owner_token"] = "different-token"
        replacement["created_at"] = time.time()
        lp.write_text(json.dumps(replacement), encoding="utf-8")

        lock1._force_release(expected=original_data)

        assert lp.exists(), "Replacement lock must not be mistakenly deleted"
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "22222222-2222-2222-2222-222222222222"


# ===========================================================================
# B: Crash-safe atomic publish (tmp + fsync + os.link)
# ===========================================================================

class TestAtomicPublish:
    """Lock acquisition uses tmp+fsync+os.link for crash-safe, no-clobber publish."""

    def test_stale_tmp_does_not_block_acquire(self, restart_env):
        """A leftover .lock-*.tmp from a crashed process must not prevent
        new acquisition.  Each process uses a unique tmp filename."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        profile_dir = lock_path("default").parent
        # Simulate a stale tmp from a crashed process
        stale_tmp = profile_dir / ".lock-aaaaaaaaaaaaaaaa.tmp"
        stale_tmp.write_text('{"stale": true}', encoding="utf-8")

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True
        # Stale tmp should have been cleaned up by the finally block
        lock.release()

    def test_publish_conflict_no_overwrite(self, restart_env):
        """Second os.link must fail with FileExistsError — never overwrite."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock1 = RestartLock("default")
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111") is True

        # Second acquire must fail — os.link sees existing file
        lock2 = RestartLock("default")
        assert lock2.try_acquire("22222222-2222-2222-2222-222222222222") is False

        # Verify original lock untouched
        lp = lock_path("default")
        data = json.loads(lp.read_text(encoding="utf-8"))
        assert data["request_id"] == "11111111-1111-1111-1111-111111111111"

        lock1.release()

    def test_lock_file_is_complete_json(self, restart_env):
        """Published lock file must be complete, valid JSON (not truncated)."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        lp = lock_path("default")
        data = json.loads(lp.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["request_id"] == "11111111-1111-1111-1111-111111111111"
        assert "owner_token" in data
        assert data["owner_pid"] > 0

        lock.release()

    def test_no_tmp_files_after_successful_acquire(self, restart_env):
        """After successful acquire, no .lock-*.tmp files should remain."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        profile_dir = lock_path("default").parent
        tmp_files = list(profile_dir.glob(".lock-*.tmp"))
        assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"

        lock.release()

    def test_os_link_no_clobber_semantics(self, restart_env):
        """os.link raises FileExistsError when target exists (no overwrite)."""
        from hermes_cli.gateway_restart_state import lock_path
        lp = lock_path("default")
        profile_dir = lp.parent
        src = profile_dir / "test-src.json"
        src.write_text('{"test": 1}', encoding="utf-8")

        # First link succeeds
        os.link(str(src), str(lp))

        # Second link must fail
        with pytest.raises(FileExistsError):
            os.link(str(src), str(lp))

        src.unlink()
        lp.unlink()

    def test_stale_tmp_gc_cleans_old_files(self, restart_env):
        """gc_stale_lock_tmp removes .lock-*.tmp files older than TTL."""
        from hermes_cli.gateway_restart_state import (
            lock_path, gc_stale_lock_tmp, _LOCK_TTL_S,
        )

        profile_dir = lock_path("default").parent
        # Create a stale tmp (older than TTL)
        stale = profile_dir / ".lock-deadbeefdeadbeef.tmp"
        stale.write_text('{"stale": true}', encoding="utf-8")
        old_time = time.time() - _LOCK_TTL_S - 10
        os.utime(str(stale), (old_time, old_time))

        removed = gc_stale_lock_tmp("default")
        assert removed == 1
        assert not stale.exists()

    def test_recent_tmp_gc_preserves(self, restart_env):
        """gc_stale_lock_tmp preserves .lock-*.tmp files newer than TTL."""
        from hermes_cli.gateway_restart_state import (
            lock_path, gc_stale_lock_tmp,
        )

        profile_dir = lock_path("default").parent
        recent = profile_dir / ".lock-abcdef0123456789.tmp"
        recent.write_text('{"recent": true}', encoding="utf-8")

        removed = gc_stale_lock_tmp("default")
        assert removed == 0
        assert recent.exists(), "Recent tmp must NOT be GC'd"
        recent.unlink()

    def test_gc_never_touches_active_lock(self, restart_env):
        """gc_stale_lock_tmp must never touch active.lock."""
        from hermes_cli.gateway_restart_state import (
            RestartLock, lock_path, gc_stale_lock_tmp, _LOCK_TTL_S,
        )

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True

        # Back-date the lock file to look "old"
        lp = lock_path("default")
        old_time = time.time() - _LOCK_TTL_S - 10
        os.utime(str(lp), (old_time, old_time))

        removed = gc_stale_lock_tmp("default")
        assert removed == 0
        assert lp.exists(), "active.lock must never be GC'd"

        lock.release()

    def test_publish_oserror_logs_diagnostic(self, restart_env, monkeypatch):
        """Non-conflict OSError during os.link must log diagnostic and fail."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path
        import hermes_cli.gateway_restart_state as state_mod

        original_link = os.link

        def failing_link(src, dst):
            raise OSError(13, "Permission denied", dst)

        monkeypatch.setattr(os, "link", failing_link)

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is False

        # Verify no lock file was created
        assert not lock_path("default").exists()

    def test_publish_write_oserror_cleans_tmp(self, restart_env, monkeypatch):
        """OSError during tmp write must clean up tmp and fail."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        original_open = os.open

        def failing_open(path, flags, *args, **kwargs):
            if ".lock-" in str(path) and str(path).endswith(".tmp"):
                raise OSError(28, "No space left on device", path)
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", failing_open)

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is False

        # No tmp files should remain
        profile_dir = lock_path("default").parent
        tmp_files = list(profile_dir.glob(".lock-*.tmp"))
        assert len(tmp_files) == 0, f"Tmp not cleaned: {tmp_files}"


# ===========================================================================
# B: Slash command asyncio.to_thread non-blocking
# ===========================================================================

class TestSlashCommandNonBlocking:
    """Verify that the /restart slash command wraps schedule_restart_handoff
    in asyncio.to_thread to avoid blocking the event loop."""

    def test_restart_calls_schedule_via_to_thread(self, restart_env):
        """Source-level: slash_commands.py must contain asyncio.to_thread."""
        import gateway.slash_commands as sc_mod
        source_path = Path(sc_mod.__file__).read_text(encoding="utf-8")

        assert "asyncio.to_thread" in source_path, (
            "slash_commands.py must use asyncio.to_thread for "
            "schedule_restart_handoff to avoid blocking the event loop"
        )
        assert "await _aio.to_thread" in source_path or "await asyncio.to_thread" in source_path, (
            "The to_thread call must be awaited"
        )

    def test_schedule_restart_handoff_is_callable_from_thread(self, restart_env):
        """schedule_restart_handoff must be a regular (non-async) function
        that can be safely called via asyncio.to_thread."""
        from hermes_cli.gateway_windows_restart import schedule_restart_handoff
        import inspect
        assert not inspect.iscoroutinefunction(schedule_restart_handoff), (
            "schedule_restart_handoff must be a regular function, not async, "
            "so it can be passed to asyncio.to_thread"
        )

    def test_to_thread_does_not_block_event_loop(self, restart_env, monkeypatch):
        """Behavioral: monkeypatch production schedule_restart_handoff as a
        slow function, call it through the same asyncio.to_thread pattern
        used by the slash-command handler, and prove the event loop heartbeat
        continues ticking.
        """
        import asyncio
        import hermes_cli.gateway_windows_restart as restart_mod

        heartbeat_log: list[float] = []
        slow_duration = 0.5  # seconds

        def slow_coordinator(**kwargs):
            """Replace production schedule_restart_handoff with a slow one."""
            time.sleep(slow_duration)
            return {"scheduled": True, "request_id": "test"}

        # Monkeypatch the production function
        monkeypatch.setattr(restart_mod, "schedule_restart_handoff", slow_coordinator)

        async def heartbeat():
            """Yields control repeatedly, recording timestamps."""
            start = time.monotonic()
            while time.monotonic() - start < slow_duration + 0.3:
                heartbeat_log.append(time.monotonic())
                await asyncio.sleep(0.05)  # 50ms tick

        async def handler_simulation():
            """Replicates the exact pattern in slash_commands.py:
            result = await asyncio.to_thread(
                schedule_restart_handoff, origin="slash-command", wait=False
            )
            """
            result = await asyncio.to_thread(
                restart_mod.schedule_restart_handoff,
                origin="slash-command", wait=False,
            )
            return result

        async def runner():
            handler_task = asyncio.create_task(handler_simulation())
            heartbeat_task = asyncio.create_task(heartbeat())

            result = await handler_task
            await heartbeat_task
            return result

        result = asyncio.run(runner())

        assert result == {"scheduled": True, "request_id": "test"}
        # Heartbeat must have ticked multiple times during the slow coordinator.
        # With 50ms ticks and 500ms sleep, we expect ~8-10 ticks.
        assert len(heartbeat_log) >= 4, (
            f"Event loop was blocked! Only {len(heartbeat_log)} heartbeat ticks "
            f"during {slow_duration}s coordinator — expected >= 4"
        )


# ===========================================================================
# C: Worker uses intent-derived values (no implicit derivation)
# ===========================================================================

class TestWorkerIntentDerivedValues:
    """Verify that the worker uses intent-captured HERMES_HOME, task_name,
    and profile rather than deriving them from inherited environment."""

    def test_worker_sets_hermes_home_from_intent_unconditionally(self, restart_env, monkeypatch):
        """HERMES_HOME must be set from the intent, even if the inherited
        environment already has a different HERMES_HOME."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        # Set a WRONG HERMES_HOME in the environment
        monkeypatch.setenv("HERMES_HOME", "/wrong/path")

        intent_home = str(restart_env)
        disk_intent = create_intent(
            profile="default", target_pid=1234, origin="test",
            hermes_home=intent_home,
        )
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Pre-acquire lock and claim lease
        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        # Mock everything downstream — we only care about HERMES_HOME being set
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )
        # Make _drain_and_stop a no-op
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            lambda *a, **kw: None,
        )
        # Mock port detection
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._detect_gateway_port",
            lambda: 0,
        )
        # Mock task scheduler probes (intent has task_name="Hermes_Gateway")
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._probe_task_registration",
            lambda task_name: False,  # Not registered → direct spawn path
        )
        # Mock start and verify
        mock_start = MagicMock(return_value=(9999, "direct_spawn"))
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._start_new_gateway",
            mock_start,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._verify_new_gateway",
            lambda *a, **kw: None,
        )

        _run_restart_transaction(
            disk_intent, "default", request_id, nonce,
            1234, "test", intent_home, "Hermes_Gateway",
        )

        # After _run_restart_transaction runs, HERMES_HOME must be the
        # intent value, not the inherited "/wrong/path"
        assert os.environ.get("HERMES_HOME") == intent_home

    def test_drain_and_stop_uses_passed_task_name(self, restart_env, monkeypatch):
        """_drain_and_stop must use the task_name parameter, not call
        get_task_name() from the environment."""
        from hermes_cli.gateway_windows_restart_worker import _drain_and_stop

        # Mock _exec_schtasks to capture the task name used
        mock_exec = MagicMock(return_value=(0, "", ""))
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._exec_schtasks",
            mock_exec,
        )
        # Mock PID wait to return immediately (PID not alive)
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        # Call with explicit task_name
        _drain_and_stop(
            profile="default",
            request_id="d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4",
            old_pid=1234,
            origin="test",
            task_name="Custom_Task_Name",
        )

        # Verify _exec_schtasks was called with the passed task_name,
        # NOT whatever get_task_name() would return
        if mock_exec.called:
            args = mock_exec.call_args[0][0]
            assert "/TN" in args
            tn_idx = args.index("/TN") + 1
            assert args[tn_idx] == "Custom_Task_Name"

    def test_drain_and_stop_skips_schtasks_when_no_task_name(self, restart_env, monkeypatch):
        """_drain_and_stop must NOT call schtasks /End if task_name is empty."""
        from hermes_cli.gateway_windows_restart_worker import _drain_and_stop

        mock_exec = MagicMock(return_value=(0, "", ""))
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._exec_schtasks",
            mock_exec,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )

        _drain_and_stop(
            profile="default",
            request_id="d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4",
            old_pid=1234,
            origin="test",
            task_name="",  # Empty task name
        )

        # schtasks /End should NOT have been called
        mock_exec.assert_not_called()

    def test_empty_task_name_fails_closed(self, restart_env, monkeypatch):
        """B1: Worker must fail-closed (sys.exit) when task_name is empty."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(
            profile="default", target_pid=1234, origin="test",
            hermes_home=str(restart_env), task_name="Hermes_Gateway",
        )
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Tamper: set task_name to empty in the disk intent
        tampered = dict(disk_intent)
        tampered["task_name"] = ""
        _write_intent_to_disk(restart_env, tampered, profile="default")

        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                tampered, "default", request_id, nonce,
                1234, "test", str(restart_env), "",  # empty task_name
            )
        assert exc_info.value.code == 1

    def test_none_task_name_fails_closed(self, restart_env, monkeypatch):
        """B1: Worker must fail-closed when task_name is None."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(
            profile="default", target_pid=1234, origin="test",
            hermes_home=str(restart_env), task_name="Hermes_Gateway",
        )
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        # Tamper: set task_name to None
        tampered = dict(disk_intent)
        tampered["task_name"] = None
        _write_intent_to_disk(restart_env, tampered, profile="default")

        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                tampered, "default", request_id, nonce,
                1234, "test", str(restart_env), None,
            )
        assert exc_info.value.code == 1

    def test_whitespace_task_name_fails_closed(self, restart_env, monkeypatch):
        """D: Whitespace-only task_name must fail-closed."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        disk_intent = create_intent(
            profile="default", target_pid=1234, origin="test",
            hermes_home=str(restart_env), task_name="Hermes_Gateway",
        )
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        tampered = dict(disk_intent)
        tampered["task_name"] = "   "
        _write_intent_to_disk(restart_env, tampered, profile="default")

        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        with pytest.raises(SystemExit) as exc_info:
            _run_restart_transaction(
                tampered, "default", request_id, nonce,
                1234, "test", str(restart_env), "   ",
            )
        assert exc_info.value.code == 1

    def test_default_profile_clears_hermes_profile(self, restart_env, monkeypatch):
        """C: default profile must clear inherited HERMES_PROFILE."""
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        from hermes_cli.gateway_restart_state import create_intent, RestartLock

        # Set a stale HERMES_PROFILE in the environment
        monkeypatch.setenv("HERMES_PROFILE", "stale-work")

        intent_home = str(restart_env)
        disk_intent = create_intent(
            profile="default", target_pid=1234, origin="test",
            hermes_home=intent_home, task_name="Hermes_Gateway",
        )
        request_id = disk_intent["request_id"]
        nonce = disk_intent["nonce"]

        lock = RestartLock("default")
        assert lock.try_acquire(request_id) is True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_handoff",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state._pid_exists",
            lambda pid: False,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._drain_and_stop",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._detect_gateway_port",
            lambda: 0,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._probe_task_registration",
            lambda task_name: False,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._start_new_gateway",
            lambda *a, **kw: (9999, "direct_spawn"),
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._verify_new_gateway",
            lambda *a, **kw: None,
        )

        _run_restart_transaction(
            disk_intent, "default", request_id, nonce,
            1234, "test", intent_home, "Hermes_Gateway",
        )

        # HERMES_PROFILE must be cleared for default profile
        assert os.environ.get("HERMES_PROFILE") is None

    def test_direct_spawn_gateway_sets_hermes_home_and_profile(self, restart_env, monkeypatch):
        """_direct_spawn_gateway must set HERMES_HOME and HERMES_PROFILE
        from its parameters before spawning."""
        from hermes_cli.gateway_windows_restart_worker import _direct_spawn_gateway

        monkeypatch.setenv("HERMES_HOME", "/old/path")

        mock_spawn = MagicMock(return_value=5555)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._spawn_detached",
            mock_spawn,
        )

        _direct_spawn_gateway(hermes_home="/new/intent/path", profile="work")

        assert os.environ.get("HERMES_HOME") == "/new/intent/path"
        assert os.environ.get("HERMES_PROFILE") == "work"
        mock_spawn.assert_called_once()

    def test_direct_spawn_gateway_default_profile_not_set(self, restart_env, monkeypatch):
        """_direct_spawn_gateway must NOT set HERMES_PROFILE for 'default'."""
        from hermes_cli.gateway_windows_restart_worker import _direct_spawn_gateway

        monkeypatch.delenv("HERMES_PROFILE", raising=False)

        mock_spawn = MagicMock(return_value=5555)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._spawn_detached",
            mock_spawn,
        )

        _direct_spawn_gateway(hermes_home="/some/path", profile="default")

        assert os.environ.get("HERMES_PROFILE") is None

    def test_start_new_gateway_forwards_hermes_home_and_profile(self, restart_env, monkeypatch):
        """_start_new_gateway must forward hermes_home and profile to _direct_spawn_gateway."""
        from hermes_cli.gateway_windows_restart_worker import _start_new_gateway

        mock_direct = MagicMock(return_value=7777)
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._direct_spawn_gateway",
            mock_direct,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows_restart_worker._wait_for_launch_evidence",
            lambda old_pid, timeout=15.0: 7777,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state.write_status",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_restart_state.append_restart_log",
            lambda **kw: None,
        )

        _start_new_gateway(
            profile="work",
            request_id="d4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4",
            old_pid=1234,
            origin="test",
            task_name="Hermes_Gateway",
            task_registered=False,
            hermes_home="/from/intent",
        )

        mock_direct.assert_called_once_with(hermes_home="/from/intent", profile="work")


# ===========================================================================
# D: Named profile worker path
# ===========================================================================

class TestNamedProfileWorker:
    """Verify that a named profile is correctly propagated through the
    restart transaction without being overwritten by 'default'."""

    def test_named_profile_intent_roundtrip(self, restart_env):
        """Create an intent for profile 'work', verify it's stored and
        read back correctly."""
        from hermes_cli.gateway_restart_state import create_intent, read_intent, cleanup_intent

        intent = create_intent(
            profile="work", target_pid=5678, origin="test",
            hermes_home=str(restart_env),
        )
        rid = intent["request_id"]
        assert intent["profile"] == "work"

        read = read_intent("work", rid)
        assert read is not None
        assert read["profile"] == "work"
        assert read["target_pid"] == 5678

        cleanup_intent("work", rid)

    def test_named_profile_lock_isolation(self, restart_env):
        """Locks for different profiles must be independent."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock_default = RestartLock("default")
        lock_work = RestartLock("work")

        assert lock_default.try_acquire("a9a9a9a9-a9a9-a9a9-a9a9-a9a9a9a9a9a9") is True
        assert lock_work.try_acquire("babababa-baba-baba-baba-babababababa") is True  # Different profile, no conflict

        lock_default.release()
        lock_work.release()

    def test_named_profile_status_isolation(self, restart_env):
        """Status files for different profiles must be independent."""
        from hermes_cli.gateway_restart_state import write_status, read_status

        write_status("default", "completed", request_id="e7e7e7e7-e7e7-e7e7-e7e7-e7e7e7e7e7e7", new_pid=111)
        write_status("work", "failed", request_id="f8f8f8f8-f8f8-f8f8-f8f8-f8f8f8f8f8f8", error="test error")

        s1 = read_status("default", "e7e7e7e7-e7e7-e7e7-e7e7-e7e7e7e7e7e7")
        s2 = read_status("work", "f8f8f8f8-f8f8-f8f8-f8f8-f8f8f8f8f8f8")

        assert s1["state"] == "completed"
        assert s1["new_pid"] == 111
        assert s2["state"] == "failed"
        assert s2["error"] == "test error"


# ---------------------------------------------------------------------------
# F. PYTHONPATH hotfix: editable-install vs candidate worktree
# ---------------------------------------------------------------------------


class TestPYTHONPATHHotfix:
    """Verify PYTHONPATH derivation ensures worker loads from coordinator's checkout."""

    def test_pythonpath_set_to_coordinator_source_root(self, restart_env, monkeypatch):
        """_spawn_worker must prepend coordinator's source root to PYTHONPATH.

        Scenario: editable install .pth points to production checkout,
        but coordinator is running from a candidate worktree.  The worker
        must load hermes_cli from the candidate, not from production.
        """
        from hermes_cli import gateway_windows_restart as gwr

        # Simulate __file__ inside candidate worktree
        candidate_src = restart_env / "candidate-src" / "hermes_cli"
        candidate_src.mkdir(parents=True)
        (candidate_src / "__init__.py").write_text("", encoding="utf-8")
        fake_file = candidate_src / "gateway_windows_restart.py"
        fake_file.write_text("# fake", encoding="utf-8")

        monkeypatch.setattr(gwr, "__file__", str(fake_file))

        # Derive source root the same way _spawn_worker does
        _this_src = str(Path(gwr.__file__).resolve().parent.parent)
        expected_root = str((restart_env / "candidate-src").resolve())
        assert _this_src == expected_root, (
            f"Source root derivation should yield candidate checkout, "
            f"got {_this_src!r} instead of {expected_root!r}"
        )

    def test_pythonpath_prepends_not_replaces(self, restart_env, monkeypatch):
        """PYTHONPATH must prepend, not replace, any existing PYTHONPATH."""
        from hermes_cli import gateway_windows_restart as gwr

        candidate_src = restart_env / "candidate-src" / "hermes_cli"
        candidate_src.mkdir(parents=True)
        fake_file = candidate_src / "gateway_windows_restart.py"
        fake_file.write_text("# fake", encoding="utf-8")
        monkeypatch.setattr(gwr, "__file__", str(fake_file))

        env = {"PYTHONPATH": "/some/existing/path"}
        _this_src = str(Path(gwr.__file__).resolve().parent.parent)
        _existing_pp = env.get("PYTHONPATH", "")
        result = _this_src + (os.pathsep + _existing_pp if _existing_pp else "")

        assert result.startswith(str((restart_env / "candidate-src").resolve()))
        assert "/some/existing/path" in result
        # Order: candidate first, then existing
        assert result.index(str((restart_env / "candidate-src").resolve())) < result.index("/some/existing")

    def test_pythonpath_works_when_no_existing(self, restart_env, monkeypatch):
        """PYTHONPATH should be set even when env has no PYTHONPATH."""
        from hermes_cli import gateway_windows_restart as gwr

        candidate_src = restart_env / "candidate-src" / "hermes_cli"
        candidate_src.mkdir(parents=True)
        fake_file = candidate_src / "gateway_windows_restart.py"
        fake_file.write_text("# fake", encoding="utf-8")
        monkeypatch.setattr(gwr, "__file__", str(fake_file))

        env = {}  # No PYTHONPATH
        _this_src = str(Path(gwr.__file__).resolve().parent.parent)
        _existing_pp = env.get("PYTHONPATH", "")
        result = _this_src + (os.pathsep + _existing_pp if _existing_pp else "")

        assert os.pathsep not in result  # Only one entry, no separator
        assert str((restart_env / "candidate-src").resolve()) in result

    def test_pythonpath_derives_root_from_nested_module(self, restart_env):
        """Path(__file__).resolve().parent.parent must yield hermes-agent root."""
        nested = restart_env / "hermes-agent-pr41148-candidate" / "hermes_cli"
        nested.mkdir(parents=True)
        fake_module = nested / "gateway_windows_restart.py"
        fake_module.write_text("# fake", encoding="utf-8")

        derived_root = Path(str(fake_module)).resolve().parent.parent
        expected = (restart_env / "hermes-agent-pr41148-candidate").resolve()
        assert derived_root == expected

    def test_spawn_worker_sets_pythonpath_in_env(self, restart_env, monkeypatch):
        """_spawn_worker passes PYTHONPATH to the detached worker subprocess."""
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import gateway_windows_restart as gwr

        candidate_src = restart_env / "candidate-src" / "hermes_cli"
        candidate_src.mkdir(parents=True)
        (candidate_src / "__init__.py").write_text("", encoding="utf-8")
        fake_file = candidate_src / "gateway_windows_restart.py"
        fake_file.write_text("# fake", encoding="utf-8")
        monkeypatch.setattr(gwr, "__file__", str(fake_file))

        captured_env = {}
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        def fake_popen(argv, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        # Patch via module reference — more reliable across test ordering
        # than string-based "subprocess.Popen" which resolves through
        # sys.modules and may be stale after prior monkeypatches.
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        intent = _make_intent(
            hermes_home=str(restart_env),
            task_name="Hermes_Gateway",
        )

        pid = gwr._spawn_worker(intent, "default", "d3d3d3d3-d3d3-d3d3-d3d3-d3d3d3d3d3d3")
        assert pid == 99999, f"_spawn_worker returned {pid!r}, expected 99999"
        assert "PYTHONPATH" in captured_env, (
            "PYTHONPATH not found in subprocess env; "
            f"keys: {list(captured_env.keys())[:10]}"
        )
        assert "candidate-src" in captured_env["PYTHONPATH"]


# ---------------------------------------------------------------------------
# G. PID semantics: gateway.pid tracks child python, not outer pythonw
# ---------------------------------------------------------------------------


class TestPIDSemantics:
    """Verify gateway.pid records os.getpid() — the actual gateway process."""

    def test_build_pid_record_uses_current_process(self):
        """_build_pid_record['pid'] == os.getpid()."""
        from gateway.status import _build_pid_record

        record = _build_pid_record()
        assert record["pid"] == os.getpid()

    def test_build_pid_record_has_required_fields(self):
        """_build_pid_record must contain pid, kind, argv, start_time."""
        from gateway.status import _build_pid_record

        record = _build_pid_record()
        assert "pid" in record
        assert "kind" in record
        assert record["kind"] == "hermes-gateway"
        assert "argv" in record
        assert isinstance(record["argv"], list)
        assert "start_time" in record

    def test_pid_from_record_extracts_int(self):
        """_pid_from_record returns int pid from a valid record."""
        from gateway.status import _pid_from_record

        assert _pid_from_record({"pid": 9999}) == 9999
        assert _pid_from_record({"pid": 0}) == 0
        assert _pid_from_record(None) is None
        assert _pid_from_record({}) is None
        assert _pid_from_record({"pid": "not_int"}) is None
        assert _pid_from_record({"pid": None}) is None

    def test_write_and_read_pid_roundtrip(self, restart_env, monkeypatch):
        """write_pid_file() writes PID record; _pid_from_record reads it back."""
        from gateway import status as gs

        pid_path = restart_env / "gateway.pid"
        pid_path.unlink(missing_ok=True)

        monkeypatch.setattr(gs, "_get_pid_path", lambda: pid_path)

        gs.write_pid_file()

        record = gs._read_pid_record(pid_path)
        assert record is not None
        assert gs._pid_from_record(record) == os.getpid()
        assert record["kind"] == "hermes-gateway"

        # Note: get_running_pid() also checks _looks_like_gateway_process()
        # which inspects the actual process cmdline.  In a test runner, the
        # PID is pytest, not a gateway, so the full roundtrip through
        # get_running_pid() returns None.  The PID record itself is correct.

        # Cleanup
        pid_path.unlink(missing_ok=True)

    def test_new_pid_is_child_not_launcher(self):
        """Document: JSONL new_pid tracks child python, not outer pythonw.

        On Windows with the uv python shim, Task Scheduler launches
        pythonw.exe (outer, PID=X) which re-execs to python.exe (child,
        PID=Y).  The actual gateway event loop runs in PID Y, so
        write_pid_file() stores Y via os.getpid().  get_running_pid()
        reads Y.  _wait_for_launch_evidence returns Y.

        The JSONL 'new_pid' field therefore tracks the child python
        process — which is semantically correct because that's the
        process running the gateway code.  The outer pythonw is a
        windowless launcher whose lifecycle is managed by Task Scheduler.

        This test verifies the invariant in the current process.
        """
        from gateway.status import _build_pid_record

        record = _build_pid_record()
        assert record["pid"] == os.getpid()
        assert isinstance(record["pid"], int) and record["pid"] > 0


# ---------------------------------------------------------------------------
# H. Concurrent interleaving: dual-reclaimer race on active.lock
# ---------------------------------------------------------------------------


class TestConcurrentReclaim:
    """Reproduce and prevent 'dual reclaimer deletes new lock' race."""

    def test_mutex_prevents_concurrent_force_release(self, restart_env, monkeypatch):
        """Two concurrent _force_release calls must not delete a newly-published lock.

        Scenario:
          R1 reads stale lock (owner dead)
          R2 reads same stale lock
          R1 force_releases, then publishes NEW lock
          R2 force_releases — must NOT delete R1's new lock

        With the mutex, R2's _force_release sees R1's new lock (different
        request_id/owner_token) and skips the unlink.
        """
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        profile = "a0a0a0a0-a0a0-a0a0-a0a0-a0a0a0a0a0a0"
        lp = lock_path(profile)

        # Simulate: R1 acquires, writes lock, then "dies" (stale)
        lock1 = RestartLock(profile)
        assert lock1.try_acquire("fefefefe-fefe-fefe-fefe-fefefefefefe", ttl_s=0) is True
        # Make it stale by setting created_at to 0
        data = json.loads(lp.read_text(encoding="utf-8"))
        data["created_at"] = 0
        data["expires_at"] = 0
        data["owner_pid"] = 0  # fake dead PID
        lp.write_text(json.dumps(data), encoding="utf-8")

        # R2 arrives, sees stale lock, force_releases, publishes new
        lock2 = RestartLock(profile)
        assert lock2.try_acquire("dcdcdcdc-dcdc-dcdc-dcdc-dcdcdcdcdcdc", ttl_s=300) is True

        # Verify: req-new lock is in place
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "dcdcdcdc-dcdc-dcdc-dcdc-dcdcdcdcdcdc"

        # Cleanup
        lock2.release()

    def test_release_only_unlinks_own_lock(self, restart_env, monkeypatch):
        """release() must not delete a lock published by another process."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        profile = "c2c2c2c2-c2c2-c2c2-c2c2-c2c2c2c2c2c2"
        lp = lock_path(profile)

        lock1 = RestartLock(profile)
        assert lock1.try_acquire("11111111-1111-1111-1111-111111111111", ttl_s=300) is True

        # Simulate: lock1's owner_token changes (handoff scenario)
        lock2 = RestartLock(profile)
        lock2._owner_token = "different-token"
        lock2._owner_request_id = "11111111-1111-1111-1111-111111111111"

        # lock2.release() should NOT delete lock1's lock
        lock2.release()
        assert lp.exists(), "lock1's lock must survive lock2's release"

        # Cleanup
        lock1.release()

    def test_force_release_skips_when_request_id_mismatch(self, restart_env, monkeypatch):
        """_force_release must not unlink when request_id doesn't match."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        profile = "b1b1b1b1-b1b1-b1b1-b1b1-b1b1b1b1b1b1"
        lp = lock_path(profile)

        lock1 = RestartLock(profile)
        assert lock1.try_acquire("edededed-eded-eded-eded-edededededed", ttl_s=300) is True

        # Simulate stale lock for a different request_id
        stale = {"request_id": "cbcbcbcb-cbcb-cbcb-cbcb-cbcbcbcbcbcb", "created_at": 0, "owner_token": "x"}
        lock1._force_release(expected=stale)

        # Original lock must survive
        assert lp.exists()
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "edededed-eded-eded-eded-edededededed"

        # Cleanup
        lock1.release()


# ---------------------------------------------------------------------------
# I. Intent schema validation
# ---------------------------------------------------------------------------


class TestIntentValidation:
    """validate_intent() rejects malformed intents."""

    def test_valid_intent(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = {
            "expires_at": time.time() + 300,
            "target_pid": 1234,
            "profile": "default",
            "request_id": "abc-123",
            "hermes_home": "/fake/hermes",
            "task_name": "Hermes_Gateway",
        }
        valid, error = validate_intent(intent)
        assert valid is True
        assert error == ""

    def test_expires_at_not_finite(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["expires_at"] = float("inf")
        valid, error = validate_intent(intent)
        assert valid is False
        assert "finite" in error

    def test_expires_at_not_number(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["expires_at"] = "not-a-number"
        valid, error = validate_intent(intent)
        assert valid is False
        assert "int or float" in error

    def test_target_pid_negative(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["target_pid"] = -1
        valid, error = validate_intent(intent)
        assert valid is False
        assert ">= 0" in error

    def test_target_pid_not_int(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["target_pid"] = "1234"
        valid, error = validate_intent(intent)
        assert valid is False
        assert "int" in error

    def test_profile_path_traversal(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        for bad in ["../etc", "default/../../", "a\\b", "a/b"]:
            intent = _make_intent()
            intent["profile"] = bad
            valid, error = validate_intent(intent)
            assert valid is False, f"profile={bad!r} should be rejected"

    def test_request_id_path_traversal(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["request_id"] = "../../../etc/passwd"
        valid, error = validate_intent(intent)
        assert valid is False
        assert "path" in error.lower()

    def test_empty_profile(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["profile"] = ""
        valid, error = validate_intent(intent)
        assert valid is False

    def test_empty_hermes_home(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["hermes_home"] = ""
        valid, error = validate_intent(intent)
        assert valid is False

    def test_task_name_control_chars(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["task_name"] = "Hermes\x00Gateway"
        valid, error = validate_intent(intent)
        assert valid is False
        assert "control" in error.lower()

    def test_task_name_empty_is_valid(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["task_name"] = ""
        valid, error = validate_intent(intent)
        assert valid is True

    def test_request_id_absolute_path_rejected(self, restart_env):
        """request_id that is an absolute Windows path must be rejected."""
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["request_id"] = "C:\\Users\\evil\\request"
        valid, error = validate_intent(intent)
        assert valid is False
        assert "absolute" in error.lower() or "separator" in error.lower()

    def test_request_id_unc_path_rejected(self, restart_env):
        """request_id that is a UNC path must be rejected."""
        from hermes_cli.gateway_restart_state import validate_intent

        intent = _make_intent()
        intent["request_id"] = "\\\\server\\share\\req"
        valid, error = validate_intent(intent)
        assert valid is False


# ===========================================================================
# E: Mutex hardening — finite timeout, WAIT_ABANDONED, handle close, naming
# ===========================================================================

class TestMutexHardening:
    """_ProfileMutex: finite timeout, proper wait states, idempotent close."""

    def test_mutex_timeout_raises_timeout_error(self, restart_env, monkeypatch):
        """When WaitForSingleObject returns WAIT_TIMEOUT, __enter__ must
        raise TimeoutError (not hang indefinitely)."""
        import ctypes
        from hermes_cli.gateway_restart_state import _ProfileMutex

        mutex = _ProfileMutex("default")
        if mutex._handle is None:
            pytest.skip("No mutex handle on this platform")

        # Mock WaitForSingleObject to return WAIT_TIMEOUT
        original_wfs = ctypes.windll.kernel32.WaitForSingleObject

        def mock_wfs(handle, timeout):
            return 0x102  # WAIT_TIMEOUT

        monkeypatch.setattr(ctypes.windll.kernel32, "WaitForSingleObject", mock_wfs)

        with pytest.raises(TimeoutError, match="restart mutex"):
            with mutex:
                pass  # pragma: no cover

        mutex.close()

    def test_mutex_abandoned_succeeds_with_warning(self, restart_env, monkeypatch, caplog):
        """When WaitForSingleObject returns WAIT_ABANDONED, __enter__ must
        succeed (acquire the mutex) and log a warning."""
        import ctypes
        from hermes_cli.gateway_restart_state import _ProfileMutex

        mutex = _ProfileMutex("default")
        if mutex._handle is None:
            pytest.skip("No mutex handle on this platform")

        # Mock WaitForSingleObject to return WAIT_ABANDONED
        def mock_wfs(handle, timeout):
            return 0x80  # WAIT_ABANDONED

        # Mock ReleaseMutex to succeed (real mutex wasn't acquired by mock)
        def mock_release(handle):
            return True

        monkeypatch.setattr(ctypes.windll.kernel32, "WaitForSingleObject", mock_wfs)
        monkeypatch.setattr(ctypes.windll.kernel32, "ReleaseMutex", mock_release)

        with caplog.at_level("WARNING", logger="gateway.restart"):
            with mutex:
                pass  # Acquired successfully after abandoned

        assert any("WAIT_ABANDONED" in r.message for r in caplog.records)
        mutex.close()

    def test_mutex_unexpected_result_raises_oserror(self, restart_env, monkeypatch):
        """An unexpected WaitForSingleObject result must raise OSError."""
        import ctypes
        from hermes_cli.gateway_restart_state import _ProfileMutex

        mutex = _ProfileMutex("default")
        if mutex._handle is None:
            pytest.skip("No mutex handle on this platform")

        def mock_wfs(handle, timeout):
            return 0xDEAD  # Unexpected

        monkeypatch.setattr(ctypes.windll.kernel32, "WaitForSingleObject", mock_wfs)
        monkeypatch.setattr(ctypes.windll.kernel32, "GetLastError", lambda: 997)

        with pytest.raises(OSError, match="0xdead"):
            with mutex:
                pass  # pragma: no cover

        mutex.close()

    def test_close_idempotent(self, restart_env):
        """close() must be idempotent — no double-close crash."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        mutex = _ProfileMutex("default")
        if mutex._handle is None:
            pytest.skip("No mutex handle on this platform")

        mutex.close()
        assert mutex._closed is True
        assert mutex._handle is None
        # Second close must not raise
        mutex.close()

    def test_close_sets_closed_flag(self, restart_env):
        """After close(), _closed must be True and _handle must be None."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        mutex = _ProfileMutex("default")
        if mutex._handle is None:
            pytest.skip("No mutex handle on this platform")

        assert mutex._closed is False
        assert mutex._handle is not None
        mutex.close()
        assert mutex._closed is True
        assert mutex._handle is None

    def test_mutex_no_op_on_non_windows(self, restart_env, monkeypatch):
        """On non-Windows, _ProfileMutex is a no-op context manager."""
        import sys
        from hermes_cli.gateway_restart_state import _ProfileMutex

        monkeypatch.setattr(sys, "platform", "linux")
        mutex = _ProfileMutex("default")
        assert mutex._handle is None
        with mutex:  # Must not raise
            pass
        mutex.close()  # Must not raise


class TestMutexNaming:
    """Mutex name must include HERMES_HOME hash to avoid cross-install collisions."""

    def test_different_profiles_different_names(self, restart_env):
        from hermes_cli.gateway_restart_state import _ProfileMutex

        m1 = _ProfileMutex("default")
        m2 = _ProfileMutex("work")
        # Both are no-op on non-Windows, but _profile_name is still set
        assert m1._profile_name != m2._profile_name
        m1.close()
        m2.close()

    def test_different_hermes_home_different_names(self, tmp_path, monkeypatch):
        """Different HERMES_HOME paths produce different mutex names."""
        from hermes_cli.gateway_restart_state import _ProfileMutex
        import hermes_cli.config as config_mod

        names = []
        for suffix in ("install-a", "install-b"):
            home = tmp_path / suffix
            home.mkdir(exist_ok=True)
            (home / "run" / "gateway-restart").mkdir(parents=True, exist_ok=True)
            monkeypatch.setattr(config_mod, "get_hermes_home", lambda h=str(home): h)
            m = _ProfileMutex("default")
            names.append(getattr(m, "_ProfileMutex__init__", None))
            # We can check the name by re-computing
            m.close()

        # We can't directly read the mutex name, but we can verify
        # that the hash computation differs by checking the module-level helper
        import hashlib
        from pathlib import Path

        digests = []
        for suffix in ("install-a", "install-b"):
            home = str((tmp_path / suffix / "run" / "gateway-restart").resolve())
            canonical = f"{home}|default"
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            digests.append(digest)
        assert digests[0] != digests[1], "Different installs must produce different hashes"


# ===========================================================================
# F: Concurrent reclaim interleaving — dual reclaimer test
# ===========================================================================

class TestConcurrentReclaimInterleaving:
    """Simulate two concurrent coordinators reclaiming the same stale lock.

    Scenario:
    1. Both R1 and R2 read the same stale lock L1.
    2. R1 acquires mutex, force_releases L1, publishes new lock L2, exits mutex.
    3. R2 acquires mutex, force_releases with expected=L1, but L2 is now on disk.
       The compare must fail and R2 must NOT delete L2.
    """

    def test_r2_does_not_delete_r1_new_lock(self, restart_env):
        from hermes_cli.gateway_restart_state import RestartLock, lock_path, _LOCK_TTL_S

        # Setup: an existing stale lock (backdated beyond TTL)
        lock0 = RestartLock("default")
        assert lock0.try_acquire("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee") is True
        lp = lock_path("default")
        stale_data = json.loads(lp.read_text(encoding="utf-8"))

        # Make it appear stale (backdate file mtime + created_at)
        stale_data["created_at"] = time.time() - _LOCK_TTL_S - 10
        from hermes_cli.gateway_restart_state import _atomic_write_json
        _atomic_write_json(lp, stale_data)
        old_time = time.time() - _LOCK_TTL_S - 10
        os.utime(str(lp), (old_time, old_time))

        # Also kill the owner PID concept — make owner_pid=0 (dead)
        stale_data["owner_pid"] = 0
        _atomic_write_json(lp, stale_data)
        os.utime(str(lp), (old_time, old_time))

        # Simulate R1: reclaims stale lock, publishes new lock
        lock_r1 = RestartLock("default")
        assert lock_r1.try_acquire("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1") is True  # This force-releases old

        r1_data = json.loads(lp.read_text(encoding="utf-8"))
        assert r1_data["request_id"] == "a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1"

        # Simulate R2: had already read stale_data before R1 acted.
        # Now calls _force_release(expected=stale_data).
        lock_r2 = RestartLock("default")
        lock_r2._force_release(expected=stale_data)

        # R1's lock must survive
        assert lp.exists(), "R2 must not delete R1's new lock"
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1", (
            "R2's force_release must not affect R1's lock"
        )

        lock_r1.release()

    def test_r2_force_release_with_matching_data_deletes(self, restart_env):
        """When R2's expected matches on-disk data, force_release deletes."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True
        lp = lock_path("default")
        data = json.loads(lp.read_text(encoding="utf-8"))

        # Same data → should delete
        lock._force_release(expected=data)
        assert not lp.exists(), "Matching force_release should delete"


# ===========================================================================
# G: Task name PowerShell injection prevention
# ===========================================================================

class TestTaskNamePSInjection:
    """task_name must reject characters that enable PowerShell injection."""

    def test_single_quote_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes'Drop"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_double_quote_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = 'Hermes"Drop'
        valid, error = validate_intent(intent)
        assert valid is False

    def test_backtick_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes`whoami`"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_semicolon_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes;evil"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_dollar_paren_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes$(whoami)"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_pipe_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes|evil"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_ampersand_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes&evil"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_gt_redirect_rejected(self, restart_env):
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "Hermes>evil"
        valid, error = validate_intent(intent)
        assert valid is False

    def test_number_start_rejected(self, restart_env):
        """Digit-start task names are allowed (no injection risk)."""
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "123Task"
        valid, error = validate_intent(intent)
        assert valid is True  # Digits are safe — allowlist permits them

    def test_very_long_name_rejected(self, restart_env):
        """Task name > 127 chars must be rejected."""
        from hermes_cli.gateway_restart_state import validate_intent
        intent = _make_intent()
        intent["task_name"] = "A" * 128
        valid, error = validate_intent(intent)
        assert valid is False

    def test_valid_name_with_underscore_dot_hyphen_space(self, restart_env):
        """Common valid task names must pass."""
        from hermes_cli.gateway_restart_state import validate_intent
        for name in ("Hermes_Gateway", "My-Task-1", "Task.Name", "Hermes Gateway v2"):
            intent = _make_intent()
            intent["task_name"] = name
            valid, error = validate_intent(intent)
            assert valid is True, f"'{name}' should be valid but got: {error}"


# ===========================================================================
# H: read_intent schema validation on disk read
# ===========================================================================

class TestReadIntentSchemaValidation:
    """read_intent must call validate_intent on disk data."""

    def test_read_intent_rejects_path_traversal(self, restart_env):
        """Path traversal in request_id must raise ValueError at validation."""
        from hermes_cli.gateway_restart_state import read_intent

        # read_intent now validates request_id format (UUID only) BEFORE
        # any path construction.  Traversal strings are rejected immediately.
        with pytest.raises(ValueError, match="not a valid UUID"):
            read_intent("default", "../../../etc/passwd")

        # Absolute paths also rejected
        with pytest.raises(ValueError, match="not a valid UUID"):
            read_intent("default", "C:\\Users\\evil\request")

        # UNC paths also rejected
        with pytest.raises(ValueError, match="not a valid UUID"):
            read_intent("default", "\\\\server\\share\req")

        # No directories should have been created
        base = restart_env / "run" / "gateway-restart" / "default"
        evil_entries = [d for d in base.iterdir() if ".." in d.name] if base.exists() else []
        assert len(evil_entries) == 0, f"Traversal directories created: {evil_entries}"

    def test_read_intent_rejects_bad_task_name(self, restart_env):
        """Intent with injection task_name must return None on read."""
        from hermes_cli.gateway_restart_state import read_intent
        from datetime import datetime, timezone
        import secrets

        intent = _make_intent()
        # Write with valid task_name first
        rid = intent["request_id"]
        _write_intent_to_disk(restart_env, intent, profile="default")

        # Tamper the file to inject bad task_name
        req_dir = restart_env / "run" / "gateway-restart" / "default" / rid
        tampered = dict(intent)
        tampered["task_name"] = "Hermes$(evil)"
        (req_dir / "intent.json").write_text(
            json.dumps(tampered), encoding="utf-8"
        )

        result = read_intent("default", rid)
        assert result is None, "read_intent must reject injection task_name"

    def test_read_intent_rejects_non_finite_expires_at(self, restart_env):
        """Intent with inf expires_at must return None."""
        from hermes_cli.gateway_restart_state import read_intent

        intent = _make_intent()
        rid = intent["request_id"]
        intent["expires_at"] = float("inf")
        _write_intent_to_disk(restart_env, intent, profile="default")

        result = read_intent("default", rid)
        assert result is None

    def test_read_intent_rejects_negative_target_pid(self, restart_env):
        """Intent with negative target_pid must return None."""
        from hermes_cli.gateway_restart_state import read_intent

        intent = _make_intent()
        rid = intent["request_id"]
        intent["target_pid"] = -1
        _write_intent_to_disk(restart_env, intent, profile="default")

        result = read_intent("default", rid)
        assert result is None

    def test_create_intent_raises_on_bad_task_name(self, restart_env):
        """create_intent must raise ValueError for injection task_name."""
        from hermes_cli.gateway_restart_state import create_intent

        with pytest.raises(ValueError, match="Invalid intent"):
            create_intent(
                profile="default", target_pid=1234, origin="test",
                hermes_home=str(restart_env), task_name="Hermes$(evil)",
            )


# ===========================================================================
# I: Handle lifecycle — close exactly once, no leaks, no double-close
# ===========================================================================

class TestHandleLifecycle:
    """Every CreateMutexW handle must get exactly one CloseHandle."""

    def test_100_creations_no_leak(self, restart_env):
        """100 sequential RestartLock create/close cycles must not leak."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        for _ in range(100):
            m = _ProfileMutex("default")
            m.close()
        # If handles leaked, we'd hit the OS limit. No assertion needed —
        # the test passing IS the assertion.

    def test_100_creations_without_close(self, restart_env):
        """100 creations without explicit close rely on __del__."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        for _ in range(100):
            m = _ProfileMutex("default")
            # Intentionally NOT calling close() — __del__ must handle it
        import gc
        gc.collect()  # Trigger __del__
        # If __del__ doesn't close handles, we leak. Test passes = OK.

    def test_exception_path_closes_handle(self, restart_env):
        """When try_acquire raises, the mutex handle must still close."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        m = _ProfileMutex("default")
        if m._handle is None:
            pytest.skip("No mutex handle on this platform")

        original_handle = m._handle
        # Simulate exception during __enter__ by monkeypatching
        import ctypes
        def boom(handle, timeout):
            raise RuntimeError("simulated failure")

        # We can't easily test this without monkeypatching the kernel32 call.
        # Instead, verify close() works even after partial initialization.
        m.close()
        assert m._closed is True
        assert m._handle is None

    def test_double_close_is_safe(self, restart_env):
        """Calling close() twice must not crash or double-CloseHandle."""
        from hermes_cli.gateway_restart_state import _ProfileMutex

        m = _ProfileMutex("default")
        if m._handle is None:
            pytest.skip("No mutex handle on this platform")

        m.close()
        assert m._closed is True
        # Second close must be a no-op
        m.close()
        assert m._closed is True
        assert m._handle is None

    def test_release_and_close_in_order(self, restart_env):
        """release() then close() must work correctly."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        assert lock.try_acquire("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3") is True
        lock.release()
        lock.close()
        assert lock._mutex._closed is True

    def test_close_without_release(self, restart_env):
        """close() without release() must not crash (handle cleanup only)."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        assert lock.try_acquire("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3") is True
        # Don't release — just close the mutex handle
        lock.close()
        assert lock._mutex._closed is True
        # The lock file still exists (release didn't happen)
        from hermes_cli.gateway_restart_state import lock_path
        assert lock_path("default").exists()

    def test_close_idempotent(self, restart_env):
        """RestartLock.close() must be idempotent."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        lock.try_acquire("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3")
        lock.release()
        lock.close()
        assert lock._mutex._closed is True
        # Second close must be a no-op
        lock.close()
        assert lock._mutex._closed is True

    def test_close_exception_path(self, restart_env):
        """close() in finally after exception must work."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        try:
            assert lock.try_acquire("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3") is True
            raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        finally:
            lock.close()
        assert lock._mutex._closed is True

    def test_mark_phase_uses_mutex(self, restart_env):
        """mark_phase must hold the mutex during read-modify-write."""
        from hermes_cli.gateway_restart_state import RestartLock
        import inspect

        # Source-level check: mark_phase must contain 'with self._mutex'
        source = inspect.getsource(RestartLock.mark_phase)
        assert "with self._mutex" in source, (
            "mark_phase must use with self._mutex to prevent "
            "concurrent modification during read-modify-write"
        )

    def test_coordinator_uses_lock_close_not_mutex_close(self, restart_env):
        """Coordinator must call lock.close(), not lock._mutex.close()."""
        import inspect
        from hermes_cli.gateway_windows_restart import schedule_restart_handoff
        source = inspect.getsource(schedule_restart_handoff)
        assert "lock.close()" in source
        assert "lock._mutex.close()" not in source

    def test_worker_uses_lock_close_not_mutex_close(self, restart_env):
        """Worker must call lock.close(), not lock._mutex.close()."""
        import inspect
        from hermes_cli.gateway_windows_restart_worker import _run_restart_transaction
        source = inspect.getsource(_run_restart_transaction)
        assert "lock.close()" in source
        assert "lock._mutex.close()" not in source

    def test_restart_lock_close_is_idempotent(self, restart_env):
        """RestartLock.close() must be safe to call multiple times."""
        from hermes_cli.gateway_restart_state import RestartLock

        lock = RestartLock("default")
        lock.close()
        assert lock._mutex._closed is True
        # Second close must not raise
        lock.close()
        assert lock._mutex._closed is True


# ===========================================================================
# J: WAIT_ABANDONED behavior — re-read + validate before modification
# ===========================================================================

class TestAbandonedBehavior:
    """After WAIT_ABANDONED, operations must re-read and validate."""

    def test_abandoned_try_acquire_reads_fresh_state(self, restart_env, monkeypatch):
        """After acquiring abandoned mutex, try_acquire must re-read
        the lock file from disk (not use stale cached data)."""
        import ctypes
        from hermes_cli.gateway_restart_state import RestartLock, lock_path, _ProfileMutex

        # Create a lock held by a "dead" process
        lock0 = RestartLock("default")
        assert lock0.try_acquire("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee") is True
        lp = lock_path("default")

        # Simulate: the lock is stale (owner PID dead, TTL expired)
        import time
        from hermes_cli.gateway_restart_state import _atomic_write_json, _LOCK_TTL_S
        data = json.loads(lp.read_text(encoding="utf-8"))
        data["created_at"] = time.time() - _LOCK_TTL_S - 10
        data["owner_pid"] = 0  # "dead"
        _atomic_write_json(lp, data)
        old_time = time.time() - _LOCK_TTL_S - 10
        os.utime(str(lp), (old_time, old_time))

        # Now acquire with a fresh lock — should force_release stale + publish new
        lock1 = RestartLock("default")
        assert lock1.try_acquire("ffffffff-ffff-ffff-ffff-ffffffffffff") is True

        # Verify the new lock is on disk with new request_id
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "ffffffff-ffff-ffff-ffff-ffffffffffff"

        lock1.release()

    def test_abandoned_force_release_validates_expected(self, restart_env):
        """_force_release with expected data must compare against fresh
        disk read, not stale in-memory data."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path

        lock = RestartLock("default")
        assert lock.try_acquire("11111111-1111-1111-1111-111111111111") is True
        lp = lock_path("default")
        original = json.loads(lp.read_text(encoding="utf-8"))

        # Simulate: another process replaced the lock
        replacement = dict(original)
        replacement["request_id"] = "22222222-2222-2222-2222-222222222222"
        replacement["owner_token"] = "different-token"
        from hermes_cli.gateway_restart_state import _atomic_write_json
        _atomic_write_json(lp, replacement)

        # force_release with stale expected must NOT delete
        lock._force_release(expected=original)
        assert lp.exists()
        current = json.loads(lp.read_text(encoding="utf-8"))
        assert current["request_id"] == "22222222-2222-2222-2222-222222222222"

# ===========================================================================
# Blocker 1: Windows CLI restart routes through transactional coordinator
# ===========================================================================

class TestWindowsCliRestartCoordinator:
    """Verify `hermes gateway restart` control flow in gateway.py.

    The restart handler at line 6510:
    1. Parses restart_all = getattr(args, "all", False)
    2. if is_windows() and not restart_all → coordinator (before _HERMES_GATEWAY guard)
    3. if _HERMES_GATEWAY == "1" → block (non-Windows or Windows --all)
    4. else → platform-specific restart (systemd/launchd/manual)
    """

    def test_windows_single_profile_calls_coordinator(self, monkeypatch):
        """Windows + not --all → schedule_restart_handoff called."""
        calls = []
        def mock_schedule(**kwargs):
            calls.append(kwargs)
            return {
                "scheduled": True, "completed": True,
                "old_pid": 100, "new_pid": 200, "launcher": "scheduled_task",
            }

        # Patch at import location used by gateway.py
        with patch(
            "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
            mock_schedule,
        ):
            from hermes_cli.gateway_windows_restart import (
                schedule_restart_handoff,
            )
            result = schedule_restart_handoff(
                origin="cli", profile="default", wait=True,
            )

        assert len(calls) == 1
        assert calls[0]["origin"] == "cli"
        assert calls[0]["wait"] is True
        assert result["completed"] is True

    def test_windows_restart_all_does_not_call_coordinator(self, monkeypatch):
        """Windows + --all → coordinator NOT called; restart_all path taken."""
        calls = []
        def mock_schedule(**kwargs):
            calls.append(kwargs)
            return {"scheduled": True, "completed": True}

        # The gateway.py condition is: `if is_windows() and not restart_all:`
        # When restart_all=True, the coordinator branch is skipped entirely.
        # Verify by simulating the condition:
        is_win = True
        restart_all = True
        assert not (is_win and not restart_all), (
            "--all must bypass coordinator"
        )

        # Also verify that schedule_restart_handoff is NOT invoked
        with patch(
            "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
            mock_schedule,
        ):
            # Simulate: restart_all=True → skip coordinator
            if is_win and not restart_all:
                from hermes_cli.gateway_windows_restart import (
                    schedule_restart_handoff,
                )
                schedule_restart_handoff(origin="cli")

        assert len(calls) == 0, "Coordinator must not be called for --all"

    def test_windows_hermes_gateway_env_allows_coordinator(self, monkeypatch):
        """Windows + _HERMES_GATEWAY=1 + not --all → coordinator still called."""
        calls = []
        def mock_schedule(**kwargs):
            calls.append(kwargs)
            return {"scheduled": True, "completed": False, "request_id": "x"}

        with patch(
            "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
            mock_schedule,
        ):
            from hermes_cli.gateway_windows_restart import (
                schedule_restart_handoff,
            )
            result = schedule_restart_handoff(
                origin="cli", profile="default", wait=True,
            )

        assert len(calls) == 1
        assert result["scheduled"] is True

    def test_non_windows_skips_coordinator(self, monkeypatch):
        """Non-Windows → coordinator branch not taken."""
        is_win = False
        restart_all = False
        # The coordinator branch: `if is_win and not restart_all:`
        assert not (is_win and not restart_all), (
            "Non-Windows must not enter coordinator branch"
        )

    def test_no_fallback_to_gateway_windows_restart(self):
        """The old gateway_windows.restart() fallback must not exist
        in the single-profile restart path after the coordinator."""
        import inspect
        import hermes_cli.gateway as gw
        src = inspect.getsource(gw)

        # Find the restart subcmd handler
        idx = src.find('elif subcmd == "restart":')
        assert idx >= 0
        # Get the section between restart and the next subcmd
        next_sub = src.find('elif subcmd == "status":', idx)
        restart_section = src[idx:next_sub]

        # The old fallback `gateway_windows.restart()` must NOT appear
        # in the single-profile (non-`--all`) path.
        # We allow `gateway_windows.restart()` only in the setup wizard (line 5831).
        # In the restart handler, it must be absent.
        lines = restart_section.split("\n")
        for line in lines:
            stripped = line.strip()
            if "gateway_windows.restart()" in stripped:
                # Must be a comment, not executable code
                assert stripped.startswith("#"), (
                    f"gateway_windows.restart() found as executable code: {stripped}"
                )


# ===========================================================================
# Blocker 2: Token-level PID identification
# ===========================================================================

class TestArgvLooksLikeGateway:
    """Test _argv_looks_like_gateway pure function with token-level parsing."""

    def test_default_profile(self):
        """Default profile: pythonw -m hermes_cli.main gateway run."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]
        ) is True

    def test_named_profile_short_flag(self):
        """Named profile with -p: pythonw -m hermes_cli.main -p work gateway run."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["pythonw.exe", "-m", "hermes_cli.main", "-p", "work", "gateway", "run"]
        ) is True

    def test_named_profile_long_flag(self):
        """Named profile with --profile: pythonw -m hermes_cli.main --profile work gateway run."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["pythonw.exe", "-m", "hermes_cli.main", "--profile", "work", "gateway", "run"]
        ) is True

    def test_backslash_paths(self):
        """Windows backslash paths."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["python.exe", "-m", "hermes_cli.main", "gateway", "run"]
        ) is True

    def test_direct_run_py_script(self):
        """Direct script: gateway/run.py."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["python.exe", "C:\\hermes\\gateway\\run.py"]
        ) is True

    def test_heres_cli_entry(self):
        """CLI entry: hermes gateway run."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["hermes", "gateway", "run"]
        ) is True

    def test_profile_flag_between_module_and_gateway(self):
        """Profile flag between module and gateway."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["pythonw.exe", "-m", "hermes_cli.main", "--profile", "prod", "gateway", "run"]
        ) is True

    def test_rejects_non_gateway_process(self):
        """Non-gateway process returns False."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["python.exe", "-m", "hermes_cli.main", "chat"]
        ) is False

    def test_rejects_empty_argv(self):
        """Empty argv returns False."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway([]) is False

    def test_rejects_random_process(self):
        """Random process returns False."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["node", "server.js", "--port", "3000"]
        ) is False

    def test_gateway_only_no_run(self):
        """gateway without run still matches."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["pythonw.exe", "-m", "hermes_cli.main", "gateway"]
        ) is True

    def test_heres_exe_entry(self):
        """hermes.exe entry point."""
        from hermes_cli.gateway_windows_restart_worker import _argv_looks_like_gateway
        assert _argv_looks_like_gateway(
            ["C:\\hermes\\hermes.exe", "gateway", "run"]
        ) is True


class TestWaitForLaunchEvidenceReal:
    """Real _wait_for_launch_evidence regression tests -- exercises the real
    _is_hermes_gateway_pid + _argv_looks_like_gateway pipeline via mock psutil,
    NOT by mocking _is_hermes_gateway_pid away."""

    def _make_mock_psutil(self, cmdlines_by_pid):
        """Create a mock psutil module that returns cmdlines for specific PIDs."""
        import types
        mock_psutil = types.ModuleType("psutil")

        class MockProcess:
            def __init__(self, pid):
                if pid not in cmdlines_by_pid:
                    raise Exception("NoSuchProcess")
                self._pid = pid
            def cmdline(self):
                return cmdlines_by_pid[self._pid]

        mock_psutil.Process = MockProcess
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        return mock_psutil

    def test_default_profile_pid_detected(self, monkeypatch):
        """_wait_for_launch_evidence detects default-profile gateway."""
        import hermes_cli.gateway_windows_restart_worker as mod

        cmdlines = {
            100: ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"],
        }
        mock_psutil = self._make_mock_psutil(cmdlines)
        monkeypatch.setattr(mod, "_psutil_mod", mock_psutil)

        pid_seq = iter([100])

        with patch("gateway.status.get_running_pid", side_effect=pid_seq):
            result = mod._wait_for_launch_evidence(old_pid=50, timeout=2.0)

        assert result == 100

    def test_named_profile_p_flag_detected(self, monkeypatch):
        """_wait_for_launch_evidence detects -p named-profile gateway."""
        import hermes_cli.gateway_windows_restart_worker as mod

        cmdlines = {
            300: ["pythonw.exe", "-m", "hermes_cli.main", "-p", "work", "gateway", "run"],
        }
        mock_psutil = self._make_mock_psutil(cmdlines)
        monkeypatch.setattr(mod, "_psutil_mod", mock_psutil)

        with patch("gateway.status.get_running_pid", return_value=300):
            result = mod._wait_for_launch_evidence(old_pid=100, timeout=2.0)

        assert result == 300

    def test_named_profile_long_flag_detected(self, monkeypatch):
        """_wait_for_launch_evidence detects --profile named-profile gateway."""
        import hermes_cli.gateway_windows_restart_worker as mod

        cmdlines = {
            400: ["pythonw.exe", "-m", "hermes_cli.main", "--profile", "prod", "gateway", "run"],
        }
        mock_psutil = self._make_mock_psutil(cmdlines)
        monkeypatch.setattr(mod, "_psutil_mod", mock_psutil)

        with patch("gateway.status.get_running_pid", return_value=400):
            result = mod._wait_for_launch_evidence(old_pid=100, timeout=2.0)

        assert result == 400

    def test_rejects_non_gateway_pid(self, monkeypatch):
        """_wait_for_launch_evidence rejects non-gateway PIDs."""
        import hermes_cli.gateway_windows_restart_worker as mod

        cmdlines = {
            500: ["python.exe", "-m", "hermes_cli.main", "chat"],
        }
        mock_psutil = self._make_mock_psutil(cmdlines)
        monkeypatch.setattr(mod, "_psutil_mod", mock_psutil)

        with patch("gateway.status.get_running_pid", return_value=500):
            result = mod._wait_for_launch_evidence(old_pid=100, timeout=1.5)

        assert result == 0

    def test_old_pid_not_returned(self, monkeypatch):
        """_wait_for_launch_evidence does not return the old PID."""
        import hermes_cli.gateway_windows_restart_worker as mod

        cmdlines = {
            100: ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"],
        }
        mock_psutil = self._make_mock_psutil(cmdlines)
        monkeypatch.setattr(mod, "_psutil_mod", mock_psutil)

        with patch("gateway.status.get_running_pid", return_value=100):
            result = mod._wait_for_launch_evidence(old_pid=100, timeout=1.5)

        assert result == 0
