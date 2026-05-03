"""Tests for gateway runtime status tracking."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from gateway import status


class TestGatewayPidState:
    def test_write_pid_file_records_gateway_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_pid_file()

        payload = json.loads((tmp_path / "gateway.pid").read_text())
        assert payload["pid"] == os.getpid()
        assert payload["kind"] == "hermes-gateway"
        assert isinstance(payload["argv"], list)
        assert payload["argv"]

    def test_write_pid_file_is_atomic_against_concurrent_writers(self, tmp_path, monkeypatch):
        """Regression: two concurrent --replace invocations must not both win.

        Without O_CREAT|O_EXCL, two processes racing through start_gateway()'s
        termination-wait would both write to gateway.pid, silently overwriting
        each other and leaving multiple gateway instances alive (#11718).
        """
        import pytest

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # First write wins.
        status.write_pid_file()
        assert (tmp_path / "gateway.pid").exists()

        # Second write (simulating a racing --replace that missed the earlier
        # guards) must raise FileExistsError rather than clobber the record.
        with pytest.raises(FileExistsError):
            status.write_pid_file()

        # Original record is preserved.
        payload = json.loads((tmp_path / "gateway.pid").read_text())
        assert payload["pid"] == os.getpid()

    def test_get_running_pid_rejects_live_non_gateway_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(str(os.getpid()))

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_cleans_stale_record_from_dead_process(self, tmp_path, monkeypatch):
        # Simulates the aftermath of a crash: the PID file still points at a
        # process that no longer exists. The next gateway startup must be
        # able to unlink it so ``write_pid_file``'s O_EXCL create succeeds —
        # otherwise systemd's restart loop hits "PID file race lost" forever.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        dead_pid = 999999  # not our pid, and below we simulate it's dead
        pid_path.write_text(json.dumps({
            "pid": dead_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "start_time": 111,
        }))

        def _dead_process(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(status.os, "kill", _dead_process)

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_accepts_gateway_metadata_when_cmdline_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.acquire_gateway_runtime_lock() is True
        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_accepts_script_style_gateway_cmdline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["/venv/bin/python", "/repo/hermes_cli/main.py", "gateway", "run", "--replace"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: "/venv/bin/python /repo/hermes_cli/main.py gateway run --replace",
        )

        assert status.acquire_gateway_runtime_lock() is True
        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_accepts_explicit_pid_path_without_cleanup(self, tmp_path, monkeypatch):
        other_home = tmp_path / "profile-home"
        other_home.mkdir()
        pid_path = other_home / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        lock_path = other_home / "gateway.lock"
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))
        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path=None: True)

        assert status.get_running_pid(pid_path, cleanup_stale=False) == os.getpid()
        assert pid_path.exists()

    def test_runtime_lock_claims_and_releases_liveness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        assert status.is_gateway_runtime_lock_active() is False
        assert status.acquire_gateway_runtime_lock() is True
        assert status.is_gateway_runtime_lock_active() is True

        status.release_gateway_runtime_lock()

        assert status.is_gateway_runtime_lock_active() is False

    def test_get_running_pid_treats_pid_file_as_stale_without_runtime_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": os.getpid(),
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        assert status.get_running_pid() is None
        assert not pid_path.exists()

    def test_get_running_pid_cleans_stale_metadata_from_dead_foreign_pid(self, tmp_path, monkeypatch):
        """Stale PID file from a *different* PID (crashed process) must still be cleaned.

        Regression for: ``remove_pid_file()`` defensively refuses to delete a
        PID file whose pid != ``os.getpid()`` to protect ``--replace``
        handoffs.  Stale-cleanup must not go through that path or real
        crashed-process PID files never get removed.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        lock_path = tmp_path / "gateway.lock"

        # PID that is guaranteed not alive and not our own.
        dead_foreign_pid = 999999
        assert dead_foreign_pid != os.getpid()

        pid_path.write_text(json.dumps({
            "pid": dead_foreign_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))
        lock_path.write_text(json.dumps({
            "pid": dead_foreign_pid,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        # No live lock holder → get_running_pid should clean both files.
        assert status.get_running_pid() is None
        assert not pid_path.exists()
        assert not lock_path.exists()

    def test_get_running_pid_falls_back_to_live_lock_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({
            "pid": 99999,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
        }))

        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)
        monkeypatch.setattr(
            status,
            "_build_pid_record",
            lambda: {
                "pid": os.getpid(),
                "kind": "hermes-gateway",
                "argv": ["python", "-m", "hermes_cli.main", "gateway"],
                "start_time": 123,
            },
        )
        assert status.acquire_gateway_runtime_lock() is True

        def fake_kill(pid, sig):
            if pid == 99999:
                raise ProcessLookupError
            return None

        monkeypatch.setattr(status.os, "kill", fake_kill)

        try:
            assert status.get_running_pid() == os.getpid()
        finally:
            status.release_gateway_runtime_lock()


class TestGatewayRuntimeStatus:
    def test_write_json_file_uses_atomic_json_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        calls = []

        def _fake_atomic_json_write(path, payload, **kwargs):
            calls.append((Path(path), payload, kwargs))

        monkeypatch.setattr(status, "atomic_json_write", _fake_atomic_json_write)

        payload = {"gateway_state": "running"}
        target = tmp_path / "gateway_state.json"
        status._write_json_file(target, payload)

        assert calls == [
            (
                target,
                payload,
                {"indent": None, "separators": (",", ":")},
            )
        ]

    def test_write_runtime_status_overwrites_stale_pid_on_restart(self, tmp_path, monkeypatch):
        """Regression: setdefault() preserved stale PID from previous process (#1631)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Simulate a previous gateway run that left a state file with a stale PID
        state_path = tmp_path / "gateway_state.json"
        state_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 1000.0,
            "kind": "hermes-gateway",
            "platforms": {},
            "updated_at": "2025-01-01T00:00:00Z",
        }))

        status.write_runtime_status(gateway_state="running")

        payload = status.read_runtime_status()
        assert payload["pid"] == os.getpid(), "PID should be overwritten, not preserved via setdefault"
        assert payload["start_time"] != 1000.0, "start_time should be overwritten on restart"

    def test_write_runtime_status_records_platform_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="telegram conflict",
            platform="telegram",
            platform_state="fatal",
            error_code="telegram_polling_conflict",
            error_message="another poller is active",
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "startup_failed"
        assert payload["exit_reason"] == "telegram conflict"
        assert payload["platforms"]["telegram"]["state"] == "fatal"
        assert payload["platforms"]["telegram"]["error_code"] == "telegram_polling_conflict"
        assert payload["platforms"]["telegram"]["error_message"] == "another poller is active"

    def test_write_runtime_status_explicit_none_clears_stale_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="stale error",
            platform="discord",
            platform_state="fatal",
            error_code="discord_timeout",
            error_message="stale platform error",
        )

        status.write_runtime_status(
            gateway_state="running",
            exit_reason=None,
            platform="discord",
            platform_state="connected",
            error_code=None,
            error_message=None,
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "running"
        assert payload["exit_reason"] is None
        assert payload["platforms"]["discord"]["state"] == "connected"
        assert payload["platforms"]["discord"]["error_code"] is None
        assert payload["platforms"]["discord"]["error_message"] is None


class TestTerminatePid:
    def test_force_uses_taskkill_on_windows(self, monkeypatch):
        calls = []
        monkeypatch.setattr(status, "_IS_WINDOWS", True)

        def fake_run(cmd, capture_output=False, text=False, timeout=None):
            calls.append((cmd, capture_output, text, timeout))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(status.subprocess, "run", fake_run)

        status.terminate_pid(123, force=True)

        assert calls == [
            (["taskkill", "/PID", "123", "/T", "/F"], True, True, 10)
        ]

    def test_force_falls_back_to_sigterm_when_taskkill_missing(self, monkeypatch):
        calls = []
        monkeypatch.setattr(status, "_IS_WINDOWS", True)

        def fake_run(*args, **kwargs):
            raise FileNotFoundError

        def fake_kill(pid, sig):
            calls.append((pid, sig))

        monkeypatch.setattr(status.subprocess, "run", fake_run)
        monkeypatch.setattr(status.os, "kill", fake_kill)

        status.terminate_pid(456, force=True)

        assert calls == [(456, status.signal.SIGTERM)]


class TestScopedLocks:
    def test_windows_file_lock_uses_high_offset(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "gateway.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        fd = handle.fileno()
        calls = []

        def fake_locking(fd, mode, size):
            calls.append((fd, mode, size, handle.tell()))

        monkeypatch.setattr(status, "_IS_WINDOWS", True)
        monkeypatch.setattr(
            status,
            "msvcrt",
            SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=fake_locking),
            raising=False,
        )

        try:
            assert status._try_acquire_file_lock(handle) is True
            status._release_file_lock(handle)
        finally:
            handle.close()

        assert calls == [
            (fd, 1, 1, status._WINDOWS_LOCK_OFFSET),
            (fd, 2, 1, status._WINDOWS_LOCK_OFFSET),
        ]
        assert lock_path.read_text(encoding="utf-8") == "\n"

    def test_acquire_scoped_lock_rejects_live_other_process(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 123,
            "kind": "hermes-gateway",
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing["pid"] == 99999

    def test_acquire_scoped_lock_replaces_stale_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 123,
            "kind": "hermes-gateway",
        }))

        def fake_kill(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(status.os, "kill", fake_kill)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "telegram"

    def test_acquire_scoped_lock_recovers_empty_lock_file(self, tmp_path, monkeypatch):
        """Empty lock file (0 bytes) left by a crashed process should be treated as stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "slack-app-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("")  # simulate crash between O_CREAT and json.dump

        acquired, existing = status.acquire_scoped_lock("slack-app-token", "secret", metadata={"platform": "slack"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "slack"

    def test_acquire_scoped_lock_recovers_corrupt_lock_file(self, tmp_path, monkeypatch):
        """Lock file with invalid JSON should be treated as stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "slack-app-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("{truncated")  # simulate partial write

        acquired, existing = status.acquire_scoped_lock("slack-app-token", "secret", metadata={"platform": "slack"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()

    def test_release_scoped_lock_only_removes_current_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))

        acquired, _ = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})
        assert acquired is True
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        assert lock_path.exists()

        status.release_scoped_lock("telegram-bot-token", "secret")
        assert not lock_path.exists()

    def test_release_all_scoped_locks_can_target_single_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        target_lock = lock_dir / "telegram-bot-token-target.lock"
        other_lock = lock_dir / "slack-app-token-other.lock"
        target_lock.write_text(json.dumps({
            "pid": 111,
            "start_time": 222,
            "kind": "hermes-gateway",
        }))
        other_lock.write_text(json.dumps({
            "pid": 999,
            "start_time": 333,
            "kind": "hermes-gateway",
        }))

        removed = status.release_all_scoped_locks(
            owner_pid=111,
            owner_start_time=222,
        )

        assert removed == 1
        assert not target_lock.exists()
        assert other_lock.exists()

    def test_release_all_scoped_locks_skips_pid_reuse_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        reused_pid_lock = lock_dir / "telegram-bot-token-reused.lock"
        reused_pid_lock.write_text(json.dumps({
            "pid": 111,
            "start_time": 999,
            "kind": "hermes-gateway",
        }))

        removed = status.release_all_scoped_locks(
            owner_pid=111,
            owner_start_time=222,
        )

        assert removed == 0
        assert reused_pid_lock.exists()


class TestProcessInfoHelpers:
    """Tests for cross-platform process info helpers (_get_process_start_time, _read_process_cmdline)."""

    def test_get_process_start_time_returns_int_for_current_process(self):
        """Should return an integer for the current (live) process on any platform."""
        result = status._get_process_start_time(os.getpid())
        assert result is None or isinstance(result, int)

    def test_get_process_start_time_ps_fallback_returns_int(self, monkeypatch):
        """When /proc is unavailable, ps lstart= fallback returns an integer."""
        import types
        from types import SimpleNamespace

        fake_run_result = SimpleNamespace(returncode=0, stdout="Thu Apr 24 14:48:56 2026\n")

        def fake_run(cmd, **kwargs):
            return fake_run_result

        # Force the /proc path to raise so the subprocess fallback runs.
        monkeypatch.setattr(status.subprocess, "run", fake_run)
        # Simulate non-Windows so the ps branch is entered.
        monkeypatch.setattr(status.sys, "platform", "darwin")

        # Directly test the fallback by calling with a PID whose /proc won't exist.
        # On macOS the /proc branch is always skipped; on Linux we mock subprocess.
        # We rely on _get_process_start_time gracefully handling the /proc miss.
        result = status._get_process_start_time(99999999)
        # The ps mock returns a valid date so we expect an int.
        assert isinstance(result, int)

    def test_get_process_start_time_ps_fallback_returns_none_when_ps_fails(self, monkeypatch):
        """ps returning non-zero (dead PID) → None."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            status.subprocess, "run",
            lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout=""),
        )
        monkeypatch.setattr(status.sys, "platform", "darwin")

        result = status._get_process_start_time(99999999)
        assert result is None

    def test_read_process_cmdline_ps_fallback(self, monkeypatch):
        """When /proc is unavailable, ps command= fallback returns the cmdline string."""
        from types import SimpleNamespace

        monkeypatch.setattr(
            status.subprocess, "run",
            lambda cmd, **kwargs: SimpleNamespace(
                returncode=0, stdout="python -m hermes_cli.main gateway run\n"
            ),
        )
        monkeypatch.setattr(status.sys, "platform", "darwin")

        result = status._read_process_cmdline(99999999)
        assert result == "python -m hermes_cli.main gateway run"

    def test_read_process_cmdline_ps_fallback_returns_none_on_dead_pid(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            status.subprocess, "run",
            lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout=""),
        )
        monkeypatch.setattr(status.sys, "platform", "darwin")

        result = status._read_process_cmdline(99999999)
        assert result is None


class TestScopedLocksStalePidReuse:
    """Tests for the PID-reuse stale-detection fix (issue #15115).

    On macOS, /proc does not exist so start_time is None in lock files written
    by older Hermes versions.  When a crashed gateway's PID is reused by a
    non-gateway process, the lock must be detected as stale via cmdline check.
    """

    def test_acquire_treats_as_stale_when_pid_reused_by_non_gateway(self, tmp_path, monkeypatch):
        """Lock with null start_time whose PID is held by a non-gateway process → stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "weixin-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Simulate a macOS-era lock: start_time is null because /proc wasn't available.
        lock_path.write_text(json.dumps({
            "pid": 78452,
            "start_time": None,
            "kind": "hermes-gateway",
        }))

        # PID 78452 is alive (reused by a non-gateway process).
        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        # Cmdline returns something that is NOT a gateway.
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "/usr/bin/some_daemon --start")

        acquired, _ = status.acquire_scoped_lock("weixin-bot-token", "secret", metadata={"platform": "weixin"})

        assert acquired is True, "Lock with reused PID (non-gateway cmdline) should be treated as stale"
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()

    def test_acquire_rejects_when_live_gateway_holds_lock_no_start_time(self, tmp_path, monkeypatch):
        """Lock with null start_time whose PID is a live gateway → NOT stale."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "weixin-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 78452,
            "start_time": None,
            "kind": "hermes-gateway",
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        # Cmdline confirms this IS a live gateway process.
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: "python -m hermes_cli.main gateway run --replace",
        )

        acquired, existing = status.acquire_scoped_lock("weixin-bot-token", "secret", metadata={"platform": "weixin"})

        assert acquired is False, "Live gateway holding lock must NOT be evicted"
        assert existing["pid"] == 78452

    def test_acquire_treats_as_stale_when_cmdline_unreadable_but_start_time_mismatch(self, tmp_path, monkeypatch):
        """Start-time mismatch should still win even when cmdline is unreadable."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "weixin-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 78452,
            "start_time": 1000,
            "kind": "hermes-gateway",
        }))

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        # Different start_time → PID was reused.
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 9999)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

        acquired, _ = status.acquire_scoped_lock("weixin-bot-token", "secret", metadata={"platform": "weixin"})

        assert acquired is True

    def test_acquire_does_not_trigger_cmdline_check_when_start_times_match(self, tmp_path, monkeypatch):
        """When start times match, cmdline check must NOT run (live owner confirmed)."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "weixin-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 78452,
            "start_time": 1000,
            "kind": "hermes-gateway",
        }))

        cmdline_calls = []

        def fake_read_cmdline(pid):
            cmdline_calls.append(pid)
            return "/usr/bin/some_daemon"  # non-gateway, but should not be reached

        monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 1000)
        monkeypatch.setattr(status, "_read_process_cmdline", fake_read_cmdline)

        acquired, existing = status.acquire_scoped_lock("weixin-bot-token", "secret")

        assert acquired is False
        assert not cmdline_calls, "cmdline check should not run when start_times match"


class TestTakeoverMarker:
    """Tests for the --replace takeover marker.

    The marker breaks the post-#5646 flap loop between two gateway services
    fighting for the same bot token. The replacer writes a file naming the
    target PID + start_time; the target's shutdown handler sees it and exits
    0 instead of 1, so systemd's Restart=on-failure doesn't revive it.
    """

    def test_write_marker_records_target_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        ok = status.write_takeover_marker(target_pid=12345)

        assert ok is True
        marker = tmp_path / ".gateway-takeover.json"
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["target_pid"] == 12345
        assert payload["target_start_time"] == 42
        assert payload["replacer_pid"] == os.getpid()
        assert "written_at" in payload

    def test_consume_returns_true_when_marker_names_self(self, tmp_path, monkeypatch):
        """Primary happy path: planned takeover is recognised."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Mark THIS process as the target
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_takeover_marker(target_pid=os.getpid())
        assert ok is True

        # Call consume as if this process just got SIGTERMed
        result = status.consume_takeover_marker_for_self()

        assert result is True
        # Marker must be unlinked after consumption
        assert not (tmp_path / ".gateway-takeover.json").exists()

    def test_consume_returns_false_for_different_pid(self, tmp_path, monkeypatch):
        """A marker naming a DIFFERENT process must not be consumed as ours."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        # Marker names a different PID
        other_pid = os.getpid() + 9999
        ok = status.write_takeover_marker(target_pid=other_pid)
        assert ok is True

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Marker IS unlinked even on non-match (the record has been consumed
        # and isn't relevant to us — leaving it around would grief a later
        # legitimate check).
        assert not (tmp_path / ".gateway-takeover.json").exists()

    def test_consume_returns_false_on_start_time_mismatch(self, tmp_path, monkeypatch):
        """PID reuse defence: old marker's start_time mismatches current process."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Marker says target started at time 100 with our PID
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        status.write_takeover_marker(target_pid=os.getpid())

        # Now change the reported start_time to simulate PID reuse
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 9999)

        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_returns_false_when_marker_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_returns_false_for_stale_marker(self, tmp_path, monkeypatch):
        """A marker older than 60s must be ignored."""
        from datetime import datetime, timezone, timedelta

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        # Hand-craft a marker written 2 minutes ago
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 123,
            "replacer_pid": 99999,
            "written_at": stale_time,
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Stale markers are unlinked so a later legit shutdown isn't griefed
        assert not marker_path.exists()

    def test_consume_handles_malformed_marker_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        marker_path.write_text("not valid json{")

        # Must not raise
        result = status.consume_takeover_marker_for_self()

        assert result is False

    def test_consume_handles_marker_with_missing_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        marker_path.write_text(json.dumps({"only_replacer_pid": 99999}))

        result = status.consume_takeover_marker_for_self()

        assert result is False
        # Malformed marker should be cleaned up
        assert not marker_path.exists()

    def test_clear_takeover_marker_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Nothing to clear — must not raise
        status.clear_takeover_marker()

        # Write then clear
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        status.write_takeover_marker(target_pid=12345)
        assert (tmp_path / ".gateway-takeover.json").exists()

        status.clear_takeover_marker()
        assert not (tmp_path / ".gateway-takeover.json").exists()

        # Clear again — still no error
        status.clear_takeover_marker()

    def test_write_marker_returns_false_on_write_failure(self, tmp_path, monkeypatch):
        """write_takeover_marker is best-effort; returns False but doesn't raise."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def raise_oserror(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(status, "_write_json_file", raise_oserror)

        ok = status.write_takeover_marker(target_pid=12345)

        assert ok is False

    def test_consume_ignores_marker_for_different_process_and_prevents_stale_grief(
        self, tmp_path, monkeypatch
    ):
        """Regression: a stale marker from a dead replacer naming a dead
        target must not accidentally cause an unrelated future gateway to
        exit 0 on legitimate SIGTERM.

        The distinguishing check is ``target_pid == our_pid AND
        target_start_time == our_start_time``. Different PID always wins.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-takeover.json"
        # Fresh marker (timestamp is recent) but names a totally different PID
        from datetime import datetime, timezone
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid() + 10000,
            "target_start_time": 42,
            "replacer_pid": 99999,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        result = status.consume_takeover_marker_for_self()

        # We are not the target — must NOT consume as planned
        assert result is False
