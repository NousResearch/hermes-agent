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

    def test_write_runtime_status_overwrites_stale_argv_on_restart(self, tmp_path, monkeypatch):
        """Regression: gateway_state.json must not keep the previous launch argv."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        state_path = tmp_path / "gateway_state.json"
        state_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": 1000.0,
            "kind": "hermes-gateway",
            "argv": ["/old/path/hermes", "gateway", "run"],
            "platforms": {},
            "updated_at": "2025-01-01T00:00:00Z",
        }))

        monkeypatch.setattr(status.sys, "argv", ["/new/path/hermes", "gateway", "run"])
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 2000)

        status.write_runtime_status(gateway_state="running")

        payload = status.read_runtime_status()
        assert payload["argv"] == ["/new/path/hermes", "gateway", "run"]
        assert payload["pid"] == os.getpid()
        assert payload["start_time"] == 2000

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

        # Post-#21561 the liveness probe routes through
        # ``gateway.status._pid_exists`` (psutil-first, safe on Windows).
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is False
        assert existing["pid"] == 99999

    def test_acquire_scoped_lock_replaces_pid_reused_by_unrelated_process(self, tmp_path, monkeypatch):
        """macOS regression: PID reused by an unrelated process with start_time=None.

        On macOS /proc is unavailable, so both the lock record and the live
        process report start_time=None.  The live PID is alive (os.kill
        succeeds) but belongs to a completely different program.  The lock
        must be treated as stale.
        """
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 873,
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["/Users/user/.hermes/hermes-agent/hermes_cli/main.py", "gateway", "run", "--replace"],
        }))

        # Post-#21561 the liveness probe routes through
        # ``gateway.status._pid_exists`` (psutil-first, safe on Windows),
        # not ``os.kill``.
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: False)

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})

        assert acquired is True
        payload = json.loads(lock_path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["metadata"]["platform"] == "telegram"

    def test_acquire_scoped_lock_keeps_lock_when_pid_reused_by_gateway(self, tmp_path, monkeypatch):
        """When start_time is None but the live PID still looks like a gateway, keep the lock."""
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "pid": 99999,
            "start_time": None,
            "kind": "hermes-gateway",
            "argv": ["/Users/user/.hermes/hermes-agent/hermes_cli/main.py", "gateway", "run", "--replace"],
        }))

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

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

        # Post-#21561: simulate "PID gone" via _pid_exists returning False.
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)

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


class TestPlannedStopMarker:
    """Tests for intentional service/manual gateway stop markers."""

    def test_write_marker_records_target_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

        ok = status.write_planned_stop_marker(target_pid=12345)

        assert ok is True
        marker = tmp_path / ".gateway-planned-stop.json"
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["target_pid"] == 12345
        assert payload["target_start_time"] == 42
        assert payload["stopper_pid"] == os.getpid()
        assert "written_at" in payload

    def test_consume_returns_true_when_marker_names_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_planned_stop_marker(target_pid=os.getpid())
        assert ok is True

        result = status.consume_planned_stop_marker_for_self()

        assert result is True
        assert not (tmp_path / ".gateway-planned-stop.json").exists()

    def test_consume_returns_false_for_different_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)
        ok = status.write_planned_stop_marker(target_pid=os.getpid() + 9999)
        assert ok is True

        result = status.consume_planned_stop_marker_for_self()

        assert result is False
        assert not (tmp_path / ".gateway-planned-stop.json").exists()

    def test_consume_returns_false_for_stale_marker(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone, timedelta

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker_path = tmp_path / ".gateway-planned-stop.json"
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        marker_path.write_text(json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": 123,
            "stopper_pid": 99999,
            "written_at": stale_time,
        }))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 123)

        result = status.consume_planned_stop_marker_for_self()

        assert result is False
        assert not marker_path.exists()

    def test_clear_planned_stop_marker_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 100)

        status.clear_planned_stop_marker()
        status.write_planned_stop_marker(target_pid=12345)
        assert (tmp_path / ".gateway-planned-stop.json").exists()

        status.clear_planned_stop_marker()

        assert not (tmp_path / ".gateway-planned-stop.json").exists()
        status.clear_planned_stop_marker()

    def test_write_marker_returns_false_on_write_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def raise_oserror(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(status, "_write_json_file", raise_oserror)

        ok = status.write_planned_stop_marker(target_pid=12345)

        assert ok is False


class TestReadProcessCmdlinePsFallback:
    """Tests for _read_process_cmdline falling back to ps on non-Linux."""

    def test_ps_fallback_when_proc_unavailable(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="/usr/libexec/bluetoothuserd\n"),
        )
        result = status._read_process_cmdline(873)
        assert result == "/usr/libexec/bluetoothuserd"

    def test_ps_fallback_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
        )
        result = status._read_process_cmdline(99999)
        assert result is None

    def test_proc_cmdline_takes_priority_over_ps(self, monkeypatch):
        calls = []

        def fake_read_bytes(self):
            calls.append("proc")
            return b"python\x00hermes_cli/main.py\x00gateway\x00"

        monkeypatch.setattr(status.Path, "read_bytes", fake_read_bytes)
        result = status._read_process_cmdline(12345)
        assert "hermes_cli/main.py" in result
        assert calls == ["proc"]

    def test_ps_fallback_used_when_proc_returns_empty(self, monkeypatch):
        monkeypatch.setattr(status.Path, "read_bytes", lambda self: b"")
        monkeypatch.setattr(
            status.subprocess, "run",
            lambda args, **kwargs: SimpleNamespace(returncode=0, stdout="python hermes_cli/main.py gateway run\n"),
        )
        result = status._read_process_cmdline(12345)
        assert "hermes_cli/main.py" in result
