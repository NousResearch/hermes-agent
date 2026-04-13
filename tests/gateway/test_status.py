"""Tests for gateway runtime status tracking."""

import json
import os
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

    def test_get_running_pid_rejects_live_non_gateway_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(str(os.getpid()))

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

        assert status.get_running_pid() == os.getpid()

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

        assert status.get_running_pid() == os.getpid()


class TestGatewayRuntimeStatus:
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
