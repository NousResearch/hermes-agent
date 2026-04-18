"""Tests for gateway runtime status tracking."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from gateway import status
from gateway.config import Platform
from gateway.run import GatewayRunner


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

    def test_write_runtime_status_clears_stale_error_fields_when_passed_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="telegram conflict",
            platform="telegram",
            platform_state="fatal",
            error_code="telegram_polling_conflict",
            error_message="another poller is active",
        )
        status.write_runtime_status(
            gateway_state="running",
            exit_reason=None,
            platform="telegram",
            platform_state="connected",
            error_code=None,
            error_message=None,
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "running"
        assert payload["exit_reason"] is None
        assert payload["platforms"]["telegram"]["state"] == "connected"
        assert payload["platforms"]["telegram"]["error_code"] is None
        assert payload["platforms"]["telegram"]["error_message"] is None

    def test_write_runtime_status_can_reset_stale_platforms(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="startup_failed",
            exit_reason="stale failure",
            platform="telegram",
            platform_state="fatal",
            error_code="telegram_connect_error",
            error_message="timed out",
        )
        status.write_runtime_status(
            gateway_state="starting",
            exit_reason=None,
            reset_platforms=True,
        )

        payload = status.read_runtime_status()
        assert payload["gateway_state"] == "starting"
        assert payload["exit_reason"] is None
        assert payload["platforms"] == {}

    def test_write_runtime_status_persists_runtime_summary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        runtime_summary = {
            "active_sessions_count": 1,
            "background_jobs": {"active_count": 2, "counts": {"running": 2}},
        }

        status.write_runtime_status(
            gateway_state="running",
            runtime_summary=runtime_summary,
        )

        payload = status.read_runtime_status()
        assert payload["runtime_summary"] == runtime_summary

    def test_write_runtime_status_clears_runtime_summary_when_passed_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="running",
            runtime_summary={"active_sessions_count": 3},
        )
        status.write_runtime_status(
            gateway_state="stopped",
            runtime_summary=None,
        )

        payload = status.read_runtime_status()
        assert payload["runtime_summary"] == {}

    def test_write_runtime_status_preserves_extended_operator_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        runtime_summary = {
            "model": {
                "configured_model": "claude-opus-4.6",
                "active_model": "gpt-5.4-mini",
                "active_provider": "openrouter",
                "fallback_active": True,
                "fallback_pinned": False,
            },
            "approvals": {"pending_count": 2},
            "qq_monitoring": {
                "active_collect_only_groups": 1,
                "groups": [
                    {
                        "group_id": "726109087",
                        "group_name": "项目群",
                        "mode": "collect_only",
                        "worker_names": ["钢镚"],
                    }
                ],
            },
            "direct_shortcuts": {
                "recent_count": 1,
                "recent": [
                    {
                        "matched_handler": "_try_handle_admin_qq_group_control",
                        "text_preview": "停止QQ 群 192903718 的监听采集",
                    }
                ],
            },
        }

        status.write_runtime_status(
            gateway_state="running",
            runtime_summary=runtime_summary,
        )

        payload = status.read_runtime_status()
        assert payload["runtime_summary"] == runtime_summary
        assert payload["runtime_health"]["status"] == "healthy"
        assert payload["runtime_health"]["summary"] == "runtime canary healthy"

    def test_write_runtime_status_persists_runtime_health_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        status.write_runtime_status(
            gateway_state="running",
            runtime_summary={
                "model": {
                    "active_provider": "custom",
                    "active_model": "gpt-5.4",
                    "degraded_provider": "custom",
                    "degraded_model": "gpt-5.4",
                    "degraded_reason": "empty_response",
                    "degraded_failures": 3,
                    "degraded_cooldown_until": "2099-01-01T00:00:00+00:00",
                }
            },
        )

        payload = status.read_runtime_status()
        assert payload["runtime_health"]["healthy"] is False
        assert payload["runtime_health"]["status"] == "warning"
        assert payload["runtime_health"]["issue_count"] == 1
        assert payload["runtime_health"]["issues"][0]["code"] == "provider_degraded"


class TestGatewayRuntimeModelSummary:
    def test_build_runtime_model_summary_includes_primary_degraded_state(self):
        runner = object.__new__(GatewayRunner)
        runner._effective_model = None
        runner._effective_provider = None
        runner._running_agents = {}
        runner._agent_cache = {}

        with (
            patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "provider": "custom",
                    "base_url": "https://pay.kxaug.xyz/v1",
                    "api_mode": "chat_completions",
                },
            ),
            patch(
                "run_agent.get_provider_health_snapshot",
                return_value={
                    "count": 1,
                    "runtimes": [
                        {
                            "provider": "custom",
                            "model": "gpt-5.4",
                            "base_url": "https://pay.kxaug.xyz/v1",
                            "api_mode": "chat_completions",
                            "reason": "empty_response",
                            "cooldown_seconds": 118.7,
                            "updated_at": 1000.0,
                        }
                    ],
                },
            ),
        ):
            summary = GatewayRunner._build_runtime_model_summary(runner)

        assert summary["configured_model"] == "gpt-5.4"
        assert summary["configured_provider"] == "custom"
        assert summary["configured_base_url"] == "https://pay.kxaug.xyz/v1"
        assert summary["primary_degraded"] is True
        assert summary["primary_degraded_reason"] == "empty_response"
        assert summary["primary_degraded_cooldown_seconds"] == 118
        assert summary["degraded_runtime_count"] == 1
        assert summary["degraded_runtimes"][0]["base_url"] == "https://pay.kxaug.xyz/v1"


class TestGatewayRuntimeSnapshotHeartbeat:
    def test_write_runtime_status_snapshot_refreshes_live_platform_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        runner = object.__new__(GatewayRunner)
        runner._running = True
        runner.adapters = {Platform.QQ_NAPCAT: SimpleNamespace()}
        runner._build_runtime_status_summary = lambda: {}

        status.write_runtime_status(
            gateway_state="running",
            platform="qq_napcat",
            platform_state="connected",
            error_code=None,
            error_message=None,
        )
        initial_payload = status.read_runtime_status()
        initial_updated_at = initial_payload["platforms"]["qq_napcat"]["updated_at"]

        runner._write_runtime_status_snapshot()

        payload = status.read_runtime_status()
        assert payload["platforms"]["qq_napcat"]["state"] == "connected"
        assert payload["platforms"]["qq_napcat"]["updated_at"] >= initial_updated_at


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

    def test_release_scoped_lock_only_removes_current_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))

        acquired, _ = status.acquire_scoped_lock("telegram-bot-token", "secret", metadata={"platform": "telegram"})
        assert acquired is True
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        assert lock_path.exists()

        status.release_scoped_lock("telegram-bot-token", "secret")
        assert not lock_path.exists()
