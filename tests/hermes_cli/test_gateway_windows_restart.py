"""Tests for gateway_windows_restart.py — coordinator logic."""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


@pytest.fixture
def coordinator_env(tmp_path, monkeypatch):
    """Set up environment for coordinator tests."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "run").mkdir()
    (hermes_home / "logs").mkdir()
    (hermes_home / "profiles").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: str(hermes_home))

    return hermes_home


class TestPreflight:
    def test_preflight_passes_with_valid_env(self, coordinator_env, monkeypatch):
        """Preflight should pass when all dependencies are available."""
        monkeypatch.setattr(sys, "platform", "win32")

        mock_gw = MagicMock()
        mock_gw.get_task_name.return_value = "Hermes_Gateway"
        mock_gw.is_task_registered.return_value = True
        mock_gw._derive_venv_pythonw.return_value = sys.executable  # pretend pythonw exists

        # Mock _get_restart_base to use tmp_path
        mock_restart_state = MagicMock()
        restart_base = coordinator_env / "run" / "gateway-restart"
        restart_base.mkdir(parents=True, exist_ok=True)
        mock_restart_state._get_restart_base.return_value = restart_base
        logs_dir = coordinator_env / "logs"
        mock_restart_state._get_logs_dir.return_value = logs_dir

        with patch.dict("sys.modules", {
            "hermes_cli.gateway_windows": mock_gw,
            "hermes_cli.gateway_windows_restart_worker": MagicMock(),
            "hermes_cli.gateway_restart_state": mock_restart_state,
        }):
            from hermes_cli.gateway_windows_restart import preflight_check
            ok, detail = preflight_check(profile="default", target_pid=1234)
            assert ok is True

    def test_preflight_fails_without_pythonw(self, coordinator_env, monkeypatch):
        """Preflight should fail when pythonw.exe is not found."""
        monkeypatch.setattr(sys, "platform", "win32")

        # Mock _derive_venv_pythonw to return None (no pythonw)
        mock_gw = MagicMock()
        mock_gw._derive_venv_pythonw.return_value = None
        mock_gw.get_task_name.return_value = "Hermes_Gateway"

        # Mock _get_restart_base to use tmp_path
        mock_restart_state = MagicMock()
        restart_base = coordinator_env / "run" / "gateway-restart"
        restart_base.mkdir(parents=True, exist_ok=True)
        mock_restart_state._get_restart_base.return_value = restart_base
        logs_dir = coordinator_env / "logs"
        mock_restart_state._get_logs_dir.return_value = logs_dir

        with patch.dict("sys.modules", {
            "hermes_cli.gateway_windows": mock_gw,
            "hermes_cli.gateway_windows_restart_worker": MagicMock(),
            "hermes_cli.gateway_restart_state": mock_restart_state,
        }):
            from hermes_cli.gateway_windows_restart import preflight_check as pc
            ok, detail = pc(profile="default", target_pid=1234)
            assert ok is False
            assert "pythonw" in detail.lower()


class TestScheduleRestartHandoff:
    def test_returns_request_id(self, coordinator_env, monkeypatch):
        """schedule_restart_handoff should return a request_id."""
        monkeypatch.setattr(sys, "platform", "win32")

        mock_lock = MagicMock()
        mock_lock.try_acquire.return_value = True

        mock_restart_state = MagicMock()
        mock_restart_state.RestartLock.return_value = mock_lock
        mock_restart_state.create_intent.return_value = {
            "request_id": "dddddddd-dddd-dddd-dddd-dddddddddddd", "profile": "default",
        }
        mock_restart_state.write_status.return_value = None
        mock_restart_state.cleanup_intent.return_value = None
        mock_restart_state.append_restart_log.return_value = None

        mock_gw = MagicMock()
        mock_gw.get_task_name.return_value = "Hermes_Gateway"

        mock_gateway_status = MagicMock()
        mock_gateway_status.get_running_pid.return_value = 1234

        with patch.dict("sys.modules", {
            "hermes_cli.gateway_windows": mock_gw,
            "hermes_cli.gateway_restart_state": mock_restart_state,
            "gateway": MagicMock(),
            "gateway.status": mock_gateway_status,
        }):
            import hermes_cli.gateway_windows_restart as mod

            # Save originals to restore after test
            orig_preflight = mod.preflight_check
            orig_spawn = mod._spawn_worker
            orig_wait_claim = mod._wait_for_worker_claim
            orig_read_lease = mod._read_lease_data
            orig_wait_comp = mod._wait_for_completion
            orig_read_final = mod._read_final_status
            try:
                mod.preflight_check = lambda **kw: (True, "ok")
                mod._spawn_worker = lambda intent, profile, request_id: 5678
                mod._wait_for_worker_claim = lambda profile, request_id, timeout_s=10.0: True
                mod._read_lease_data = lambda profile, request_id: {
                    "request_id": request_id, "owner_token": "lease-token",
                    "worker_pid": 5678,
                }
                mod._wait_for_completion = lambda profile, timeout_s, request_id="": (True, "completed")
                mod._read_final_status = lambda profile, request_id: {
                    "state": "completed", "new_pid": 5678, "launcher": "direct_spawn",
                }

                result = mod.schedule_restart_handoff(origin="test", wait=True)
                assert "request_id" in result
                assert result["scheduled"] is True
            finally:
                mod.preflight_check = orig_preflight
                mod._spawn_worker = orig_spawn
                mod._wait_for_worker_claim = orig_wait_claim
                mod._read_lease_data = orig_read_lease
                mod._wait_for_completion = orig_wait_comp
                mod._read_final_status = orig_read_final

    def test_intermediate_states_do_not_report_success(self, coordinator_env, monkeypatch):
        """P1-2: intermediate states like 'draining' should not be treated as success."""
        monkeypatch.setattr(sys, "platform", "win32")

        mock_lock = MagicMock()
        mock_lock.try_acquire.return_value = True

        mock_restart_state = MagicMock()
        mock_restart_state.RestartLock.return_value = mock_lock
        mock_restart_state.create_intent.return_value = {
            "request_id": "dddddddd-dddd-dddd-dddd-dddddddddddd", "profile": "default",
        }
        mock_restart_state.write_status.return_value = None
        mock_restart_state.cleanup_intent.return_value = None
        mock_restart_state.append_restart_log.return_value = None

        mock_gw = MagicMock()
        mock_gw.get_task_name.return_value = "Hermes_Gateway"

        mock_gateway_status = MagicMock()
        mock_gateway_status.get_running_pid.return_value = 1234

        with patch.dict("sys.modules", {
            "hermes_cli.gateway_windows": mock_gw,
            "hermes_cli.gateway_restart_state": mock_restart_state,
            "gateway": MagicMock(),
            "gateway.status": mock_gateway_status,
        }):
            import hermes_cli.gateway_windows_restart as mod

            orig_preflight = mod.preflight_check
            orig_spawn = mod._spawn_worker
            orig_wait_claim = mod._wait_for_worker_claim
            orig_read_lease = mod._read_lease_data
            orig_wait_comp = mod._wait_for_completion
            orig_read_final = mod._read_final_status
            try:
                mod.preflight_check = lambda **kw: (True, "ok")
                mod._spawn_worker = lambda intent, profile, request_id: 5678
                mod._wait_for_worker_claim = lambda profile, request_id, timeout_s=10.0: True
                mod._read_lease_data = lambda profile, request_id: {
                    "request_id": request_id, "owner_token": "lease-token",
                    "worker_pid": 5678,
                }
                # _wait_for_completion returns False with intermediate state
                mod._wait_for_completion = lambda profile, timeout_s, request_id="": (False, "draining")
                # _read_final_status returns an intermediate state
                mod._read_final_status = lambda profile, request_id: {"state": "draining"}

                result = mod.schedule_restart_handoff(origin="test", wait=True)
                assert result["scheduled"] is True
                assert result["completed"] is False
                # Intermediate state should not produce a success message
                assert "successfully" not in result["detail"].lower()
            finally:
                mod.preflight_check = orig_preflight
                mod._spawn_worker = orig_spawn
                mod._wait_for_worker_claim = orig_wait_claim
                mod._read_lease_data = orig_read_lease
                mod._wait_for_completion = orig_wait_comp
                mod._read_final_status = orig_read_final


class TestWorkerSpawn:
    def test_worker_env_cleans_hermes_gateway(self, coordinator_env, monkeypatch):
        """Worker spawn should remove _HERMES_GATEWAY from env."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("_HERMES_GATEWAY", "1")

        captured_env = {}
        captured_argv = []

        def fake_popen(argv, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            captured_argv.extend(argv)
            proc = MagicMock()
            proc.pid = 9999
            return proc

        import subprocess
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        mock_gw = MagicMock()
        mock_gw._derive_venv_pythonw.return_value = "pythonw.exe"

        with patch.dict("sys.modules", {
            "hermes_cli.gateway_windows": mock_gw,
        }):
            from hermes_cli.gateway_windows_restart import _spawn_worker
            intent = {"request_id": "dddddddd-dddd-dddd-dddd-dddddddddddd", "profile": "default"}
            pid = _spawn_worker(intent, "default", "dddddddd-dddd-dddd-dddd-dddddddddddd")

            assert "_HERMES_GATEWAY" not in captured_env
            assert captured_env.get("HERMES_GATEWAY_RESTART_WORKER") == "1"
            assert pid == 9999
            # Worker is now invoked with --profile and --request-id
            assert "--profile" in captured_argv
            assert "default" in captured_argv
            assert "--request-id" in captured_argv
            assert "dddddddd-dddd-dddd-dddd-dddddddddddd" in captured_argv


# ---------------------------------------------------------------------------
# P0-3: Coordinator init exception releases active.lock
# ---------------------------------------------------------------------------

class TestCoordinatorInitFailure:
    """P0-3: Coordinator initialization failure releases active.lock."""

    def test_create_intent_failure_releases_lock(self, coordinator_env, monkeypatch):
        """create_intent() OSError → active.lock released."""
        from hermes_cli.gateway_restart_state import RestartLock, lock_path
        import hermes_cli.gateway_windows_restart as mod

        monkeypatch.setattr(sys, "platform", "win32")
        mod.preflight_check = lambda **kw: (True, "ok")

        import hermes_cli.gateway_restart_state as state_mod
        monkeypatch.setattr(state_mod, "create_intent",
                          MagicMock(side_effect=OSError("disk full")))
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 1234)
        monkeypatch.setattr("hermes_cli.gateway_windows.get_task_name", lambda: "Hermes_Gateway")

        result = mod.schedule_restart_handoff(origin="test", wait=False)
        assert result["scheduled"] is False
        assert not lock_path("default").exists()

    def test_write_status_failure_releases_lock(self, coordinator_env, monkeypatch):
        """write_status("scheduled") OSError → active.lock released."""
        from hermes_cli.gateway_restart_state import lock_path
        import hermes_cli.gateway_windows_restart as mod

        monkeypatch.setattr(sys, "platform", "win32")
        mod.preflight_check = lambda **kw: (True, "ok")

        call_count = [0]
        def failing_write_status(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call is "scheduled"
                raise OSError("disk full")

        import hermes_cli.gateway_restart_state as state_mod
        monkeypatch.setattr(state_mod, "write_status", failing_write_status)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 1234)
        monkeypatch.setattr("hermes_cli.gateway_windows.get_task_name", lambda: "Hermes_Gateway")

        result = mod.schedule_restart_handoff(origin="test", wait=False)
        assert result["scheduled"] is False
        assert not lock_path("default").exists()

    def test_spawn_worker_failure_releases_lock(self, coordinator_env, monkeypatch):
        """_spawn_worker() exception → active.lock released + cleanup."""
        from hermes_cli.gateway_restart_state import lock_path
        import hermes_cli.gateway_windows_restart as mod

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(mod, "preflight_check", lambda **kw: (True, "ok"))
        monkeypatch.setattr(mod, "_spawn_worker", MagicMock(side_effect=OSError("spawn failed")))
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 1234)
        monkeypatch.setattr("hermes_cli.gateway_windows.get_task_name", lambda: "Hermes_Gateway")

        result = mod.schedule_restart_handoff(origin="test", wait=False)
        assert result["scheduled"] is False
        assert not lock_path("default").exists()


# ---------------------------------------------------------------------------
# P1-3: Handoff failure returns clear status
# ---------------------------------------------------------------------------

class TestHandoffFailure:
    """P1-3: handoff failure returns clear status."""

    def test_handoff_failure_returns_not_scheduled(self, coordinator_env, monkeypatch):
        """handoff_active_lock() returns False → scheduled=False."""
        import hermes_cli.gateway_windows_restart as mod
        import hermes_cli.gateway_restart_state as state_mod

        monkeypatch.setattr(sys, "platform", "win32")
        orig_preflight = mod.preflight_check
        orig_spawn = mod._spawn_worker
        orig_wait_claim = mod._wait_for_worker_claim
        orig_read_lease = mod._read_lease_data
        try:
            mod.preflight_check = lambda **kw: (True, "ok")
            mod._spawn_worker = lambda *a, **kw: 5678
            mod._wait_for_worker_claim = lambda *a, **kw: True
            mod._read_lease_data = lambda *a, **kw: {"worker_pid": 5678, "owner_token": "bad"}
            monkeypatch.setattr("gateway.status.get_running_pid", lambda: 1234)
            monkeypatch.setattr("hermes_cli.gateway_windows.get_task_name", lambda: "Hermes_Gateway")

            # Make handoff_active_lock return False
            monkeypatch.setattr(state_mod.RestartLock, "handoff_active_lock", lambda *a, **kw: False)

            result = mod.schedule_restart_handoff(origin="test", wait=False)
            assert result["scheduled"] is False
            assert "handoff" in result.get("detail", "").lower()
        finally:
            mod.preflight_check = orig_preflight
            mod._spawn_worker = orig_spawn
            mod._wait_for_worker_claim = orig_wait_claim
            mod._read_lease_data = orig_read_lease

    def test_lease_disappears_handoff_failure(self, coordinator_env, monkeypatch):
        """lease.json disappears → no handoff, scheduled=False."""
        import hermes_cli.gateway_windows_restart as mod

        monkeypatch.setattr(sys, "platform", "win32")
        orig_preflight = mod.preflight_check
        orig_spawn = mod._spawn_worker
        orig_wait_claim = mod._wait_for_worker_claim
        orig_read_lease = mod._read_lease_data
        try:
            mod.preflight_check = lambda **kw: (True, "ok")
            mod._spawn_worker = lambda *a, **kw: 5678
            mod._wait_for_worker_claim = lambda *a, **kw: True
            mod._read_lease_data = lambda *a, **kw: None  # lease disappeared
            monkeypatch.setattr("gateway.status.get_running_pid", lambda: 1234)
            monkeypatch.setattr("hermes_cli.gateway_windows.get_task_name", lambda: "Hermes_Gateway")

            result = mod.schedule_restart_handoff(origin="test", wait=False)
            assert result["scheduled"] is False
        finally:
            mod.preflight_check = orig_preflight
            mod._spawn_worker = orig_spawn
            mod._wait_for_worker_claim = orig_wait_claim
            mod._read_lease_data = orig_read_lease
