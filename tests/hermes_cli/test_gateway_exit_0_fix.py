"""Regression tests for PR #73210 — exit 0 when gateway already running.

Core contributor review (PR #73210, review #4820521856) requested:
1. Exit code tests for both CLI preflight guards
2. Async start_gateway() test for existing-PID return value
"""

import pytest
from unittest.mock import MagicMock

import hermes_cli.gateway as gateway


# ── _guard_supervised_gateway_conflict ──────────────────────────


class TestGuardSupervisedConflict:
    """Exit contract: SystemExit(0) when a supervised gateway is already running."""

    def test_exits_0_when_supervised(self, monkeypatch):
        """Installed+running service manager → sys.exit(0)."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        snapshot = gateway.GatewayRuntimeSnapshot(
            manager="systemd (user)", service_installed=True, service_running=True
        )
        monkeypatch.setattr(gateway, "get_gateway_runtime_snapshot", lambda: snapshot)

        with pytest.raises(SystemExit) as exc:
            gateway._guard_supervised_gateway_conflict()
        assert exc.value.code == 0

    def test_returns_none_when_no_service(self, monkeypatch):
        """No installed service → returns None."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        snapshot = gateway.GatewayRuntimeSnapshot(
            manager="systemd (user)", service_installed=False, service_running=False
        )
        monkeypatch.setattr(gateway, "get_gateway_runtime_snapshot", lambda: snapshot)

        assert gateway._guard_supervised_gateway_conflict() is None

    def test_returns_none_when_probe_fails(self, monkeypatch):
        """Probe exception → returns None (best-effort)."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        monkeypatch.setattr(
            gateway, "get_gateway_runtime_snapshot",
            lambda: (_ for _ in ()).throw(Exception("probe failed")),
        )

        assert gateway._guard_supervised_gateway_conflict() is None

    def test_returns_none_when_force(self, monkeypatch):
        """--force bypasses the guard."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        assert gateway._guard_supervised_gateway_conflict(force=True) is None

    def test_returns_none_when_already_under_supervisor(self, monkeypatch):
        """Under supervisor → skip guard."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: True)
        assert gateway._guard_supervised_gateway_conflict() is None


# ── _guard_existing_gateway_process_conflict ────────────────────


class TestGuardExistingProcessConflict:
    """Exit contract: SystemExit(0) when a PID file exists."""

    def test_exits_0_when_pid_exists(self, monkeypatch):
        """Existing PID → sys.exit(0)."""
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 12345)

        with pytest.raises(SystemExit) as exc:
            gateway._guard_existing_gateway_process_conflict()
        assert exc.value.code == 0

    def test_returns_none_when_no_pid(self, monkeypatch):
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)

        assert gateway._guard_existing_gateway_process_conflict() is None

    def test_returns_none_when_probe_fails(self, monkeypatch):
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: (_ for _ in ()).throw(Exception("probe failed")),
        )

        assert gateway._guard_existing_gateway_process_conflict() is None

    def test_returns_none_when_replace(self, monkeypatch):
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: False)
        assert gateway._guard_existing_gateway_process_conflict(replace=True) is None

    def test_returns_none_when_under_supervisor(self, monkeypatch):
        monkeypatch.setattr(gateway, "_running_under_gateway_supervisor", lambda: True)
        assert gateway._guard_existing_gateway_process_conflict() is None


# ── start_gateway existing-PID branch ──────────────────────────


@pytest.mark.asyncio
class TestStartGatewayExistingPidBranch:
    """Return contract: True (not False) when an existing PID is found."""

    async def test_returns_true_when_existing_pid(self, monkeypatch, tmp_path):
        """An existing PID → start_gateway returns True (not False)."""
        class FakeRunner:
            def __init__(self, *args, **kwargs): pass
            async def start(self): return True
            def stop(self): pass
            should_exit_cleanly = False
            _running = True

        monkeypatch.setattr("gateway.run.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 123_456)
        monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet: None)
        monkeypatch.setattr("gateway.run.GatewayRunner", FakeRunner)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [123])
        monkeypatch.setattr(gateway, "_profile_suffix", lambda: "")
        
        # Add a mock to track GatewayRunner constructor calls
        mock_gateway_runner = MagicMock(side_effect=FakeRunner)
        monkeypatch.setattr("gateway.run.GatewayRunner", mock_gateway_runner)

        from gateway.run import start_gateway

        result = await start_gateway()
        assert result is True
        
        # Assert that GatewayRunner was NOT called
        mock_gateway_runner.assert_not_called()

    async def test_returns_false_on_genuine_failure(self, monkeypatch, tmp_path):
        """A genuine startup failure should still return False."""

        async def failing_start_gateway(*args, **kwargs):
            # Simulate a real failure inside the gateway logic
            return False

        monkeypatch.setattr("gateway.run.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
        monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet: None)
        monkeypatch.setattr("hermes_logging.setup_logging", lambda **kw: None)
        monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
        
        # Mock GatewayRunner to simulate an internal startup failure
        class FakeFailingRunner:
            def __init__(self, *args, **kwargs): pass
            async def start(self): return False
            def stop(self): pass
            should_exit_cleanly = False
            _running = False
        
        monkeypatch.setattr("gateway.run.GatewayRunner", FakeFailingRunner)

        from gateway.run import start_gateway
        
        result = await start_gateway()
        assert result is False
