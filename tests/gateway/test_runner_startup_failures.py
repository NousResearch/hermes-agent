import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _RetryableFailureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        self._set_fatal_error(
            "telegram_connect_error",
            "Telegram startup failed: temporary DNS resolution failure.",
            retryable=True,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _DisabledAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=False, token="***"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        raise AssertionError("connect should not be called for disabled platforms")

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SuccessfulAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_returns_failure_for_retryable_startup_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _RetryableFailureAdapter())

    ok = await runner.start()

    assert ok is False
    assert runner.should_exit_cleanly is False
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    assert "temporary DNS resolution failure" in state["exit_reason"]
    assert state["platforms"]["telegram"]["state"] == "retrying"
    assert state["platforms"]["telegram"]["error_code"] == "telegram_connect_error"


@pytest.mark.asyncio
async def test_runner_allows_cron_only_mode_when_no_platforms_are_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.adapters == {}
    state = read_runtime_status()
    assert state["gateway_state"] == "running"


@pytest.mark.asyncio
async def test_runner_records_connected_platform_state_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _SuccessfulAdapter())
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    ok = await runner.start()

    assert ok is True
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["platforms"]["discord"]["state"] == "connected"
    assert state["platforms"]["discord"]["error_code"] is None
    assert state["platforms"]["discord"]["error_message"] is None


@pytest.mark.asyncio
async def test_start_gateway_verbosity_imports_redacting_formatter(monkeypatch, tmp_path):
    """Verbosity != None must not crash with NameError on RedactingFormatter (#8044)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    # verbosity=1 triggers the code path that uses RedactingFormatter.
    # Before the fix this raised NameError.
    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=1)

    assert ok is True


@pytest.mark.asyncio
async def test_start_gateway_replace_force_uses_terminate_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls = []

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda: 0)
    monkeypatch.setattr("gateway.status.terminate_pid", lambda pid, force=False: calls.append((pid, force)))
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("gateway.run.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    assert calls == [(42, False), (42, True)]


@pytest.mark.asyncio
async def test_start_gateway_replace_writes_takeover_marker_before_sigterm(
    monkeypatch, tmp_path
):
    """--replace must write a takeover marker BEFORE sending SIGTERM.

    The marker lets the target's shutdown handler identify the signal as a
    planned takeover (→ exit 0) rather than an unexpected kill (→ exit 1).
    Without the marker, PR #5646's signal-recovery path would revive the
    target via systemd Restart=on-failure, starting a flap loop.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Record the ORDER of marker-write + terminate_pid calls
    events: list[str] = []
    marker_paths_seen: list = []

    def record_write_marker(target_pid: int) -> bool:
        events.append(f"write_marker(target_pid={target_pid})")
        # Also check that the marker file actually exists after this call
        marker_paths_seen.append(
            (tmp_path / ".gateway-takeover.json").exists() is False  # not yet
        )
        # Actually write the marker so we can verify cleanup later
        from gateway.status import _get_takeover_marker_path, _write_json_file, _get_process_start_time
        _write_json_file(_get_takeover_marker_path(), {
            "target_pid": target_pid,
            "target_start_time": 0,
            "replacer_pid": 100,
            "written_at": "2026-04-17T00:00:00+00:00",
        })
        return True

    def record_terminate(pid, force=False):
        events.append(f"terminate_pid(pid={pid}, force={force})")

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_all_scoped_locks", lambda: 0)
    monkeypatch.setattr("gateway.status.write_takeover_marker", record_write_marker)
    monkeypatch.setattr("gateway.status.terminate_pid", record_terminate)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    # Simulate old process exiting on first check so we don't loop into force-kill
    monkeypatch.setattr(
        "gateway.run.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    # Ordering: marker written BEFORE SIGTERM
    assert events[0] == "write_marker(target_pid=42)"
    assert any(e.startswith("terminate_pid(pid=42") for e in events[1:])
    # Marker file cleanup: replacer cleans it after loop completes
    assert not (tmp_path / ".gateway-takeover.json").exists()


@pytest.mark.asyncio
async def test_start_gateway_replace_clears_marker_on_permission_denied(
    monkeypatch, tmp_path
):
    """If we fail to kill the existing PID (permission denied), clean up the
    marker so it doesn't grief an unrelated future shutdown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write_marker(target_pid: int) -> bool:
        from gateway.status import _get_takeover_marker_path, _write_json_file
        _write_json_file(_get_takeover_marker_path(), {
            "target_pid": target_pid,
            "target_start_time": 0,
            "replacer_pid": 100,
            "written_at": "2026-04-17T00:00:00+00:00",
        })
        return True

    def raise_permission(pid, force=False):
        raise PermissionError("simulated EPERM")

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.write_takeover_marker", write_marker)
    monkeypatch.setattr("gateway.status.terminate_pid", raise_permission)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)

    from gateway.run import start_gateway

    # Should return False due to permission error
    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is False
    # Marker must NOT be left behind
    assert not (tmp_path / ".gateway-takeover.json").exists()
