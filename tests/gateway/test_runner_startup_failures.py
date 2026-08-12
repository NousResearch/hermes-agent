import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _RetryableFailureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        raise AssertionError("connect should not be called for disabled platforms")

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SuccessfulAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_start_gateway_verbosity_imports_redacting_formatter(monkeypatch, tmp_path):
    """Verbosity != None must not crash with NameError on RedactingFormatter (#8044)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            assert self._platform_lock_takeover_on_start is False
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
async def test_start_gateway_replace_aborts_when_force_killed_pid_still_alive(
    monkeypatch, tmp_path
):
    """Regression for #19471 (duplicate-gateway half).

    If SIGKILL fails to reap the old gateway, --replace must NOT clear the PID
    file / scoped locks and start a fresh instance — that leaves two live
    gateways fighting over the same token. It should abort instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls = []
    removed_pid = False
    released_locks = False

    class _RunnerShouldNotStart:
        def __init__(self, config):
            raise AssertionError("replacement must not start while old PID is alive")

    def _mock_remove_pid_file():
        nonlocal removed_pid
        removed_pid = True

    def _mock_release_all_scoped_locks(**kwargs):
        nonlocal released_locks
        released_locks = True
        return 0

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.remove_pid_file", _mock_remove_pid_file)
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks",
        _mock_release_all_scoped_locks,
    )
    monkeypatch.setattr(
        "gateway.status.terminate_pid",
        lambda pid, force=False: calls.append((pid, force)),
    )
    # _pid_exists never goes False — the force-kill did not take.
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("gateway.run.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _RunnerShouldNotStart)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is False
    assert calls == [(42, False), (42, True)]
    assert removed_pid is False
    assert released_locks is False


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
        from gateway.status import _get_takeover_marker_path, _write_json_file
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
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    _pid_state = {"alive": True}
    def _mock_get_running_pid():
        return 42 if _pid_state["alive"] else None
    def _mock_remove_pid_file():
        _pid_state["alive"] = False
    monkeypatch.setattr("gateway.status.get_running_pid", _mock_get_running_pid)
    monkeypatch.setattr("gateway.status.remove_pid_file", _mock_remove_pid_file)
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks",
        lambda **kwargs: 0,
    )
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


@pytest.mark.asyncio
async def test_runner_degrades_gracefully_when_all_adapters_missing(monkeypatch, tmp_path, caplog):
    """When all enabled platforms have no adapter (missing library or credentials),
    the gateway should NOT return failure — it should warn and continue running for
    cron job execution, matching the behaviour of 'no platforms enabled' (#5196).

    In fleet deployments the same config.yaml is shared across nodes that may only
    have credentials for a subset of platforms.  Requiring perfect credentials on
    every node makes fleet operation impossible."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    # Simulate _create_adapter returning None for ALL platforms (missing library /
    # missing credentials — no connection attempt ever made).
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    import logging
    with caplog.at_level(logging.WARNING):
        ok = await runner.start()

    # Must NOT return False — gateway should keep running for cron.
    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.adapters == {}
    # Runtime state must remain "running", not "startup_failed".
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    # A warning must be emitted explaining why no platforms connected.
    assert any(
        "No adapter could be created" in record.message
        for record in caplog.records
    ), "Expected degraded-mode warning when all adapters are missing"


class _NonRetryableFailureAdapter(BasePlatformAdapter):
    """Simulates a fatal config error like token collision."""
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "discord-bot-token_lock",
            "Discord bot token already in use (PID 999). Stop the other gateway first.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_exits_with_ex_config_on_nonretryable_startup_error(monkeypatch, tmp_path):
    """Non-retryable startup errors (token collision) must set exit_code to 78
    (EX_CONFIG) so the s6 finish script can translate it to exit 125
    (permanent failure).  See #51228."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _NonRetryableFailureAdapter())

    ok = await runner.start()

    assert ok is True  # start() returns True (clean exit requested)
    assert runner.should_exit_cleanly is True
    assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"


@pytest.mark.asyncio
async def test_start_gateway_propagates_fatal_config_exit_code(monkeypatch, tmp_path):
    """A clean exit carrying GATEWAY_FATAL_CONFIG_EXIT_CODE must surface as a
    process-level SystemExit(78) — NOT a truthy return — so main() exits 78
    and the s6 finish script can translate it to 125 (no restart).

    This guards the propagation gap: runner.start() stamps exit_code=78 and
    requests a clean exit, but start_gateway()'s clean-exit branch used to
    `return True` before the SystemExit(exit_code) site, so main() exited 0
    and s6 crash-looped anyway (#51228)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _FatalConfigRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = "discord: Discord bot token already in use"
            self.exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _FatalConfigRunner)

    from gateway.run import start_gateway

    with pytest.raises(SystemExit) as exc_info:
        await start_gateway(config=GatewayConfig(), replace=False, verbosity=0)

    assert exc_info.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE




def _optional_platform() -> Platform:
    try:
        return Platform("buzz")
    except Exception:
        return Platform.TELEGRAM


class _ProductionTokenLockAdapter(BasePlatformAdapter):
    """Calls the real ``_acquire_platform_lock`` path (retryable=True)."""

    def __init__(
        self,
        platform: Platform = Platform.DISCORD,
        *,
        token: str = "live-foreign-token-xyz",
        scope: str = "discord-bot-token",
        resource_desc: str = "Discord bot token",
    ):
        super().__init__(PlatformConfig(enabled=True, token=token), platform)
        self._lock_token = token
        self._lock_scope = scope
        self._lock_resource_desc = resource_desc

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # Production shape: adapters call _acquire_platform_lock which emits
        # ``{scope}_lock`` with retryable=True. Classification must still treat
        # a live foreign holder as a global single-writer conflict.
        if not self._acquire_platform_lock(
            self._lock_scope, self._lock_token, self._lock_resource_desc
        ):
            return False
        return True

    async def disconnect(self) -> None:
        self._release_platform_lock()
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _PlatformLocalAuthFailureAdapter(BasePlatformAdapter):
    """Optional adapter auth/config failure (e.g. Buzz membership rejected)."""

    def __init__(self, platform: Platform | None = None):
        super().__init__(
            PlatformConfig(enabled=True, token="***"),
            platform or _optional_platform(),
        )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # Production shape: Buzz CLI exit 3 auth_error, non-retryable,
        # not a single-writer lock conflict.
        self._set_fatal_error(
            "connect_failed",
            "auth_error: relay error 403: relay_membership_required (exit 3)",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SecondaryLockConflictAdapter(BasePlatformAdapter):
    """Secondary multiplex adapter that hits a real token lock."""

    def __init__(self, platform: Platform = Platform.TELEGRAM):
        super().__init__(
            PlatformConfig(enabled=True, token="secondary-token"),
            platform,
        )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._acquire_platform_lock(
            "telegram-bot-token",
            "secondary-token",
            "Telegram bot token",
        ):
            return False
        return True

    async def disconnect(self) -> None:
        self._release_platform_lock()
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _seed_live_foreign_token_lock(
    monkeypatch,
    *,
    scope: str,
    identity: str,
    foreign_pid: int = 424242,
) -> None:
    """Plant a live foreign gateway lock that production acquire will refuse."""
    import gateway.status as status_mod

    lock_path = status_mod._get_scope_lock_path(scope, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    status_mod._write_json_file(
        lock_path,
        {
            "pid": foreign_pid,
            "start_time": 1_700_000_000,
            "scope": scope,
            "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run"],
            "metadata": {"platform": "discord"},
            "updated_at": "2026-08-10T00:00:00+00:00",
        },
    )

    real_pid_exists = status_mod._pid_exists
    real_start = status_mod._get_process_start_time
    real_looks = status_mod._looks_like_gateway_process

    def _pid_exists(pid: int) -> bool:
        if pid == foreign_pid:
            return True
        return real_pid_exists(pid)

    def _start_time(pid: int):
        if pid == foreign_pid:
            return 1_700_000_000
        return real_start(pid)

    def _looks(pid: int) -> bool:
        if pid == foreign_pid:
            return True
        return real_looks(pid)

    monkeypatch.setattr(status_mod, "_pid_exists", _pid_exists)
    monkeypatch.setattr(status_mod, "_get_process_start_time", _start_time)
    monkeypatch.setattr(status_mod, "_looks_like_gateway_process", _looks)


@pytest.mark.asyncio
async def test_runner_exits_78_on_real_foreign_token_lock(monkeypatch, tmp_path):
    """Live foreign holder via real ``_acquire_platform_lock`` must exit 78.

    Production emits discord/telegram/... token-lock failures with
    retryable=True. Classification must inspect conflict semantics, not the
    retryable flag, and must not queue a retry storm.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "live-foreign-token-xyz"
    _seed_live_foreign_token_lock(
        monkeypatch, scope="discord-bot-token", identity=token
    )
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token=token),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: _ProductionTokenLockAdapter(token=token),
    )

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert Platform.DISCORD not in runner._failed_platforms
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    plat = state.get("platforms", {}).get("discord", {})
    assert plat.get("state") == "fatal"
    assert plat.get("error_code") == "discord-bot-token_lock"
    assert "already in use" in (plat.get("error_message") or "").lower()


@pytest.mark.asyncio
async def test_runner_secondary_multiplex_lock_conflict_exits_78(monkeypatch, tmp_path):
    """Secondary multiplex token-lock must follow the same fatal exit-78 contract."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "secondary-token"
    _seed_live_foreign_token_lock(
        monkeypatch, scope="telegram-bot-token", identity=token, foreign_pid=515151
    )

    config = GatewayConfig(
        platforms={},  # primary has nothing enabled
        sessions_dir=tmp_path / "sessions",
        multiplex_profiles=True,
    )
    runner = GatewayRunner(config)

    secondary_home = tmp_path / "profiles" / "work"
    secondary_home.mkdir(parents=True)

    def _profiles_to_serve(multiplex=True):
        return [("work", secondary_home)]

    def _create(platform, platform_config):
        return _SecondaryLockConflictAdapter(platform)

    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profiles_to_serve", _profiles_to_serve)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    secondary_cfg = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token=token)},
        sessions_dir=secondary_home / "sessions",
    )

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: secondary_cfg)
    monkeypatch.setattr(runner, "_create_adapter", _create)
    monkeypatch.setattr(
        "gateway.run._own_policy_open_startup_violation",
        lambda cfg: None,
    )

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    plat = state.get("platforms", {}).get("telegram", {})
    assert plat.get("state") == "fatal"
    assert plat.get("error_code") == "telegram-bot-token_lock"


@pytest.mark.asyncio
async def test_runner_stays_alive_on_platform_local_auth_failure(monkeypatch, tmp_path, caplog):
    """Optional adapter auth/config failure must not kill gateway duties.

    Production outage: Buzz returned relay_membership_required (non-retryable)
    and the gateway exited 78, stopping scheduler/Kanban ownership until the
    platform was manually disabled. Platform-local failures park the adapter
    and leave the process running for cron/Kanban and other platforms.
    Lifecycle stays ``running`` so busy/drain contracts remain valid.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    local_platform = _optional_platform()
    config = GatewayConfig(
        platforms={
            local_platform: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    spawned: list[str] = []

    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: _PlatformLocalAuthFailureAdapter(platform),
    )

    def _capture_spawn(coro_factory, name, **kwargs):
        spawned.append(name)
        # Do not schedule real background loops in the unit test.
        return None

    monkeypatch.setattr(runner, "_spawn_supervised", _capture_spawn)

    import logging
    with caplog.at_level(logging.ERROR):
        ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.exit_code in (None, 0)
    assert runner.adapters == {}
    # Non-retryable local auth must be parked, not queued for retry storms.
    assert local_platform not in runner._failed_platforms
    state = read_runtime_status()
    # Lifecycle stays running — per-platform fatal holds the detail.
    assert state["gateway_state"] == "running"
    plat_state = state.get("platforms", {}).get(local_platform.value, {})
    assert plat_state.get("state") == "fatal"
    assert plat_state.get("error_code") == "connect_failed"
    assert "relay_membership_required" in (plat_state.get("error_message") or "")
    assert any(
        "fatally misconfigured and parked" in record.message
        for record in caplog.records
    ), "expected platform-local failure to be parked, not exit-78"
    # Unrelated gateway duties still wire up.
    assert "kanban_dispatcher_watcher" in spawned
    assert "kanban_notifier_watcher" in spawned
    assert "session_expiry_watcher" in spawned

    # Busy/drain contract: lifecycle running + active work must still work
    # while the optional platform is parked (cron/Kanban in flight).
    from gateway.status import (
        derive_gateway_busy,
        derive_gateway_drainable,
        write_runtime_status,
    )

    write_runtime_status(active_agents=2)
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["active_agents"] == 2
    assert derive_gateway_busy(
        gateway_running=True,
        gateway_state=state["gateway_state"],
        active_agents=state["active_agents"],
    ) is True
    assert derive_gateway_drainable(
        gateway_running=True,
        gateway_state=state["gateway_state"],
    ) is True
    # Explicit regression: degraded would break busy/drain — must not be used.
    assert derive_gateway_busy(
        gateway_running=True, gateway_state="degraded", active_agents=2
    ) is False
    assert derive_gateway_drainable(
        gateway_running=True, gateway_state="degraded"
    ) is False


@pytest.mark.asyncio
async def test_runner_keeps_healthy_platform_when_optional_adapter_fails(
    monkeypatch, tmp_path
):
    """One failed optional platform must not prevent other platforms or duties."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    local_platform = _optional_platform()
    healthy = Platform.DISCORD if local_platform != Platform.DISCORD else Platform.TELEGRAM
    config = GatewayConfig(
        platforms={
            local_platform: PlatformConfig(enabled=True, token="***"),
            healthy: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    spawned: list[str] = []

    def _create(platform, platform_config):
        if platform == local_platform:
            return _PlatformLocalAuthFailureAdapter(local_platform)
        return _SuccessfulAdapter(healthy)

    monkeypatch.setattr(runner, "_create_adapter", _create)
    monkeypatch.setattr(
        runner,
        "_spawn_supervised",
        lambda coro_factory, name, **kwargs: spawned.append(name) or None,
    )

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.exit_code in (None, 0)
    assert healthy in runner.adapters
    assert local_platform not in runner.adapters
    assert local_platform not in runner._failed_platforms
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["platforms"][local_platform.value]["state"] == "fatal"
    assert state["platforms"][healthy.value]["state"] == "connected"
    assert "kanban_dispatcher_watcher" in spawned


@pytest.mark.asyncio
async def test_runner_stays_alive_on_mixed_retryable_and_nonretryable_errors(
    monkeypatch, tmp_path, caplog
):
    """Mixed startup failures must NOT exit with EX_CONFIG (NS-609)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    def _make_adapter(platform, platform_config):
        if platform == Platform.DISCORD:
            # Platform-local non-retryable (not a global lock conflict).
            return _PlatformLocalAuthFailureAdapter(Platform.DISCORD)
        return _RetryableFailureAdapter()

    monkeypatch.setattr(runner, "_create_adapter", _make_adapter)
    monkeypatch.setattr(
        runner,
        "_spawn_supervised",
        lambda coro_factory, name, **kwargs: None,
    )

    import logging
    with caplog.at_level(logging.ERROR):
        ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.exit_code in (None, 0)
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert Platform.TELEGRAM in runner._failed_platforms
    assert state["platforms"]["telegram"]["state"] == "retrying"
    assert Platform.DISCORD not in runner._failed_platforms
    assert state["platforms"]["discord"]["state"] == "fatal"
    assert any(
        "fatally misconfigured" in record.message for record in caplog.records
    )


def test_is_global_startup_conflict_contract():
    from gateway.restart import is_global_startup_conflict

    assert is_global_startup_conflict("discord-bot-token_lock", "token already in use")
    assert is_global_startup_conflict("lock_conflict", "Buzz identity in use by another profile")
    assert is_global_startup_conflict("telegram_polling_conflict", "getUpdates conflict")
    assert not is_global_startup_conflict(
        "connect_failed",
        "auth_error: relay error 403: relay_membership_required (exit 3)",
    )
    assert not is_global_startup_conflict("config_missing", "BUZZ_PRIVATE_KEY must be set")
    assert not is_global_startup_conflict("missing_credentials", "No bot token configured")
    # Message fallback when adapters omit a structured code.
    assert is_global_startup_conflict(
        None,
        "Telegram bot token already in use (PID 42). Stop the other gateway first.",
    )
    # Production Telegram connect can wrap a live 409 getUpdates conflict in
    # the generic telegram_connect_error envelope — still a single-writer gate.
    assert is_global_startup_conflict(
        "telegram_connect_error",
        "Telegram startup failed: Conflict: terminated by other getUpdates request",
    )
    assert is_global_startup_conflict(
        "telegram_connect_error",
        "Telegram startup failed: HTTP 409 Conflict: another getUpdates request is active",
    )
    assert is_global_startup_conflict(
        None,
        "Conflict: terminated by other getUpdates request",
    )
    # Unrelated transient telegram_connect_error must stay retryable, not fatal.
    assert not is_global_startup_conflict(
        "telegram_connect_error",
        "Telegram startup failed: temporary DNS resolution failure.",
    )


def test_production_acquire_platform_lock_still_retryable_for_reconnect(monkeypatch, tmp_path):
    """Adapter-level lock failures remain retryable for post-startup reconnect (#54167)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "reconnect-token"
    _seed_live_foreign_token_lock(
        monkeypatch, scope="telegram-bot-token", identity=token, foreign_pid=606060
    )
    adapter = _ProductionTokenLockAdapter(
        Platform.TELEGRAM,
        token=token,
        scope="telegram-bot-token",
        resource_desc="Telegram bot token",
    )
    assert adapter._acquire_platform_lock(
        "telegram-bot-token", token, "Telegram bot token"
    ) is False
    assert adapter.fatal_error_retryable is True


class _SecondaryRetryableFailureAdapter(BasePlatformAdapter):
    """Secondary multiplex adapter with a transient (non-conflict) failure."""

    def __init__(self, platform: Platform = Platform.TELEGRAM):
        super().__init__(
            PlatformConfig(enabled=True, token="secondary-retry-token"),
            platform,
        )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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


class _Telegram409ConnectErrorAdapter(BasePlatformAdapter):
    """Production shape: outer connect exception records telegram_connect_error
    while the body still carries a Bot API 409 getUpdates conflict."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_connect_error",
            "Telegram startup failed: Conflict: terminated by other getUpdates request",
            retryable=True,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_exits_78_on_telegram_connect_error_409(monkeypatch, tmp_path):
    """Production 409 getUpdates under telegram_connect_error must fail closed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: _Telegram409ConnectErrorAdapter(),
    )

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert Platform.TELEGRAM not in runner._failed_platforms
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    plat = state.get("platforms", {}).get("telegram", {})
    assert plat.get("state") == "fatal"
    assert plat.get("error_code") == "telegram_connect_error"
    assert "getupdates" in (plat.get("error_message") or "").lower()


@pytest.mark.asyncio
async def test_runner_secondary_retryable_enters_profile_retry_registry(
    monkeypatch, tmp_path
):
    """Secondary retryable startup failure must arm profile-scoped reconnect.

    Previously queue_retry=False left startup_retryable_errors non-empty
    (suppressing exit 78) while neither ``_failed_platforms`` nor
    ``_profile_failed_platforms`` received the secondary — live but deaf.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={},  # primary has nothing enabled
        sessions_dir=tmp_path / "sessions",
        multiplex_profiles=True,
    )
    runner = GatewayRunner(config)

    secondary_home = tmp_path / "profiles" / "work"
    secondary_home.mkdir(parents=True)

    def _profiles_to_serve(multiplex=True):
        return [("work", secondary_home)]

    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profiles_to_serve", _profiles_to_serve)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    secondary_cfg = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="secondary-retry-token")},
        sessions_dir=secondary_home / "sessions",
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: secondary_cfg)
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: _SecondaryRetryableFailureAdapter(platform),
    )
    monkeypatch.setattr(
        "gateway.run._own_policy_open_startup_violation",
        lambda cfg: None,
    )
    # Do not run real reconnect loops in the unit test.
    monkeypatch.setattr(
        runner,
        "_spawn_supervised",
        lambda coro_factory, name, **kwargs: None,
    )
    scheduled: list[tuple[str, Platform]] = []

    def _capture_schedule(profile_name, platform, adapter):
        scheduled.append((profile_name, platform))
        # Mirror production bookkeeping without starting the reconnect task.
        pending = runner._profile_failed_platforms
        if not isinstance(pending, dict):
            pending = {}
            runner._profile_failed_platforms = pending
        profile_pending = pending.setdefault(profile_name, {})
        # Placeholder non-task marker so the registry is observable.
        profile_pending[platform] = object()

    monkeypatch.setattr(runner, "_schedule_secondary_profile_reconnect", _capture_schedule)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.exit_code in (None, 0)
    # Must not claim a primary-slot retry for a secondary failure.
    assert Platform.TELEGRAM not in runner._failed_platforms
    # Must enter the profile-scoped retry registry.
    assert scheduled == [("work", Platform.TELEGRAM)]
    assert Platform.TELEGRAM in runner._profile_failed_platforms.get("work", {})
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    plat = state.get("platforms", {}).get("telegram", {})
    assert plat.get("state") == "retrying"


@pytest.mark.asyncio
async def test_production_scoped_lock_tuple_unreachable_without_unpack(monkeypatch, tmp_path):
    """Buzz/IRC/LINE must unpack acquire_scoped_lock's (False, existing) tuple.

    A non-empty failure tuple is truthy in Python; ``if not acquire_scoped_lock()``
    never enters the lock_conflict branch. Production shape: real status helper
    returning (False, holder) must set lock_conflict and refuse connect.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import gateway.status as status_mod

    holder = {"pid": 777001, "scope": "buzz"}

    def _fail_lock(scope, identity, metadata=None):
        return (False, holder)

    monkeypatch.setattr(status_mod, "acquire_scoped_lock", _fail_lock)

    from tests.gateway._plugin_adapter_loader import load_plugin_adapter

    # --- Buzz ---
    buzz_mod = load_plugin_adapter("buzz")
    from gateway.config import PlatformConfig as PC

    buzz = buzz_mod.BuzzAdapter(
        PC(enabled=True, extra={"relay_url": "wss://relay.example", "private_key": "nsec1test"})
    )
    # Minimal scripted connect path up to the lock.
    buzz.cli_path = "/fake/buzz"
    monkeypatch.setattr(buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")

    async def _buzz_cli(args):
        if args[:2] == ["users", "get"]:
            return 0, '[{"pubkey":"' + ("ab" * 32) + '","display_name":"T"}]', ""
        return 1, "", "unexpected"

    buzz._run_cli = _buzz_cli
    assert await buzz.connect() is False
    assert buzz.has_fatal_error is True
    assert buzz.fatal_error_code == "lock_conflict"
    assert buzz._lock_key is None

    # --- IRC ---
    irc_mod = load_plugin_adapter("irc")
    irc = irc_mod.IRCAdapter(
        PC(
            enabled=True,
            extra={
                "server": "irc.example",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
            },
        )
    )
    assert await irc.connect() is False
    assert irc.has_fatal_error is True
    assert irc.fatal_error_code == "lock_conflict"
    assert getattr(irc, "_lock_key", None) in (None, "")

    # --- LINE ---
    line_mod = load_plugin_adapter("line")
    line = line_mod.LineAdapter(
        PC(
            enabled=True,
            extra={
                "channel_access_token": "line-token-xyz",
                "channel_secret": "line-secret-xyz",
            },
        )
    )
    # LINE may read token/secret from env or extra depending on version.
    if not getattr(line, "channel_access_token", None):
        line.channel_access_token = "line-token-xyz"
    if not getattr(line, "channel_secret", None):
        line.channel_secret = "line-secret-xyz"
    assert await line.connect() is False
    assert line.has_fatal_error is True
    assert line.fatal_error_code == "lock_conflict"
    assert getattr(line, "_lock_key", None) in (None, "")
