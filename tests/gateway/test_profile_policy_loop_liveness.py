"""A blocked profile policy open must not stall the shared gateway loop.

Regression for the 2026-10-12 reconnect freeze. Use real scope/config readers;
only the network adapter and a controlled filesystem delay are substituted.
"""
import asyncio
import builtins
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager

import pytest

from gateway import run
from tools import terminal_scope
from agent.secret_scope import current_secret_scope, get_secret, set_multiplex_context, reset_multiplex_context
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home
from tools.terminal_scope import TerminalPolicyUnavailable, terminal_env


def _profile(home, backend):
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"PROFILE_MARKER={backend}\nDISCORD_BOT_TOKEN=FAKE_{backend}\n")
    (home / "config.yaml").write_text(
        f"terminal:\n  backend: {backend}\nagent:\n  reconnect_attention_after: 0\n"
        "platforms:\n  discord:\n    enabled: true\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["async_scope", "startup", "reconnect", "backoff"])
@pytest.mark.parametrize("policy_file", [".env", "config.yaml"])
async def test_blocked_policy_open_keeps_loop_live(tmp_path, monkeypatch, entry, policy_file):
    home = tmp_path / "profiles" / "served"
    _profile(home, "docker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    started = threading.Event()
    released = threading.Event()
    loop_thread = threading.get_ident()
    loop_opens = []
    probe_count = 0
    stop = asyncio.Event()
    original_open = Path.open
    original_builtin_open = builtins.open
    phase = "attempt"
    held = False
    in_policy = threading.local()
    original_build = terminal_scope.build_profile_terminal_scope

    def build_policy(*args, **kwargs):
        in_policy.active = True
        try:
            return original_build(*args, **kwargs)
        finally:
            in_policy.active = False

    def hold_open(path):
        nonlocal held
        path = Path(path)
        if path == home / policy_file:
            if threading.get_ident() == loop_thread:
                loop_opens.append(str(path))
            if (not held and getattr(in_policy, "active", False)
                    and (entry != "backoff" or phase == "backoff")):
                held = True
                started.set()
                # A finite fail-safe lets the broken loop resume and report RED.
                released.wait(2.0)

    def blocked_open(path, *args, **kwargs):
        hold_open(path)
        return original_open(path, *args, **kwargs)

    def blocked_builtin_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            hold_open(path)
        return original_builtin_open(path, *args, **kwargs)

    async def probe():
        nonlocal probe_count
        while not stop.is_set():
            if started.is_set() and not released.is_set():
                probe_count += 1
                released.set()
            await asyncio.sleep(0)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._routed_profile_home = lambda _name: home
    runner._snapshot_profile_busy_modes = lambda *a: None
    runner._subscribe_plugin_rewire = lambda *a: None
    runner._register_config_hooks = lambda *a, **k: None
    runner._configure_profile_adapter = lambda *a: None
    runner._sync_voice_mode_state_to_adapter = lambda *a: None
    runner._schedule_planned_restart_replay = lambda: None
    runner._schedule_resume_pending_sessions = lambda **k: None
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    runner._safe_adapter_disconnect = AsyncMock()
    runner._update_platform_runtime_status = lambda *a, **k: None
    adapter = SimpleNamespace(fatal_error_retryable=True, has_fatal_error=False)
    created = 0

    def create(*_args):
        nonlocal created
        created += 1
        assert get_hermes_home() == home
        assert get_secret("PROFILE_MARKER") == "docker"
        assert terminal_env("TERMINAL_ENV") == "docker"
        return adapter if created == 1 else None

    async def connect(*_args, **_kwargs):
        nonlocal phase
        assert get_hermes_home() == home
        assert terminal_env("TERMINAL_ENV") == "docker"
        phase = "backoff"
        return entry != "backoff"

    runner._create_adapter = create
    runner._connect_adapter_with_timeout = connect
    runner._connect_initial_adapter_with_timeout = connect
    monkeypatch.setattr(run, "_reconnect_backoff", lambda _: 0)
    monkeypatch.setattr(Path, "open", blocked_open)
    monkeypatch.setattr(builtins, "open", blocked_builtin_open)
    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", build_policy)
    token = set_multiplex_context(True)
    ticker = asyncio.create_task(probe())
    before_home, before_secrets = get_hermes_home(), current_secret_scope()
    try:
        if entry == "async_scope":
            async with run._async_profile_runtime_scope(home):
                assert get_hermes_home() == home
                assert terminal_env("TERMINAL_ENV") == "docker"
        elif entry == "startup":
            assert await runner._start_one_profile_adapters("served", home, {}) == 1
        else:
            await asyncio.wait_for(
                runner._run_secondary_profile_reconnect("served", Platform.DISCORD), 5.0)
        if entry != "backoff" or started.is_set():
            assert started.is_set(), "the real policy open was not reached"
            assert probe_count > 0, "gateway loop missed every liveness probe during a blocked policy open"
        assert not loop_opens, f"policy filesystem work still ran on the gateway loop: {loop_opens}"
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
    finally:
        released.set()
        stop.set()
        await ticker
        reset_multiplex_context(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped_threshold", [None, "37"])
async def test_secondary_backoff_never_expands_launch_environment(tmp_path, monkeypatch, scoped_threshold):
    home = tmp_path / "served"
    _profile(home, "docker")
    (home / "config.yaml").write_text(
        "terminal:\n  backend: docker\nagent:\n"
        "  reconnect_attention_after: ${RECONNECT_THRESHOLD}\n"
        "platforms:\n  discord:\n    enabled: true\n")
    if scoped_threshold is not None:
        with (home / ".env").open("a") as stream:
            stream.write(f"RECONNECT_THRESHOLD={scoped_threshold}\n")
    monkeypatch.setenv("RECONNECT_THRESHOLD", "0.001")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._routed_profile_home = lambda _name: home
    runner._configure_profile_adapter = lambda *a: None
    runner._safe_adapter_disconnect = AsyncMock()
    runner._connect_adapter_with_timeout = AsyncMock(return_value=False)
    adapter = SimpleNamespace(has_fatal_error=False, fatal_error_retryable=True)
    runner._create_adapter = lambda *a: adapter
    thresholds = []

    def attention(*args, **kwargs):
        thresholds.append(kwargs["attention_after"])
        runner._running = False

    runner._flag_reconnect_needs_attention = attention
    monkeypatch.setattr(run, "_reconnect_backoff", lambda _: 0)
    token = set_multiplex_context(True)
    before_home, before_secrets = get_hermes_home(), current_secret_scope()
    try:
        assert before_secrets is None, "exercise startup reconnect with no inherited secret scope"
        await asyncio.wait_for(runner._run_secondary_profile_reconnect("served", Platform.DISCORD), 5)
        assert thresholds == [7200.0 if scoped_threshold is None else 37.0]
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
        assert os.environ["RECONNECT_THRESHOLD"] == "0.001"
    finally:
        reset_multiplex_context(token)


@pytest.mark.asyncio
async def test_async_policy_scope_isolated_refreshes_and_fails_closed(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    _profile(a, "docker")
    _profile(b, "ssh")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("PROFILE_MARKER", "ambient")
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    before_home, before_secrets = get_hermes_home(), current_secret_scope()
    token = set_multiplex_context(True)
    try:
        for home, expected in [(a, "docker"), (b, "ssh"), (a, "docker")]:
            async with run._async_profile_runtime_scope(home):
                assert get_hermes_home() == home
                assert get_secret("PROFILE_MARKER") == expected
                assert terminal_env("TERMINAL_ENV") == expected
                # Worker ContextVars do not bind this task; its children must inherit the result.
                assert await asyncio.to_thread(terminal_env, "TERMINAL_ENV") == expected
            assert get_hermes_home() == before_home
            assert current_secret_scope() is before_secrets
        async def overlapping(home, expected, ready, peer_ready):
            async with run._async_profile_runtime_scope(home):
                ready.set()
                await peer_ready.wait()
                assert get_hermes_home() == home
                assert get_secret("PROFILE_MARKER") == expected
                assert terminal_env("TERMINAL_ENV") == expected

        ready_a, ready_b = asyncio.Event(), asyncio.Event()
        await asyncio.gather(overlapping(a, "docker", ready_a, ready_b),
                             overlapping(b, "ssh", ready_b, ready_a))
        (b / "config.yaml").write_text("terminal:\n  backend: docker\n")
        async with run._async_profile_runtime_scope(b):
            assert terminal_env("TERMINAL_ENV") == "docker"
        (b / "config.yaml").write_text("terminal: [unclosed\n")
        async with run._async_profile_runtime_scope(b):
            assert get_hermes_home() == b
            with pytest.raises(TerminalPolicyUnavailable):
                terminal_env("TERMINAL_ENV")
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
        original_open = Path.open

        def unreadable(path, *args, **kwargs):
            if path == b / ".env":
                raise OSError("policy file is locked")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", unreadable)
        async with run._async_profile_runtime_scope(b):
            with pytest.raises(TerminalPolicyUnavailable):
                terminal_env("TERMINAL_ENV")
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
    finally:
        reset_multiplex_context(token)


@pytest.mark.asyncio
async def test_async_scope_pairing_grant_refreshes_bound_secrets(tmp_path, monkeypatch):
    from gateway.pairing import _sync_allowlist_add

    home = tmp_path / "served"
    _profile(home, "docker")
    (home / ".env").write_text("DISCORD_ALLOWED_USERS=profile-admin\n")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "ambient-admin")
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    token = set_multiplex_context(True)
    try:
        async with run._async_profile_runtime_scope(home):
            _sync_allowlist_add("discord", "111")
            assert get_secret("DISCORD_ALLOWED_USERS") == "profile-admin,111"
        assert (home / ".env").read_text().strip() == "DISCORD_ALLOWED_USERS=profile-admin,111"
        assert __import__("os").environ["DISCORD_ALLOWED_USERS"] == "ambient-admin"
    finally:
        reset_multiplex_context(token)


@asynccontextmanager
async def _held_read_witness():
    started, released = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    observed = SimpleNamespace(probes=0, on_loop=False)
    stop = asyncio.Event()

    def hold():
        observed.on_loop |= threading.get_ident() == loop_thread
        if started.is_set():
            return
        started.set()
        released.wait(2.0)  # Finite red-path escape; a live loop releases it immediately.

    async def probe():
        while not stop.is_set():
            if started.is_set() and not released.is_set():
                observed.probes += 1
                released.set()
            await asyncio.sleep(0)

    observed.hold = hold
    ticker = asyncio.create_task(probe())
    try:
        yield observed
        assert started.is_set(), "the real reader was not reached"
        assert observed.probes > 0, "gateway loop starved while the reader was held"
        assert not observed.on_loop, "blocking preparation still ran on the loop thread"
    finally:
        released.set()
        stop.set()
        await ticker


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_file", [".env", "config.yaml"])
async def test_launch_scope_blocked_read_keeps_loop_live(tmp_path, monkeypatch, policy_file):
    from gateway.run_adapters import GatewayAdapterLifecycleMixin
    from tui_gateway import launch_profile_policy as launch

    home = tmp_path / "launch"
    _profile(home, "docker")
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: home)
    monkeypatch.setattr(launch, "_snapshot", {"PROFILE_MARKER": "frozen", "LAUNCH_ONLY": "env-only"})
    monkeypatch.setenv("LAUNCH_ONLY", "polluted")
    original_path_open, original_builtin_open = Path.open, builtins.open
    token = set_multiplex_context(True)
    before_home, before_secrets = get_hermes_home(), current_secret_scope()
    try:
        async with _held_read_witness() as witness:
            def hold(path):
                if Path(path) == home / policy_file:
                    witness.hold()

            def path_open(path, *args, **kwargs):
                hold(path)
                return original_path_open(path, *args, **kwargs)

            def builtin_open(path, *args, **kwargs):
                if isinstance(path, (str, Path)):
                    hold(path)
                return original_builtin_open(path, *args, **kwargs)

            monkeypatch.setattr(Path, "open", path_open)
            monkeypatch.setattr(builtins, "open", builtin_open)
            scope = GatewayAdapterLifecycleMixin._async_scope_or_null(run._async_profile_runtime_scope, None)
            async with scope:
                assert get_hermes_home() == home
                assert get_secret("PROFILE_MARKER") == "docker"
                assert get_secret("LAUNCH_ONLY") == "env-only"
                assert terminal_env("TERMINAL_ENV") == "docker"
                assert await asyncio.to_thread(get_secret, "LAUNCH_ONLY") == "env-only"
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
    finally:
        reset_multiplex_context(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["startup", "reconnect", "primary_prefilter", "primary_reconnect"])
@pytest.mark.parametrize("reader", ["raw_config", "liveness"])
async def test_real_discord_constructor_keeps_loop_live(tmp_path, monkeypatch, entry, reader):
    from gateway import platform_registry as registry_module
    from gateway.platform_registry import PlatformEntry, PlatformRegistry
    from gateway.config import PlatformConfig
    from plugins.platforms.discord import adapter as discord_adapter
    import hermes_cli.config as config_module

    secondary = entry in ("startup", "reconnect")
    home = tmp_path / "profiles" / "served"
    _profile(home, "docker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: home)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    registry = PlatformRegistry()
    monkeypatch.setattr(registry_module, "platform_registry", registry)
    monkeypatch.setattr(discord_adapter, "discord_deps_present", lambda: True)
    discord_adapter.register(SimpleNamespace(register_platform=lambda **kwargs: registry.register(PlatformEntry(**kwargs))))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._snapshot_profile_busy_modes = lambda *a: None
    runner._subscribe_plugin_rewire = lambda *a: None
    runner._register_config_hooks = lambda *a, **k: None
    runner._configure_profile_adapter = lambda *a: None
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
    runner._connect_initial_adapter_with_timeout = AsyncMock(return_value=True)
    runner._sync_voice_mode_state_to_adapter = lambda adapter: None
    # Primary paths: the launch profile's own Discord adapter on a standalone host.
    primary_config = PlatformConfig(enabled=True, token="FAKE_primary_discord_token")
    runner.config = GatewayConfig(multiplex_profiles=secondary, platforms={Platform.DISCORD: primary_config})
    runner.adapters = {}
    runner._failed_platforms = {Platform.DISCORD: {"config": primary_config, "attempts": 0, "next_retry": 0}}
    runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
    runner._wire_adapter_handlers = lambda *a, **k: None
    runner._flag_reconnect_needs_attention = lambda *a, **k: None
    installed = {}

    async def install(platform, adapter):
        installed[platform] = adapter

    runner._install_reconnected_adapter = install
    token = set_multiplex_context(secondary)
    before_home, before_secrets = get_hermes_home(), current_secret_scope()
    try:
        async with _held_read_witness() as witness:
            original = config_module.read_raw_config if reader == "raw_config" else discord_adapter.DiscordAdapter._config_value

            def held_reader(*args, **kwargs):
                witness.hold()
                if secondary:
                    assert get_hermes_home() == home
                    assert get_secret("PROFILE_MARKER") == "docker"
                return original(*args, **kwargs)

            if reader == "raw_config":
                monkeypatch.setattr(config_module, "read_raw_config", held_reader)
            else:
                monkeypatch.setattr(discord_adapter.DiscordAdapter, "_config_value", held_reader)
            if entry == "startup":
                assert await runner._start_one_profile_adapters("served", home, {}) == 1
                adapter = runner._profile_adapters["served"][Platform.DISCORD]
            elif entry == "reconnect":
                adapter, success = await runner._secondary_reconnect_attempt("served", Platform.DISCORD)
                assert success
            elif entry == "primary_prefilter":
                aborted, _count, _skipped, pending = await runner._start_prefilter_platforms()
                assert not aborted and len(pending) == 1
                adapter = pending[0][2]
            else:
                await runner._reconnect_failed_platform(Platform.DISCORD, 1.0)
                adapter = installed[Platform.DISCORD]
            assert isinstance(adapter, discord_adapter.DiscordAdapter)
            assert adapter._liveness_interval_seconds == 15.0
            # Events created by the prepared adapter remain usable on the owning loop.
            ready = asyncio.create_task(adapter._ready_event.wait())
            asyncio.get_running_loop().call_soon(adapter._ready_event.set)
            assert await asyncio.wait_for(ready, 2.0)
        assert get_hermes_home() == before_home
        assert current_secret_scope() is before_secrets
    finally:
        reset_multiplex_context(token)


@pytest.mark.asyncio
async def test_non_opted_in_adapter_factory_keeps_owning_loop(tmp_path, monkeypatch):
    from gateway import platform_registry as registry_module
    from gateway.platform_registry import PlatformEntry, PlatformRegistry
    from gateway.config import PlatformConfig

    home = tmp_path / "served"
    _profile(home, "docker")
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _home: None)
    registry = PlatformRegistry()
    monkeypatch.setattr(registry_module, "platform_registry", registry)
    loop = asyncio.get_running_loop()
    adapter = object()

    def loop_bound_factory(platform, config):
        assert asyncio.get_running_loop() is loop
        assert get_hermes_home() == home
        return adapter

    registry.register(PlatformEntry("loop-bound", "Loop bound", lambda config: adapter, lambda: True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._create_adapter = loop_bound_factory
    async with run._async_profile_runtime_scope(home):
        assert await runner._create_adapter_off_loop(Platform("loop-bound"), PlatformConfig()) is adapter


def _launch_home(tmp_path, monkeypatch):
    from tui_gateway import launch_profile_policy as launch

    home = tmp_path / "launch"
    _profile(home, "docker")
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: home)
    monkeypatch.setattr(launch, "_snapshot", {"LAUNCH_ONLY": "env-only"})
    return home


@pytest.mark.asyncio
async def test_async_launch_scope_prepares_on_the_gateway_executor(tmp_path, monkeypatch):
    """The launch scope's preparation joins the runner's fenced pool, which shutdown drains."""
    from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed

    _launch_home(tmp_path, monkeypatch)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock, runner._executor_closing, runner._executor = threading.Lock(), False, None
    used = []
    original = GatewayRunner._run_in_executor_with_context

    async def recording(self, func, *args):
        used.append(getattr(func, "__name__", func))
        return await original(self, func, *args)

    monkeypatch.setattr(GatewayRunner, "_run_in_executor_with_context", recording)
    monkeypatch.setattr(run, "_gateway_runner_ref", lambda: runner)
    token = set_multiplex_context(True)
    try:
        async with async_launch_profile_scope_if_multiplexed():
            assert get_secret("LAUNCH_ONLY") == "env-only"
        assert used, "launch-scope preparation bypassed the gateway executor"
    finally:
        reset_multiplex_context(token)
        runner._shutdown_executor()


@pytest.mark.asyncio
async def test_standalone_handler_prepares_launch_scope_off_loop(tmp_path, monkeypatch):
    """A standalone host whose hosted room flipped multiplexing binds the launch scope per event;
    its .env/config.yaml reads must not run on the loop."""
    home = _launch_home(tmp_path, monkeypatch)
    original_builtin_open, original_path_open = builtins.open, Path.open
    runner = GatewayRunner.__new__(GatewayRunner)
    seen = []

    async def handler(event):
        seen.append((get_hermes_home(), get_secret("LAUNCH_ONLY")))
        return "ok"

    token = set_multiplex_context(True)
    try:
        async with _held_read_witness() as witness:
            def hold(path):
                if Path(path) == home / ".env":
                    witness.hold()

            monkeypatch.setattr(Path, "open", lambda p, *a, **k: (hold(p), original_path_open(p, *a, **k))[1])
            monkeypatch.setattr(builtins, "open", lambda p, *a, **k: (
                hold(p) if isinstance(p, (str, Path)) else None, original_builtin_open(p, *a, **k))[1])
            assert await runner._standalone_scoped(handler)(object()) == "ok"
        assert seen == [(home, "env-only")]
    finally:
        reset_multiplex_context(token)
