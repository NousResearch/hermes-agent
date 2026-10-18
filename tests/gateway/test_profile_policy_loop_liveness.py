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

import pytest

import gateway.run as run
import tools.terminal_scope as terminal_scope
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
