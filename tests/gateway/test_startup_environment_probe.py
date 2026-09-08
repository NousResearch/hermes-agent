"""The first gateway prompt reuses the enabled, profile-scoped startup probe."""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from gateway import run as gateway_run
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import env_probe
from tools.terminal_scope import install_and_reset_profile_terminal_scope, terminal_env


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,backend", [(None, "local"), (True, "local"), (False, "local"), (True, "ssh")])
async def test_startup_reuses_only_enabled_local_environment_probe(tmp_path, monkeypatch, enabled, backend):
    """Real config, terminal scope, worker and cache; replace only the host inspection."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "ssh" if backend == "local" else "local")
    monkeypatch.setattr(gateway_run, "_warm_turn_machinery_sync", lambda: 0)
    profile = tmp_path / "profile"
    profile.mkdir()
    config = {"terminal": {"backend": backend}, "agent": {}}
    if enabled is not None:
        config["agent"]["environment_probe"] = enabled
    config_path = profile / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = config_path.read_bytes()
    loop_thread = threading.get_ident()
    calls = []

    def inspect_host():
        assert threading.get_ident() != loop_thread
        calls.append(True)
        return "Python toolchain: fixture."

    monkeypatch.setattr(env_probe, "_build_probe_line", inspect_host)
    env_probe._reset_cache_for_tests()
    home_token = set_hermes_home_override(profile)
    try:
        with install_and_reset_profile_terminal_scope(profile):
            assert terminal_env("TERMINAL_ENV") == backend
            runner = object.__new__(gateway_run.GatewayRunner)
            await runner._warm_turn_prerequisites()
            should_probe = enabled is not False and backend == "local"
            assert bool(calls) is should_probe
            if should_probe:
                assert env_probe.get_environment_probe_line() == "Python toolchain: fixture."
                await runner._warm_turn_prerequisites()
                assert len(calls) == 1
            assert config_path.read_bytes() == before
    finally:
        reset_hermes_home_override(home_token)
        env_probe._reset_cache_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["slow", "disabled", "error"])
async def test_environment_warmup_overlaps_tooling_and_keeps_startup_bounded(tmp_path, monkeypatch, case):
    """A slow worker cannot hold the inbound gate or serialize the other warm-up."""
    from hermes_cli import config as config_module

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text('{"agent":{"environment_probe":true}}', encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    tooling_done = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def inspect_host():
        calls.append(True)
        entered.set()
        assert release.wait(10)
        return "Python toolchain: fixture."

    def tooling():
        if case == "slow":
            assert entered.wait(5)
        loop.call_soon_threadsafe(tooling_done.set)
        return 0

    def bad_config():
        raise OSError("fixture config unavailable")

    monkeypatch.setattr(env_probe, "_build_probe_line", inspect_host)
    monkeypatch.setattr(gateway_run, "_warm_turn_machinery_sync", tooling)
    monkeypatch.setattr(gateway_run, "_startup_warmup_timeout_secs", lambda: 0 if case == "disabled" else 0.01)
    if case == "error":
        monkeypatch.setattr(config_module, "load_config_readonly", bad_config)
    env_probe._reset_cache_for_tests()
    runner = object.__new__(gateway_run.GatewayRunner)
    try:
        with install_and_reset_profile_terminal_scope(tmp_path):
            runner._start_startup_warmup()
            if case == "disabled":
                assert runner._startup_warmup_task is None
                assert calls == []
                return
            await asyncio.wait_for(tooling_done.wait(), timeout=6)
            await runner._await_startup_warmup()
            if case == "slow":
                assert calls == [True]
                assert not runner._startup_warmup_task.done()
            release.set()
            await asyncio.wait_for(runner._startup_warmup_task, timeout=5)
            if case == "slow":
                assert env_probe.get_environment_probe_line() == "Python toolchain: fixture."
                assert calls == [True]
            else:
                assert calls == []
    finally:
        release.set()
        task = getattr(runner, "_startup_warmup_task", None)
        if task is not None and not task.done():
            await asyncio.wait_for(task, timeout=5)
        env_probe._reset_cache_for_tests()
