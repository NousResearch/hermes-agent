"""#132993: a running gateway registers a shell hook once its consent is recorded, and drops it
once revoked, without a restart or a session open, for the launch profile of a single-profile
gateway and for a profile a multiplexer serves. Hooks only: adapters are never rebuilt for it."""
import asyncio
from pathlib import Path

import pytest

from agent import shell_hooks
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.waiting import async_wait_until


def _hooks_in(home: Path) -> list:
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        return [cb for cb in get_plugin_manager()._hooks.get("pre_tool_call", [])
                if getattr(cb, "shell_hook_key", None)]
    finally:
        reset_hermes_home_override(token)


def _approve(home: Path, command: str, approve: bool) -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        shell_hooks._record_approval("pre_tool_call", command) if approve else shell_hooks.revoke(command)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [False, True])
async def test_consent_change_registers_and_revokes_hook_without_restart(tmp_path, monkeypatch, multiplex):
    launch = tmp_path / ".hermes"
    profile = launch / "profiles" / "guarded"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    command = str(tmp_path / "guard.sh")
    hook_cfg = f"hooks:\n  pre_tool_call:\n    - command: {command}\n      fail_closed: true\n"
    target = profile if multiplex else launch
    (target / "config.yaml").write_text(hook_cfg, encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner._running = True
    runner._served_profile_homes = {"guarded": profile} if multiplex else None
    passes = []

    async def _reconcile(**_kw):  # runs once per watcher pass when multiplexing
        passes.append(1)
        return {}

    runner.reconcile_served_profiles = _reconcile

    async def _passes(n):
        start = len(passes)
        await async_wait_until(lambda: len(passes) >= start + n, timeout=10, what="watcher passes")

    watcher = asyncio.create_task(runner._profile_reconcile_watcher(interval=0.01))
    try:
        if multiplex:
            await _passes(2)
        assert _hooks_in(target) == []  # no consent yet: skipped

        _approve(target, command, True)
        await async_wait_until(lambda: len(_hooks_in(target)) == 1, timeout=10, what="hook registered")
        if multiplex:
            await _passes(3)  # later passes with unchanged consent never double-register
        assert len(_hooks_in(target)) == 1

        _approve(target, command, False)
        await async_wait_until(lambda: _hooks_in(target) == [], timeout=10, what="hook unregistered")
        assert bool(passes) == multiplex

        # Consent landing while config.yaml is unreadable applies nothing; its recovery alone,
        # with no further allowlist edit, must still register the hook.
        (target / "config.yaml").rename(target / "config.yaml.bak")
        _approve(target, command, True)
        if multiplex:
            await _passes(2)
        assert _hooks_in(target) == []
        (target / "config.yaml.bak").rename(target / "config.yaml")
        await async_wait_until(lambda: len(_hooks_in(target)) == 1, timeout=10, what="hook after config recovery")
    finally:
        runner._running = False
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        shell_hooks.reset_for_tests()
