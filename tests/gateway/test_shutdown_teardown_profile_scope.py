"""Multiplex invariant: gateway-hosted teardown fires memory-provider lifecycle hooks under the
OWNING profile's scope (#110622).

``on_session_end`` / ``shutdown`` / ``close`` read credentials and home at call time. Shutdown
runs on the main loop outside any adapter handler, so its executor hop carried an EMPTY scope and
a secondary profile's ``get_secret("OPENVIKING_API_KEY")`` failed closed — the end-of-session
commit was skipped and logged as a WARNING + traceback on every gateway-hosted session close.
"""
from __future__ import annotations

import asyncio
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


def _runner(profile_homes: dict[str, Path]) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = SimpleNamespace(_profile_home_for_key=lambda key: profile_homes.get(key))
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)

    async def _run_in_executor_with_context(func, *args):
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(executor, lambda: ctx.run(func, *args))

    runner._run_in_executor_with_context = _run_in_executor_with_context
    return runner


def _recording_agent(seen: dict):
    class _Provider:
        name = "recorder"

        def on_session_end(self, messages):
            seen["scope"] = secret_scope.current_secret_scope()
            seen["home"] = get_hermes_home()

    def shutdown_memory_provider(messages=None):
        _Provider().on_session_end(messages)

    return SimpleNamespace(
        shutdown_memory_provider=shutdown_memory_provider, close=lambda: None, _session_messages=[],
    )


@pytest.mark.asyncio
async def test_shutdown_teardown_commits_memory_under_the_owning_profile(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("OPENVIKING_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    seen: dict = {}
    secret_scope.set_multiplex_active(True)
    try:
        # Shutdown runs on the main loop with NO scope installed.
        assert secret_scope.current_secret_scope() is None
        await _runner({"agent:b:telegram:dm:1": prof_b})._cleanup_agent_resources_off_loop(
            _recording_agent(seen), context="shutdown finalize", session_key="agent:b:telegram:dm:1",
        )
    finally:
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == prof_b
    assert seen["scope"] and seen["scope"].get("OPENVIKING_API_KEY") == "key-of-b"


@pytest.mark.asyncio
async def test_in_turn_teardown_keeps_the_callers_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen: dict = {}
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"MARKER": "turn-scope"})
    try:
        await _runner({})._cleanup_agent_resources_off_loop(_recording_agent(seen), context="session hygiene")
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)
    assert seen["scope"] == {"MARKER": "turn-scope"}
