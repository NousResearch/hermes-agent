"""Multiplex invariant: session-end work a gateway hands to a worker runs in the OWNING profile's scope.

Memory providers' ``on_session_end`` and plugin ``on_session_finalize`` / ``on_session_reset``
read the caller profile (home + secrets) at call time. ``/new`` and ``/reset`` run on the main
loop with no scope installed (the confirm button), so their old-agent cleanup and session hooks
ran unscoped: on a multiplexed gateway ``on_session_end`` was refused
(``memory_caller_profile_unresolved``) and a provider that extracts at session end lost the
conversation (#151672). The opposite slip: an in-turn caller that passes no session key (session
hygiene, deferred cleanup) must keep its own profile's scope, never be moved to the default one.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import memory_caller_scope, secret_scope
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


class _RecordingAgent:
    """What ``_cleanup_agent_resources`` calls on a real agent: records where it ran and what the
    memory-caller guard decides for a manager bound to ``bound_home`` from that scope."""

    def __init__(self, seen: dict, bound_home: Path):
        self.seen, self.bound_home = seen, bound_home

    def shutdown_memory_provider(self, messages=None):
        self.seen["cleanup"] = str(get_hermes_home())
        self.seen["secret"] = (secret_scope.current_secret_scope() or {}).get("HINDSIGHT_LLM_API_KEY")
        self.seen["guard"] = memory_caller_scope.caller_scope(
            memory_caller_scope.bound_scope(self.bound_home), "hook on_session_end")

    def close(self):
        pass


def _runner(profile_homes: dict[str, Path]) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = SimpleNamespace(_profile_home_for_key=lambda key: profile_homes.get(key))
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    return runner


def _profile_b(tmp_path, monkeypatch) -> Path:
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return prof_b


@pytest.mark.asyncio
async def test_reset_runs_memory_cleanup_and_session_hooks_under_the_owning_profile(tmp_path, monkeypatch):
    prof_b = _profile_b(tmp_path, monkeypatch)
    key = "agent:b:telegram:dm:1"
    seen: dict = {}
    monkeypatch.setattr("hermes_cli.lifecycle.finalize_session",
                        lambda **_kw: seen.__setitem__("finalize", str(get_hermes_home())))
    runner = _runner({key: prof_b})
    runner._agent_cache[key] = (_RecordingAgent(seen, prof_b), "sig")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1", chat_type="dm")
    secret_scope.set_multiplex_active(True)
    try:
        assert secret_scope.current_secret_scope() is None  # the reset path is unscoped
        await runner._cleanup_old_agent_for_reset(key)
        await runner._fire_session_reset_hooks(source, key, "old", "new")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert (seen["cleanup"], seen["secret"]) == (str(prof_b), "key-of-b")
    assert seen["guard"] is not None, "on_session_end refused: the old session's memory is lost (#151672)"
    assert seen["finalize"] == str(prof_b)


@pytest.mark.asyncio
async def test_unkeyed_in_turn_cleanup_keeps_the_callers_named_profile(tmp_path, monkeypatch):
    """Session hygiene and deferred cleanup pass no session key from inside profile b's turn."""
    prof_b = _profile_b(tmp_path, monkeypatch)
    seen: dict = {}
    runner = _runner({})
    secret_scope.set_multiplex_active(True)
    try:
        with _profile_runtime_scope(prof_b):
            await runner._cleanup_agent_resources_off_loop(_RecordingAgent(seen, prof_b), context="session hygiene")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert (seen["cleanup"], seen["secret"]) == (str(prof_b), "key-of-b")
    assert seen["guard"] is not None
