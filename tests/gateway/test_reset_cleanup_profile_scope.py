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

from hermes_agent.agent import memory_caller_scope, secret_scope
from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_agent.gateway.session import SessionSource
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
    monkeypatch.setattr("hermes_agent.hermes_cli.lifecycle.finalize_session",
                        lambda **_kw: seen.__setitem__("finalize", str(get_hermes_home())))
    monkeypatch.setattr("hermes_agent.hermes_cli.lifecycle.invoke_hook", lambda name, **_kw: seen.__setitem__(name, (
        str(get_hermes_home()), (secret_scope.current_secret_scope() or {}).get("HINDSIGHT_LLM_API_KEY"))))
    runner = _runner({key: prof_b})
    runner._agent_cache[key] = (_RecordingAgent(seen, prof_b), "sig")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1", chat_type="dm")
    secret_scope.set_multiplex_active(True)
    try:
        assert secret_scope.current_secret_scope() is None  # the reset path is unscoped
        await runner._cleanup_old_agent_for_reset(key)
        await runner._fire_session_reset_hooks(source, key, "old", "new")
        await runner._fire_plugin_session_reset(source, key, "old", "new")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert (seen["cleanup"], seen["secret"]) == (str(prof_b), "key-of-b")
    assert seen["guard"] is not None, "on_session_end refused: the old session's memory is lost (#151672)"
    assert seen["finalize"] == str(prof_b)
    assert seen["on_session_reset"] == (str(prof_b), "key-of-b")


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

@pytest.mark.asyncio
@pytest.mark.parametrize("source_profile", ["b", None])
async def test_reset_gateway_events_reach_only_the_owning_profile_hooks(tmp_path, monkeypatch, source_profile):
    """An unscoped /new must not dispatch B's session events to default's installed hooks. The key
    names the owner of the session that ended, so a source that routes elsewhere (no profile on it)
    must not move the hooks either."""
    from hermes_agent.gateway.hooks import HookRegistry, ProfileHookRegistries
    from hermes_constants import hermes_home_key

    prof_b = _profile_b(tmp_path, monkeypatch)
    key = "agent:b:telegram:dm:1"
    runner = _runner({key: prof_b})
    # The confirm callback itself is unscoped; the source routes to B, or to the launch profile.
    runner._profile_scope_key_for_source = lambda _source: prof_b if source_profile else tmp_path / ".hermes"
    runner._finalize_session_off_loop = AsyncMock(return_value=[])
    runner._deliver_session_end_messages = AsyncMock()
    seen = []
    hooks = ProfileHookRegistries()
    for home, label in ((tmp_path / ".hermes", "default"), (prof_b, "b")):
        registry = HookRegistry()

        def record(event_type, payload, *, _label=label):
            seen.append((
                _label, event_type, payload["session_key"], str(get_hermes_home()),
                (secret_scope.current_secret_scope() or {}).get("HINDSIGHT_LLM_API_KEY"),
            ))

        registry._handlers["session:end"] = [record]
        registry._handlers["session:reset"] = [record]
        hooks._by_home[hermes_home_key(home)] = registry
    runner.hooks = hooks
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1",
                           chat_type="dm", profile=source_profile)

    secret_scope.set_multiplex_active(True)
    try:
        assert secret_scope.current_secret_scope() is None
        await runner._fire_session_reset_hooks(source, key, "old", "new")
    finally:
        secret_scope.set_multiplex_active(False)

    assert seen == [
        ("b", "session:end", key, str(prof_b), "key-of-b"),
        ("b", "session:reset", key, str(prof_b), "key-of-b"),
    ]


def test_a_key_whose_profile_home_is_gone_warns_before_releasing_under_root(tmp_path, monkeypatch, caplog):
    """``agent:ghost:…`` names a profile with no directory: the release falls back to the default root,
    where the memory guard refuses its hooks. That loss must be logged, not silent."""
    _profile_b(tmp_path, monkeypatch)
    runner = _runner({})
    runner.session_store._named_profile_for_key = lambda key: key.split(":")[1]
    seen: dict = {}
    secret_scope.set_multiplex_active(True)
    try:
        with caplog.at_level("WARNING", logger="hermes_agent.gateway.run"):
            runner._run_release_in_profile_scope(lambda: seen.__setitem__("home", str(get_hermes_home())), (),
                                                 "agent:ghost:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == str(tmp_path / ".hermes")
    assert any("'ghost'" in r.getMessage() and "no home" in r.getMessage() for r in caplog.records), caplog.text
