"""Multiplex invariant: /reset and /new finalize the old agent's memory in the OWNING profile's scope.

``_cleanup_old_agent_for_reset`` used to hand ``_cleanup_agent_resources`` straight to the
housekeeping executor. That executor carries the CALLING context along, and the reset path
runs on the main loop where no profile scope is installed, so on a multiplexed gateway the
memory provider's ``on_session_end`` hook saw no caller profile and refused it
(``memory_caller_profile_unresolved``): the transcript of the old session was never
extracted. The shutdown path already runs the same cleanup through
``_run_release_in_profile_scope`` (run_shutdown.py); these tests pin the same contract on the
reset path (#151672).
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


class _ScopeRecordingAgent:
    """Agent double whose cleanup records the profile scope the cleanup ran under.

    ``guard_result`` mirrors what the memory-caller guard (agent/memory_caller_scope) decides
    for a manager bound to ``bound_home`` when the hook fires from THIS cleanup's scope: the
    recorded scope, or None when the guard refuses (no named caller profile, or a mismatch
    between the caller and the manager's bound profile).
    """

    def __init__(self, bound_home: Path | None = None):
        self.seen: dict = {}
        self.bound_home = bound_home
        self.guard_result: dict | None | str = "unset"
        self._done = threading.Event()

    # The cleanup chokepoint: what ``_cleanup_agent_resources`` calls on a real agent.
    def shutdown_memory_provider(self, messages=None):
        from agent import memory_caller_scope

        self.seen["scope"] = secret_scope.current_secret_scope()
        self.seen["home"] = str(get_hermes_home())
        if self.bound_home is not None:
            bound = memory_caller_scope.bound_scope(self.bound_home)
            self.guard_result = memory_caller_scope.caller_scope(
                bound, "hook on_session_end"
            )
        self._done.set()

    def close(self):
        pass


def _multiplex_runner(profile_homes: dict[str, Path | None]) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = SimpleNamespace(
        _profile_home_for_key=lambda key: profile_homes.get(key)
    )
    return runner


def _reset_cleanup(runner: GatewayRunner, key: str, agent=None) -> _ScopeRecordingAgent:
    agent = agent or _ScopeRecordingAgent()
    runner._agent_cache[key] = (agent, "sig")
    return agent


@pytest.mark.asyncio
async def test_reset_cleanup_finalizes_memory_in_the_owning_profile_scope(
    tmp_path, monkeypatch
):
    """Unscoped reset path: the old agent's end-of-session memory must run under the profile
    that OWNS the session key, not under no profile at all (refusal) or the launch profile."""
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    runner = _multiplex_runner({"agent:b:telegram:dm:1": prof_b})
    agent = _reset_cleanup(
        runner, "agent:b:telegram:dm:1", _ScopeRecordingAgent(bound_home=prof_b)
    )
    try:
        assert secret_scope.current_secret_scope() is None
        await runner._cleanup_old_agent_for_reset("agent:b:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert agent._done.wait(10)
    assert agent.seen["home"] == str(prof_b), (
        "reset cleanup ran outside the owning profile's scope; on_session_end is refused "
        "on a multiplexed gateway and the old session's memory is lost (#151672)"
    )
    assert (
        agent.seen["scope"]
        and agent.seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"
    )
    # Positive control inside the same cleanup: a manager bound to the OWNING profile is
    # allowed to finalize from that scope.
    assert agent.guard_result is not None and agent.guard_result["hermes_home"] == str(
        prof_b
    )  # type: ignore[index]


@pytest.mark.asyncio
async def test_reset_cleanup_of_a_default_profile_session_runs_in_default_scope(
    tmp_path, monkeypatch
):
    """``agent:main:`` keys name no profile: the owner IS the default profile, and its cleanup
    must run under it (positive control for the owning-profile resolution, not a refusal)."""
    default_home = tmp_path / ".hermes"
    default_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    runner = _multiplex_runner({})
    agent = _reset_cleanup(runner, "agent:main:telegram:dm:1")
    try:
        await runner._cleanup_old_agent_for_reset("agent:main:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert agent._done.wait(10)
    assert agent.seen["home"] == str(default_home)


@pytest.mark.asyncio
async def test_reset_cleanup_of_an_unresolved_owner_refuses_a_manager_bound_to_another_profile(
    tmp_path, monkeypatch
):
    """The profile guard stays fail-closed for cross-profile hooks.

    A session key whose owning profile resolves to no home enters the DEFAULT home under
    ``_run_release_in_profile_scope`` (the helper's existing fallback, shared with the
    shutdown path). A manager bound to ANOTHER profile must still be refused from there:
    the guard's equality check (agent/memory_caller_scope) rejects the mismatch. What the
    wrapper closes is the no-scope refusal of the OWNING case, not the guard itself. A
    default-bound manager is allowed, matching the shutdown path's shipped semantics
    (#110622); that allowance is why the refusal is asserted through the guard here and
    not through the wrapper.
    """
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    runner = _multiplex_runner({"agent:ghost:telegram:dm:1": None})
    agent = _reset_cleanup(
        runner, "agent:ghost:telegram:dm:1", _ScopeRecordingAgent(bound_home=prof_b)
    )
    try:
        await runner._cleanup_old_agent_for_reset("agent:ghost:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
        runner._housekeeping_executor.shutdown(wait=True)
    assert agent._done.wait(10)
    # The wrapper resolved the unresolvable owner to the default home (existing fallback).
    assert agent.seen["home"] == str(default_home)
    # ... and the memory-caller guard refused the hook for a manager bound to profile b.
    assert agent.guard_result is None, (
        "the guard must refuse an end-of-session hook whose caller profile (default, via the "
        "unresolved-owner fallback) is not the manager's bound profile"
    )
