"""Multiplex invariant: agent-cache eviction commits end-of-session memory under the OWNING profile.

``_spawn_release_thread`` used to start a bare ``threading.Thread``; threads begin with an EMPTY
context, so provider ``on_session_end`` (credentials/home read at call time) ran under the launch
profile — a secondary's transcript was extracted into the default profile's memory namespace, or
failed closed (``UnscopedSecretError``) and the memories were lost.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


def _runner(profile_homes: dict[str, Path]) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = SimpleNamespace(_profile_home_for_key=lambda key: profile_homes.get(key))
    return runner


def _seen_after_release(runner, key, *, wait_scope=None) -> dict:
    seen: dict = {}
    done = threading.Event()

    def target(agent, k):
        seen["scope"] = secret_scope.current_secret_scope()
        seen["home"] = get_hermes_home()
        done.set()

    runner._spawn_release_thread(target, (None, key), f"t-{key}", inline_fallback=False, session_key=key)
    assert done.wait(5)
    return seen


def test_unscoped_housekeeping_sweep_enters_the_owning_profile_scope(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    try:
        # The housekeeping watcher runs with NO scope installed.
        assert secret_scope.current_secret_scope() is None
        seen = _seen_after_release(_runner({"agent:b:telegram:dm:1": prof_b}), "agent:b:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == prof_b
    assert seen["scope"] and seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"


def test_in_turn_cap_eviction_keeps_the_callers_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"MARKER": "turn-scope"})
    try:
        seen = _seen_after_release(_runner({}), "agent:main:cli:1")
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)
    assert seen["scope"] == {"MARKER": "turn-scope"}
