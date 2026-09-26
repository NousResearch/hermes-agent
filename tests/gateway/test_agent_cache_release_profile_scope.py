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


def test_in_turn_cap_eviction_of_another_profiles_agent_enters_the_owners_scope(tmp_path, monkeypatch):
    """The LRU cap is enforced inside the REQUESTING turn (profile A's scope), and the agent it
    evicts may be profile B's. B's end-of-session commit must run under B, not under A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / ".hermes"
    prof_a, prof_b = default_home / "profiles" / "a", default_home / "profiles" / "b"
    prof_a.mkdir(parents=True), prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_a))
    scope_token = secret_scope.set_secret_scope({"HINDSIGHT_LLM_API_KEY": "key-of-a"})
    try:
        seen = _seen_after_release(_runner({"agent:b:telegram:dm:1": prof_b}), "agent:b:telegram:dm:1")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == prof_b
    assert seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"


def test_in_turn_cap_eviction_of_a_default_profile_agent_leaves_the_secondarys_scope(tmp_path, monkeypatch):
    """Inverse: secondary A's turn evicts a default-profile session (``agent:main:`` — no named
    owner in the key). The commit runs under the default home, not under A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / ".hermes"
    prof_a = default_home / "profiles" / "a"
    prof_a.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_a))
    scope_token = secret_scope.set_secret_scope({"MARKER": "profile-a"})
    try:
        seen = _seen_after_release(_runner({}), "agent:main:telegram:dm:9")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == default_home
    assert seen["scope"] is not None and seen["scope"].get("MARKER") is None


def test_default_profile_owner_is_the_root_even_when_launched_under_a_named_profile(tmp_path, monkeypatch):
    """``hermes -p x gateway`` sets HERMES_HOME to x's home and serves the default profile as a
    secondary. An ``agent:main:`` session still belongs to the default profile at the ROOT, not to
    the launch profile x — otherwise x's turn would commit default's transcript under x."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    prof_x = root / "profiles" / "x"
    prof_x.mkdir(parents=True)
    (root / ".env").write_text("MARKER=default-root\n")
    monkeypatch.setenv("HERMES_HOME", str(prof_x))  # launched under x
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_x))
    scope_token = secret_scope.set_secret_scope({"MARKER": "profile-x"})
    try:
        seen = _seen_after_release(_runner({}), "agent:main:telegram:dm:3")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == root
    assert seen["scope"].get("MARKER") == "default-root"
