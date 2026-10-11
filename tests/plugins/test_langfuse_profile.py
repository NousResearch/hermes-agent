"""The Langfuse root trace names the profile the turn runs for.

Several profiles can share one Langfuse project (and one gateway, when
multiplexed). Without the profile on the trace, their turns are
indistinguishable. Driven through the real hooks against a temp HERMES_HOME.
"""
from __future__ import annotations

import contextlib
import importlib
import sys

import pytest


def _fresh_plugin():
    sys.modules.pop("plugins.observability.langfuse", None)
    return importlib.import_module("plugins.observability.langfuse")


class _Span:
    def update(self, **kw):
        pass

    def update_trace(self, **kw):
        pass

    def end(self, **kw):
        pass

    def start_observation(self, **kw):
        return _Span()


class _RootCM:
    def __enter__(self):
        return _Span()

    def __exit__(self, *exc):
        return False


class _Client:
    def __init__(self, roots):
        self._roots = roots

    def create_trace_id(self, seed=None):
        return f"trace::{seed}"

    def start_as_current_observation(self, **kw):
        self._roots.append(kw)
        return _RootCM()

    def flush(self):
        pass


@pytest.fixture
def traced(tmp_path, monkeypatch):
    """Plugin wired to a recording client; returns (module, roots, propagated)."""
    home = tmp_path / "hermes"
    (home / "profiles" / "arkana").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_PROFILE", "HERMES_PROFILE_NAME"):
        monkeypatch.delenv(name, raising=False)

    mod = _fresh_plugin()
    roots: list = []
    propagated: list = []

    @contextlib.contextmanager
    def _propagate(**kw):
        propagated.append(kw)
        yield

    monkeypatch.setattr(mod, "_get_langfuse", lambda: _Client(roots))
    monkeypatch.setattr(mod, "propagate_attributes", _propagate)
    mod._TRACE_STATE.clear()
    return mod, roots, propagated, home


def _open_turn(mod, session="s1"):
    mod.on_pre_llm_request(
        task_id=session, session_id=session, model="m", provider="p", api_mode="chat",
        api_call_count=1, request_messages=[{"role": "user", "content": "hi"}],
        turn_id=f"{session}:t1", api_request_id=f"{session}:t1:api:1",
    )


def test_default_profile_is_named(traced):
    mod, roots, propagated, _ = traced
    _open_turn(mod)
    assert roots[0]["metadata"]["profile"] == "default"
    assert propagated[0]["metadata"] == {"profile": "default"}


def test_routed_turn_names_the_served_profile_not_the_launch_one(traced, monkeypatch):
    mod, roots, propagated, home = traced
    # The launch profile's env pin must not relabel a routed turn.
    monkeypatch.setenv("HERMES_PROFILE", "default")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home / "profiles" / "arkana"))
    try:
        _open_turn(mod, session="s2")
    finally:
        reset_hermes_home_override(token)

    assert roots[0]["metadata"]["profile"] == "arkana"
    assert propagated[0]["metadata"] == {"profile": "arkana"}
