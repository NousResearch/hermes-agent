"""Work the SQLite backend runs under a store's own profile home (opening the store, the ``on_session_delete`` dispatch after a delete) binds that home alone: the caller's bound scope stays the caller's, so its secrets and terminal policy never move to the store's profile, and the home is restored afterwards."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import session_retention
from agent.runtime_scope import bind_state, scope_for_profile
from hermes_constants import get_hermes_home
from hermes_scope import current_scope
from persistence.sqlite._backend import _home_bound


@pytest.fixture
def caller_scope(tmp_path):
    """A local scope bound for the caller's own profile, the way a turn or a gateway request runs."""
    home = tmp_path / "caller"
    home.mkdir()
    scope = scope_for_profile(home)
    with bind_state(scope, hydrate_secrets=False, terminal=False):
        yield home, scope


def _same(a, b) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def test_a_store_opening_under_its_own_home_keeps_the_callers_scope(caller_scope, tmp_path):
    caller_home, scope = caller_scope
    store_home = tmp_path / "store"
    store_home.mkdir()
    (store_home / "config.yaml").write_text("database:\n  cache_size: -2222\n")
    with _home_bound(store_home):
        assert _same(get_hermes_home(), store_home)
        assert current_scope() == scope
    assert _same(get_hermes_home(), caller_home)
    assert current_scope() == scope


def test_the_delete_hook_runs_under_the_stores_home_and_keeps_the_callers_scope(caller_scope, monkeypatch):
    caller_home, scope = caller_scope
    store_home = Path(os.environ["HERMES_HOME"]) / "profiles" / "coder"
    store_home.mkdir(parents=True, exist_ok=True)
    seen: list = []

    def _invoke(name, **payload):
        seen.append((name, payload["profile"], str(get_hermes_home()), current_scope()))

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _invoke)
    session_retention.emit_session_deleted(store_home, ["s1"], reason="delete")
    assert len(seen) == 1
    name, profile, home, during = seen[0]
    assert (name, profile) == ("on_session_delete", "coder")
    assert _same(home, store_home)
    assert during == scope
    assert _same(get_hermes_home(), caller_home)
