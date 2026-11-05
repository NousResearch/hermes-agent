"""The dashboard hands the TUI project helpers the storage interface reader it opened on the profile's store; the
helpers read through it, on one state.db connection of its own, released with the call."""
import sqlite3

import pytest

from hermes_constants import get_hermes_home
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture
def connects(monkeypatch):
    calls = []
    real = sqlite3.connect

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("database"))
        return real(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    return calls


@pytest.fixture
def seen(monkeypatch, connects):
    """``(reader handed to the helper, state.db connections opened during the helper's call)`` per call."""
    from tui_gateway import server

    calls = []
    real_repos, real_tree = server._discover_repos_payload, server._build_project_tree

    def recorded(real):
        def helper(db, **kwargs):
            before = len(connects)
            try:
                return real(db, **kwargs)
            finally:
                calls.append((kwargs.get("persistence"), sum("state.db" in str(t) for t in connects[before:])))
        return helper

    repos, tree = recorded(real_repos), recorded(real_tree)

    monkeypatch.setattr(server, "_discover_repos_payload", repos)
    monkeypatch.setattr(server, "_build_project_tree", tree)
    return calls


def _store():
    path = get_hermes_home() / "state.db"
    with open_persistence(profile_scope()) as store:
        store.ready()
    return path


def test_workspace_picker_hands_the_repo_helper_a_reader(monkeypatch, connects, seen):
    from hermes_cli.web_routers import chat_workspaces

    path = _store()
    chat_workspaces._collect_workspaces(None, False)
    handle, opened = seen[-1]
    assert opened == 1  # the reader's own connection
    assert handle.read_only is True and handle.scope.profile_dir == path.parent.resolve()
    with pytest.raises(RuntimeError):
        handle.ready()  # released with the call
    from persistence_assembly import open_persistence_reader
    before = len(connects)
    with open_persistence_reader(path) as reader:
        reader.ready()
    assert len(connects) > before  # the count above would see an open


def test_profile_project_tree_hands_the_builder_a_reader(monkeypatch, connects, seen):
    from hermes_cli.web_routers import profiles as profiles_routes

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    path = _store()
    profiles_routes.get_profiles_projects_tree(profile="default")
    handle, opened = seen[-1]
    assert opened == 1  # the reader's own connection
    assert handle.read_only is True and handle.scope.profile_dir == path.parent.resolve()
    with pytest.raises(RuntimeError):
        handle.ready()
