"""Store publication never calls scope builders or application code under its install lock."""

import builtins
import logging

from agent.runtime_scope import bind_state, scope_for_profile
from agent.secret_scope import get_secret

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_imports_and_logging_hold_no_snapshot_lock(store, monkeypatch):
    provider, home = store
    provider.seed_external(home, {"LOCK_KEY": "old-marker"})
    provider.refresh(home)
    cell = provider.cells[provider._key(home)]
    original_import = builtins.__import__
    original_handle = logging.Logger.handle

    def checked_import(*args, **kwargs):
        assert not cell.lock.locked()
        return original_import(*args, **kwargs)

    def checked_handle(*args, **kwargs):
        assert not cell.lock.locked()
        return original_handle(*args, **kwargs)

    with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
        monkeypatch.setattr(builtins, "__import__", checked_import)
        monkeypatch.setattr(logging.Logger, "handle", checked_handle)
        provider.seed_external(home, {"LOCK_KEY": "new-marker"})
        provider.refresh(home)
        assert get_secret("LOCK_KEY") == "new-marker"


def test_publication_does_not_rebuild_any_scope(store, monkeypatch):
    from tui_gateway import launch_profile_policy
    provider, home = store
    provider.seed_external(home, {"LOCK_KEY": "old-marker"})
    provider.refresh(home)
    with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
        def refuse(path):
            raise AssertionError("publication called a scope builder")
        monkeypatch.setattr(launch_profile_policy, "owner_secret_scope", refuse)
        provider.seed_external(home, {"LOCK_KEY": "new-marker"})
        provider.refresh(home)
        assert get_secret("LOCK_KEY") == "new-marker"
