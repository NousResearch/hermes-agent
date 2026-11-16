"""Scope recomposition runs without the publication lock, including logging and imports."""

import builtins
import logging
import threading

from agent import secret_scope as ss
from agent.runtime_scope import bind_state, scope_for_profile

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_rebuild_logging_and_imports_hold_no_install_lock(store, tmp_path, monkeypatch):
    provider, home = store
    from hermes_cli import managed_scope

    provider.seed_external(home, {"LOCK_KEY": "old-marker"})
    provider.refresh(home)
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    observed = []

    class Handler(logging.Handler):
        def emit(self, record):
            observed.append(ss._INSTALLED_SCOPES_LOCK._is_owned())

    original_import = builtins.__import__

    def checked_import(*args, **kwargs):
        assert not ss._INSTALLED_SCOPES_LOCK._is_owned()
        return original_import(*args, **kwargs)

    handler = Handler()
    managed_scope.logger.addHandler(handler)
    try:
        with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
            (managed / ".env").write_bytes(b"KEY=\xff\n")
            managed_scope.invalidate_managed_cache()
            with monkeypatch.context() as patch:
                patch.setattr(builtins, "__import__", checked_import)
                provider.seed_external(home, {"LOCK_KEY": "new-marker"})
                provider.refresh(home)
        assert observed and not any(observed)
    finally:
        managed_scope.logger.removeHandler(handler)
        managed_scope.invalidate_managed_cache()


def test_a_rebuild_can_wait_for_another_threads_registration(store, monkeypatch):
    provider, home = store
    from tui_gateway import launch_profile_policy as policy

    provider.seed_external(home, {"LOCK_KEY": "old-marker"})
    provider.refresh(home)
    original = policy.owner_secret_scope
    entered, registered = threading.Event(), threading.Event()
    errors = []

    def register():
        try:
            assert entered.wait(5)
            with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
                registered.set()
        except BaseException as exc:
            errors.append(exc)
            registered.set()

    armed = False

    def builder(path):
        if armed and threading.current_thread() is threading.main_thread():
            entered.set()
            assert registered.wait(5)
        return original(path)

    monkeypatch.setattr(policy, "owner_secret_scope", builder)
    with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
        armed = True
        worker = threading.Thread(target=register, daemon=True)
        worker.start()
        try:
            provider.seed_external(home, {"LOCK_KEY": "new-marker"})
            provider.refresh(home)
        finally:
            worker.join(5)
        assert not worker.is_alive() and not errors
