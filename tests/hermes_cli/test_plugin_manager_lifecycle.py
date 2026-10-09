"""Profile delete/rename retire cached plugin managers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import plugins
from hermes_cli.profiles import (
    create_profile,
    delete_profile,
    get_profile_dir,
    rename_profile,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_delete_evicts_cached_plugin_manager(profile_env):
    create_profile("gone", no_alias=True, no_skills=True)
    profile_dir = get_profile_dir("gone")
    token = set_hermes_home_override(profile_dir)
    try:
        manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)

    with patch("hermes_cli.profiles._cleanup_gateway_service"), \
         patch("hermes_cli.profiles._maybe_unregister_gateway_service"), \
         patch("hermes_cli.profiles._check_gateway_running", return_value=False), \
         patch("hermes_cli.profiles._stop_profile_backends"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False), \
         patch("hermes_cli.profiles._notify_multiplexer"), \
         patch("hermes_cli.profiles._purge_identity", return_value=True), \
         patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers"):
        delete_profile("gone", yes=True)

    assert not profile_dir.exists()
    assert profile_dir.resolve() not in plugins._plugin_managers_by_home
    assert plugins._plugin_manager is not manager


def test_delete_logs_plugin_teardown_failure_and_releases_lookups(profile_env, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    create_profile("broken", no_alias=True, no_skills=True)
    profile_dir = get_profile_dir("broken")
    home_key = profile_dir.resolve()

    def fail_unload():
        raise RuntimeError("plugin host shutdown failed")

    manager = SimpleNamespace(home_path=profile_dir, unload=fail_unload)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home_key: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_clear_plugin_submodules", lambda _manager: None)
    monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *_args: False)
    monkeypatch.setattr("hermes_cli.profiles._maybe_unregister_gateway_service", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _home: False)
    monkeypatch.setattr("hermes_cli.profiles._stop_profile_backends", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._stop_bot_desktop", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._live_default_multiplexer", lambda: False)
    monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._purge_identity", lambda *_args: True)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_kwargs: None)

    with caplog.at_level(logging.WARNING):
        delete_profile("broken", yes=True)

    assert not profile_dir.exists()
    assert "plugin host shutdown failed" in caplog.text
    assert home_key not in plugins._plugin_manager_teardown_owners
    token = set_hermes_home_override(profile_dir)
    try:
        assert plugins.get_plugin_manager() is not manager
    finally:
        reset_hermes_home_override(token)


def test_rename_retires_old_home_plugin_manager(profile_env):
    create_profile("oldname", no_alias=True, no_skills=True)
    old_dir = get_profile_dir("oldname")
    token = set_hermes_home_override(old_dir)
    try:
        old_manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
         patch("hermes_cli.profiles._check_gateway_running", return_value=False), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False), \
         patch("hermes_cli.profiles._notify_multiplexer"), \
         patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers"):
        new_dir = rename_profile("oldname", "newname")

    assert old_dir.resolve() not in plugins._plugin_managers_by_home
    token = set_hermes_home_override(new_dir)
    try:
        new_manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    assert new_manager is not old_manager


def test_rename_rolls_back_tombstone_when_plugin_teardown_fails(profile_env, monkeypatch):
    from contextlib import contextmanager
    from hermes_constants import named_profile_is_deleted

    create_profile("oldname", no_alias=True, no_skills=True)
    old_dir = get_profile_dir("oldname")

    @contextmanager
    def fail_teardown(_home):
        yield False, RuntimeError("plugin host shutdown failed")

    monkeypatch.setattr("hermes_cli.profiles.check_alias_collision", lambda _name: "skip")
    monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _home: False)
    monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *_args: False)
    monkeypatch.setattr("hermes_cli.profiles._maybe_unregister_gateway_service", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._maybe_register_gateway_service", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._live_default_multiplexer", lambda: True)
    monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", lambda *_args: None)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins_lifecycle.reserve_plugin_manager_for_home", fail_teardown)

    with pytest.raises(RuntimeError, match="plugin host shutdown failed"):
        rename_profile("oldname", "newname")

    assert old_dir.is_dir()
    assert not get_profile_dir("newname").exists()
    assert not named_profile_is_deleted(old_dir)


def test_unload_profile_manager_waits_for_same_home_lookup(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = (tmp_path / "profile").resolve()
    unload_started = threading.Event()
    release_unload = threading.Event()
    lookup_returned = threading.Event()
    result = []

    def unload():
        unload_started.set()
        assert release_unload.wait(timeout=5.0)

    manager = SimpleNamespace(home_path=home, unload=unload)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_clear_plugin_submodules", lambda _manager: None)

    def retire():
        unload_plugin_manager_for_home(home)

    def lookup():
        token = set_hermes_home_override(home)
        try:
            result.append(plugins.get_plugin_manager())
        finally:
            reset_hermes_home_override(token)
            lookup_returned.set()

    retiring = threading.Thread(target=retire)
    looking = threading.Thread(target=lookup)
    retiring.start()
    try:
        assert unload_started.wait(timeout=5.0)
        looking.start()
        assert not lookup_returned.wait(timeout=2.0)
    finally:
        release_unload.set()
        retiring.join(timeout=5.0)
        looking.join(timeout=5.0)

    assert not retiring.is_alive()
    assert not looking.is_alive()
    assert lookup_returned.is_set()
    assert result and result[0] is not manager


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.parametrize("manager_cached", [False, True], ids=["no-manager", "cached-manager"])
def test_profile_mutation_keeps_same_home_lookup_blocked_until_commit(
    profile_env, monkeypatch, operation, manager_cached
):
    import threading

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    plugins._reset_plugin_managers_for_tests()
    create_profile("race", no_alias=True, no_skills=True)
    profile_dir = get_profile_dir("race")
    original_manager = None
    if manager_cached:
        token = set_hermes_home_override(profile_dir)
        try:
            original_manager = plugins.get_plugin_manager()
        finally:
            reset_hermes_home_override(token)
    else:
        assert profile_dir.resolve() not in plugins._plugin_managers_by_home

    mutation_started = threading.Event()
    release_mutation = threading.Event()
    lookup_waiting = threading.Event()
    lookup_returned = threading.Event()
    unrelated_returned = threading.Event()
    lookup_result = []
    unrelated_result = []
    operation_errors = []

    def hold_mutation():
        mutation_started.set()
        assert release_mutation.wait(timeout=10.0)

    if operation == "rename":
        original_rename = Path.rename

        def blocked_rename(path, target):
            if path == profile_dir:
                hold_mutation()
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", blocked_rename)
        monkeypatch.setattr("hermes_cli.profiles.check_alias_collision", lambda _name: "skip")
        monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _home: False)
        monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *_args: False)
        monkeypatch.setattr("hermes_cli.profiles._maybe_unregister_gateway_service", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._maybe_register_gateway_service", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._live_default_multiplexer", lambda: False)
        monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", lambda *_args: None)
        monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_kwargs: None)

        def mutate():
            rename_profile("race", "renamed")
    else:
        from hermes_cli import profiles as profiles_mod

        original_rmtree = profiles_mod._rmtree_with_retry

        def blocked_rmtree(path, *args):
            if path == profile_dir:
                hold_mutation()
            return original_rmtree(path, *args)

        monkeypatch.setattr(profiles_mod, "_rmtree_with_retry", blocked_rmtree)
        monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *_args: False)
        monkeypatch.setattr("hermes_cli.profiles._maybe_unregister_gateway_service", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _home: False)
        monkeypatch.setattr("hermes_cli.profiles._stop_profile_backends", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._stop_bot_desktop", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._live_default_multiplexer", lambda: False)
        monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", lambda *_args: None)
        monkeypatch.setattr("hermes_cli.profiles._purge_identity", lambda *_args: True)
        monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_kwargs: None)

        def mutate():
            delete_profile("race", yes=True)

    condition = plugins._plugin_manager_teardown_condition
    original_wait = condition.wait

    def observe_lookup_wait(timeout=None):
        if threading.current_thread().name == "same-home-lookup":
            lookup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observe_lookup_wait)

    def run_mutation():
        try:
            mutate()
        except Exception as exc:
            operation_errors.append(exc)

    def lookup():
        token = set_hermes_home_override(profile_dir)
        try:
            lookup_result.append(plugins.get_plugin_manager())
        finally:
            reset_hermes_home_override(token)
            lookup_returned.set()

    def lookup_unrelated_home():
        token = set_hermes_home_override(profile_dir.parent / "other")
        try:
            unrelated_result.append(plugins.get_plugin_manager())
        finally:
            reset_hermes_home_override(token)
            unrelated_returned.set()

    mutating = threading.Thread(target=run_mutation)
    looking = threading.Thread(target=lookup, name="same-home-lookup")
    unrelated = threading.Thread(target=lookup_unrelated_home)
    mutating.start()
    try:
        assert mutation_started.wait(timeout=5.0)
        looking.start()
        assert lookup_waiting.wait(timeout=5.0)
        assert not lookup_returned.is_set()
        unrelated.start()
        assert unrelated_returned.wait(timeout=5.0)
    finally:
        release_mutation.set()
        mutating.join(timeout=10.0)
        if looking.ident is not None:
            looking.join(timeout=10.0)
        if unrelated.ident is not None:
            unrelated.join(timeout=10.0)

    assert not mutating.is_alive()
    assert not looking.is_alive()
    assert not operation_errors
    assert lookup_returned.is_set()
    assert lookup_result and lookup_result[0] is not original_manager
    assert unrelated_returned.is_set()
    assert unrelated_result
    plugins._reset_plugin_managers_for_tests()


def test_unload_profile_manager_owner_can_reenter_lookup(tmp_path, monkeypatch):
    import threading

    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = (tmp_path / "profile").resolve()
    manager = plugins.PluginManager(scope_key=str(home))
    unload = manager.unload
    reentered = []

    def unload_with_reentry():
        token = set_hermes_home_override(home)
        try:
            reentered.append(plugins.get_plugin_manager() is manager)
        finally:
            reset_hermes_home_override(token)
        unload()

    monkeypatch.setattr(manager, "unload", unload_with_reentry)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_clear_plugin_submodules", lambda _manager: None)

    completed = threading.Event()
    thread = threading.Thread(target=lambda: (unload_plugin_manager_for_home(home), completed.set()))
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert completed.is_set()
    assert reentered == [True]


def test_reservation_owner_can_create_manager_when_none_is_cached(tmp_path):
    from hermes_cli.plugins_lifecycle import reserve_plugin_manager_for_home

    plugins._reset_plugin_managers_for_tests()
    home = (tmp_path / "profile").resolve()
    token = set_hermes_home_override(home)
    try:
        with reserve_plugin_manager_for_home(home) as (had_manager, teardown_error):
            assert not had_manager and teardown_error is None
            manager = plugins.get_plugin_manager()
            assert manager is not None
        assert home not in plugins._plugin_managers_by_home
        plugins._reset_plugin_managers_for_tests()
    finally:
        reset_hermes_home_override(token)


def test_delayed_background_discovery_cannot_load_a_reserved_manager(
    profile_env, monkeypatch
):
    from agent import memory_provider
    from hermes_cli.plugins_discovery import start_background_plugin_discovery
    from hermes_cli.plugins_lifecycle import reserve_plugin_manager_for_home

    plugins._reset_plugin_managers_for_tests()
    home = (profile_env / "profiles" / "delayed-discovery").resolve()
    home.mkdir(parents=True)
    token = set_hermes_home_override(home)
    try:
        manager = plugins.get_plugin_manager()
        discovery_calls = []
        persisted = []
        manager._discover_and_load_inner = lambda: discovery_calls.append("load")
        manager._refresh_secret_sources_after_discovery = lambda: None
        monkeypatch.setattr(
            plugins, "_persist_plugin_toolset_keys", lambda **_kwargs: persisted.append("persist")
        )

        class DeferredThread:
            def __init__(self, target):
                self.target = target
                self.alive = False

            def start(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def run(self):
                self.alive = False
                self.target()

        deferred_threads = []

        def defer(target, **_kwargs):
            thread = DeferredThread(target)
            deferred_threads.append(thread)
            return thread

        monkeypatch.setattr(memory_provider, "spawn_context_thread", defer)
        monkeypatch.setattr(plugins, "_background_discovery_thread", None)
        start_background_plugin_discovery()
    finally:
        reset_hermes_home_override(token)

    assert len(deferred_threads) == 1
    with reserve_plugin_manager_for_home(home) as (had_manager, teardown_error):
        assert had_manager
        assert teardown_error is None

    deferred_threads[0].run()

    assert discovery_calls == []
    assert persisted == []
    assert home not in plugins._plugin_managers_by_home
    plugins._reset_plugin_managers_for_tests()


def test_background_persistence_finishes_before_same_home_reservation_yields(
    profile_env, monkeypatch
):
    import threading

    from hermes_cli.plugins_discovery import start_background_plugin_discovery
    from hermes_cli.plugins_lifecycle import reserve_plugin_manager_for_home

    plugins._reset_plugin_managers_for_tests()
    home = (profile_env / "profiles" / "persist-race").resolve()
    home.mkdir(parents=True)
    token = set_hermes_home_override(home)
    try:
        manager = plugins.get_plugin_manager()
        manager._discover_and_load_inner = lambda: None
        manager._refresh_secret_sources_after_discovery = lambda: None

        persistence_started = threading.Event()
        release_persistence = threading.Event()
        persisted_manager = []
        original_persist = plugins._persist_plugin_toolset_keys

        def delayed_persist(*, manager=None, home=None):
            persisted_manager.append((manager, home))
            persistence_started.set()
            assert release_persistence.wait(timeout=5.0)
            if manager is None and home is None:
                return original_persist()
            return original_persist(manager=manager, home=home)

        monkeypatch.setattr(plugins, "_persist_plugin_toolset_keys", delayed_persist)
        monkeypatch.setattr(plugins, "_background_discovery_thread", None)
        start_background_plugin_discovery()
    finally:
        reset_hermes_home_override(token)

    background = plugins._background_discovery_thread
    assert background is not None
    assert persistence_started.wait(timeout=5.0)

    reservation_yielded = threading.Event()
    reservation_errors = []

    def reserve_home():
        scoped = set_hermes_home_override(home)
        try:
            with reserve_plugin_manager_for_home(home):
                reservation_yielded.set()
        except Exception as exc:
            reservation_errors.append(exc)
        finally:
            reset_hermes_home_override(scoped)

    reservation = threading.Thread(target=reserve_home)
    reservation.start()
    yielded_during_persistence = reservation_yielded.wait(timeout=2.0)
    release_persistence.set()
    background.join(timeout=5.0)
    reservation.join(timeout=5.0)

    assert not background.is_alive()
    assert not reservation.is_alive()
    assert not reservation_errors
    assert not yielded_during_persistence
    assert persisted_manager == [(manager, manager.home_path)]
    assert home not in plugins._plugin_managers_by_home
    plugins._reset_plugin_managers_for_tests()
