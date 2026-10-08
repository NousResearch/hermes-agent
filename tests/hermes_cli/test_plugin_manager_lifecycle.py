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
    from hermes_constants import named_profile_is_deleted

    create_profile("oldname", no_alias=True, no_skills=True)
    old_dir = get_profile_dir("oldname")

    def fail_teardown(_home):
        raise RuntimeError("plugin host shutdown failed")

    monkeypatch.setattr("hermes_cli.profiles.check_alias_collision", lambda _name: "skip")
    monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _home: False)
    monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *_args: False)
    monkeypatch.setattr("hermes_cli.profiles._maybe_unregister_gateway_service", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._maybe_register_gateway_service", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.profiles._live_default_multiplexer", lambda: True)
    monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", lambda *_args: None)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins_lifecycle.unload_plugin_manager_for_home", fail_teardown)

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
