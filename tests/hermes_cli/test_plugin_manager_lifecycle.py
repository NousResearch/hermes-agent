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
