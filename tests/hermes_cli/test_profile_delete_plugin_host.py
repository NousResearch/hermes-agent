"""Profile deletion must stop the profile's plugin host in this process (#159497).

Under ``plugins.isolation: host`` each profile's third-party plugins run in a host process started
with the profile home as its working directory. On Windows a directory that is a process's working
directory cannot be removed, so ``delete_profile`` run inside the Desktop's ``hermes serve`` emptied
the profile and then failed the final rmdir with ``[WinError 32]``. The stop and the removal are
asserted on every OS (the rmdir failure only reproduces on Windows).
"""

from pathlib import Path

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli import profiles
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """A default home plus a created profile; host services are not touched."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_maybe_unregister_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        yield profiles.create_profile("plugin-host-delete", no_alias=True)
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


def _start_host(home: Path):
    token = set_hermes_home_override(home)
    try:
        host = plugins_mod.get_plugin_manager()._plugin_host()
        host.ensure_started()
    finally:
        reset_hermes_home_override(token)
    return host


def test_delete_profile_stops_that_profiles_plugin_host(profile):
    host = _start_host(profile)
    proc = host._proc
    try:
        assert proc.poll() is None

        profiles.delete_profile("plugin-host-delete", yes=True)

        assert proc.poll() is not None, "the deleted profile's plugin host is still running"
        assert not profile.exists()
    finally:
        host.shutdown()


def test_delete_profile_leaves_other_profiles_plugin_hosts_running(profile):
    other = profiles.create_profile("plugin-host-keep", no_alias=True)
    kept = _start_host(other)
    try:
        profiles.delete_profile("plugin-host-delete", yes=True)
        assert kept.alive
    finally:
        kept.shutdown()
