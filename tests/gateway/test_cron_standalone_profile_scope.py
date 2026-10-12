"""A ``gateway.standalone: true`` profile's own gateway ticks only its own cron store.

``gateway.standalone`` opts a named profile out of the host multiplexer: that profile runs its
own gateway process. Its in-process cron ticker must not also enumerate the host's profile set:
``_cron_profile_gate`` only stands a home down when another gateway owns it, so a profile no
gateway serves (such as the default profile on a standalone-only host) would be ticked by it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """A scratch HOME so profile enumeration can never read or write the live install."""
    fake_home = tmp_path / "fakehome"
    hermes_home = fake_home / ".hermes"
    for name in ("solo", "other"):
        (hermes_home / "profiles" / name).mkdir(parents=True)
    (hermes_home / "profiles" / "solo" / "config.yaml").write_text("gateway:\n  standalone: true\n")
    (hermes_home / "profiles" / "other" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import profiles as profiles_mod

    assert str(profiles_mod._get_profiles_root()).startswith(str(tmp_path))
    return hermes_home


def _host_set(hermes_home):
    return [("default", hermes_home), ("other", hermes_home / "profiles" / "other")]


def test_standalone_profile_ticks_only_its_own_store(isolated_profiles, monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "solo")
    monkeypatch.setattr(
        gateway_run, "_multiplex_profile_homes", lambda _config: _host_set(isolated_profiles))

    homes = gateway_run._cron_tick_profile_homes(object())

    assert [name for name, _home in homes] == ["solo"]
    assert homes[0][1] == profiles_mod.get_profile_dir("solo")


def test_non_standalone_profile_keeps_host_enumeration(isolated_profiles, monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "other")
    monkeypatch.setattr(
        gateway_run, "_multiplex_profile_homes", lambda _config: _host_set(isolated_profiles))

    homes = gateway_run._cron_tick_profile_homes(object())

    assert [name for name, _home in homes] == ["default", "other"]
