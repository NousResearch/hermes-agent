"""Deleting a profile from a multiplexed process removes THAT profile's service, never the host's.

A multi-profile dashboard/tui-gateway process has ``set_multiplex_active(True)`` and the launch
home pinned for routed-profile decisions. ``DELETE /api/profiles/<name>`` then runs
``profiles._cleanup_gateway_service`` in that same process, which switches the home to the victim
to resolve ``get_service_name()``. If that switch were invisible to ``get_hermes_home()`` the name
would resolve to the launch home's bare ``hermes-gateway`` and the cleanup would disable, stop and
unlink the HOST multiplexer's own unit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_cli import profiles


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    victim = launch / "profiles" / "victim"
    victim.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    unit_dir = tmp_path / "xdg" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    host_unit = unit_dir / "hermes-gateway.service"
    victim_unit = unit_dir / "hermes-gateway-victim.service"
    host_unit.write_text("[Service]\n")
    victim_unit.write_text("[Service]\n")
    return launch, victim, host_unit, victim_unit


@pytest.mark.linux_only
def test_delete_from_a_pinned_multiplexer_targets_the_victims_unit_only(two_homes, monkeypatch):
    launch, victim, host_unit, victim_unit = two_homes
    calls: list[list[str]] = []
    monkeypatch.setattr(profiles.subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    set_multiplex_active(True)  # the dashboard has served a second profile: launch home pinned

    profiles._cleanup_gateway_service("victim", victim)

    assert not victim_unit.exists()
    assert host_unit.exists(), "the host multiplexer's own unit must survive a profile delete"
    assert ["systemctl", "--user", "disable", "hermes-gateway-victim"] in calls
    assert not any(c[:3] == ["systemctl", "--user", "disable"] and c[3] == "hermes-gateway" for c in calls)
    # The caller's process home is restored: the process still is the launch profile afterwards.
    assert profiles.os.environ["HERMES_HOME"] == str(launch)
