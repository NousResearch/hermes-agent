"""Deleting or renaming a profile removes its system-scope systemd unit and Windows Scheduled
Task, not only the user unit / launchd plist. Both carry ``--profile <old>`` with HERMES_HOME
pinned to a dir that no longer exists, so a survivor restarts the removed profile at boot.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hermes_cli import gateway, profiles


@pytest.fixture
def victim(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    victim_dir = launch / "profiles" / "victim"
    victim_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return victim_dir


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("euid", [0, 1000])
def test_cleanup_handles_the_system_scope_unit(victim, tmp_path, monkeypatch, capsys, euid):
    system_dir = tmp_path / "etc-systemd-system"
    system_dir.mkdir()
    unit = system_dir / "hermes-gateway-victim.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(gateway, "_SYSTEM_UNIT_DIR", system_dir)
    monkeypatch.setattr(profiles.os, "geteuid", lambda: euid)
    calls: list[list[str]] = []
    import hermes_cli.profiles_service_cleanup as psc
    monkeypatch.setattr(psc.subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(profiles.subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))

    removed = profiles._cleanup_gateway_service("victim", victim)

    if euid == 0:
        assert removed and not unit.exists()
        assert ["systemctl", "disable", "hermes-gateway-victim"] in calls
    else:
        # Not root: the unit cannot be unlinked, so the exact leftover and command are named.
        assert not removed and unit.exists()
        out = capsys.readouterr().out
        assert str(unit) in out and "sudo systemctl disable --now hermes-gateway-victim" in out


def test_cleanup_deletes_the_windows_scheduled_task(victim, tmp_path, monkeypatch):
    registered = {"HermesGateway_victim"}
    startup = tmp_path / "Startup" / "HermesGateway_victim.vbs"
    startup.parent.mkdir()
    startup.write_text("")
    fake = types.SimpleNamespace(
        get_task_name=lambda: "HermesGateway_victim",
        is_task_registered=lambda: bool(registered),
        _exec_schtasks=lambda args: (registered.discard(args[-1]), (0, "", ""))[1],
        get_startup_entry_path=lambda: startup,
        _legacy_startup_entry_path=lambda: tmp_path / "Startup" / "legacy.vbs",
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", fake)
    # `from hermes_cli import gateway_windows` reads the package attribute first when an earlier
    # test already imported the real module, so the fake must be bound there too.
    import hermes_cli
    monkeypatch.setattr(hermes_cli, "gateway_windows", fake, raising=False)
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    assert profiles._cleanup_gateway_service("victim", victim) is True
    assert registered == set()
    assert not startup.exists()
