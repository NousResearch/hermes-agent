"""Renaming a profile removes the gateway service registered under its old name.

The launchd/systemd unit and the s6 slot are named after the profile and start ``--profile
<old>``. A rename that kept them left a service that the next login or container boot would
start for a profile that no longer exists, while the renamed profile had none.
"""
from __future__ import annotations

import os
import platform
import pwd
import subprocess
from pathlib import Path

import pytest

from hermes_cli import gateway, profiles
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    real = pwd.getpwuid(os.getuid())
    fake = pwd.struct_passwd((real.pw_name, real.pw_passwd, real.pw_uid, real.pw_gid,
                              real.pw_gecos, str(tmp_path), real.pw_shell))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: fake)  # launchd plists live under the account home
    calls: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(profiles.subprocess, "run", _run)
    return tmp_path


def _host_unit_path(profile_dir: Path) -> Path:
    token = set_hermes_home_override(str(profile_dir))
    try:
        if platform.system() == "Darwin":
            return gateway.get_launchd_plist_path()
        return gateway.user_systemd_unit_dir() / f"{gateway.get_service_name()}.service"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("posix")
def test_rename_removes_the_old_names_service_while_the_gateway_is_stopped(profile_env):
    old_dir = profiles.create_profile("coder", no_alias=True)
    unit = _host_unit_path(old_dir)
    assert "coder" in unit.name
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("installed by `hermes -p coder gateway install`\n", encoding="utf-8")

    new_dir = profiles.rename_profile("coder", "dev")

    assert not unit.exists()
    assert not _host_unit_path(new_dir).exists()


def test_rename_moves_the_s6_slot_to_the_new_name(profile_env, monkeypatch):
    class _S6:
        def __init__(self):
            self.slots = {"coder"}

        def supports_runtime_registration(self):
            return True

        def register_profile_gateway(self, name, *, start_now=True):
            self.slots.add(name)

        def unregister_profile_gateway(self, name):
            self.slots.discard(name)

    s6 = _S6()
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.get_service_manager", lambda: s6)
    profiles.create_profile("coder", no_alias=True)

    profiles.rename_profile("coder", "dev")

    assert s6.slots == {"dev"}
