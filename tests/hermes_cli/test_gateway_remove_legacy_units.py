"""Direct unit tests for legacy gateway unit removal exit-code and exception handling."""

from __future__ import annotations

import subprocess
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_migrate_backends_host as host_mod
from hermes_cli.gateway_command_errors import SystemctlUnavailableError
from hermes_cli.gateway_migrate_backends_host import SystemHost

_LEGACY_UNIT = (
    "[Unit]\nDescription=Hermes Gateway\n[Service]\n"
    "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
)


def _seed_legacy_unit(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    unit_path = user_dir / "hermes.service"
    unit_path.write_bytes(_LEGACY_UNIT.encode("utf-8"))
    monkeypatch.setattr(gw, "_legacy_unit_search_paths", lambda: [(False, user_dir)])
    return unit_path


@pytest.mark.parametrize("failed_action", ["stop", "disable"])
def test_remove_legacy_units_retains_unit_on_nonzero_systemctl(tmp_path, monkeypatch, capsys, failed_action):
    unit_path = _seed_legacy_unit(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def fake_systemctl(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 1 if args[0] == failed_action else 0)

    monkeypatch.setattr(gw, "_run_systemctl", fake_systemctl)

    removed, remaining = gw.remove_legacy_hermes_units(interactive=False)

    assert (removed, remaining, unit_path.exists()) == (0, [unit_path], True)
    assert unit_path.read_bytes() == _LEGACY_UNIT.encode("utf-8")
    assert [failed_action, "hermes.service"] in calls
    if failed_action == "stop":
        assert ["disable", "hermes.service"] not in calls
    assert ["daemon-reload"] in calls
    out = capsys.readouterr().out
    assert f"systemctl {failed_action} hermes.service exited with status 1" in out


@pytest.mark.parametrize(
    "exc",
    [
        SystemctlUnavailableError(),
        subprocess.TimeoutExpired(["systemctl", "--user", "stop", "hermes.service"], 90),
    ],
)
def test_remove_legacy_units_retains_unit_when_systemctl_raises(tmp_path, monkeypatch, capsys, exc):
    unit_path = _seed_legacy_unit(tmp_path, monkeypatch)

    def fake_systemctl(args, **kwargs):
        if args[0] == "stop":
            raise exc
        return CompletedProcess(args, 0)

    monkeypatch.setattr(gw, "_run_systemctl", fake_systemctl)

    removed, remaining = gw.remove_legacy_hermes_units(interactive=False)

    assert (removed, remaining, unit_path.exists()) == (0, [unit_path], True)
    assert "Could not remove" in capsys.readouterr().out


@pytest.mark.platforms("linux")
def test_system_host_remove_legacy_units_succeeds_and_refuses_on_timeout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    unit_path = _seed_legacy_unit(tmp_path, monkeypatch)
    host = SystemHost(home)
    monkeypatch.setattr(host, "gateway_kind", lambda: "systemd")
    monkeypatch.setattr(host, "unit_props", lambda unit, scope: {})
    host_calls: list[list[str]] = []
    monkeypatch.setattr(
        host_mod,
        "_systemctl",
        lambda scope, args, **kw: (host_calls.append(args), CompletedProcess(args, 0))[1],
    )

    # Timeout during stop retains the file and refuses removal (no reset-failed).
    def failing_systemctl(args, **kw):
        if args[0] == "stop":
            raise subprocess.TimeoutExpired(args, 90)
        return CompletedProcess(args, 0)

    monkeypatch.setattr(gw, "_run_systemctl", failing_systemctl)
    with pytest.raises(RuntimeError, match="could not remove hermes.service"):
        host.remove_legacy_units()
    assert unit_path.exists()
    assert host_calls == []

    # Preserved-behavior control: succeeding stop + disable deletes unit and resets failed state.
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **kw: CompletedProcess(args, 0))
    host.remove_legacy_units()
    assert not unit_path.exists()
    assert ["reset-failed", "hermes.service"] in host_calls


def test_migrate_legacy_and_systemd_install_retain_unit_when_stop_fails(tmp_path, monkeypatch, capsys):
    unit_path = _seed_legacy_unit(tmp_path, monkeypatch)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "is_macos", lambda: False)
    monkeypatch.setattr(
        gw,
        "_run_systemctl",
        lambda args, **kw: CompletedProcess(args, 1 if args[0] == "stop" else 0),
    )

    gw._cmd_migrate_legacy(SimpleNamespace(dry_run=False, yes=True))
    assert unit_path.exists()
    assert "1 legacy unit(s) still present" in capsys.readouterr().out

    new_unit = tmp_path / "hermes-gateway.service"
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: new_unit)
    monkeypatch.setattr(gw, "generate_systemd_unit", lambda system=False, run_as_user=None: "[Service]\n")
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(gw, "_ensure_linger_enabled", lambda: None)
    monkeypatch.setattr(gw, "print_systemd_scope_conflict_warning", lambda: None)

    gw.systemd_install(non_interactive=True)
    assert unit_path.exists()
    assert new_unit.exists()
    assert "1 legacy unit(s) still present" in capsys.readouterr().out
