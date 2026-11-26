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
    assert ("start it again with: systemctl --user start hermes.service" in out) == (failed_action == "disable")


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("is_system", [False, True])
def test_remove_legacy_units_names_the_start_command_after_a_stop(tmp_path, monkeypatch, capsys, is_system):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / "hermes.service").write_bytes(_LEGACY_UNIT.encode("utf-8"))
    monkeypatch.setattr(gw, "_legacy_unit_search_paths", lambda: [(is_system, unit_dir)])
    monkeypatch.setattr(gw.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **kw: CompletedProcess(args, 1 if args[0] == "disable" else 0))

    gw.remove_legacy_hermes_units(interactive=False)
    scope = "" if is_system else "--user "
    assert f"start it again with: systemctl {scope}start hermes.service" in capsys.readouterr().out


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
def test_system_host_remove_legacy_units_succeeds_and_refuses_on_timeout(tmp_path, monkeypatch, capsys):
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
    assert "timed out after 90 seconds" in capsys.readouterr().out  # the Move shows why, not only which unit

    # Preserved-behavior control: succeeding stop + disable deletes unit and resets failed state.
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **kw: CompletedProcess(args, 0))
    host.remove_legacy_units()
    assert not unit_path.exists()
    assert ["reset-failed", "hermes.service"] in host_calls


def test_migrate_legacy_retains_unit_when_stop_fails(tmp_path, monkeypatch, capsys):
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


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(
    ("failed_action", "failure"),
    [
        ("stop", "timeout"),
        ("disable", "timeout"),
        ("stop", "nonzero"),
        ("disable", "nonzero"),
    ],
)
def test_systemd_install_and_setup_abort_when_legacy_removal_fails(
    tmp_path, monkeypatch, capsys, failed_action, failure
):
    from hermes_cli import gateway_setup_service as setup
    import hermes_constants

    unit_path = _seed_legacy_unit(tmp_path, monkeypatch)
    new_unit = tmp_path / "hermes-gateway.service"
    started: list[str] = []

    def fake_systemctl(args, **kw):
        if args[0] == failed_action:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 90)
            return CompletedProcess(args, 1)
        return CompletedProcess(args, 0)

    monkeypatch.setattr(gw, "_run_systemctl", fake_systemctl)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: new_unit)
    monkeypatch.setattr(gw, "generate_systemd_unit", lambda **kw: "[Service]\n")
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(gw, "_ensure_linger_enabled", lambda: None)
    monkeypatch.setattr(gw, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gw, "_is_service_installed", new_unit.exists)
    monkeypatch.setattr(gw, "_is_service_running", lambda: False)
    monkeypatch.setattr(gw, "_served_profile_needs_no_service", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    monkeypatch.setattr(setup, "record_service_choice", lambda *a: None)
    monkeypatch.setattr(gw, "systemd_start", lambda **kw: started.append("start"))

    # Setup path must not write the replacement unit or call systemd_start().
    result = setup.ensure_gateway_service(install=True)
    assert (result, unit_path.exists(), new_unit.exists(), started) == (False, True, False, [])

    # CLI install path must abort before writing the replacement or calling systemd_start().
    monkeypatch.setattr(gw, "is_wsl", lambda: False)
    cli_args = SimpleNamespace(start_now=True, start_on_login=True, force_unit_path=False)
    with pytest.raises(SystemExit) as excinfo:
        gw._install_systemd_from_cli(cli_args, force=False, system=False, run_as_user=None)
    assert excinfo.value.code == 1
    assert (unit_path.exists(), new_unit.exists(), started) == (True, False, [])


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("euid", [1000, 0])
def test_systemd_install_aborts_on_an_eligible_failure_beside_another_scope(tmp_path, monkeypatch, euid):
    """Non-root: a failed user unit aborts although the skipped system unit may stay. Root: a failed system unit aborts."""
    dirs = {scope: tmp_path / scope for scope in ("user", "system")}
    for unit_dir in dirs.values():
        unit_dir.mkdir()
        (unit_dir / "hermes.service").write_bytes(_LEGACY_UNIT.encode("utf-8"))
    new_unit = tmp_path / "hermes-gateway.service"
    failing_system = euid == 0  # root also removes system scope; fail the scope it is answerable for

    def fake_systemctl(args, system=False, **kw):
        return CompletedProcess(args, 1 if args[0] == "stop" and system == failing_system else 0)

    monkeypatch.setattr(gw, "_legacy_unit_search_paths", lambda: [(False, dirs["user"]), (True, dirs["system"])])
    monkeypatch.setattr(gw.os, "geteuid", lambda: euid)
    monkeypatch.setattr(gw, "_run_systemctl", fake_systemctl)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: new_unit)
    monkeypatch.setattr(gw, "generate_systemd_unit", lambda **kw: "[Service]\n")
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(gw, "_ensure_linger_enabled", lambda: None)
    monkeypatch.setattr(gw, "print_systemd_scope_conflict_warning", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        gw.systemd_install(system=False, non_interactive=True)
    assert (excinfo.value.code, new_unit.exists()) == (1, False)


@pytest.mark.platforms("linux")
def test_systemd_install_continues_when_non_root_system_legacy_unit_is_skipped(tmp_path, monkeypatch):
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    sys_unit = system_dir / "hermes.service"
    sys_unit.write_bytes(_LEGACY_UNIT.encode("utf-8"))
    new_unit = tmp_path / "hermes-gateway.service"

    monkeypatch.setattr(gw, "_legacy_unit_search_paths", lambda: [(True, system_dir)])
    monkeypatch.setattr(gw.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(gw, "_run_systemctl", lambda args, **kw: CompletedProcess(args, 0))
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: new_unit)
    monkeypatch.setattr(gw, "generate_systemd_unit", lambda **kw: "[Service]\n")
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(gw, "_ensure_linger_enabled", lambda: None)
    monkeypatch.setattr(gw, "print_systemd_scope_conflict_warning", lambda: None)

    gw.systemd_install(system=False, non_interactive=True)
    assert sys_unit.exists()
    assert new_unit.exists()
