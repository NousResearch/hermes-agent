"""Service manager consumers preserve persisted control-only gateway launchers."""

from __future__ import annotations

import plistlib
import shlex

import pytest


def _write_unit(path, argv: str) -> bytes:
    data = f"[Unit]\nDescription=operator --control-only note\n\n[Service]\nExecStart={argv}\nRestart=always\n".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _write_plist(path, argv: list[str], **fields) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = plistlib.dumps({"Label": "ai.hermes.gateway-test", "ProgramArguments": argv, **fields})
    path.write_bytes(data)
    return data


def test_systemd_refresh_install_start_and_restart_preserve_control_only_unit(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli.gateway_service_mode import systemd_definition_mode

    unit = tmp_path / "hermes-gateway.service"
    before = _write_unit(unit, "/opt/hermes gateway run --control-only")
    effects = []
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gateway, "_systemd_scope_preamble", lambda *args, **kwargs: False)
    monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_systemd_unit", lambda *args: None)
    monkeypatch.setattr(gateway.gateway_service_owner, "unit_belongs_to_caller", lambda *args: True)
    monkeypatch.setattr(gateway, "_run_systemctl", lambda *args, **kwargs: effects.append(args))

    assert systemd_definition_mode(unit) == "control_only"
    assert gateway.refresh_systemd_unit_if_needed() is False
    assert unit.read_bytes() == before
    for action in (
        lambda: gateway.systemd_install(force=True, enable_on_startup=True),
        gateway.systemd_start,
        gateway.systemd_restart,
    ):
        with pytest.raises(SystemExit):
            action()
        assert unit.read_bytes() == before
    assert effects == []


def test_launchd_refresh_install_start_and_restart_preserve_control_only_plist(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli import gateway_launchd
    from hermes_cli.gateway_service_mode import launchd_definition_mode

    plist = tmp_path / "ai.hermes.gateway-test.plist"
    before = _write_plist(plist, ["/opt/hermes", "gateway", "run", "--control-only"])
    effects = []
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway-test")
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_launchd_plist", lambda *args: None)
    monkeypatch.setattr(gateway.gateway_service_owner, "definition_belongs_to_home", lambda *args: True)
    monkeypatch.setattr(gateway, "_launchctl_bootstrap", lambda *args, **kwargs: effects.append(args))
    monkeypatch.setattr(gateway_launchd, "_launchctl_kickstart_current", lambda *args: effects.append(args))
    monkeypatch.setattr(gateway_launchd.subprocess, "run", lambda *args, **kwargs: effects.append(args))

    assert launchd_definition_mode(plist) == "control_only"
    assert gateway_launchd._rewrite_stale_launchd_plist() is False
    assert plist.read_bytes() == before
    for action in (
        lambda: gateway_launchd.launchd_install(force=True),
        gateway_launchd.launchd_start,
        gateway_launchd.launchd_restart,
    ):
        with pytest.raises(SystemExit):
            action()
        assert plist.read_bytes() == before
    assert effects == []


def test_generated_launchd_program_arguments_allow_serving_replace_through_start_consumer(tmp_path, monkeypatch):
    from hermes_cli import gateway, gateway_launchd
    from hermes_cli.gateway_service_mode import launchd_definition_mode

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    path = tmp_path / "ai.hermes.gateway-test.plist"
    command = gateway_launchd._gateway_run_command()
    assert "--replace" in command
    arguments = gateway_launchd.launchd_program_arguments(
        command, tmp_path / "gateway.log", tmp_path / "gateway.error.log"
    )
    before = _write_plist(path, arguments)
    assert launchd_definition_mode(path) == "serving"

    effects = []
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: path)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway-test")
    monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_launchd_plist", lambda *args: None)
    monkeypatch.setattr(gateway, "refresh_launchd_plist_if_needed", lambda: False)
    monkeypatch.setattr(gateway, "_clear_launchd_unsupported_marker", lambda: None)
    monkeypatch.setattr(gateway_launchd, "_launchctl_kickstart_current", lambda *args: effects.append(args))

    gateway_launchd.launchd_start()

    assert effects == [("ai.hermes.gateway-test",)]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "argv",
    [
        ["/opt/hermes", "gateway", "run", "--replace", "--profile"],
        ["/opt/hermes", "gateway", "run", "--replace", "-p"],
        ["/opt/hermes", "--profile", "", "gateway", "run", "--replace"],
        ["/opt/hermes", "-p", "--control-only", "gateway", "run", "--replace"],
        ["/opt/hermes", "--profile", "--control-only", "gateway", "run", "--replace"],
        ["/opt/hermes", "--profile=", "gateway", "run", "--replace"],
        ["/opt/hermes", "--profile=--control-only", "gateway", "run", "--replace"],
        ["/opt/hermes", "-p", "writer_2", "--profile=writer_3", "gateway", "run", "--replace"],
    ],
    ids=["missing-profile", "missing-short-profile", "empty-profile", "short-option-value",
         "option-value", "empty-equals", "option-equals", "duplicate-profile"],
)
@pytest.mark.parametrize("manager", ["systemd", "launchd"])
def test_malformed_profile_launchers_are_unknown_and_start_is_refused(
    tmp_path, monkeypatch, argv, manager
):
    from hermes_cli import gateway, gateway_launchd
    from hermes_cli.gateway_service_mode import (
        launchd_definition_mode, systemd_definition_mode,
    )

    effects = []
    if manager == "systemd":
        path = tmp_path / "hermes-gateway.service"
        before = _write_unit(path, shlex.join(argv))
        monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: path)
        monkeypatch.setattr(gateway, "_systemd_scope_preamble", lambda *args, **kwargs: False)
        monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_systemd_unit", lambda *args: None)
        monkeypatch.setattr(gateway, "refresh_systemd_unit_if_needed", lambda *args, **kwargs: effects.append("refresh"))
        monkeypatch.setattr(gateway, "_run_systemctl", lambda *args, **kwargs: effects.append(args))
        classify = systemd_definition_mode
        start = gateway.systemd_start
    else:
        path = tmp_path / "ai.hermes.gateway-test.plist"
        before = _write_plist(path, argv)
        monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: path)
        monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway-test")
        monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_launchd_plist", lambda *args: None)
        monkeypatch.setattr(
            gateway, "refresh_launchd_plist_if_needed", lambda *args, **kwargs: effects.append("refresh")
        )
        monkeypatch.setattr(gateway_launchd, "_launchctl_kickstart_current", lambda *args: effects.append(args))
        classify = launchd_definition_mode
        start = gateway_launchd.launchd_start

    assert classify(path) == "unknown"
    with pytest.raises(SystemExit):
        start()
    assert path.read_bytes() == before
    assert effects == []


def test_generated_named_profile_serving_and_control_only_remain_classifiable(tmp_path, monkeypatch):
    from hermes_cli import gateway, gateway_launchd
    from hermes_cli.gateway_service_mode import launchd_definition_mode, systemd_definition_mode

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "writer_2"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    unit = tmp_path / "profiled.service"
    serving_unit = ["/opt/hermes", "--profile", "writer_2", "gateway", "run"]
    _write_unit(unit, shlex.join(serving_unit))
    assert systemd_definition_mode(unit) == "serving"
    control_unit = [*serving_unit, "--control-only"]
    _write_unit(unit, shlex.join(control_unit))
    assert systemd_definition_mode(unit) == "control_only"

    command = gateway_launchd._gateway_run_command()
    profile_index = command.index("--profile")
    assert command[profile_index + 1] == "writer_2"
    serving_plist = plistlib.loads(gateway.generate_launchd_plist().encode("utf-8"))
    launchd_path = tmp_path / "profiled.plist"
    launchd_path.write_bytes(plistlib.dumps(serving_plist))
    assert launchd_definition_mode(launchd_path) == "serving"

    control_command = [arg for arg in command if arg != "--replace"]
    control_command.append("--control-only")
    control_args = gateway_launchd.launchd_program_arguments(
        control_command, tmp_path / "gateway.log", tmp_path / "gateway.error.log"
    )
    launchd_path.write_bytes(plistlib.dumps({"Label": "ai.hermes.gateway-test", "ProgramArguments": control_args}))
    assert launchd_definition_mode(launchd_path) == "control_only"


@pytest.mark.parametrize(
    "program",
    ["/custom/wrapper", 17, ["/custom/wrapper"]],
    ids=["mismatch", "wrong-type", "non-string"],
)
def test_launchd_program_mismatch_or_malformed_refuses_start_without_touching_artifact(
    tmp_path, monkeypatch, program
):
    from hermes_cli import gateway, gateway_launchd
    from hermes_cli.gateway_service_mode import launchd_definition_mode

    path = tmp_path / "ai.hermes.gateway-test.plist"
    before = _write_plist(path, ["/opt/hermes", "gateway", "run", "--replace"], Program=program)
    effects = []
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: path)
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway-test")
    monkeypatch.setattr(gateway.gateway_service_owner, "refuse_foreign_launchd_plist", lambda *args: None)
    monkeypatch.setattr(gateway, "refresh_launchd_plist_if_needed", lambda: effects.append("refresh"))
    monkeypatch.setattr(gateway_launchd, "_launchctl_kickstart_current", lambda *args: effects.append(args))

    assert launchd_definition_mode(path) == "unknown"
    with pytest.raises(SystemExit):
        gateway_launchd.launchd_start()
    assert path.read_bytes() == before
    assert effects == []


@pytest.mark.parametrize("suffix", ["--control-only --replace", "--control-only --force"])
def test_control_only_cannot_be_classified_serving_with_replace_or_force(tmp_path, suffix):
    from hermes_cli.gateway_service_mode import launchd_definition_mode

    path = tmp_path / "control-only.plist"
    _write_plist(path, ["/opt/hermes", "gateway", "run", *suffix.split()])
    assert launchd_definition_mode(path) == "unknown"


def test_plist_serialization_of_actual_generated_launchd_definition_is_portable(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli.gateway_service_mode import launchd_definition_mode

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    plist_path = tmp_path / "generated-launchd.plist"
    plist_path.write_text(gateway.generate_launchd_plist(), encoding="utf-8")
    assert launchd_definition_mode(plist_path) == "serving"


def test_serving_definitions_still_refresh_and_comments_do_not_set_mode(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli import gateway_launchd
    from hermes_cli.gateway_service_mode import launchd_definition_mode, systemd_definition_mode

    unit = tmp_path / "hermes-gateway.service"
    _write_unit(unit, "/opt/hermes gateway run")
    plist = tmp_path / "ai.hermes.gateway-test.plist"
    _write_plist(plist, ["/opt/hermes", "gateway", "run"])
    assert systemd_definition_mode(unit) == "serving"
    assert launchd_definition_mode(plist) == "serving"

    systemd_effects = []
    replacement_unit = "[Unit]\nDescription=updated\n[Service]\nExecStart=/opt/hermes gateway run\n"
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gateway.gateway_service_owner, "unit_belongs_to_caller", lambda *args: True)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: False)
    monkeypatch.setattr(gateway, "generate_systemd_unit", lambda **kwargs: replacement_unit)
    monkeypatch.setattr(gateway, "_prepare_service_launcher", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_run_systemctl", lambda args, **kwargs: systemd_effects.append(args))
    assert gateway.refresh_systemd_unit_if_needed() is True
    assert unit.read_text(encoding="utf-8") == replacement_unit
    assert systemd_effects == [["daemon-reload"]]

    replacement_plist = plistlib.dumps({
        "Label": "ai.hermes.gateway-test", "ProgramArguments": ["/opt/hermes", "gateway", "run", "--quiet"]
    })
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gateway, "generate_launchd_plist", lambda: replacement_plist.decode())
    monkeypatch.setattr(gateway_launchd.gateway_service_owner, "definition_belongs_to_home", lambda *args: True)
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.setattr(gateway, "_prepare_service_launcher", lambda: None)
    assert gateway_launchd._rewrite_stale_launchd_plist() is True
    assert plistlib.loads(plist.read_bytes())["ProgramArguments"] == [
        "/opt/hermes", "gateway", "run", "--quiet"
    ]


@pytest.mark.platforms("posix")
def test_generated_systemd_and_launchd_serving_definitions_remain_recognized(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli.gateway_service_mode import launchd_definition_mode, systemd_definition_mode

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    systemd = tmp_path / "generated.service"
    launchd = tmp_path / "generated.plist"
    systemd.write_text(gateway.generate_systemd_unit(), encoding="utf-8")
    launchd.write_text(gateway.generate_launchd_plist(), encoding="utf-8")

    assert systemd_definition_mode(systemd) == "serving"
    assert launchd_definition_mode(launchd) == "serving"


def test_update_fleet_does_not_restart_control_only_effective_systemd_command(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import update_cmd_fleet

    effects = []
    failures = []
    def systemctl(args, **kwargs):
        effects.append(args)
        return SimpleNamespace(stdout=(
            "active\n" if "is-active" in args else
            "{ path=/opt/hermes ; argv[]=/opt/hermes gateway run --control-only ; ignore_errors=no ; start_time=[n/a]; }\n"
        ))
    monkeypatch.setattr(update_cmd_fleet, "_systemctl", systemctl)
    monkeypatch.setattr(update_cmd_fleet, "_systemd_unit_owned_by_update", lambda *args: True)
    monkeypatch.setattr(update_cmd_fleet, "_repair_unit_without_fatal_exit_park", lambda *args: effects.append("repair"))

    update_cmd_fleet._restart_one_systemd_gateway_unit(
        "hermes-gateway", scope="user", scope_cmd=["systemctl", "--user"], drain_budget=1,
        _manage_cmd_cache={}, restarted_services=[], failed_or_stale_units=failures,
    )

    assert effects == [
        ["systemctl", "--user", "is-active", "hermes-gateway"],
        ["systemctl", "--user", "show", "hermes-gateway", "--property=ExecStart", "--value"],
    ]
    assert failures == ["hermes-gateway"]

