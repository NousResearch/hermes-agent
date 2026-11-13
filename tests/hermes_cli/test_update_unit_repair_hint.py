"""Service-specific exit-78 repair guidance (the dashboard observation in #146577)."""

import shlex

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli import update_cmd_fleet as fleet


@pytest.fixture
def unit_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_cli, "_SYSTEM_UNIT_DIR", tmp_path)
    monkeypatch.setattr(gateway_cli, "user_systemd_unit_dir", lambda: tmp_path)
    monkeypatch.setattr(gateway_cli, "get_service_name", lambda: "hermes-gateway")
    return tmp_path


@pytest.mark.parametrize("scope", ["system", "user"])
@pytest.mark.parametrize("service", ["hermes-dashboard", "custom-hermes", "hermes-gateway-other"])
def test_custom_unit_repair_targets_its_own_drop_in(unit_dir, monkeypatch, capsys, scope, service):
    original = "[Service]\nExecStart=/opt/custom/hermes dashboard\nRestart=on-failure\n"
    unit = unit_dir / f"{service}.service"
    unit.write_text(original, encoding="utf-8")
    monkeypatch.setattr(fleet, "_needs_sudo", lambda scope: False)

    def refuse_gateway_refresh(**kwargs):
        pytest.fail("A custom service must not be repaired by regenerating the gateway unit")

    monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", refuse_gateway_refresh)
    fleet._repair_unit_without_fatal_exit_park(service, scope)

    output = capsys.readouterr().out
    repair = output.split("Repair: ", 1)[1].splitlines()[0]
    command = ["sudo", "systemctl"] if scope == "system" else ["systemctl", "--user"]
    assert shlex.split(repair) == [*command, "edit", f"{service}.service"]
    lines = [line.strip() for line in output.splitlines()]
    assert "[Service]" in lines
    assert f"RestartPreventExitStatus={gateway_cli.GATEWAY_FATAL_CONFIG_EXIT_CODE}" in lines
    assert unit.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("scope", ["system", "user"])
@pytest.mark.parametrize("service", ["hermes-gateway", "hermes-gateway-ops"])
def test_managed_gateway_keeps_its_installer_repair(unit_dir, monkeypatch, capsys, scope, service):
    # The installer repairs the active profile's service, including named profiles.
    monkeypatch.setattr(gateway_cli, "get_service_name", lambda: service)
    unit = unit_dir / f"{service}.service"
    unit.write_text("[Service]\nRestart=on-failure\n", encoding="utf-8")
    monkeypatch.setattr(fleet, "_needs_sudo", lambda scope: True)

    fleet._repair_unit_without_fatal_exit_park(service, scope)

    repair = capsys.readouterr().out.split("Repair: ", 1)[1].splitlines()[0]
    expected = ["sudo", "hermes", "gateway", "install", "--system"] if scope == "system" else [
        "hermes", "gateway", "install",
    ]
    assert shlex.split(repair) == expected
