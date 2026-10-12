"""Recovery instructions follow failed unit kinds, independently of the control host."""
import pytest
from hermes_cli.update_cmd_fleet import _warn_incomplete_gateway_fleet_restart


@pytest.mark.parametrize("units, launchd, systemd", [
    ([], False, False),
    (["ai.hermes.gateway.default"], True, False),
    (["hermes-gateway.service"], False, True),
    (["ai.hermes.gateway.default", "hermes-gateway.service"], True, True),
])
def test_recovery_hints_follow_failed_unit_kinds(units, launchd, systemd, capsys):
    _warn_incomplete_gateway_fleet_restart(units)
    output = capsys.readouterr().out
    assert ("launchctl kickstart" in output) is launchd
    assert ("systemctl --user restart" in output) is systemd
