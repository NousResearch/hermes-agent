from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import gateway
from hermes_cli import update_cmd_fleet


def test_root_update_excludes_foreign_user_service_gateway():
    cmdlines = {
        17595: "python\x00-m\x00hermes_cli.main\x00gateway\x00run\x00",
        17596: "python\x00-m\x00hermes_cli.main\x00--profile\x00writer\x00gateway\x00run\x00",
        17597: "python\x00-m\x00hermes_cli.main\x00gateway\x00status\x00",
    }
    cgroups = {
        17595: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n",
        17596: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway-writer.service\n",
        17597: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service\n",
    }

    with (
        patch.object(gateway.os, "geteuid", return_value=0),
        patch.object(gateway.os, "listdir", return_value=[str(pid) for pid in cmdlines]),
        patch(
            "builtins.open",
            side_effect=lambda path, *args, **kwargs: _BytesReader(
                cgroups[int(path.split("/")[2])] if path.endswith("/cgroup") else cmdlines[int(path.split("/")[2])]
            ),
        ),
    ):
        assert gateway._foreign_user_systemd_gateway_pids(all_profiles=True) == {17595, 17596}


def test_service_pid_exclusion_includes_foreign_user_service(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "_foreign_user_systemd_gateway_pids", lambda all_profiles: {17595})
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=0),
    )

    assert gateway._get_service_pids(all_profiles=True) == {17595}


def test_update_does_not_relaunch_sibling_user_services(monkeypatch):
    service_pids = {17595, 17596}
    monkeypatch.setattr(gateway, "_get_service_pids", lambda all_profiles: service_pids)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kwargs: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **kwargs: [])
    monkeypatch.setattr(
        gateway,
        "_prepare_profile_gateway_update_restart",
        lambda *args: (_ for _ in ()).throw(AssertionError("service gateways must not be relaunched manually")),
    )
    outcome = update_cmd_fleet._GatewayRestartOutcome(
        incomplete=False,
        phase_errors=[],
        pre_restart_gateway_pids=[],
        restarted_services=[],
        failed_or_stale_units=[],
        relaunched_profiles=[],
        externally_supervised_profiles=[],
        killed_pids=set(),
    )

    update_cmd_fleet._restart_manual_gateways(outcome, 45.0)

    assert outcome.relaunched_profiles == []


class _BytesReader:
    def __init__(self, value):
        self.value = value.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.value
