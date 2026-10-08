"""Custom systemd-owned serve inventory and restart contracts."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from hermes_cli import main_dashboard
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
from hermes_cli import update_inventory as inventory
from hermes_cli import update_restart_recovery as recovery


_UNIT = "hermes-desktop.service"
_CGROUP = f"/system.slice/{_UNIT}"


def _entry(pid: int = 4321, purpose: str = "serve") -> dict:
    return {
        "pid": pid,
        "purpose": purpose,
        "profile": "default",
        "argv": "hermes serve --host 127.0.0.1 --port 9119",
        "create_time": 123.0,
    }


def _systemd_runtime(pid: int = 4321, unit: str = _UNIT, scope: str = "system") -> RuntimeRecord:
    cgroup = f"/{'system.slice' if scope == 'system' else 'user.slice/user-1000.slice/user@1000.service/app.slice'}/{unit}"
    return RuntimeRecord(
        kind="serve", profile="default", pid=pid, supervisor="systemd", restart_via="systemd",
        detail={"systemd_scope": scope, "systemd_unit": unit, "systemd_cgroup": cgroup},
    )


def _collect_one_ledger_runtime(monkeypatch, *, main_pid: int, spawner_dead: bool = True,
                                ssh: bool = False, launchd=None) -> RuntimeRecord:
    entry = _entry()
    monkeypatch.setattr(inventory, "_loaded_backend_launchd_jobs", lambda: [launchd] if launchd else [])
    monkeypatch.setattr(inventory, "_launchd_owner_for_ledger_entry", lambda *a: launchd)
    monkeypatch.setattr(inventory, "_is_desktop_ssh_ledger_entry", lambda _entry: ssh)
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda: [entry])
    monkeypatch.setattr("hermes_cli.process_identity.spawner_is_dead", lambda _entry: spawner_dead)
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway._ensure_user_systemd_env", lambda: None)
    monkeypatch.setattr(main_dashboard, "_pid_unified_cgroup_entries", lambda _pid: iter([_CGROUP]))
    monkeypatch.setattr(
        main_dashboard,
        "_run_probe",
        lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout=str(main_pid), stderr=""),
    )
    plan = UpdatePlan()
    inventory._collect_ledger_runtimes(plan, set())
    assert len(plan.runtimes) == 1
    return plan.runtimes[0]


@pytest.mark.platforms("linux")
def test_custom_systemd_mainpid_is_in_plan_with_scope_and_restart_command(monkeypatch, capsys):
    runtime = _collect_one_ledger_runtime(monkeypatch, main_pid=4321)
    assert runtime.supervisor == "systemd"
    assert runtime.restart_via == "systemd"
    assert runtime.detail["systemd_scope"] == "system"
    assert runtime.detail["systemd_unit"] == _UNIT
    assert runtime.detail["systemd_cgroup"] == _CGROUP

    inventory.print_update_plan(UpdatePlan(runtimes=[runtime]))
    output = capsys.readouterr().out
    assert "— systemd" in output
    assert f"systemctl --no-ask-password restart {_UNIT}" in output


@pytest.mark.platforms("linux")
def test_inherited_service_cgroup_without_matching_mainpid_stays_manual(monkeypatch):
    runtime = _collect_one_ledger_runtime(monkeypatch, main_pid=991)
    assert runtime.supervisor == "manual-serve"
    assert runtime.restart_via == "respawn-argv"
    assert "systemd_unit" not in runtime.detail


@pytest.mark.parametrize(
    ("spawner_dead", "ssh", "launchd", "expected"),
    [
        (False, False, None, "desktop"),
        (True, True, None, "desktop-ssh"),
        (True, False, ("gui/1000", "custom.backend", 4321), "launchd"),
        (True, False, None, "manual-serve"),
    ],
)
def test_existing_serve_supervisors_keep_their_classification(
    monkeypatch, spawner_dead, ssh, launchd, expected,
):
    runtime = _collect_one_ledger_runtime(
        monkeypatch, main_pid=999, spawner_dead=spawner_dead, ssh=ssh, launchd=launchd,
    )
    assert runtime.supervisor == expected


class _TargetSystemctl:
    def __init__(self, pid: int, *, restart_rc: int = 0):
        self.pid = pid
        self.restart_rc = restart_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if "show" in argv:
            return SimpleNamespace(returncode=0, stdout=str(self.pid), stderr="")
        if "is-active" in argv:
            return SimpleNamespace(returncode=0, stdout="active", stderr="")
        if "restart" in argv:
            if self.restart_rc == 0:
                self.pid += 1000
            return SimpleNamespace(returncode=self.restart_rc, stdout="", stderr="denied")
        raise AssertionError(f"unexpected systemctl request: {argv}")


@pytest.fixture
def custom_systemctl(monkeypatch):
    from hermes_cli import update_cmd_fleet

    monkeypatch.setattr(recovery.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        update_cmd_fleet, "_resolve_manage_cmd",
        lambda _cache, _scope, scope_cmd, _unit: list(scope_cmd) + ["--no-ask-password"],
    )
    fake = _TargetSystemctl(4321)
    monkeypatch.setattr(recovery, "_pid_cgroup", lambda _pid: _CGROUP)
    return fake


@pytest.mark.platforms("linux")
def test_arbitrary_custom_unit_restarts_once_after_mainpid_and_cgroup_checks(monkeypatch, custom_systemctl):
    targets = [
        {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP},
        # A second runtime row for the same owning unit cannot trigger a second restart.
        {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP},
    ]
    result = recovery.restart_serve_units(
        targets=targets, include_discovered=False, run=custom_systemctl, sleep=lambda _seconds: None,
    )
    assert result == {"verified": ["system/hermes-desktop"], "failed": []}
    restarts = [argv for argv in custom_systemctl.calls if "restart" in argv]
    assert len(restarts) == 1
    assert restarts[0] == ["/usr/bin/systemctl", "--no-ask-password", "restart", _UNIT]


@pytest.mark.parametrize(
    ("current_pid", "current_cgroup"),
    [(4321, "/system.slice/other.service"), (991, _CGROUP)],
    ids=["inherited-cgroup", "changed-mainpid"],
)
@pytest.mark.platforms("linux")
def test_target_without_current_ownership_is_never_restarted(
    monkeypatch, custom_systemctl, current_pid, current_cgroup,
):
    custom_systemctl.pid = current_pid
    monkeypatch.setattr(recovery, "_pid_cgroup", lambda _pid: current_cgroup)
    target = {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP}
    result = recovery.restart_serve_units(
        targets=[target], include_discovered=False, run=custom_systemctl, sleep=lambda _seconds: None,
    )
    assert result == {"verified": [], "failed": ["system/hermes-desktop"]}
    assert not any("restart" in argv for argv in custom_systemctl.calls)


@pytest.mark.platforms("linux")
def test_system_unit_restart_without_authorization_fails_closed(monkeypatch, custom_systemctl):
    from hermes_cli import update_cmd_fleet

    monkeypatch.setattr(update_cmd_fleet, "_resolve_manage_cmd", lambda *_args: None)
    target = {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP}
    result = recovery.restart_serve_units(
        targets=[target], include_discovered=False, run=custom_systemctl, sleep=lambda _seconds: None,
    )
    assert result == {"verified": [], "failed": ["system/hermes-desktop"]}
    restart_calls = [argv for argv in custom_systemctl.calls if "restart" in argv]
    assert restart_calls == []


@pytest.mark.platforms("linux")
def test_user_unit_restart_preserves_user_manager_scope(monkeypatch, custom_systemctl):
    from hermes_cli import update_cmd_fleet

    unit = "hermes-custom-serve.service"
    cgroup = f"/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}"
    monkeypatch.setattr(recovery, "_pid_cgroup", lambda _pid: cgroup)
    monkeypatch.setattr(
        update_cmd_fleet, "_resolve_manage_cmd",
        lambda _cache, _scope, scope_cmd, _unit: list(scope_cmd) + ["--no-ask-password"],
    )
    target = {"scope": "user", "unit": unit, "pid": 4321, "cgroup": cgroup}

    result = recovery.restart_serve_units(
        targets=[target], include_discovered=False, run=custom_systemctl, sleep=lambda _seconds: None,
    )

    assert result == {"verified": ["user/hermes-custom-serve"], "failed": []}
    assert [argv for argv in custom_systemctl.calls if "restart" in argv] == [
        ["/usr/bin/systemctl", "--user", "--no-ask-password", "restart", unit]
    ]


def test_interrupted_update_recovery_carries_verified_custom_unit_target(monkeypatch):
    from hermes_cli import update_abort_recovery

    runtime = _systemd_runtime()
    plan = UpdatePlan(runtimes=[runtime])
    monkeypatch.setattr(update_abort_recovery, "_serve_unit_recovery_available", lambda: False)
    captured = {}

    def fake_run(argv, **kwargs):
        captured["payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "verified": [], "relaunch_attempted": [], "failed": [],
                "serve_units": {"verified": ["system/hermes-desktop"], "failed": []},
            }),
        )

    monkeypatch.setattr(update_abort_recovery.subprocess, "run", fake_run)
    result = update_abort_recovery._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert captured["payload"]["serve_units"]["recover"] is True
    assert captured["payload"]["serve_units"]["targets"] == [
        {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP},
    ]
    assert result["serve_units"]["verified"] == ["system/hermes-desktop"]


def test_failed_interrupted_recovery_records_custom_unit_as_unrecovered(monkeypatch):
    from hermes_cli import update_abort_recovery

    plan = UpdatePlan(runtimes=[_systemd_runtime()])
    monkeypatch.setattr(update_abort_recovery, "_serve_unit_recovery_available", lambda: False)
    monkeypatch.setattr(update_abort_recovery, "_run_fresh_recovery_process", lambda *args, **kwargs: None)

    result = update_abort_recovery._recover_gateway_restart_after_abort(plan, gateway_mode=False)

    assert result["serve_units"] == {
        "verified": [], "failed": ["system/hermes-desktop"],
    }


def test_recovery_payload_accepts_only_plan_verified_custom_skip_targets():
    target = {"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP}
    payload = {
        "profiles": [],
        "serve_units": {
            "recover": True,
            "skip": [
                {"scope": "system", "unit": _UNIT},
                {"scope": "user", "unit": "unrelated.service"},
            ],
            "targets": [target],
        },
    }
    _, _, recover_serve, skip, targets = recovery._parse_payload(
        io.StringIO(json.dumps(payload))
    )
    assert recover_serve is True
    assert skip == ["system/hermes-desktop"]
    assert targets == [target]
