"""Regression coverage for update-time gateway PID classification."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway
from hermes_cli import update_cmd
from hermes_cli import update_cmd_fleet as fleet


def _write_cgroup(proc_root: Path, pid: int, content: str) -> None:
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True)
    (pid_dir / "cgroup").write_text(content, encoding="utf-8")


def test_cgroup_classifies_custom_named_systemd_service(tmp_path):
    proc_root = tmp_path / "proc"
    _write_cgroup(
        proc_root,
        101,
        "0::/system.slice/acme-operations-agent.service\n",
    )

    assert (
        update_cmd._systemd_service_membership_for_pid(101, proc_root=proc_root) is True
    )


def test_cgroup_classifies_legacy_named_systemd_service(tmp_path):
    proc_root = tmp_path / "proc"
    _write_cgroup(
        proc_root,
        105,
        "12:name=systemd:/system.slice/gaiasignal-agent-hermes.service\n",
    )

    assert (
        update_cmd._systemd_service_membership_for_pid(105, proc_root=proc_root) is True
    )


def test_cgroup_classification_accounts_for_reader_namespace():
    from hermes_cli.update_cmd_gateway_ownership import _classify_systemd_cgroup

    manual = "0::/user.slice/user-1000.slice/session-8.scope\n"
    hidden_ancestor = "0::/worker\n"
    visible_service = "0::/custom.service/worker\n"
    assert _classify_systemd_cgroup(manual, initial_namespace=True) is False
    assert _classify_systemd_cgroup(hidden_ancestor, initial_namespace=False) is None
    assert _classify_systemd_cgroup(visible_service, initial_namespace=False) is True


def test_cgroup_unreadable_or_ambiguous_is_unknown(tmp_path):
    proc_root = tmp_path / "proc"
    (proc_root / "103" / "cgroup").mkdir(parents=True)
    _write_cgroup(proc_root, 104, "7:cpu,cpuacct:/legacy/path\n")

    assert (
        update_cmd._systemd_service_membership_for_pid(103, proc_root=proc_root) is None
    )
    assert (
        update_cmd._systemd_service_membership_for_pid(104, proc_root=proc_root) is None
    )


def test_update_manual_sweep_only_selects_confirmed_non_service(monkeypatch, tmp_path):
    monkeypatch.setattr(
        update_cmd,
        "_systemd_service_membership_for_pid",
        lambda pid: {201: True, 202: False, 203: None}[pid],
    )

    selected = update_cmd._select_update_manual_gateway_pids(
        [200, 201, 202, 203, 202],
        known_service_pids={200},
        systemd_supported=True,
    )

    assert selected == [202]


@pytest.mark.platforms("linux")
def test_update_manual_sweep_fails_safe_when_capability_probe_is_false(
    monkeypatch, tmp_path
):
    proc_root = tmp_path / "proc"
    _write_cgroup(proc_root, 204, "0::/system.slice/custom-gateway.service\n")
    (proc_root / "205" / "cgroup").mkdir(parents=True)
    classify = update_cmd._systemd_service_membership_for_pid
    monkeypatch.setattr(
        update_cmd,
        "_systemd_service_membership_for_pid",
        lambda pid: classify(pid, proc_root=proc_root),
    )

    selected = update_cmd._select_update_manual_gateway_pids(
        [204, 205],
        known_service_pids=set(),
        systemd_supported=False,
    )

    assert selected == []


def test_update_manual_sweep_preserves_non_systemd_behavior(monkeypatch):
    classified: list[int] = []

    def classify(pid: int) -> bool:
        classified.append(pid)
        return False

    monkeypatch.setattr(
        update_cmd,
        "_systemd_service_membership_for_pid",
        classify,
    )

    selected = update_cmd._select_update_manual_gateway_pids(
        [300, 301, 301],
        known_service_pids={300},
        systemd_supported=False,
    )

    assert selected == [301]
    assert classified == [301]


def test_service_pid_discovery_remains_targeted(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert "hermes-gateway*" in command
        assert "--type=service" not in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    assert gateway._get_service_pids(all_profiles=True) == set()
    assert calls == [
        [
            "systemctl",
            "--user",
            "list-units",
            "hermes-gateway*",
            "--plain",
            "--no-legend",
            "--no-pager",
        ],
        [
            "systemctl",
            "list-units",
            "hermes-gateway*",
            "--plain",
            "--no-legend",
            "--no-pager",
        ],
    ]


@pytest.mark.platforms("linux")
def test_fleet_sweep_spares_unmanaged_services_and_unknown_ownership(monkeypatch, capsys):
    """Neither a profile-mapped custom service nor an unknown PID can be signaled."""
    monkeypatch.setattr(gateway, "_get_service_pids", lambda **kw: {100})
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [101, 102, 103])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **kw: [
        SimpleNamespace(pid=101, profile="custom"), SimpleNamespace(pid=102, profile="unknown"),
        SimpleNamespace(pid=103, profile="manual"),
    ])
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(update_cmd, "_systemd_service_membership_for_pid", lambda pid: {101: True, 102: None, 103: False}[pid])
    monkeypatch.setattr("hermes_cli.update_cmd_posix_pause.already_restarted", lambda: {"pids": set()})
    monkeypatch.setattr(fleet, "_scoped_manual_gateway_pids", lambda pids, **kw: list(pids))
    armed, drained = [], []
    monkeypatch.setattr(gateway, "_prepare_profile_gateway_update_restart", lambda profile, pid: armed.append(pid) or "watcher")
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **kw: None)
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", lambda pid, *a, **kw: drained.append(pid) or True)
    out = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )

    fleet._restart_manual_gateways(out, 5)

    assert armed == drained == [103]
    assert out.killed_pids == {103}
    assert out.incomplete is True
    assert "101, 102" in capsys.readouterr().out


def test_cgroup_root_is_ambiguous(tmp_path):
    _write_cgroup(tmp_path, 700, "0::/\n")
    assert update_cmd._systemd_service_membership_for_pid(700, proc_root=tmp_path) is None


def test_unreadable_reader_namespace_keeps_non_service_path_unknown(tmp_path):
    _write_cgroup(tmp_path, 704, "0::/worker\n")
    assert update_cmd._systemd_service_membership_for_pid(704, proc_root=tmp_path) is None


@pytest.mark.platforms("linux")
def test_real_reader_namespace_is_used_for_live_pid():
    import os
    from hermes_cli.update_cmd_gateway_ownership import _classify_systemd_cgroup

    proc_root = Path("/proc")
    text = (proc_root / "self" / "cgroup").read_text(encoding="utf-8-sig")
    initial = (proc_root / "self" / "ns" / "cgroup").stat().st_ino == 0xEFFFFFFB
    assert update_cmd._systemd_service_membership_for_pid(os.getpid()) == _classify_systemd_cgroup(
        text, initial_namespace=initial,
    )


def test_service_cgroup_identity_includes_ancestor_unit(tmp_path):
    from hermes_cli.update_cmd_gateway_ownership import _systemd_service_cgroup_for_pid
    _write_cgroup(tmp_path, 701, "0::/system.slice/custom.service/child.scope\n")
    _write_cgroup(tmp_path, 702, "0::/user.slice/session-1.scope\n")
    assert _systemd_service_cgroup_for_pid(701, proc_root=tmp_path) == "/system.slice/custom.service"
    assert _systemd_service_cgroup_for_pid(702, proc_root=tmp_path) is None
    assert _systemd_service_cgroup_for_pid(703, proc_root=tmp_path) is None
