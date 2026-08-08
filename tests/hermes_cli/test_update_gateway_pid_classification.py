"""Regression coverage for update-time gateway PID classification."""

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.gateway as gateway
import hermes_cli.update_cmd as update_cmd


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


def test_cgroup_classifies_ordinary_manual_gateway(tmp_path):
    proc_root = tmp_path / "proc"
    _write_cgroup(
        proc_root,
        102,
        "0::/user.slice/user-1000.slice/session-8.scope\n",
    )

    assert (
        update_cmd._systemd_service_membership_for_pid(102, proc_root=proc_root)
        is False
    )


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
    proc_root = tmp_path / "proc"
    _write_cgroup(proc_root, 201, "0::/system.slice/custom-gateway.service\n")
    _write_cgroup(proc_root, 202, "0::/user.slice/user-1000.slice/session-8.scope\n")
    (proc_root / "203" / "cgroup").mkdir(parents=True)
    classify = update_cmd._systemd_service_membership_for_pid
    monkeypatch.setattr(
        update_cmd,
        "_systemd_service_membership_for_pid",
        lambda pid: classify(pid, proc_root=proc_root),
    )

    selected = update_cmd._select_update_manual_gateway_pids(
        [200, 201, 202, 203, 202],
        known_service_pids={200},
        systemd_supported=True,
    )

    assert selected == [202]


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

    assert gateway._get_service_pids() == set()
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
