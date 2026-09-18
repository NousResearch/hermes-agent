"""Invariant tests for the bounded Windows gateway supervisor."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import hermes_cli.gateway_windows_supervisor as supervisor_module
from hermes_cli.gateway_windows_supervisor import (
    GatewayWindowsSupervisor,
    build_failure_send_argv,
    build_gateway_child_argv,
    claim_recovery_marker,
    format_recovery_message,
    mark_recovery_notification_delivered,
    read_recovery_marker,
    recovery_marker_for_pid,
)


@dataclass
class _Clock:
    monotonic_value: float = 100.0
    wall_value: float = 1_000.0

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> float:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += seconds


class _Process:
    def __init__(self, pid: int, returncode: int, lifetime: float, clock: _Clock):
        self.pid = pid
        self.returncode = returncode
        self.lifetime = lifetime
        self.clock = clock

    def wait(self) -> int:
        self.clock.advance(self.lifetime)
        return self.returncode


class _ProcessFactory:
    def __init__(self, clock: _Clock, outcomes):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        pid, returncode, lifetime = self.outcomes.pop(0)
        return _Process(pid, returncode, lifetime, self.clock)


def _supervisor(tmp_path, outcomes, *, backoffs=(5, 15, 30, 60, 60), stable=120):
    clock = _Clock()
    factory = _ProcessFactory(clock, outcomes)
    sleeps = []
    notices = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    supervisor = GatewayWindowsSupervisor(
        home=tmp_path,
        python_exe="python.exe",
        profile="coding-local",
        popen=factory,
        sleep=sleep,
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        notifier=lambda python, profile, message: notices.append((python, profile, message)) or True,
        backoffs=backoffs,
        stable_seconds=stable,
    )
    return supervisor, factory, sleeps, notices, clock


@pytest.mark.parametrize(
    ("first_exit", "expected_spawns", "expected_result", "expected_kind"),
    [
        (0, 1, 0, None),
        (75, 2, 0, "planned"),
        (78, 1, 78, "crash"),
    ],
)
def test_supervisor_semantic_exit_contract(
    tmp_path, first_exit, expected_spawns, expected_result, expected_kind
):
    outcomes = [(101, first_exit, 1)]
    if first_exit == 75:
        outcomes.append((202, 0, 1))
    supervisor, factory, sleeps, notices, _clock = _supervisor(tmp_path, outcomes)

    assert supervisor.run() == expected_result
    assert len(factory.calls) == expected_spawns
    assert sleeps == []
    assert all("--external-supervisor" in call[0] for call in factory.calls)
    assert all(call[1]["env"]["HERMES_GATEWAY_EXTERNAL_SUPERVISOR"] == "1" for call in factory.calls)
    marker = read_recovery_marker(tmp_path)
    assert (marker or {}).get("kind") == expected_kind
    assert bool(notices) is (first_exit == 78)


def test_supervisor_bounds_crash_recovery_and_alerts_once(tmp_path):
    # Initial life + five failed recovery children.
    outcomes = [(100 + i, 1, 1) for i in range(6)]
    supervisor, factory, sleeps, notices, _clock = _supervisor(tmp_path, outcomes)

    assert supervisor.run() == 1
    assert len(factory.calls) == 6
    assert sleeps == [5, 15, 30, 60, 60]
    assert len(notices) == 1
    assert "5 попыток" in notices[0][2]
    marker = read_recovery_marker(tmp_path)
    assert marker is not None
    assert marker["state"] == "failed"
    assert marker["attempt"] == 5
    assert marker["notification_delivered"] is False


def test_stable_child_resets_incident_budget(tmp_path):
    outcomes = [
        (101, 1, 1),     # initial incident
        (102, 1, 121),   # stable recovery, then a new incident
        (103, 0, 1),
    ]
    supervisor, _factory, sleeps, _notices, _clock = _supervisor(tmp_path, outcomes)

    assert supervisor.run() == 0
    assert sleeps == [5, 5]
    marker = read_recovery_marker(tmp_path)
    assert marker is not None
    assert marker["old_pid"] == 102
    assert marker["attempt"] == 1


def test_recovery_marker_is_pid_scoped_and_formats_downtime(tmp_path):
    marker = {
        "schema": "hermes.gateway-recovery.r3",
        "incident_id": "incident",
        "kind": "crash",
        "state": "starting",
        "detected_at": 100.0,
        "old_pid": 10,
        "new_pid": 20,
        "attempt": 2,
        "max_attempts": 5,
        "notification_delivered": False,
    }
    (tmp_path / ".gateway_recovery.json").write_text(json.dumps(marker), encoding="utf-8")

    assert recovery_marker_for_pid(tmp_path, 99) is None
    assert "за 25 с" in format_recovery_message(marker, now=125.0)
    assert "попытка 2/5" in format_recovery_message(marker, now=125.0)
    assert mark_recovery_notification_delivered(tmp_path, 20, now=126.0) is True
    delivered = read_recovery_marker(tmp_path)
    assert delivered is not None
    assert delivered["state"] == "recovered"
    assert delivered["notification_delivered"] is True


def test_gateway_runtime_claim_replaces_windows_launcher_pid(tmp_path):
    marker = {
        "schema": "hermes.gateway-recovery.r3",
        "incident_id": "incident",
        "kind": "planned",
        "state": "starting",
        "detected_at": 100.0,
        "old_pid": 10,
        "new_pid": None,
        "attempt": 0,
        "max_attempts": 5,
        "notification_delivered": False,
    }
    (tmp_path / ".gateway_recovery.json").write_text(json.dumps(marker), encoding="utf-8")

    assert claim_recovery_marker(tmp_path, "wrong-incident", 30) is False
    assert claim_recovery_marker(tmp_path, "incident", 30) is True
    assert recovery_marker_for_pid(tmp_path, 20) is None
    assert recovery_marker_for_pid(tmp_path, 30)["new_pid"] == 30


def test_supervisor_command_builders_preserve_profile_and_isolate_send():
    child = build_gateway_child_argv("python.exe", "coding-local")
    alert = build_failure_send_argv("python.exe", "coding-local", "failed")

    assert child == [
        "python.exe", "-m", "hermes_cli.main", "--profile", "coding-local",
        "gateway", "run", "--external-supervisor",
    ]
    assert alert == [
        "python.exe", "-m", "hermes_cli.main", "--profile", "coding-local",
        "send", "--to", "telegram", "--quiet", "failed",
    ]


def test_spawn_failure_uses_the_same_bounded_recovery_budget(tmp_path):
    clock = _Clock()
    successful_child = _Process(202, 0, 1, clock)
    calls = []
    sleeps = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            raise OSError("synthetic spawn failure")
        return successful_child

    def sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    supervisor = GatewayWindowsSupervisor(
        home=tmp_path,
        python_exe="python.exe",
        profile="coding-local",
        popen=popen,
        sleep=sleep,
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        notifier=lambda *_args: True,
        backoffs=(5,),
        stable_seconds=120,
    )

    assert supervisor.run() == 0
    assert len(calls) == 2
    assert sleeps == [5]
    marker = read_recovery_marker(tmp_path)
    assert marker is not None
    assert marker["old_pid"] == 0
    assert marker["new_pid"] is None
    assert marker["attempt"] == 1


def test_supervisor_records_runtime_pid_instead_of_windows_shim_pid(tmp_path):
    supervisor, _factory, _sleeps, _notices, _clock = _supervisor(
        tmp_path, [(101, 75, 1), (202, 0, 1)]
    )
    (tmp_path / "gateway_state.json").write_text(json.dumps({"pid": 303}), encoding="utf-8")

    assert supervisor.run() == 0
    marker = read_recovery_marker(tmp_path)
    assert marker is not None
    assert marker["old_pid"] == 303
    assert marker["new_pid"] is None


def test_failure_notification_opt_out_never_spawns_sender(monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor_module, "_failure_notification_enabled", lambda: False)

    assert supervisor_module.send_failure_notification(
        "python.exe",
        "coding-local",
        "failed",
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
    ) is False
    assert calls == []
