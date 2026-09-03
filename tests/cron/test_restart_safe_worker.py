"""Restart-safe cron worker handoff and ownership contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def execution_ledger(tmp_path, monkeypatch):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    return executions


def test_execution_owner_moves_to_external_worker_before_running(
    execution_ledger, monkeypatch
):
    record = execution_ledger.create_execution("job-1", source="builtin")
    monkeypatch.setattr(execution_ledger.os, "getpid", lambda: 4242)
    monkeypatch.setattr(execution_ledger, "_process_start_time", lambda pid: 9876)

    adopted = execution_ledger.adopt_claimed_execution(record["id"])

    assert adopted is not None
    assert adopted["pid"] == 4242
    assert adopted["process_started_at"] == 9876
    assert adopted["status"] == "running"
    assert execution_ledger.adopt_claimed_execution(record["id"]) is None
    assert execution_ledger.mark_execution_running(record["id"]) is None


def test_restart_safe_gateway_child_fails_closed_without_scope(monkeypatch):
    import tools.process_registry as process_registry

    monkeypatch.setattr(process_registry, "_IS_WINDOWS", False)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setenv("INVOCATION_ID", "managed-service")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)

    with pytest.raises(RuntimeError, match="systemd-run --user --scope is unavailable"):
        process_registry.restart_safe_gateway_child_argv(
            ["python", "worker.py"], unit_suffix="cron-job-1"
        )


def test_restart_safe_gateway_child_is_unchanged_outside_managed_gateway(monkeypatch):
    import tools.process_registry as process_registry

    command = ["python", "worker.py"]
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)

    assert process_registry.restart_safe_gateway_child_argv(
        command, unit_suffix="cron-job-1"
    ) is command


def test_external_worker_adopts_execution_and_runs_payload_once(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    payload = tmp_path / "payload.json"
    ack = tmp_path / "ready.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": "exec-1"},
            "profile_home": str(tmp_path / "profile"),
        }),
        encoding="utf-8",
    )
    from hermes_constants import get_hermes_home

    observed_homes = []
    adopted = Mock(
        side_effect=lambda execution_id: (
            observed_homes.append(get_hermes_home().resolve())
            or {"id": execution_id, "status": "running"}
        )
    )
    run = Mock(
        side_effect=lambda *_args, **_kwargs: (
            observed_homes.append(get_hermes_home().resolve()) or True
        )
    )
    monkeypatch.setattr("cron.executions.adopt_claimed_execution", adopted)
    monkeypatch.setattr(scheduler, "run_one_job", run)

    assert scheduler._run_external_worker_payload(payload, ack) is True

    adopted.assert_called_once_with("exec-1")
    run.assert_called_once()
    assert run.call_args.args[0]["id"] == "job-1"
    expected_home = (tmp_path / "profile").resolve()
    assert observed_homes == [expected_home, expected_home]
    assert ack.exists()
    assert not payload.exists()


def test_external_worker_refuses_to_run_without_durable_ownership(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    payload = tmp_path / "payload.json"
    ack = tmp_path / "ready.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": "exec-1"},
            "profile_home": str(tmp_path / "profile"),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("cron.executions.adopt_claimed_execution", lambda _id: None)
    run = Mock()
    monkeypatch.setattr(scheduler, "run_one_job", run)

    assert scheduler._run_external_worker_payload(payload, ack) is False

    run.assert_not_called()
    assert not ack.exists()


def test_launch_external_worker_uses_restart_safe_scope_and_acknowledges(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    job = {"id": "job-1", "execution_id": "exec-1", "prompt": "work"}
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    wrapped_commands = []

    def wrap(command, *, unit_suffix):
        wrapped_commands.append((command, unit_suffix))
        return ["scope", "--", *command]

    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv", wrap
    )

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

    spawned = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        ack_index = command.index("--ack-file") + 1
        Path(command[ack_index]).write_text(
            json.dumps({"pid": 4321, "execution_id": "exec-1"}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-cross-profile")
    from agent.secret_scope import set_multiplex_active

    set_multiplex_active(True)
    try:
        assert scheduler._launch_external_cron_worker(job) is True
    finally:
        set_multiplex_active(False)
    assert wrapped_commands[0][1] == "cron-job-1-exec-exec-1"
    assert spawned[0][0][0:2] == ["scope", "--"]
    assert spawned[0][1]["start_new_session"] is True
    assert "ANTHROPIC_API_KEY" not in spawned[0][1]["env"]
    payload = json.loads((tmp_path / "cron/external-workers/exec-1.json").read_text())
    assert payload["multiplex_active"] is True


def test_launch_external_worker_stays_in_process_outside_managed_gateway(
    monkeypatch,
):
    import cron.scheduler as scheduler

    command_calls = []

    def unchanged(command, *, unit_suffix):
        command_calls.append((command, unit_suffix))
        return command

    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv", unchanged
    )
    popen = Mock()
    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)

    assert scheduler._launch_external_cron_worker(
        {"id": "job-1", "execution_id": "exec-1"}
    ) is False
    assert command_calls
    popen.assert_not_called()


def test_shared_run_path_hands_gateway_fire_to_external_worker(monkeypatch):
    import cron.scheduler as scheduler

    launch = Mock(return_value=True)
    run = Mock(side_effect=AssertionError("agent ran inside gateway"))
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", launch)
    monkeypatch.setattr(scheduler, "run_job", run)
    job = {"id": "job-1", "execution_id": "exec-1"}

    assert scheduler.run_one_job(job, adapters={"discord": object()}) is True

    launch.assert_called_once_with(job)
    run.assert_not_called()


def test_shared_run_path_creates_execution_before_managed_handoff(monkeypatch):
    import cron.scheduler as scheduler

    created = Mock(return_value={"id": "exec-new"})
    launch = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "create_execution", created)
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", launch)
    job = {"id": "manual-job"}

    assert scheduler.run_one_job(job, adapters={"discord": object()}) is True

    created.assert_called_once_with("manual-job", source="direct")
    assert job["execution_id"] == "exec-new"
    launch.assert_called_once_with(job)


def test_lost_execution_start_cas_prevents_side_effects(monkeypatch):
    import cron.scheduler as scheduler

    run = Mock(side_effect=AssertionError("side effect ran without ownership"))
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(scheduler, "run_job", run)

    assert scheduler.run_one_job(
        {"id": "job-1", "execution_id": "exec-1"}, adapters=None
    ) is True
    run.assert_not_called()
