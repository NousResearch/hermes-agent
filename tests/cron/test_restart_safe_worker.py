"""Restart-safe cron worker handoff and ownership contracts."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
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
    assert execution_ledger.mark_execution_handoff_pending(record["id"]) is not None
    monkeypatch.setattr(execution_ledger.os, "getpid", lambda: 4242)
    monkeypatch.setattr(execution_ledger, "_process_start_time", lambda pid: 9876)

    adopted = execution_ledger.adopt_claimed_execution(record["id"])

    assert adopted is not None
    assert adopted["pid"] == 4242
    assert adopted["process_started_at"] == 9876
    assert adopted["status"] == "running"
    assert execution_ledger.adopt_claimed_execution(record["id"]) is None
    assert execution_ledger.mark_execution_running(record["id"]) is None


def test_external_worker_cannot_adopt_execution_without_handoff_fence(
    execution_ledger, monkeypatch
):
    record = execution_ledger.create_execution("job-unfenced", source="builtin")
    monkeypatch.setattr(execution_ledger.os, "getpid", lambda: 4242)
    monkeypatch.setattr(execution_ledger, "_process_start_time", lambda _pid: 9876)

    assert execution_ledger.adopt_claimed_execution(record["id"]) is None
    assert execution_ledger.get_execution(record["id"])["status"] == "claimed"


def test_genuine_external_worker_crash_is_recovered_unknown(
    execution_ledger, monkeypatch
):
    record = execution_ledger.create_execution("job-crash", source="builtin")
    assert execution_ledger.mark_execution_handoff_pending(record["id"]) is not None
    script = (
        "import os\n"
        "from pathlib import Path\n"
        "import cron.executions as executions\n"
        f"executions.EXECUTIONS_FILE = Path({str(execution_ledger.EXECUTIONS_FILE)!r})\n"
        f"assert executions.adopt_claimed_execution({record['id']!r}) is not None\n"
        "os._exit(9)\n"
    )

    crashed = subprocess.run([sys.executable, "-c", script], check=False)
    assert crashed.returncode == 9

    monkeypatch.setattr(execution_ledger, "_PROCESS_ID", "replacement-scheduler")
    assert execution_ledger.recover_interrupted_executions() == 1
    recovered = execution_ledger.latest_execution("job-crash")
    assert recovered["status"] == "unknown"
    assert "whether side effects ran is unknown" in recovered["error"]


@pytest.mark.linux_only
def test_restart_safe_gateway_child_fails_closed_without_scope(monkeypatch):
    import tools.process_registry as process_registry

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


def test_restart_safe_gateway_child_never_probes_systemd_off_linux(monkeypatch):
    import tools.process_registry as process_registry

    command = ["python", "worker.py"]
    probe = Mock(side_effect=AssertionError("systemd probe ran off Linux"))
    monkeypatch.setattr(process_registry, "_IS_LINUX", False)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", probe)
    monkeypatch.setenv("INVOCATION_ID", "managed-service")

    assert process_registry.restart_safe_gateway_child_argv(
        command, unit_suffix="cron-job-1"
    ) is command
    probe.assert_not_called()


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

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            return self.returncode

    spawned = []

    payloads = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        payload_index = command.index("--external-worker-file") + 1
        payloads.append(json.loads(Path(command[payload_index]).read_text()))
        ack_index = command.index("--ack-file") + 1
        Path(command[ack_index]).write_text(
            json.dumps({"pid": 4321, "execution_id": "exec-1"}),
            encoding="utf-8",
        )
        return FakeProcess()

    handoff = Mock(return_value={"id": "exec-1", "handoff_pending": 1})
    monkeypatch.setattr(scheduler, "mark_execution_handoff_pending", handoff)
    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    observed_statuses = iter(
        [
            {"id": "exec-1", "status": "running"},
            {"id": "exec-1", "status": "completed"},
        ]
    )
    get = Mock(side_effect=lambda _execution_id: next(observed_statuses))
    monkeypatch.setattr(scheduler, "get_execution", get)
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
    handoff.assert_called_once_with("exec-1")
    assert get.call_count == 2
    assert payloads[0]["multiplex_active"] is True
    # Once the attempt is terminal the parent reaps its own handoff artifacts.
    assert not (tmp_path / "cron/external-workers/exec-1.json").exists()


def test_external_worker_exit_rechecks_exact_execution_before_failure(monkeypatch):
    import cron.scheduler as scheduler

    statuses = iter(
        [
            {"id": "exec-1", "status": "running"},
            {"id": "exec-1", "status": "completed"},
        ]
    )
    get = Mock(side_effect=lambda _execution_id: next(statuses))
    monkeypatch.setattr(scheduler, "get_execution", get, raising=False)
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0

    assert scheduler._wait_for_external_cron_worker(
        process, execution_id="exec-1"
    ) is True
    assert get.call_count == 2


def test_external_worker_crash_recovers_uncertain_attempt(monkeypatch):
    import cron.scheduler as scheduler

    statuses = iter(
        [
            {"id": "exec-1", "status": "running"},
            {"id": "exec-1", "status": "unknown"},
        ]
    )
    get = Mock(side_effect=lambda _execution_id: next(statuses))
    recover = Mock(return_value=1)
    monkeypatch.setattr(scheduler, "get_execution", get)
    monkeypatch.setattr(
        scheduler, "recover_interrupted_executions", recover, raising=False
    )
    process = Mock()
    process.poll.return_value = 9
    process.wait.return_value = 9

    assert scheduler._wait_for_external_cron_worker(
        process, execution_id="exec-1"
    ) is True
    recover.assert_called_once_with()
    assert get.call_count == 2


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


def test_shutdown_does_not_interrupt_restart_safe_waiter():
    import cron.scheduler as scheduler

    job_id = "external-waiter"
    scheduler._running_job_ids.add(job_id)
    scheduler._restart_safe_waiter_job_ids.add(job_id)
    try:
        assert scheduler.mark_running_jobs_interrupted("gateway restart") == []
        assert job_id not in scheduler._interrupted_job_ids
    finally:
        scheduler._restart_safe_waiter_job_ids.discard(job_id)
        scheduler._running_job_ids.discard(job_id)
        scheduler._interrupted_job_ids.discard(job_id)


def test_worker_delivery_queue_is_keyed_by_the_delivering_jobs_own_execution(
    monkeypatch, tmp_path
):
    """A nested in-process dispatch inside a worker (e.g. a script running
    ``hermes cron run <other>``) must not queue under the OUTER execution id."""
    import cron.scheduler as scheduler
    import cron.scheduler_delivery as scheduler_delivery

    queued = []
    monkeypatch.setattr(
        "cron.delivery_queue.enqueue_and_wait",
        lambda execution_id, job, content, **kw: (
            queued.append(execution_id) or "queued-marker"
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job, for_failure=False: [{"platform": "telegram", "chat_id": "123"}],
    )
    monkeypatch.setattr(
        scheduler_delivery,
        "_resolve_delivery_targets",
        lambda job, for_failure=False: [{"platform": "telegram", "chat_id": "123"}],
    )

    def _standalone(*_args, **_kwargs):
        raise RuntimeError("standalone path reached")

    # First call the standalone (non-queue) path makes after the guard; the
    # failure is reported as the delivery error string.
    monkeypatch.setattr("gateway.config.load_gateway_config", _standalone)
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-outer")

    # Own attempt: routed through the durable queue.
    assert scheduler._deliver_result(
        {"id": "job-1", "execution_id": "exec-outer", "deliver": "telegram:123"},
        "done",
        adapters=None,
        loop=None,
    ) == "queued-marker"
    assert queued == ["exec-outer"]

    # A different job's attempt: must NOT be queued under exec-outer; it falls
    # through to the standalone path.
    error = scheduler._deliver_result(
        {"id": "job-2", "execution_id": "exec-inner", "deliver": "telegram:123"},
        "done",
        adapters=None,
        loop=None,
    )
    assert error == "failed to load gateway config: standalone path reached"
    assert queued == ["exec-outer"]


def test_gateway_tool_run_without_adapter_objects_hands_off(monkeypatch):
    import cron.scheduler as scheduler

    created = Mock(return_value={"id": "exec-tool"})
    launch = Mock(return_value=True)
    run = Mock(side_effect=AssertionError("agent ran inside gateway"))
    monkeypatch.setattr(scheduler, "create_execution", created)
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", launch)
    monkeypatch.setattr(scheduler, "run_job", run)
    job = {"id": "tool-job"}

    assert scheduler.run_one_job(job, adapters=None) is True

    created.assert_called_once_with("tool-job", source="direct")
    assert job["execution_id"] == "exec-tool"
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


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_managed_gateway_restart_preserves_active_worker_and_single_side_effect(
    tmp_path, monkeypatch
):
    import cron.delivery_queue as delivery_queue
    import cron.executions as executions
    import cron.scheduler as scheduler
    from cron.jobs import create_job, use_cron_store
    from gateway.config import Platform, PlatformConfig
    from gateway.status import _pid_exists
    from tools import process_registry

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")

    home = tmp_path / "profile"
    scripts_dir = home / "scripts"
    scripts_dir.mkdir(parents=True)
    started = tmp_path / "started"
    release = tmp_path / "release"
    side_effect = tmp_path / "side-effect"
    probe = scripts_dir / "restart_probe.py"
    probe.write_text(
        "import pathlib, time\n"
        f"started = pathlib.Path({str(started)!r})\n"
        f"release = pathlib.Path({str(release)!r})\n"
        f"side_effect = pathlib.Path({str(side_effect)!r})\n"
        "started.write_text('started')\n"
        "deadline = time.monotonic() + 15\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "if not release.exists():\n"
        "    raise SystemExit('release timeout')\n"
        "with side_effect.open('a') as handle:\n"
        "    handle.write('once\\n')\n"
        "print('completed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    with use_cron_store(home):
        job = create_job(
            prompt=None,
            schedule="every 1h",
            name="restart probe",
            script=probe.name,
            no_agent=True,
            deliver="telegram:123",
        )
    payload = tmp_path / "job.json"
    launched = tmp_path / "launched.json"
    payload.write_text(json.dumps(job), encoding="utf-8")

    sent = []
    adapter = Mock()

    async def send(_chat_id, content, metadata=None):
        sent.append((content, metadata))
        return {"success": True, "message_id": "restart-delivery-1"}

    adapter.send = send
    gateway_config = Mock()
    gateway_config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True),
    }
    gateway_config.get_home_channel = lambda _platform: None
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: gateway_config
    )
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    replacement_loop = asyncio.new_event_loop()
    replacement_thread = threading.Thread(
        target=replacement_loop.run_forever,
        daemon=True,
    )
    replacement_thread.start()
    deadline = time.monotonic() + 2
    while not replacement_loop.is_running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert replacement_loop.is_running()

    harness = (
        "import json, os, pathlib, time\n"
        f"os.environ['HERMES_HOME'] = {str(home)!r}\n"
        "os.environ['INVOCATION_ID'] = 'restart-fixture'\n"
        "from cron import scheduler\n"
        "from tools import process_registry\n"
        "process_registry._is_supervised_gateway_process = lambda: True\n"
        f"job = json.loads(pathlib.Path({str(payload)!r}).read_text())\n"
        "if not scheduler.run_one_job(job, adapters=None, loop=None):\n"
        "    raise SystemExit('worker was not isolated')\n"
        f"pathlib.Path({str(launched)!r}).write_text('returned')\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", harness])
    worker_pid = None
    try:
        deadline = time.monotonic() + 10
        current = None
        while time.monotonic() < deadline:
            if parent.poll() is not None:
                pytest.fail(f"gateway fixture exited early with {parent.returncode}")
            current = executions.latest_execution(job["id"])
            if started.exists() and current and current.get("pid") != os.getpid():
                break
            time.sleep(0.05)
        assert started.exists()
        assert current is not None
        execution = current
        worker_pid = int(current["pid"])
        assert not launched.exists(), "handoff returned before execution completed"

        # Replacing a managed gateway kills its old process tree. The active
        # cron owner must remain in its transient scope and keep the same PID.
        parent.terminate()
        parent.wait(timeout=5)
        assert _pid_exists(worker_pid)

        release.write_text("go", encoding="utf-8")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = delivery_queue.get_status(execution["id"])
            if row and row["status"] == "pending":
                scheduler.drain_delivery_queue(
                    {Platform.TELEGRAM: adapter}, replacement_loop
                )
            current = executions.latest_execution(job["id"])
            if current and current["status"] == "completed":
                break
            time.sleep(0.05)
        assert executions.latest_execution(job["id"])["status"] == "completed"
        assert side_effect.read_text(encoding="utf-8").splitlines() == ["once"]
        assert delivery_queue.get_status(execution["id"])["status"] == "delivered"
        assert len(sent) == 1
        assert "completed" in sent[0][0]
    finally:
        replacement_loop.call_soon_threadsafe(replacement_loop.stop)
        replacement_thread.join(timeout=2)
        replacement_loop.close()
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
        if worker_pid is not None and _pid_exists(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)
