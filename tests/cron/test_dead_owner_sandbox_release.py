"""Dead-owner cron recovery releases the run's Docker sandbox (#75467).

A session-scoped cron sandbox idles in ``sleep infinity``. When the run's owner
process dies before teardown (a wall-clock ``timeout`` around ``hermes cron tick``
sends SIGTERM; SIGKILL; OOM), nothing removes it: the in-process idle reaper died
with the owner and the orphan reaper only considers ``status=exited``. One
container leaks per such run. Recovery already proves the owner gone, so it is the
point that must release the sandbox — and must never do so for a live owner.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

import tools.environments.docker as docker_mod


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions_mod


@pytest.fixture()
def docker_cli(monkeypatch):
    """Fake docker CLI: ``ps`` lists one container per queried task label; records every argv."""
    calls: list[list[str]] = []

    def fake_run_capture(argv, timeout=None):
        calls.append(list(argv))
        stdout = "c0ffee000001\n" if argv[1] == "ps" else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(docker_mod, "find_docker", lambda: "docker")
    monkeypatch.setattr(docker_mod, "run_capture", fake_run_capture)
    return calls


def _own_row(executions, job_id: str, pid: int, started_at) -> str:
    record = executions.create_execution(job_id, source="scheduler")
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='other-process', pid=?, process_started_at=? WHERE id=?",
            (pid, started_at, record["id"]),
        )
    return record["id"]


def _dead_pid() -> int:
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True, text=True, check=True,
    )
    return int(proc.stdout.strip())


@pytest.mark.parametrize("recover", ["sweep", "targeted"])
def test_dead_owner_recovery_removes_the_runs_sandbox(executions, docker_cli, recover):
    execution_id = _own_row(executions, "job-a", _dead_pid(), 1)

    if recover == "sweep":
        assert executions.recover_interrupted_executions() == 1
    else:
        assert executions.terminalize_dead_owner(execution_id, reason="worker exited 143")

    # The sandbox is found by the same label DockerEnvironment stamps for the run's task id,
    # and that exact container is force-removed.
    task_label = docker_mod._sanitize_label_value(executions.cron_task_id("job-a", execution_id))
    ps = [argv for argv in docker_cli if argv[1] == "ps"]
    assert len(ps) == 1 and f"label=hermes-task-id={task_label}" in ps[0]
    assert ["docker", "rm", "-f", "c0ffee000001"] in docker_cli


def test_live_owner_without_fingerprint_keeps_its_sandbox(executions, docker_cli):
    """A NULL start-time fingerprint lets the ledger terminalize a live owner (#108480);
    sandbox removal is irreversible, so it must demand real proof of death."""
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _own_row(executions, "job-b", live.pid, None)
        executions.recover_interrupted_executions()
        assert live.poll() is None
        assert docker_cli == []
    finally:
        live.kill()
        live.wait()
