"""Direct `hermes cron run` must survive caller timeout.

A one-shot CLI process used to own the durable execution in-process. When
the caller was SIGTERM/SIGKILL'd mid-run, recover_interrupted_executions()
proved the owner dead and marked the attempt unknown, with no output.
Immediate runs now hand off to a start_new_session worker so the waiter
can die without taking the owner or the artifact with it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _env(home: Path, repo: Path) -> dict:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)
    return env


def _py(repo: Path, env: dict, snippet: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        **kwargs,
    )


def test_direct_run_caller_kill_leaves_worker_to_terminalize(tmp_path):
    home = tmp_path / "home"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    workdir = home / "workdir"
    workdir.mkdir()
    artifact = workdir / "report.md"
    started = home / "started.flag"
    (home / "scripts" / "slow.py").write_text(
        "import time, pathlib\n"
        f"pathlib.Path({str(started)!r}).write_text('started')\n"
        "time.sleep(4)\n"
        f"pathlib.Path({str(artifact)!r}).write_text('report complete')\n"
        "print('done')\n",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    env = _env(home, repo)

    created = _py(
        repo,
        env,
        "from cron.jobs import create_job; "
        "j=create_job(prompt=None, schedule='every 1h', name='timeout-repro', "
        "script='slow.py', no_agent=True, deliver='local', "
        f"workdir={str(workdir)!r}); print(j['id'])",
        check=True,
    )
    job_id = created.stdout.strip().splitlines()[-1]
    assert job_id

    runner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, threading, time\n"
            "from pathlib import Path\n"
            "from cron.jobs import get_job\n"
            "from tools.cronjob_tools import _execute_job_now\n"
            f"started = Path({str(started)!r})\n"
            "def abandon_caller():\n"
            "    deadline = time.monotonic() + 8\n"
            "    while time.monotonic() < deadline and not started.exists():\n"
            "        time.sleep(0.05)\n"
            "    time.sleep(0.2)\n"
            "    os._exit(9)\n"
            "threading.Thread(target=abandon_caller, daemon=True).start()\n"
            f"job=get_job({job_id!r})\n"
            "print(_execute_job_now(job), flush=True)\n",
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = runner.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        runner.terminate()
        out, err = runner.communicate(timeout=5)
        raise AssertionError(
            f"caller never abandoned; rc={runner.returncode}\nstdout={out}\nstderr={err}"
        )
    assert runner.returncode == 9, (
        f"caller should abandon with os._exit(9); rc={runner.returncode}\n"
        f"stdout={out}\nstderr={err}"
    )
    assert started.exists(), "job script never started"

    mid = _py(
        repo,
        env,
        "import json; "
        "from cron.executions import list_executions, recover_interrupted_executions; "
        f"print(recover_interrupted_executions()); "
        f"print(json.dumps(list_executions(job_id={job_id!r})))",
        check=True,
    )
    lines = [ln for ln in mid.stdout.splitlines() if ln.strip()]
    recovered = int(lines[0])
    records = json.loads(lines[1])
    assert recovered == 0, (
        f"live detached worker must not be marked unknown; records={records}"
    )
    assert records, "direct run must have created an execution row"
    assert records[0]["source"] == "direct"
    assert records[0]["status"] in {"claimed", "running"}
    worker_pid = int(records[0]["pid"])
    assert worker_pid != runner.pid

    finish_deadline = time.monotonic() + 20
    while time.monotonic() < finish_deadline:
        if artifact.exists():
            break
        time.sleep(0.1)
    assert artifact.exists(), "detached worker should have written the job artifact"
    assert artifact.read_text(encoding="utf-8") == "report complete"

    final = _py(
        repo,
        env,
        "import json; "
        "from cron.executions import list_executions, recover_interrupted_executions; "
        "print(recover_interrupted_executions()); "
        f"print(json.dumps(list_executions(job_id={job_id!r})))",
        check=True,
    )
    final_lines = [ln for ln in final.stdout.splitlines() if ln.strip()]
    assert final_lines[0] == "0"
    final_records = json.loads(final_lines[1])
    assert final_records[0]["status"] == "completed"
    assert final_records[0]["error"] is None
