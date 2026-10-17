"""The occurrence identity a cron fire exports to its child processes.

The harness knows which OCCURRENCE each fire is, and a child process cannot re-derive it: the
ledger row records the SCHEDULER's pid, so a ``no_agent`` script or a restart-safe worker can
not recognise its own row, and picking one by nearest window would put resolution logic in the
consumer. So the fire exports ``HERMES_CRON_JOB_ID``, ``HERMES_CRON_EXECUTION_ID``,
``HERMES_CRON_SCHEDULED_INSTANT`` (the occurrence, verbatim as the ledger holds it) and
``HERMES_CRON_SOURCE``, and a value that is absent is OMITTED rather than exported empty —
absence is the single readable "off-schedule / manual fire" signal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

#: Where the probe writes what its env carried. A non-Hermes name on purpose: it must survive
#: the child-env sanitizer the way any other inherited variable does.
_OUT_ENV = "CRON_IDENTITY_PROBE_OUT"

# Only the identity names the harness exports, so a present-but-EMPTY variable and an absent one
# stay distinguishable in the assertion.
_PROBE = (
    "#!/usr/bin/env bash\n"
    f'env | grep "^HERMES_CRON_" | sort > "${_OUT_ENV}"\n'
)

#: A real scheduler tick in a child process: the ledger row it creates is the expected value.
_TICK = """
import json
from cron import scheduler
from cron.executions import list_executions
scheduler.tick(verbose=False, sync=True)
print(json.dumps(list_executions(job_id="env-probe")))
"""


def _tick_no_agent_job(tmp_path, monkeypatch, job: dict) -> tuple[dict, dict[str, str]]:
    """Fire *job* through a real tick; return its ledger row and the env its script received.

    The script is a separate process, which also covers the overlay's ORDER: the subprocess
    sanitizer strips Hermes-owned names, so an overlay applied before it would arrive empty.
    """
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir()
    (home / "scripts" / "probe.sh").write_text(_PROBE, encoding="utf-8")
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}), encoding="utf-8")
    monkeypatch.setenv(_OUT_ENV, str(home / "identity.txt"))

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "_HERMES_"))
           and not k.endswith(("_API_KEY", "_TOKEN"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run([sys.executable, "-c", _TICK], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    (execution,) = json.loads(result.stdout.strip().splitlines()[-1])
    lines = (home / "identity.txt").read_text(encoding="utf-8").splitlines()
    return execution, dict(line.split("=", 1) for line in lines)


def _no_agent_job(**overrides) -> dict:
    from hermes_time import now

    job = {
        "id": "env-probe", "name": "env probe", "script": "probe.sh", "no_agent": True,
        "deliver": "local", "schedule": {"kind": "interval", "minutes": 240},
        "next_run_at": (now() - timedelta(minutes=1)).isoformat(),
        "enabled": True, "state": "scheduled", "repeat": {"times": None, "completed": 0},
    }
    job.update(overrides)
    return job


def test_helper_exports_known_values_and_omits_the_absent():
    from cron.execution_identity import cron_execution_env

    assert cron_execution_env({
        "id": "job-1",
        "execution_id": "exec-1",
        "_scheduled_instant": "2026-10-10T06:15:00+00:00",
        "source": "builtin",
    }) == {
        "HERMES_CRON_JOB_ID": "job-1",
        "HERMES_CRON_EXECUTION_ID": "exec-1",
        "HERMES_CRON_SCHEDULED_INSTANT": "2026-10-10T06:15:00+00:00",
        "HERMES_CRON_SOURCE": "builtin",
    }

    # A manual / off-tick fire carries no occurrence. The key must be ABSENT, never present
    # with an empty value: a consumer's "no identity ⇒ refuse" test reads presence.
    manual = cron_execution_env({
        "id": "job-1", "execution_id": "exec-2", "_scheduled_instant": None, "source": "direct",
    })
    assert "HERMES_CRON_SCHEDULED_INSTANT" not in manual
    assert manual["HERMES_CRON_SOURCE"] == "direct"
    assert manual and "" not in manual.values()


def test_scheduled_no_agent_fire_exports_its_ledger_occurrence(tmp_path, monkeypatch):
    execution, exported = _tick_no_agent_job(tmp_path, monkeypatch, _no_agent_job())

    assert exported["HERMES_CRON_JOB_ID"] == "env-probe"
    assert exported["HERMES_CRON_EXECUTION_ID"] == execution["id"]
    assert exported["HERMES_CRON_SOURCE"] == execution["source"] == "builtin"
    # The occurrence the scheduler consumed, verbatim — never the wall clock at script time.
    assert exported["HERMES_CRON_SCHEDULED_INSTANT"] == execution["scheduled_instant"]


def test_manual_fire_omits_the_absent_occurrence(tmp_path, monkeypatch):
    """A manual run of a scheduled job (``manual_run_at``) is occurrence-free end to end."""
    job = _no_agent_job()
    job["manual_run_at"] = job["next_run_at"]
    execution, exported = _tick_no_agent_job(tmp_path, monkeypatch, job)

    assert execution["scheduled_instant"] is None
    assert "HERMES_CRON_SCHEDULED_INSTANT" not in exported
    assert exported["HERMES_CRON_JOB_ID"] == "env-probe"
    assert exported["HERMES_CRON_EXECUTION_ID"] == execution["id"]
    assert exported["HERMES_CRON_SOURCE"] == execution["source"] == "builtin"
