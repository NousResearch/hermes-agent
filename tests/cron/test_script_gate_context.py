"""Script-gated cron jobs are the cheap default for watchers.

A real gate script keeps its own state in ``HERMES_CRON_STATE_DIR`` across ticks, wakes the agent
only with the compact ``context`` from its gate line (never the gate line itself), and a broken
gate alerts like a ``no_agent`` watchdog instead of waking the paid agent.
"""

from __future__ import annotations

import json

import pytest

_COUNTER_GATE = """\
import json, os, pathlib
state = pathlib.Path(os.environ["HERMES_CRON_STATE_DIR"]) / "count"
n = int(state.read_text()) + 1 if state.exists() else 1
state.write_text(str(n))
print("raw noise the agent should not see")
print(json.dumps({"wakeAgent": n % 2 == 0, "context": {"run": n, "job": os.environ["HERMES_CRON_JOB_ID"]}}))
"""


@pytest.fixture
def gate_home(tmp_path, monkeypatch):
    from cron import jobs

    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with jobs.use_cron_store(home):
        yield home


def _prepare(job):
    from cron.scheduler import _prepare_job_prompt

    return _prepare_job_prompt(job, job["id"], job["name"], None, None)


def test_gate_state_persists_and_agent_sees_only_the_context(gate_home):
    from cron.jobs import create_job

    (gate_home / "scripts" / "counter.py").write_text(_COUNTER_GATE)
    job = create_job(prompt="Say the run number.", schedule="every 1h", name="counter",
                     script="counter.py", deliver="local")

    early, prompt = _prepare(job)  # run 1: odd -> gate closed, silent
    assert prompt is None and early[0] is True and early[2] == "[SILENT]"

    early, prompt = _prepare(job)  # run 2: the state file survived -> even -> wake
    assert early is None
    expected = json.dumps({"run": 2, "job": job["id"]}, indent=2)
    assert "## Script Output\nThe following data" in prompt and expected in prompt
    assert "wakeAgent" not in prompt and "raw noise" not in prompt


def test_failing_gate_alerts_without_waking_the_agent(gate_home):
    from cron.jobs import create_job

    (gate_home / "scripts" / "broken.py").write_text("import sys\nsys.exit('upstream API down')\n")
    job = create_job(prompt="Summarise.", schedule="every 1h", name="broken",
                     script="broken.py", deliver="local")

    early, prompt = _prepare(job)

    assert prompt is None
    success, _doc, alert, error = early
    assert success is False and "exited with code 1" in error
    assert alert.startswith("⚠ Cron watchdog 'broken' script failed") and "upstream API down" in alert
