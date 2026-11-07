"""A cron script can tell a real cron run from a manual one (``HERMES_CRON_RUN``).

Fire-once-then-disarm watchdogs delete their state file when they alert. Nothing in the
script's environment distinguished the runner from a human checking on it by hand, so a
manual run consumed the pending alert: it printed to the operator's terminal, the state file
was gone, and the next scheduled run had nothing left to deliver.

Real scripts run through the real ``run_job`` dispatch against a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

_WATCHDOG = """\
import os, pathlib, sys
state = pathlib.Path(os.environ["WATCHDOG_STATE"])
if state.exists():
    if os.environ.get("HERMES_CRON_RUN") != "1":
        print("[dry-run] alert pending", file=sys.stderr)
        sys.exit(0)
    state.unlink()
    print("SERVICE RECOVERED")
"""


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_CRON_RUN", raising=False)
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)
    return home


def _run_no_agent(script: str) -> str:
    from cron.jobs import create_job
    from cron.scheduler import run_job

    job = create_job(prompt=None, schedule="every 5m", script=script, no_agent=True, deliver="local")
    success, _doc, final_response, error = run_job(job)
    assert success is True, error
    return final_response


def test_cron_dispatched_script_sees_marker_without_leaking_it(hermes_env):
    (hermes_env / "scripts" / "probe.py").write_text(
        "import os\nprint('MARKER=' + os.environ.get('HERMES_CRON_RUN', 'ABSENT'))\n")

    assert "MARKER=1" in _run_no_agent("probe.py")
    # The marker rides the child's env only: the gateway process, and anything it later spawns
    # outside a cron run, must not inherit it.
    assert "HERMES_CRON_RUN" not in os.environ


def test_manual_run_keeps_the_alert_for_the_scheduled_run(hermes_env, tmp_path, monkeypatch):
    state = tmp_path / "recovered.flag"
    state.write_text("armed")
    monkeypatch.setenv("WATCHDOG_STATE", str(state))
    script = hermes_env / "scripts" / "watchdog.py"
    script.write_text(_WATCHDOG)

    manual = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60, check=True)
    assert "SERVICE RECOVERED" not in manual.stdout
    assert state.exists(), "a manual run consumed the pending alert"

    assert "SERVICE RECOVERED" in _run_no_agent("watchdog.py")
    assert not state.exists()
