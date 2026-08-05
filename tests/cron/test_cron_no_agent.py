"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: data-layer semantics
# ---------------------------------------------------------------------------


def test_create_job_no_agent_requires_script(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


def test_update_job_roundtrips_no_agent_flag(hermes_env):
    from cron.jobs import create_job, update_job, get_job

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local")

    update_job(job["id"], {"no_agent": False})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False

    update_job(job["id"], {"no_agent": True})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool: API-layer validation
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_no_agent_without_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "no_agent=True requires a script" in result.get("error", "")


# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


# ---------------------------------------------------------------------------
# no_agent jobs honoring a configured Python interpreter
# ---------------------------------------------------------------------------


def test_run_job_no_agent_uses_configured_interpreter(hermes_env):
    """A no-agent job's script must run through the configured interpreter.

    Proves the override survives the no_agent branch of ``run_job`` →
    ``_run_job_script_with_claim_heartbeat`` → ``_run_job_script``.
    """
    import os
    import stat as _stat
    import sys

    from cron.jobs import create_job
    from cron.scheduler import run_job

    # A wrapper that re-execs the real interpreter with an env marker.
    wrapper = hermes_env / "scripts" / "python-wrapper"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "env = os.environ.copy()\n"
        'env["CRON_WRAPPER_USED"] = "1"\n'
        "os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | _stat.S_IXUSR)

    script_path = hermes_env / "scripts" / "marker.py"
    script_path.write_text(
        'import os\nprint(os.environ.get("CRON_WRAPPER_USED", "0"))\n'
    )

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="marker.py",
        no_agent=True,
        deliver="local",
        interpreter=str(wrapper),
    )

    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == "1"
    assert "1" in doc
