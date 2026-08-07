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


def test_run_job_no_agent_exposes_resolved_delivery_target(hermes_env):
    """Script-only jobs receive routing metadata without entering the LLM path."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "route.py"
    script_path.write_text(
        "import json, os\n"
        "print(json.dumps({k: os.environ.get(k) for k in (\n"
        "    'HERMES_CRON_AUTO_DELIVER_PLATFORM',\n"
        "    'HERMES_CRON_AUTO_DELIVER_CHAT_ID',\n"
        "    'HERMES_CRON_AUTO_DELIVER_THREAD_ID',\n"
        ")}))\n"
    )

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="route.py",
        no_agent=True,
        deliver="discord:123456789012345678:987654321098765432",
    )
    success, _doc, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert json.loads(final_response) == {
        "HERMES_CRON_AUTO_DELIVER_PLATFORM": "discord",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "123456789012345678",
        "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "987654321098765432",
    }


def test_run_job_no_agent_clears_ambient_route_for_local_delivery(
    hermes_env, monkeypatch
):
    """A local script must not inherit another session or job's destination."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    for key, value in {
        "HERMES_CRON_AUTO_DELIVER_PLATFORM": "discord",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "ambient-chat",
        "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "ambient-thread",
    }.items():
        monkeypatch.setenv(key, value)

    script_path = hermes_env / "scripts" / "local-route.py"
    script_path.write_text(
        "import json, os\n"
        "print(json.dumps({k: os.environ.get(k) for k in (\n"
        "    'HERMES_CRON_AUTO_DELIVER_PLATFORM',\n"
        "    'HERMES_CRON_AUTO_DELIVER_CHAT_ID',\n"
        "    'HERMES_CRON_AUTO_DELIVER_THREAD_ID',\n"
        ")}))\n"
    )
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="local-route.py",
        no_agent=True,
        deliver="local",
    )
    success, _doc, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert json.loads(final_response) == {
        "HERMES_CRON_AUTO_DELIVER_PLATFORM": "",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "",
        "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "",
    }


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support and environment boundaries
# ---------------------------------------------------------------------------


def test_run_job_script_rejects_non_routing_env_overrides(hermes_env):
    """Post-sanitization overrides are limited to cron delivery metadata."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "inspect-env.py"
    script_path.write_text(
        "import os\n"
        "print(os.environ.get('UNRELATED_TEST_ONLY_ENV', '<absent>'))\n"
    )

    ok, output = _run_job_script(
        "inspect-env.py",
        env_overrides={"UNRELATED_TEST_ONLY_ENV": "must-not-pass"},
    )

    assert ok is True
    assert output == "<absent>"


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


def test_run_job_script_nul_path_fails_cleanly(hermes_env):
    """Sibling of the lifecycle-guard ingestion fix: a NUL-bearing script
    value can survive to fire time (the creation-time guard treats it as
    "nothing to scan"), and ``Path.expanduser()`` raises ValueError — not
    OSError — on it. The scheduler must fail the run with a report, not
    crash with an unhandled exception."""
    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("~user\x00bad.sh")
    assert ok is False
    assert "Blocked" in output
