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


# ---------------------------------------------------------------------------
# _summarize_cron_failure_for_delivery: no_agent jobs are never provider failures
# ---------------------------------------------------------------------------


NO_AGENT_JOB = {
    "name": "ng-mirror-sync",
    "no_agent": True,
    "script": "sync_ng_mirror.sh",
    "model": None,
    "provider": None,
}

AGENT_JOB = {"name": "daily-recap", "no_agent": False, "model": "x", "provider": "y"}


@pytest.mark.parametrize(
    "error_text",
    [
        # Google Sheets read timeout surfaced through a script's stdout.
        (
            "Script exited with code 1\nstdout:\n"
            "Sheet read failed (4x): HTTPSConnectionPool("
            "host='sheets.googleapis.com', port=443): Read timed out. "
            "(read timeout=30)"
        ),
        # Sheets quota rejection.
        (
            "Script exited with code 1\nstdout:\n"
            "APIError: [429]: Quota exceeded for quota metric 'Read requests'"
        ),
        # Remote host unreachable over ssh.
        "ssh: connect to host 10.0.0.1 port 22: Operation timed out",
        # Upstream API rejecting the script's own credentials.
        "urllib.error.HTTPError: HTTP Error 401: Unauthorized",
    ],
)
def test_no_agent_failure_never_reported_as_provider_error(hermes_env, error_text):
    """A no_agent job makes no inference call, so its failures must never be
    described as provider/fallback-chain problems.

    The scheduler classified failures by substring alone, so any script whose
    stdout contained "timed out", "429" or "401" was delivered to the operator
    as "provider timeout. Fallback chain was exhausted or unavailable" — for a
    job with model=None and provider=None that never had a fallback chain. That
    text sends the operator to debug the wrong subsystem and hides the script's
    real error.
    """
    from cron.scheduler import _summarize_cron_failure_for_delivery

    summary = _summarize_cron_failure_for_delivery(NO_AGENT_JOB, error_text)

    assert "provider" not in summary.lower()
    assert "fallback chain" not in summary.lower()
    assert "ng-mirror-sync" in summary


def test_no_agent_failure_surfaces_the_real_script_error(hermes_env):
    """The delivered message must carry the script's own error, not a
    provider-shaped substitute, so the operator can act on it."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    summary = _summarize_cron_failure_for_delivery(
        NO_AGENT_JOB,
        "Script exited with code 1\nstdout:\n"
        "Sheet read failed (4x): HTTPSConnectionPool("
        "host='sheets.googleapis.com', port=443): Read timed out. "
        "(read timeout=30)",
    )

    assert "sheets.googleapis.com" in summary


@pytest.mark.parametrize(
    "error_text,expected",
    [
        ("Provider call failed: Read timed out", "provider timeout"),
        ("HTTP 429 rate limit reached", "provider rate limit"),
        ("Authentication failed for provider", "provider authentication error"),
    ],
)
def test_agent_mode_still_classifies_provider_failures(hermes_env, error_text, expected):
    """Agent-mode jobs DO call a provider, so the compact provider summaries
    must be preserved for them."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    summary = _summarize_cron_failure_for_delivery(AGENT_JOB, error_text)

    assert expected in summary.lower()
