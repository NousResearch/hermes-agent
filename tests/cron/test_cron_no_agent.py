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
# _summarize_cron_failure_for_delivery: mode-aware failure attribution
# ---------------------------------------------------------------------------
#
# The summarizer classified failures by substring-matching the error prose and
# mapped any hit onto a provider-shaped explanation. For a no_agent job that is
# structurally impossible — run_job short-circuits before any model is reached —
# so a script whose own text happened to contain "timed out", "429" or
# "authentication" had its failure attributed to a provider it never called.
#
# Observed in practice: _run_job_script reports a timeout as "Script timed out
# after {n}s: {path}", which was delivered to chat as "provider timeout. Fallback
# chain was exhausted or unavailable." for a job that never opened a socket.
#
# The summarizer had no direct test coverage — the only test referencing it
# mocks it out and asserts on its arguments — which is why this shipped.


@pytest.mark.parametrize(
    "error",
    [
        "Script timed out after 900s: /home/u/.hermes/scripts/nightly.sh",
        "Script failed: curl returned 429 from api.example.com",
        "Script failed: gpg authentication failed for key",
        "Script failed: ReadTimeout contacting localhost",
    ],
)
def test_no_agent_failure_never_blamed_on_a_provider(error):
    """A script job's failure must never be reported as a provider/fallback failure."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "nightly-job", "no_agent": True, "script": "nightly.sh"}
    msg = _summarize_cron_failure_for_delivery(job, error)

    assert "provider" not in msg.lower()
    assert "fallback chain" not in msg.lower()
    # The operator must be pointed at what actually failed.
    assert "Script" in msg


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("ReadTimeout: provider did not respond", "provider timeout"),
        ("HTTP 429 rate limit exceeded", "provider rate limit"),
        ("HTTP 401 authentication failed", "provider authentication error"),
    ],
)
def test_agent_job_provider_classification_unchanged(error, expected):
    """Regression guard: agent-mode jobs keep the provider-shaped summaries."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "daily-digest", "no_agent": False}
    assert expected in _summarize_cron_failure_for_delivery(job, error)
