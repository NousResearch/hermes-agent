"""Anti-drift contract tests for the cronjob ``no_agent`` schema.

The ``no_agent`` parameter description in ``CRONJOB_SCHEMA`` makes four
explicit promises to the model (and therefore to users) about how a
script-only watchdog behaves:

    1. ``script`` is REQUIRED when ``no_agent=True``.
    2. EMPTY stdout means SILENT — nothing is delivered.
    3. Non-empty stdout is delivered VERBATIM.
    4. A non-zero exit / failure sends an ERROR ALERT (no silent failure).

These behaviors are implemented in ``cron/scheduler.run_job`` and validated
by ``tools/cronjob_tools.cronjob``. The behavior is already covered by
``test_cron_no_agent.py`` — but nothing pins the *documentation* to the
*behavior*. If someone changes the scheduler's silent-on-empty handling (or
the script-required rule) without updating the schema text — or rewords the
schema without changing the code — the model would confidently mislead the
user about a high-impact edge case (a watchdog that silently stops alerting).

Each test below asserts the documented claim AND the runtime behavior in the
same place, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------


def _no_agent_description() -> str:
    """Return the ``no_agent`` property description from the live schema.

    Fails loudly (KeyError) if the property is renamed or removed, which is
    itself a form of drift worth catching.
    """
    from tools.cronjob_tools import CRONJOB_SCHEMA

    return CRONJOB_SCHEMA["parameters"]["properties"]["no_agent"]["description"]


# ---------------------------------------------------------------------------
# Shared HERMES_HOME isolation (mirrors tests/cron/test_cron_no_agent.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# Promise 1: script REQUIRED when no_agent=True
# ---------------------------------------------------------------------------


def test_no_agent_script_required_doc_matches_validation(hermes_env):
    """Doc says script MUST be set; create must actually reject no_agent w/o script."""
    desc = _no_agent_description().lower()
    # Documented claim.
    assert "must be set" in desc
    assert "requirements when true" in desc

    # Enforced behavior.
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "requires a script" in result.get("error", "")


# ---------------------------------------------------------------------------
# Promise 2: EMPTY stdout means SILENT
# ---------------------------------------------------------------------------


def test_no_agent_empty_stdout_silent_doc_matches_behavior(hermes_env):
    """Doc says empty stdout is SILENT; run_job must emit SILENT_MARKER, not deliver."""
    desc = _no_agent_description().lower()
    assert "empty stdout means silent" in desc

    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\n# nothing to report\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    success, _doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER


# ---------------------------------------------------------------------------
# Promise 3: non-empty stdout delivered VERBATIM
# ---------------------------------------------------------------------------


def test_no_agent_nonempty_stdout_verbatim_doc_matches_behavior(hermes_env):
    """Doc says non-empty stdout is sent verbatim; run_job must deliver it unchanged."""
    desc = _no_agent_description().lower()
    assert "verbatim" in desc

    from cron.jobs import create_job
    from cron.scheduler import run_job

    sentinel = "RAM 92% on host db-01"
    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text(f"#!/bin/bash\necho '{sentinel}'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, _doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    # "verbatim" means the exact script text reaches the user, not a paraphrase.
    assert final_response.strip() == sentinel


# ---------------------------------------------------------------------------
# Promise 4: non-zero exit sends an ERROR ALERT (never a silent failure)
# ---------------------------------------------------------------------------


def test_no_agent_failure_alert_doc_matches_behavior(hermes_env):
    """Doc says non-zero exit sends an error alert; run_job must NOT fail silently."""
    desc = _no_agent_description().lower()
    assert "error alert" in desc

    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "broken.sh"
    script_path.write_text("#!/bin/bash\necho 'disk probe failed' >&2\nexit 3\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="broken.sh", no_agent=True, deliver="local"
    )
    success, _doc, final_response, error = run_job(job)
    assert success is False
    assert error is not None
    # A failing watchdog must surface SOMETHING — never the SILENT sentinel.
    assert final_response != SILENT_MARKER
    assert final_response.strip() != ""
