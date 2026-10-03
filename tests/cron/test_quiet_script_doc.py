"""Quiet pre-run script runs must record a non-empty document (#130950).

When a cron job's pre-run script succeeds with no output, the agent is
skipped but the run must still record an informative document — otherwise
"ran and found nothing" is indistinguishable on disk from a crash
(zero-byte output file).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with cron/output + scripts dirs."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _make_job(name="quiet-test"):
    return {
        "id": f"job_{name}",
        "name": name,
        "prompt": "Report anything notable.",
        "schedule": "*/5 * * * *",
        "script": "check.py",
    }


def test_prepare_quiet_script_returns_nonempty_doc(cron_env):
    """_prepare_job_prompt on empty script output returns a quiet_doc, not ''."""
    import cron.scheduler as scheduler

    job = _make_job()
    with patch.object(
        scheduler, "_run_job_script_with_claim_heartbeat", return_value=(True, "")
    ):
        early, prompt = scheduler._prepare_job_prompt(job, job["id"], job["name"], None, None)

    assert prompt is None
    assert early is not None
    success, doc, final, err = early
    assert success is True
    assert err is None
    assert final == scheduler.SILENT_MARKER
    # The regression: this used to be "" (zero-byte output file).
    assert doc != ""
    assert len(doc.strip()) > 0
    assert f"# Cron Job: {job['name']}" in doc
    assert job["id"] in doc
    assert "ran, nothing new" in doc
    assert "produced no output" in doc
    assert "not a failure" in doc


def test_run_job_quiet_script_skips_agent_with_quiet_doc(cron_env):
    """run_job with a quiet script skips the agent and stays silent."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

    job = _make_job(name="quiet-run-job")
    with patch.object(sched_script, "_run_job_script", return_value=(True, "")), patch(
        "run_agent.AIAgent"
    ) as agent_cls:
        success, doc, final, err = scheduler.run_job(job)

    assert success is True
    assert err is None
    assert final == scheduler.SILENT_MARKER
    agent_cls.assert_not_called()
    assert doc != ""
    assert f"# Cron Job: {job['name']}" in doc
    assert "**Status:** ran, nothing new" in doc


def test_quiet_doc_output_file_is_nonempty(cron_env):
    """The saved output file for a quiet run is non-empty and informative."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script
    from cron.jobs import OUTPUT_DIR, save_job_output

    job = _make_job(name="quiet-output-file")
    with patch.object(sched_script, "_run_job_script", return_value=(True, "")), patch(
        "run_agent.AIAgent"
    ):
        success, doc, final, err = scheduler.run_job(job)

    assert success is True
    assert final == scheduler.SILENT_MARKER
    output_file = save_job_output(job["id"], doc)
    assert output_file.exists()
    content = Path(output_file).read_text(encoding="utf-8")
    assert len(content) > 0
    assert content == doc
    assert f"# Cron Job: {job['name']}" in content
    assert f"**Job ID:** {job['id']}" in content
    assert "**Run Time:**" in content
    assert "ran, nothing new" in content
    assert "agent was not called" in content
    # Output lands under the per-job output dir, not as a zero-byte crash lookalike.
    assert output_file.parent == OUTPUT_DIR / job["id"]
    assert output_file.stat().st_size > 0


def test_quiet_doc_distinguishable_from_crash_and_wake_gate(cron_env):
    """Quiet doc carries its own status line, distinct from failures/empty."""
    import cron.scheduler as scheduler

    job = _make_job(name="quiet-distinct")
    with patch.object(
        scheduler, "_run_job_script_with_claim_heartbeat", return_value=(True, "")
    ):
        (success, doc, final, _), _prompt = scheduler._prepare_job_prompt(
            job, job["id"], job["name"], None, None
        )

    assert success is True
    assert final == "[SILENT]"
    # Not the old empty document, not a failure marker.
    assert doc.strip() != ""
    assert "[CRON_FAILURE]" not in doc
    assert "wakeAgent=false" not in doc
    assert "nothing to report" in doc
