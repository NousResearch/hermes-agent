"""Live-dispatch script carry for cron fires (#115470).

The stored job record is authoritative for the job definition. When a
dispatch snapshot reaches prompt preparation without the script field,
the run must backfill it from the store: otherwise the wake gate never
runs and the agent fires without the Script Output block.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a scripts dir and a cron store."""
    home = tmp_path / "hermes-home"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", home / "cron")
    monkeypatch.setattr(
        jobs_mod, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(
        jobs_mod, "OUTPUT_DIR", home / "cron" / "output")
    return home


def _seed_script_job(cron_home):
    from cron.jobs import create_job
    script = cron_home / "scripts" / "facts.sh"
    script.write_text("#!/bin/bash" + chr(10) + "echo FACT-42" + chr(10))
    return create_job(
        prompt="Decide the wording.",
        schedule="every 1m",
        name="live-script-carry",
        script="facts.sh",
    )


def test_prepare_prompt_backfills_script_missing_from_dispatch_snapshot(cron_home):
    """A snapshot without script must still run the stored script (#115470)."""
    from cron.scheduler import _prepare_job_prompt
    job = _seed_script_job(cron_home)
    snapshot = {
        "id": job["id"],
        "name": job["name"],
        "prompt": job["prompt"],
        "schedule": job["schedule"],
    }
    assert "script" not in snapshot
    early, prompt = _prepare_job_prompt(snapshot, job["id"], job["name"], None, None)
    assert early is None
    assert "## Script Output" in prompt
    assert "FACT-42" in prompt


def test_provider_claim_snapshot_carries_script(cron_home):
    """claim_fire snapshots must include the script field."""
    from cron.jobs import claim_job_for_fire
    job = _seed_script_job(cron_home)
    claimed = claim_job_for_fire(job["id"], force=True, return_job=True)
    assert isinstance(claimed, dict)
    assert claimed.get("script") == "facts.sh"
