"""AC5 — cron skill-ref migration fires for shared skills + composite rollback.

``cron.jobs.rewrite_skill_refs`` is name-based (not tree-based), so a shared
skill named in a cron job is rewritten exactly like a local one; this test
proves it end-to-end against a temp HERMES_HOME and proves the COMPOSITE
rollback named in the spec: jobs.json lives OUTSIDE skills-shared/, so its
restore path is the agent snapshot's captured cron-jobs.json (via
``curator_backup._restore_cron_skill_links``), not ``git revert``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    jobs = {
        "jobs": [
            {
                "id": "job-1",
                "name": "smart-home sweep",
                "skills": ["clanker-e2e", "other-skill"],
                "schedule": "0 4 * * *",
                "prompt": "run the sweep",
                "enabled": True,
            }
        ],
        "updated_at": "2026-07-13T00:00:00Z",
    }
    (home / "cron" / "jobs.json").write_text(
        json.dumps(jobs, indent=2), encoding="utf-8"
    )
    import cron.jobs as cron_jobs
    importlib.reload(cron_jobs)
    import agent.curator_backup as backup
    importlib.reload(backup)
    yield {"home": home, "cron_jobs": cron_jobs, "backup": backup}


def test_shared_skill_cron_ref_rewritten_to_umbrella(cron_env):
    cron_jobs = cron_env["cron_jobs"]
    report = cron_jobs.rewrite_skill_refs(
        consolidated={"clanker-e2e": "smart-home-umbrella"}, pruned=[],
    )
    assert report["jobs_updated"] == 1
    data = json.loads(
        (cron_env["home"] / "cron" / "jobs.json").read_text(encoding="utf-8")
    )
    skills = data["jobs"][0]["skills"]
    assert "smart-home-umbrella" in skills
    assert "clanker-e2e" not in skills
    assert "other-skill" in skills  # untouched sibling ref preserved


def test_snapshot_captures_jobs_json_and_rollback_restores_narrow_name(cron_env):
    """The composite-rollback proof: snapshot BEFORE the rewrite; rewrite;
    then the snapshot's cron restore path puts the narrow name back."""
    backup = cron_env["backup"]
    cron_jobs = cron_env["cron_jobs"]

    snap = backup.snapshot_skills(reason="pre-shared-consolidation")
    assert snap is not None
    captured = snap / backup.CRON_JOBS_FILENAME
    assert captured.exists()
    pre = json.loads(captured.read_text(encoding="utf-8"))
    assert pre["jobs"][0]["skills"] == ["clanker-e2e", "other-skill"]

    cron_jobs.rewrite_skill_refs(
        consolidated={"clanker-e2e": "smart-home-umbrella"}, pruned=[],
    )
    live = cron_env["home"] / "cron" / "jobs.json"
    assert "smart-home-umbrella" in live.read_text(encoding="utf-8")

    # rollback restores ONLY the skills/skill fields from the snapshot
    result = backup._restore_cron_skill_links(snap)
    assert result.get("restored"), result
    data = json.loads(live.read_text(encoding="utf-8"))
    assert data["jobs"][0]["skills"] == ["clanker-e2e", "other-skill"]
