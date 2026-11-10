"""Cron prompt recovery (#82990) must serialize with cron writers: it re-reads jobs.json under
the canonical cron lock and publishes via the locked, merge-aware save path, so cron edits
committed while recovery runs are never erased by a stale whole-document write."""

import json
from pathlib import Path


def _seed_jobs(path: Path, jobs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}))


def test_concurrent_cron_edits_survive_recovery(tmp_path, monkeypatch):
    """Recovery must publish under the cron writer lock from a fresh read: cron edits
    committed after recovery first read jobs.json (create, remove, schedule edit, a
    legitimate prompt edit) all survive; only the still-degraded prompt is restored."""
    import cron.jobs as cron_jobs
    from hermes_cli.backup import create_quick_snapshot
    import hermes_cli.backup_cron_prompts as bcp
    hermes_home = tmp_path / ".hermes"
    jobs_path = hermes_home / "cron" / "jobs.json"
    sched = {"kind": "interval", "minutes": 60, "display": "every 60m"}

    def job(job_id, prompt):
        return {"id": job_id, "name": job_id.upper(), "prompt": prompt,
                "schedule": sched, "enabled": True}

    _seed_jobs(jobs_path, [job("a", "A real prompt."), job("b", "B real prompt."),
                          job("c", "C real prompt.")])
    snap_id = create_quick_snapshot(label="pre-update", hermes_home=hermes_home, keep=5)
    assert snap_id
    _seed_jobs(jobs_path, [job("a", "A"), job("b", "B"), job("c", "C real prompt.")])

    real_load = bcp._load_cron_jobs_doc
    fired = []

    def load_then_race(path):
        doc = real_load(path)
        if Path(path) == jobs_path and not fired:
            fired.append(True)
            with cron_jobs.use_cron_store(hermes_home):
                new_id = cron_jobs.create_job(prompt="New prompt.", schedule="every 1h",
                                              name="new")["id"]
                fired.append(new_id)
                assert cron_jobs.remove_job("c")
                cron_jobs.update_job("a", {"schedule": cron_jobs.parse_schedule("every 2h")})
                cron_jobs.update_job("b", {"prompt": "User rewrote b."})
        return doc

    monkeypatch.setattr(bcp, "_load_cron_jobs_doc", load_then_race)
    result = bcp.restore_cron_prompt_fields_if_degraded(snap_id, hermes_home=hermes_home)

    assert len(fired) == 2
    assert result is not None and result["job_ids"] == ["a"]
    final = {j["id"]: j for j in json.loads(jobs_path.read_text())["jobs"]}
    assert set(final) == {"a", "b", fired[1]}
    assert final["a"]["prompt"] == "A real prompt."
    assert final["a"]["schedule"]["minutes"] == 120
    assert final["b"]["prompt"] == "User rewrote b."
    assert final[fired[1]]["prompt"] == "New prompt."
