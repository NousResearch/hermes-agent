"""One malformed legacy ``cron/bot_chat_pending`` record must not stop the profile's cron.

``drain_legacy_pending`` runs on every tick before ``get_due_jobs()``. A record missing a
field (or any per-record failure) is set aside as ``ambiguous`` evidence, and the tick
still fires the due job beside it.
"""
import json
from datetime import timedelta

import cron.jobs as J
import cron.scheduler as S
from cron import scheduler_authority


def test_corrupt_legacy_record_is_isolated_and_due_job_still_fires(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "scripts").mkdir(parents=True)
    pending = home / "cron" / "bot_chat_pending"
    pending.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name, value in (("HERMES_DIR", home), ("CRON_DIR", home / "cron"),
                        ("JOBS_FILE", home / "cron" / "jobs.json"),
                        ("OUTPUT_DIR", home / "cron" / "output")):
        monkeypatch.setattr(J, name, value)
    monkeypatch.setattr(S, "_hermes_home", home)
    monkeypatch.setattr(S, "_sweep_mcp_orphans", lambda: None)
    monkeypatch.setattr(scheduler_authority, "reconcile_pending", lambda *, allow_connect=True: None)
    S._running_job_ids.clear()

    corrupt = pending / "corrupt.json"  # a queued record with no ``home``/``content``
    corrupt.write_text(json.dumps({"id": "legacy-1", "status": "queued"}), encoding="utf-8")
    script = home / "scripts" / "fire.sh"
    script.write_text("#!/bin/sh\necho fired\n", encoding="utf-8")
    script.chmod(0o755)
    job = J.create_job(prompt=None, schedule="every 1h", name="script", script="fire.sh",
                       no_agent=True, deliver="local")
    stored = J.load_jobs()
    stored[0]["next_run_at"] = (J._hermes_now() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    J.save_jobs(stored)

    assert S.tick(verbose=False) == 1
    assert J.get_job(job["id"])["last_status"] == "ok"
    record = json.loads(corrupt.read_text(encoding="utf-8"))
    assert record["status"] == "ambiguous" and record["error"]
