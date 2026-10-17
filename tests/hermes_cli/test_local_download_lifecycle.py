"""Saved download plans expire and can be discarded without deleting models."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client  # noqa: PLC0414 -- pytest fixture re-export


def _saved_plan(lm):
    models = lm.bootstrap.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    first, second = models / "complete.gguf", models / "pending.gguf"
    first.write_bytes(b"completed model")
    url = "https://example.invalid/pending.gguf"
    plan = [("https://example.invalid/complete.gguf", first, first.stat().st_size), (url, second, 8)]
    job = lm._job("quickstart", "test model", model_id="test")
    key = lm.download_plans.save_plan(job, plan)
    job["plan_key"] = key
    root = lm.download_plans._root().parent
    partial_key = hashlib.sha256(url.encode()).hexdigest()
    part, side = root / f"{partial_key}.part", root / f"{partial_key}.ranges"
    part.write_bytes(b"part")
    side.write_text(json.dumps({"total": 8, "etag": '"stable"', "sha256": "", "ranges": [[0, 4]]}))
    return job, key, first, part, side


@pytest.mark.parametrize("age_days,force_gc,fresh_checkpoint,retained", [
    (2, False, False, True), (8, False, False, False), (0, True, False, False), (8, False, True, True),
])
def test_recipe_retention_matches_partial_gc(client, monkeypatch, age_days, force_gc, fresh_checkpoint, retained):
    from hermes_cli.web_routers import local_models as lm
    from pm.download_state import collect_partials

    _job, key, first, part, side = _saved_plan(lm)
    recipe = lm.download_plans._root() / f"{key}.json"
    when = time.time() - age_days * 24 * 60 * 60
    for path in (recipe, part, side):
        os.utime(path, (when, when))
    if fresh_checkpoint:
        os.utime(side, None)
    collect_partials(part.parent, **({"grace_seconds": 0} if force_gc else {}))
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert recipe.exists() is retained
    assert bool(restored) is retained
    if retained:
        assert restored[0]["done_bytes"] == first.stat().st_size + 4
    assert first.read_bytes() == b"completed model"


def test_discard_removes_a_parked_plan_and_releases_quickstart_ownership(client, monkeypatch):
    from hermes_cli.web_routers import local_models as lm
    from pm.download_state import collect_download_plans

    job, key, first, part, side = _saved_plan(lm)
    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", lock)
    lm._spawn_job(job, "parked-test", lambda: None, on_exit=lock.release, resumable=True, start_paused=True)
    view = client.get(f"/api/local-models/jobs/{job['job_id']}").json()
    assert view["can_discard"]
    controls = lm._RUNNING[job["job_id"]]
    with lm.download_plans.plan_lock(key):
        collect_download_plans(part.parent, grace_seconds=0)
        assert lm.download_plans.has_plan(key)
        busy = client.post("/api/local-models/download/discard", json={"job_id": job["job_id"]})
        assert busy.status_code == 200 and not busy.json()["discarded"]
    result = client.post("/api/local-models/download/discard", json={"job_id": job["job_id"]})
    assert result.status_code == 200 and result.json()["discarded"]
    assert not controls["discard"]() and not controls["resume"]()
    assert lock.acquire(blocking=False)
    lock.release()
    assert client.get(f"/api/local-models/jobs/{job['job_id']}").status_code == 404
    assert client.get("/api/local-models/jobs").json()["jobs"] == []
    assert not (lm.download_plans._root() / f"{key}.json").exists()
    assert first.read_bytes() == b"completed model"
    assert part.read_bytes() == b"part" and side.is_file()
