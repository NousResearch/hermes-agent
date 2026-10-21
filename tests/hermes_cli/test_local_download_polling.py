"""Job reads do not repeat filesystem reconciliation."""
import json
from threading import Event

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_recovery_runs_at_startup_instead_of_each_jobs_request(monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    scans = []
    monkeypatch.setattr(lm, "_restore_download_plans", lambda: scans.append(None))
    monkeypatch.setattr(lm, "_JOBS", {})
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app) as client:
        assert len(scans) == 1
        for _ in range(10):
            assert client.get("/api/local-models/jobs").json() == {"jobs": []}
        assert len(scans) == 1


def test_failed_startup_reconciliation_is_retried_without_a_jobs_request(monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    reconciled = Event()
    scans = []

    def restore():
        scans.append(None)
        if len(scans) == 1:
            raise OSError("saved download cache is unavailable")
        reconciled.set()

    monkeypatch.setattr(lm, "_restore_download_plans", restore)
    monkeypatch.setattr(lm, "_RECOVERY_SCAN_SECONDS", 0.01)
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app):
        assert reconciled.wait(5)


def test_failed_plan_does_not_publish_a_job_or_stop_later_recovery(tmp_path, monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr("pm.paths.partials_root", lambda: tmp_path / "partials")
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    records = []
    for name in ("unreadable", "recoverable"):
        plan = [(f"https://example.invalid/{name}", tmp_path / f"{name}.gguf", 8)]
        key = lm.download_plans.save_plan({"target": name}, plan)
        record = json.loads((lm.download_plans._root() / f"{key}.json").read_bytes())
        records.append((key, record, plan))
    monkeypatch.setattr(lm.download_plans, "pending_plans", lambda: iter(records))
    retained = lm.download_plans.retained_progress

    def progress(plan):
        if plan == records[0][2]:
            raise OSError("saved download is unreadable")
        return retained(plan)

    monkeypatch.setattr(lm.download_plans, "retained_progress", progress)
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app) as client:
        jobs = client.get("/api/local-models/jobs").json()["jobs"]
        assert len(jobs) == 1 and jobs[0]["target"] == "recoverable"
        assert jobs[0]["status"] == "paused" and jobs[0]["can_resume"]
        assert jobs[0]["can_discard"]


@pytest.mark.parametrize("invalid_file", [[], ""], ids=["empty-sequence", "empty-string"])
def test_malformed_plan_does_not_block_startup_or_other_recovery(tmp_path, monkeypatch, invalid_file):
    from hermes_cli.web_routers import local_models as lm

    partials, models = tmp_path / "partials", tmp_path / "models"
    models.mkdir()
    completed = models / "complete.gguf"
    completed.write_bytes(b"completed model")
    monkeypatch.setattr("pm.paths.partials_root", lambda: partials)
    monkeypatch.setattr(lm.bootstrap, "models_dir", lambda: models)
    monkeypatch.setattr(lm.bootstrap, "assets_dir", lambda: models / "assets")
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    plan = [("https://example.invalid/model", models / "pending.gguf", 8)]
    key = lm.download_plans.save_plan({"target": "recoverable model"}, plan)
    malformed = lm.download_plans._root() / ("0" * 64 + ".json")
    malformed.write_text(json.dumps({"files": [invalid_file]}))
    app = FastAPI()
    app.include_router(lm.router)

    with TestClient(app) as client:
        response = client.get("/api/local-models/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]
        assert len(jobs) == 1 and jobs[0]["target"] == "recoverable model"
        assert jobs[0]["status"] == "paused" and jobs[0]["can_resume"]
    assert lm.download_plans.has_plan(key)
    assert malformed.is_file()
    assert completed.read_bytes() == b"completed model"
