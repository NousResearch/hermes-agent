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


def test_background_reconciliation_runs_without_a_jobs_request(monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    reconciled = Event()
    scans = []

    def restore():
        scans.append(None)
        if len(scans) > 1:
            reconciled.set()

    monkeypatch.setattr(lm, "_restore_download_plans", restore)
    monkeypatch.setattr(lm, "_RECOVERY_SCAN_SECONDS", 0.01)
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app):
        assert reconciled.wait(5)


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
