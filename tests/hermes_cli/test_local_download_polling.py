"""Job reads do not repeat filesystem reconciliation."""
from threading import Event

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
