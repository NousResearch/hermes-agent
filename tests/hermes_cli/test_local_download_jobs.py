"""Local-model jobs use PM transfers for every entry point."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm.bootstrap, "refresh_local_runtime", lambda: False)
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app) as client:
        yield client


def poll(client, job_id, until=("done", "error", "paused")):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/api/local-models/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in until:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not settle: {job}")


def test_browsed_download_can_pause_and_resume_pm_transfer(client, monkeypatch, dl_server):
    from hermes_cli.web_routers import local_models as lm

    payload = bytes(range(256)) * (32 * 1024)
    RangeHandler.payloads["/test.gguf"] = payload
    monkeypatch.setattr(lm, "_hf_url", lambda repo, path: url(dl_server, "/test.gguf"))
    flowing, release = threading.Event(), threading.Event()
    real_download = lm.Download

    class ObservedDownload(real_download):
        def run(self, progress=None):
            def tick(done, total, ranges):
                if progress is not None:
                    progress(done, total, ranges)
                if 0 < done < total and not flowing.is_set():
                    flowing.set()
                    assert release.wait(15)
            return super().run(progress=tick)

    monkeypatch.setattr(lm, "Download", ObservedDownload)
    response = client.post("/api/local-models/download-browsed", json={"repo": "a/b", "paths": ["test.gguf"]})
    job_id = response.json()["job_id"]
    try:
        assert flowing.wait(5), "browsed download bypassed PM's downloader"
        assert client.post("/api/local-models/download/pause", json={"job_id": job_id}).json()["paused"]
    finally:
        release.set()
    job = poll(client, job_id)
    assert job["status"] == "paused", job
    assert job["can_resume"] and not job["can_pause"]
    assert job["error"] is None
    before = job["done_bytes"]
    assert before > 0
    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
    job = poll(client, job_id, until=("done", "error"))
    assert job["status"] == "done", job
    assert job["done_bytes"] == job["total_bytes"] == len(payload)
    assert (lm.bootstrap.models_dir() / "test.gguf").read_bytes() == payload
    assert not job["can_pause"] and not job["can_resume"]


def test_quickstart_plan_progress_uses_actual_whole_plan_bytes(client, monkeypatch, dl_server, tmp_path):
    from hermes_cli.web_routers import local_models as lm

    bodies = {"first": b"first", "second": b"second"}
    RangeHandler.payloads = {"/" + name: body for name, body in bodies.items()}
    plan = [(url(dl_server, "/" + name), tmp_path / (name + ".gguf"), 1000) for name in bodies]
    job = lm._job("quickstart", "test")
    lm._run_download_plan(job, plan, "test")
    expected = sum(map(len, bodies.values()))
    assert job["done_bytes"] == job["total_bytes"] == expected
    assert sum(end - start for rows in job["ranges"].values() for start, end in rows) == expected


def test_existing_weights_do_not_skip_missing_companion_download(client, monkeypatch, dl_server, tmp_path):
    from dataclasses import replace
    from hermes_cli.local_runtime.catalog import AssetFile
    from hermes_cli.web_routers import local_models as lm

    entry = replace(lm.catalog.CATALOG[0], mmproj=AssetFile("projector.gguf", 1000), draft=None)
    variant = entry.variants[0]
    monkeypatch.setattr(lm, "_download_target", lambda model_id: (entry, variant))
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: [variant.model_id])
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"already downloaded")
    companion = tmp_path / "assets" / "projector.gguf"
    RangeHandler.payloads["/projector"] = b"projector bytes"
    plan = [(url(dl_server, "/weights"), weights, 1000),
            (url(dl_server, "/projector"), companion, 1000)]
    monkeypatch.setattr(lm, "_download_plan", lambda *args: plan)
    response = client.post("/api/local-models/download", json={"model_id": entry.id})
    assert response.json().get("job_id"), "weights-only staging incorrectly skipped the projector"
    job = poll(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert companion.read_bytes() == RangeHandler.payloads["/projector"]
    assert job["done_bytes"] == weights.stat().st_size + companion.stat().st_size
