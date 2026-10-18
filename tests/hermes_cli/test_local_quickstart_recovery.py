"""Quickstart adopts saved downloads and can leave an interrupted setup safely."""
from __future__ import annotations

import threading
from urllib.error import HTTPError

from tests.hermes_cli.test_local_download_jobs import client as client  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._range_server import RangeHandler, url
from tests.pm._range_server import dl_server as dl_server  # noqa: PLC0414 -- pytest fixture re-export
from tests.waiting import wait_until


def _routes(monkeypatch, server):
    from hermes_cli.web_routers import local_models as lm

    entries = lm.catalog.CATALOG[:2]
    choices = {entry.id: (entry, entry.variants[0]) for entry in entries}
    plans = {entry.id: [(url(server, f"/{index}"), lm.bootstrap.models_dir() / f"{entry.variants[0].model_id}.gguf", 5)]
             for index, entry in enumerate(entries)}
    RangeHandler.payloads = {"/0": b"first", "/1": b"other"}
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **kw: None)
    monkeypatch.setattr(lm, "_quickstart_target", lambda body, budget: choices[body.model_id])
    monkeypatch.setattr(lm, "_download_target", lambda model_id: choices[model_id])
    monkeypatch.setattr(lm, "_runtime_target", lambda requested=None: ("test", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda backend: object())
    monkeypatch.setattr(lm, "_download_plan", lambda entry, variant: plans[entry.id])
    monkeypatch.setattr(lm, "_set_runtime_enabled", lambda enabled: {})
    calls = []
    monkeypatch.setattr(lm, "_ensure_server", lambda job, config, model_id, **kw: calls.append(("server", model_id)))
    monkeypatch.setattr(lm, "_assign_default", lambda job, model_id: calls.append(("default", model_id)))
    return lm, entries, plans, calls


def test_quickstart_adopts_restored_plan_and_keeps_its_setup_continuation(client, monkeypatch, dl_server):
    lm, entries, plans, calls = _routes(monkeypatch, dl_server)
    entry = entries[0]
    plan = plans[entry.id]
    lm.download_plans.save_plan({"target": entry.display_name, "model_id": entry.id}, plan)
    lm._restore_download_plans()
    original = client.get("/api/local-models/jobs").json()["jobs"][0]
    old_resume = lm._RUNNING[original["job_id"]]["resume"]
    flowing, release = threading.Event(), threading.Event()
    real_run = lm.Download.run

    def run(download, progress=None):
        flowing.set()
        assert release.wait(10)
        return real_run(download, progress=progress)

    monkeypatch.setattr(lm.Download, "run", run)
    response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        assert flowing.wait(5)
        rows = client.get("/api/local-models/jobs").json()["jobs"]
        assert len(rows) == 1 and rows[0]["job_id"] == job_id and rows[0]["kind"] == "quickstart"
        download = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
        assert download["job_id"] == job_id
        assert not old_resume()
    finally:
        release.set()
        wait_until(lambda: lm._JOBS[job_id]["status"] != "running", timeout=10)
    rows = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(rows) == 1 and rows[0]["status"] == "done"
    assert plan[0][1].read_bytes() == b"first"
    assert calls == [("server", entry.variants[0].model_id), ("default", entry.variants[0].model_id)]
    assert not lm._QUICKSTART_LOCK.locked()


def test_discard_interrupted_quickstart_allows_another_model_and_retires_old_resume(client, monkeypatch, dl_server):
    lm, entries, plans, calls = _routes(monkeypatch, dl_server)
    first, second = entries
    real_run = lm.Download.run

    def run(download, progress=None):
        if download.sources[0].url == plans[first.id][0][0]:
            source = download.sources[0].url
            raise lm.DownloadTransportError(source, HTTPError(source, 503, "Unavailable", {}, None))
        return real_run(download, progress=progress)

    monkeypatch.setattr(lm.Download, "run", run)
    job_id = client.post("/api/local-models/quickstart", json={"model_id": first.id}).json()["job_id"]
    wait_until(lambda: lm._JOBS[job_id]["status"] != "running", timeout=5)
    parked = client.get(f"/api/local-models/jobs/{job_id}").json()
    assert parked["status"] == "paused" and parked["phase"] == "interrupted" and parked["can_discard"]
    assert [row["job_id"] for row in client.get("/api/local-models/jobs").json()["jobs"]] == [job_id]
    old_resume = lm._RUNNING[job_id]["resume"]
    assert client.post("/api/local-models/quickstart", json={"model_id": second.id}).status_code == 409
    assert client.post("/api/local-models/download/discard", json={"job_id": job_id}).json()["discarded"]
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda backend: object())
    response = client.post("/api/local-models/quickstart", json={"model_id": second.id})
    assert response.status_code == 200, response.text
    next_id = response.json()["job_id"]
    wait_until(lambda: lm._JOBS[next_id]["status"] != "running", timeout=10)
    assert lm._JOBS[next_id]["status"] == "done"
    assert not old_resume()
    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).status_code == 404
    assert calls == [("server", second.variants[0].model_id), ("default", second.variants[0].model_id)]
    assert not lm._QUICKSTART_LOCK.locked()
