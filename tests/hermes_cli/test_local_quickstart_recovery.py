"""Quickstart adopts saved downloads and can leave an interrupted setup safely."""
from __future__ import annotations

import threading
from urllib.error import HTTPError

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client  # noqa: PLC0414 -- pytest fixture re-export
from tests.hermes_cli.test_local_download_recovery import draft_plan as draft_plan  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._range_server import RangeHandler, url
from tests.pm._range_server import dl_server as dl_server  # noqa: PLC0414 -- pytest fixture re-export
from tests.waiting import wait_until


def test_engine_interruption_preserves_draft_role_without_catalog_on_restart(client, draft_plan, monkeypatch):
    lm, entry, plan, _ = draft_plan
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **kw: None)
    monkeypatch.setattr(lm, "_quickstart_target", lambda *_: (entry, entry.variants[0]))
    monkeypatch.setattr(lm, "_runtime_target", lambda requested=None: ("test", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda _: None)

    def unavailable(job, backend):
        source = "https://example.invalid/engine"
        raise lm.DownloadTransportError(source, HTTPError(source, 503, "Unavailable", {}, None))

    monkeypatch.setattr(lm, "_install_engine_job", unavailable)
    response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
    assert response.status_code == 200, response.text
    job = lm._JOBS[response.json()["job_id"]]
    assert wait_until(lambda: job["status"] == "paused", timeout=5)
    saved = list(lm.download_plans.pending_plans())
    assert len(saved) == 1 and saved[0][2] == plan
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    monkeypatch.setattr(lm.catalog, "catalog_by_id", dict)
    lm._restore_download_plans()
    restored = next(iter(lm._JOBS.values()))
    assert restored["optional_files"] == [str(plan[1][1])]
    assert restored["download_plan"] == plan and restored["setup_stage"] == "engine"


@pytest.mark.parametrize("restart", [False, True])
def test_quickstart_adopts_a_catalog_download_with_a_draft(client, draft_plan, monkeypatch, restart):
    lm, entry, plan, failures = draft_plan
    failures.add("/required")
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    assert wait_until(lambda: lm._JOBS[started["job_id"]]["status"] != "running", timeout=5)
    if restart:
        monkeypatch.setattr(lm, "_JOBS", {})
        monkeypatch.setattr(lm, "_RUNNING", {})
        lm._restore_download_plans()
    previous = next(job for job in lm._JOBS.values() if job["status"] == "paused")
    old_resume = lm._RUNNING[previous["job_id"]]["resume"]
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **kw: None)
    monkeypatch.setattr(lm, "_quickstart_target", lambda *_: (entry, entry.variants[0]))
    monkeypatch.setattr(lm, "_runtime_target", lambda requested=None: ("test", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda _: object())
    monkeypatch.setattr(lm, "_set_runtime_enabled", lambda _: {})
    calls = []
    monkeypatch.setattr(lm, "_ensure_server", lambda *a, **kw: calls.append("server"))
    monkeypatch.setattr(lm, "_assign_default", lambda *a: calls.append("default"))
    failures.clear()
    response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    assert wait_until(lambda: lm._JOBS[job_id]["status"] != "running", timeout=5)
    jobs = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["kind"] == "quickstart" and jobs[0]["status"] == "done", jobs
    assert not old_resume()
    assert plan[0][1].read_bytes() == b"main" and plan[1][1].read_bytes() == b"draft"
    assert calls == ["server", "default"]
    assert not lm._QUICKSTART_LOCK.locked()
    assert not list(lm.download_plans._root().glob("*.json"))


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


def test_paused_handover_keeps_the_adopted_plan_until_discard_owns_it(client, monkeypatch, dl_server):
    lm, entries, plans, _ = _routes(monkeypatch, dl_server)
    entry = entries[0]
    plan = plans[entry.id]
    key = lm.download_plans.save_plan({"target": entry.display_name, "model_id": entry.id}, plan)
    lm._restore_download_plans()
    with lm.download_plans.plan_lock(key):
        response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        job = lm._JOBS[job_id]
        assert wait_until(lambda: job["phase"] == "waiting", timeout=5)
        assert client.post("/api/local-models/download/pause", json={"job_id": job_id}).json()["paused"]
        assert wait_until(lambda: job["status"] == "paused", timeout=5)
        lm._restore_download_plans()
        assert job_id in lm._JOBS and lm.download_plans.has_plan(key)
        assert not client.post("/api/local-models/download/discard", json={"job_id": job_id}).json()["discarded"]
    resume = lm._RUNNING[job_id]["resume"]
    assert client.post("/api/local-models/download/discard", json={"job_id": job_id}).json()["discarded"]
    assert not resume() and not lm._QUICKSTART_LOCK.locked()
    lm._restore_download_plans()
    assert not lm._JOBS and not list(lm.download_plans._root().glob("*.json"))


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
