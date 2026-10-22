"""Normal download routes recover a multi-file model after outage and restart."""
from __future__ import annotations

import socket
import urllib.error
from dataclasses import replace

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client, poll  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export


@pytest.fixture
def draft_plan(client, monkeypatch, dl_server):
    from hermes_cli.local_runtime.catalog import AssetFile
    from hermes_cli.web_routers import local_models as lm

    entry = replace(lm.catalog.CATALOG[0], draft=AssetFile("optional.gguf", 5))
    variant = entry.variants[0]
    plan = [(url(dl_server, "/required"), lm.bootstrap.models_dir() / "required.gguf", 4),
            (url(dl_server, "/optional"), lm.bootstrap.assets_dir() / "optional.gguf", 5)]
    RangeHandler.payloads = {"/required": b"main", "/optional": b"draft"}
    monkeypatch.setattr(lm, "_download_target", lambda _: (entry, variant))
    monkeypatch.setattr(lm, "_download_plan", lambda *_: plan)
    monkeypatch.setattr(lm.Download, "_wait_retry", lambda *_: None)
    original = RangeHandler.do_GET
    failures = set()

    def respond(handler):
        if handler.path in failures:
            handler.send_error(503)
        else:
            original(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    return lm, entry, plan, failures


def test_optional_transport_failure_does_not_restore_a_completed_model_job(client, draft_plan, monkeypatch):
    lm, entry, plan, failures = draft_plan
    failures.add("/optional")
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    settled = poll(client, started["job_id"])
    assert settled["status"] == "done", settled
    assert plan[0][1].read_bytes() == b"main" and not plan[1][1].exists()
    assert not list(lm.download_plans._root().glob("*.json"))
    lm._restore_download_plans()
    assert not any(job["status"] == "paused" for job in lm._JOBS.values())
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    lm._restore_download_plans()
    assert client.get("/api/local-models/jobs").json()["jobs"] == []


def test_draft_download_keeps_full_identity_and_restores_optional_intent(client, draft_plan, monkeypatch):
    lm, entry, plan, failures = draft_plan
    failures.add("/required")
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    parked = poll(client, started["job_id"])
    assert parked["status"] == "paused", parked
    key = lm.download_plans.plan_key(plan)
    assert lm._JOBS[started["job_id"]]["plan_key"] == key
    duplicate = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    assert duplicate["job_id"] == started["job_id"]
    saved = list(lm.download_plans.pending_plans())
    assert len(saved) == 1 and saved[0][2] == plan
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(restored) == 1 and restored[0]["status"] == "paused"
    # Recovery must use the saved optional role, not require a live catalog entry.
    monkeypatch.setattr(lm.catalog, "catalog_by_id", dict)
    failures.clear()
    resumed = client.post("/api/local-models/download/resume", json={"job_id": restored[0]["job_id"]})
    assert resumed.json()["resumed"]
    settled = poll(client, restored[0]["job_id"])
    assert settled["status"] == "done", settled
    assert plan[0][1].read_bytes() == b"main" and plan[1][1].read_bytes() == b"draft"
    assert not list(lm.download_plans._root().glob("*.json"))


@pytest.mark.parametrize("optional_available", [False, True])
def test_pause_during_optional_retry_restores_its_role(client, draft_plan, monkeypatch, optional_available):
    lm, entry, plan, failures = draft_plan
    failures.add("/optional")
    original_wait = lm.Download._wait_retry

    def pause_retry(download, delay):
        download.pause()
        download._check_pause()

    monkeypatch.setattr(lm.Download, "_wait_retry", pause_retry)
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    parked = poll(client, started["job_id"])
    assert parked["status"] == "paused" and parked["error"] is None, parked
    before = plan[0][1].stat().st_mtime_ns
    saved = list(lm.download_plans.pending_plans())
    assert len(saved) == 1 and saved[0][2] == plan
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm.catalog, "catalog_by_id", dict)
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(restored) == 1
    monkeypatch.setattr(lm.Download, "_wait_retry", original_wait)
    if optional_available:
        failures.clear()
    assert client.post("/api/local-models/download/resume", json={"job_id": restored[0]["job_id"]}).json()["resumed"]
    settled = poll(client, restored[0]["job_id"])
    assert settled["status"] == "done", settled
    assert plan[1][1].exists() is optional_available
    assert plan[0][1].read_bytes() == b"main" and plan[0][1].stat().st_mtime_ns == before
    assert not list(lm.download_plans._root().glob("*.json"))


@pytest.mark.parametrize("outage_kind", ["http", "dns"])
def test_interrupted_plan_survives_backend_restart_and_resumes_all_assets(client, monkeypatch, dl_server, outage_kind):
    from hermes_cli.web_routers import local_models as lm
    from pm import downloader

    block = 1 << 20
    payload = b"weights!" * (4 * block // 8)
    companion = b"vision and MTP companion"
    entry = lm.catalog.CATALOG[0]
    variant = entry.variants[0]
    models = lm.bootstrap.models_dir()
    models.mkdir(parents=True)
    first, second, asset = models / "first.gguf", models / "second.gguf", models / "assets" / "mtp.gguf"
    first.write_bytes(b"complete first shard")
    before = first.stat().st_mtime_ns
    RangeHandler.payloads = {"/second": payload, "/asset": companion}
    plan = [(url(dl_server, "/first"), first, 123), (url(dl_server, "/second"), second, 456),
            (url(dl_server, "/asset"), asset, 789)]
    monkeypatch.setattr(lm, "_download_target", lambda _: (entry, variant))
    monkeypatch.setattr(lm, "_download_plan", lambda *_: plan)
    real_wait = lm.Download._wait_retry
    monkeypatch.setattr(lm.Download, "_wait_retry", lambda self, _: real_wait(self, 0))
    original_get = RangeHandler.do_GET
    attempts = []
    outage = True
    dns_requests = []
    original_open = downloader._OPENER.open

    def open_request(request, **kwargs):
        if outage_kind == "dns" and outage and attempts:
            dns_requests.append(request.full_url)
            raise urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided"))
        return original_open(request, **kwargs)

    monkeypatch.setattr(downloader._OPENER, "open", open_request)

    def respond(handler):
        bounds = handler.headers.get("Range")
        if handler.path == "/second" and bounds != "bytes=0-0":
            attempts.append(bounds)
            if outage:
                if len(attempts) == 1:
                    handler.send_response(206)
                    handler.send_header("Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload[:block])
                    handler.wfile.flush()
                    handler.close_connection = True
                else:
                    handler.send_error(503)
                return
        original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    # One connection gives the interrupted shard a deterministic durable prefix.
    real_download = lm.Download

    class SerialDownload(real_download):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, connections=1, **kwargs)

    monkeypatch.setattr(lm, "Download", SerialDownload)
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    parked = poll(client, started["job_id"])
    assert parked["status"] == "paused" and parked["can_resume"], parked
    assert parked["phase"] == "interrupted"
    assert ("503" if outage_kind == "http" else "nodename") in parked["error"]
    assert len(dns_requests) == (1 if outage_kind == "dns" else 0)
    assert parked["done_bytes"] == first.stat().st_size + block
    assert not second.exists() and not asset.exists()
    duplicate = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    assert duplicate["job_id"] == started["job_id"]

    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(restored) == 1
    recovered = restored[0]
    assert recovered["job_id"] != started["job_id"]
    assert recovered["status"] == "paused" and recovered["can_resume"]
    assert recovered["done_bytes"] == parked["done_bytes"]
    assert recovered["total_bytes"] == len(payload) + len(companion) + first.stat().st_size
    outage = False
    assert client.post("/api/local-models/download/resume", json={"job_id": recovered["job_id"]}).json()["resumed"]
    settled = poll(client, recovered["job_id"], until=("done", "error"))
    assert settled["status"] == "done", settled
    assert second.read_bytes() == payload and asset.read_bytes() == companion
    assert first.stat().st_mtime_ns == before
    assert attempts[-1] == f"bytes={block}-{len(payload) - 1}"
    assert settled["done_bytes"] == settled["total_bytes"]
    assert not list(lm.download_plans._root().glob("*.json"))


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_download_failures_are_actionable_without_automatic_resume(client, monkeypatch, dl_server, status):
    from hermes_cli.web_routers import local_models as lm

    entry = lm.catalog.CATALOG[0]
    variant = entry.variants[0]
    plan = [(url(dl_server, "/missing"), lm.bootstrap.models_dir() / "missing.gguf", 1)]
    monkeypatch.setattr(lm, "_download_target", lambda _: (entry, variant))
    monkeypatch.setattr(lm, "_download_plan", lambda *_: plan)

    def denied(handler):
        handler.send_error(status)

    monkeypatch.setattr(RangeHandler, "do_GET", denied)
    started = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    settled = poll(client, started["job_id"])
    assert settled["status"] == "error" and not settled["can_resume"]
    assert str(status) in settled["error"]
    assert not list(lm.download_plans._root().glob("*.json"))
