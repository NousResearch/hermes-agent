"""Normal download routes recover a multi-file model after outage and restart."""
from __future__ import annotations

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client, poll  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export


def test_interrupted_plan_survives_backend_restart_and_resumes_all_assets(client, monkeypatch, dl_server):
    from hermes_cli.web_routers import local_models as lm

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
    assert parked["phase"] == "interrupted" and "503" in parked["error"]
    assert parked["done_bytes"] == first.stat().st_size + block
    assert not second.exists() and not asset.exists()
    duplicate = client.post("/api/local-models/download", json={"model_id": entry.id}).json()
    assert duplicate["job_id"] == started["job_id"]

    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
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
