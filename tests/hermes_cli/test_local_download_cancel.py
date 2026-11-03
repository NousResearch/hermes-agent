"""A local-model download can be cancelled from running or paused, discarding its partial bytes.

Regression for #146469: a paused download offered only Resume — no way to abandon it,
reclaim the partial bytes, or go back to choosing another model.
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export

_PAYLOAD = bytes(range(256)) * (32 * 1024)


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


def _poll(client, job_id, until):
    deadline = time.monotonic() + 15
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/local-models/jobs/{job_id}").json()
        if job["status"] in until:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not settle: {job}")


def _partials_for(source_url: str):
    from pm import paths

    key = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    root = paths.partials_root()
    return [root / f"{key}.part", root / f"{key}.ranges"]


def _start_held_download(client, monkeypatch, dl_server):
    """Start a browsed download and hold it mid-transfer; returns (job_id, source_url, release)."""
    from hermes_cli.web_routers import local_models as lm

    RangeHandler.payloads["/test.gguf"] = _PAYLOAD
    source_url = url(dl_server, "/test.gguf")
    monkeypatch.setattr(lm, "_hf_url", lambda repo, path: source_url)
    flowing, release = threading.Event(), threading.Event()
    real_download = lm.Download

    class HeldDownload(real_download):
        def run(self, progress=None):
            def tick(done, total, ranges):
                if progress is not None:
                    progress(done, total, ranges)
                if 0 < done < total and not flowing.is_set():
                    flowing.set()
                    assert release.wait(15)
            return super().run(progress=tick)

    monkeypatch.setattr(lm, "Download", HeldDownload)
    job_id = client.post("/api/local-models/download-browsed",
                         json={"repo": "a/b", "paths": ["test.gguf"]}).json()["job_id"]
    assert flowing.wait(5), "download never started moving bytes"
    return job_id, source_url, release


def _assert_cancelled_and_clean(client, job_id, source_url):
    from hermes_cli.web_routers import local_models as lm

    job = _poll(client, job_id, until=("cancelled", "done", "error"))
    assert job["status"] == "cancelled", job
    assert job["error"] is None
    assert not job["can_resume"] and not job["can_pause"] and not job["can_cancel"]
    # Nothing landed and nothing is left to resume from.
    assert not (lm.bootstrap.models_dir() / "test.gguf").exists()
    assert [p for p in _partials_for(source_url) if p.exists()] == []
    # The pane rediscovers in-flight work from /jobs — a cancelled job must not read as active.
    active = [j["job_id"] for j in client.get("/api/local-models/jobs").json()["jobs"]
              if j["status"] in ("running", "paused")]
    assert job_id not in active


def test_paused_download_can_be_cancelled_and_its_partials_discarded(client, monkeypatch, dl_server):
    job_id, source_url, release = _start_held_download(client, monkeypatch, dl_server)
    try:
        assert client.post("/api/local-models/download/pause", json={"job_id": job_id}).json()["paused"]
    finally:
        release.set()
    job = _poll(client, job_id, until=("paused", "done", "error"))
    assert job["status"] == "paused", job
    assert job["can_cancel"], "a parked download must offer a way out, not only Resume"
    assert any(p.exists() for p in _partials_for(source_url)), "precondition: the pause banked partial bytes"

    assert client.post("/api/local-models/download/cancel", json={"job_id": job_id}).json()["cancelled"]
    _assert_cancelled_and_clean(client, job_id, source_url)


def test_running_download_can_be_cancelled(client, monkeypatch, dl_server):
    job_id, source_url, release = _start_held_download(client, monkeypatch, dl_server)
    try:
        job = client.get(f"/api/local-models/jobs/{job_id}").json()
        assert job["can_cancel"], job
        assert client.post("/api/local-models/download/cancel", json={"job_id": job_id}).json()["cancelled"]
    finally:
        release.set()
    _assert_cancelled_and_clean(client, job_id, source_url)


def test_cancel_unknown_job_is_404(client):
    assert client.post("/api/local-models/download/cancel", json={"job_id": "nope"}).status_code == 404


def _hold_after_first_byte(monkeypatch, hold_dest_name: str):
    """Let every file through except ``hold_dest_name``, which parks after its first progress tick."""
    from hermes_cli.web_routers import local_models as lm

    flowing, release = threading.Event(), threading.Event()
    real_download = lm.Download

    class HeldDownload(real_download):
        def run(self, progress=None):
            def tick(done, total, ranges):
                if progress is not None:
                    progress(done, total, ranges)
                moving = any(str(dest).endswith(hold_dest_name) and rows and rows[0][1] > rows[0][0]
                             for dest, rows in ranges.items())
                if moving and not flowing.is_set():
                    flowing.set()
                    assert release.wait(15)
            return super().run(progress=tick)

    monkeypatch.setattr(lm, "Download", HeldDownload)
    return flowing, release


def test_cancel_removes_what_the_job_published_but_keeps_what_predates_it(client, monkeypatch, dl_server, tmp_path):
    """A split model cancelled after part 1 landed must not leave half a model staged — but a file that
    was already on disk (a projector shared with another build) is not the job's to delete."""
    from dataclasses import replace

    from hermes_cli.local_runtime.catalog import AssetFile
    from hermes_cli.web_routers import local_models as lm

    entry = replace(lm.catalog.CATALOG[0], mmproj=AssetFile("projector.gguf", 1000), draft=None)
    monkeypatch.setattr(lm, "_download_target", lambda model_id: (entry, entry.variants[0]))
    shared = tmp_path / "assets" / "projector.gguf"
    shared.parent.mkdir()
    shared.write_bytes(b"belongs to another build")
    part1, part2 = tmp_path / "m-00001-of-00002.gguf", tmp_path / "m-00002-of-00002.gguf"
    RangeHandler.payloads.update({"/p1": b"first part", "/p2": _PAYLOAD})
    plan = [(url(dl_server, "/p1"), part1, 10), (url(dl_server, "/p2"), part2, len(_PAYLOAD)),
            (url(dl_server, "/projector"), shared, 1000)]
    monkeypatch.setattr(lm, "_download_plan", lambda *args: plan)
    flowing, release = _hold_after_first_byte(monkeypatch, part2.name)

    job_id = client.post("/api/local-models/download", json={"model_id": entry.id}).json()["job_id"]
    try:
        assert flowing.wait(5), "part 2 never started"
        assert part1.read_bytes() == b"first part", "precondition: part 1 was published before the cancel"
        assert client.post("/api/local-models/download/cancel", json={"job_id": job_id}).json()["cancelled"]
    finally:
        release.set()

    job = _poll(client, job_id, until=("cancelled", "done", "error"))
    assert job["status"] == "cancelled", job
    assert not part1.exists() and not part2.exists()
    assert [p for p in _partials_for(url(dl_server, "/p2")) if p.exists()] == []
    assert shared.read_bytes() == b"belongs to another build"


def test_cancelling_a_paused_quickstart_frees_setup_for_another_run(client, monkeypatch, dl_server, tmp_path):
    """A paused quickstart holds the one-at-a-time setup lock; cancelling it must hand that lock back,
    or the user can never start a different setup."""
    from hermes_cli.local_runtime.binaries import Engine
    from hermes_cli.local_runtime.catalog import VariantChoice
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **kw: None)
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda *a, **k: Engine("cpu", "b10362", tmp_path))
    monkeypatch.setattr(lm.catalog, "select_variant",
                        lambda entry, budget: VariantChoice(variant=entry.variants[0], zero_spill=True,
                                                            reason_key="best-fits"))
    monkeypatch.setattr(lm, "_engine_too_old", lambda min_engine: False)
    RangeHandler.payloads["/weights"] = _PAYLOAD
    weights = tmp_path / "weights.gguf"
    monkeypatch.setattr(lm, "_download_plan", lambda *args: [(url(dl_server, "/weights"), weights, len(_PAYLOAD))])
    flowing, release = _hold_after_first_byte(monkeypatch, weights.name)

    response = client.post("/api/local-models/quickstart", json={"model_id": lm.catalog.CATALOG[0].id})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        assert flowing.wait(5), "quickstart never reached its download"
        assert client.post("/api/local-models/download/pause", json={"job_id": job_id}).json()["paused"]
    finally:
        release.set()
    assert _poll(client, job_id, until=("paused", "done", "error"))["status"] == "paused"
    assert not lm._QUICKSTART_LOCK.acquire(blocking=False), "precondition: a parked quickstart holds setup"

    assert client.post("/api/local-models/download/cancel", json={"job_id": job_id}).json()["cancelled"]
    assert _poll(client, job_id, until=("cancelled",))["status"] == "cancelled"
    assert lm._QUICKSTART_LOCK.acquire(blocking=False), "cancelled quickstart kept the setup lock"
    lm._QUICKSTART_LOCK.release()
    assert not weights.exists()
