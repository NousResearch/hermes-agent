"""Opt-in live acceptance: download the actual pinned CUDA engine and run it."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pm
from hermes_cli.local_runtime import binaries
from hermes_cli.web_routers import local_models as lm
from pm import paths
from pm.lock import Facts, Lockfile


@pytest.mark.platforms("windows", arch="arm64")
def test_live_pinned_cuda_download_pause_resume_and_execute(tmp_path, monkeypatch, record_property):
    home = tmp_path / "home"
    store = home / "tools"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    app = FastAPI()
    app.include_router(lm.router)
    pinned = Lockfile(paths.lockfile_path()).artifacts("llamacpp-cuda", pm.current_target())
    flowing = threading.Event()
    release = threading.Event()
    snapshots = []
    stage_hook = lm._runtime_progress_hook
    byte_hook = lm._download_progress_hook

    def observed_progress(job):
        original = byte_hook(job)

        def tick(done, total, ranges):
            original(done, total, ranges)
            snapshots.append((done, total))
        return tick

    def observed_stage(job):
        original = stage_hook(job)
        first_archive_bytes = 0

        def tick(stage, done, total, label):
            nonlocal first_archive_bytes
            original(stage, done, total, label)
            if stage == "download" and label == "1/2":
                first_archive_bytes = done
            if stage == "download" and label == "2/2" and first_archive_bytes < done < total and not flowing.is_set():
                flowing.set()
                assert release.wait(30)
        return tick

    monkeypatch.setattr(lm, "_runtime_progress_hook", observed_stage)
    monkeypatch.setattr(lm, "_download_progress_hook", observed_progress)

    def wait_job(client, job_id, states, timeout=240):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = client.get(f"/api/local-models/jobs/{job_id}").json()
            if row["status"] in states:
                return row
            time.sleep(0.05)
        raise AssertionError(row)

    with TestClient(app) as client:
        response = client.post("/api/local-models/runtime/install", json={"backend": "cuda"})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        try:
            assert flowing.wait(240), client.get(f"/api/local-models/jobs/{job_id}").text
            assert client.post("/api/local-models/download/pause", json={"job_id": job_id}).json()["paused"]
        finally:
            release.set()
        paused = wait_job(client, job_id, {"paused", "done", "error"})
        assert paused["status"] == "paused", paused
        assert binaries.installed_engine("cuda") is None
        assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
        completed = wait_job(client, job_id, {"done", "error"})
        assert completed["status"] == "done", completed

    engine = binaries.installed_engine("cuda", allow_outdated=False)
    assert engine is not None
    fact = Facts(paths.facts_path()).get("llamacpp-cuda")
    assert fact["artifacts"] == [asset["sha256"] for asset in pinned]
    assert [done for done, _ in snapshots] == sorted(done for done, _ in snapshots)
    assert completed["done_bytes"] == completed["total_bytes"] > 0
    receipt = {"target": pm.current_target(), "tag": engine.tag,
               "download_bytes": completed["done_bytes"], "pause_bytes": paused["done_bytes"],
               "artifacts": pinned, "binary": str(engine.binary), "commands": {}}
    for argument in ("--version", "--list-devices"):
        run = subprocess.run([str(engine.binary), argument], cwd=engine.binary.parent,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        receipt["commands"][argument] = run.stdout + run.stderr
    dlls = sorted(engine.binary.parent.glob("*.dll"))
    assert dlls
    receipt["dlls"] = [dll.name for dll in dlls]
    evidence = tmp_path / "acceptance.json"
    evidence.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    record_property("acceptance", json.dumps(receipt))
    print(json.dumps(receipt, indent=2))
