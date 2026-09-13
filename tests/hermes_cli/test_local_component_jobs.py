"""Desktop component jobs pause real PM archive transfers before activation."""

from __future__ import annotations

import hashlib
import threading

import pytest

import pm
from pm import paths, registry
from pm.lock import Facts, Lockfile
from pm.packages import LlamaCppCpu
from tests.hermes_cli.test_local_download_jobs import client, poll  # noqa: F401
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401
from tests.pm.test_install_download_control import archive


@pytest.mark.parametrize("endpoint", ["runtime/install", "quickstart"])
def test_component_job_pause_retains_pin_and_stops_sequence(client, monkeypatch, tmp_path, dl_server, endpoint):
    from hermes_cli.web_routers import local_models as lm

    class ArchiveEngine(LlamaCppCpu):
        probe_version = False

    package = ArchiveEngine()
    monkeypatch.setitem(registry._packages, package.name, package)
    root = tmp_path / "store"
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    monkeypatch.setattr(lm, "_runtime_section", lambda: {"backend": "cpu"})
    target = pm.current_target()
    binary_name = package.binary(root, target).name
    files = {binary_name: b"archive engine fixture", "runtime.dll": bytes(range(256)) * (32 * 1024)}
    pins = []
    for index, (name, body) in enumerate(files.items()):
        payload = archive({name: body})
        path = f"/component-{index}.zip"
        RangeHandler.payloads[path] = payload
        pins.append({"url": url(dl_server, path), "sha256": hashlib.sha256(payload).hexdigest()})
    lock = Lockfile(lock_path)
    lock.set_pin(package.name, "10362", {target: pins})
    lock.save()

    entry = lm.catalog.CATALOG[0]
    variant = entry.variants[0]
    monkeypatch.setattr(lm, "_quickstart_target", lambda *args: (entry, variant))
    model = lm.bootstrap.models_dir() / "test.gguf"
    RangeHandler.payloads["/model"] = b"model fixture"
    monkeypatch.setattr(lm, "_download_plan", lambda *args: [(url(dl_server, "/model"), model, 13)])
    calls = []
    monkeypatch.setattr(lm, "_ensure_server", lambda *args, **kwargs: calls.append("server"))
    monkeypatch.setattr(lm, "_assign_default", lambda *args: calls.append("default"))
    monkeypatch.setattr(lm.bootstrap, "get_supervisor", lambda: None)

    flowing, release = threading.Event(), threading.Event()
    real_hook = lm._runtime_progress_hook

    def observed_hook(job):
        hook = real_hook(job)
        first_archive_bytes = 0

        def tick(stage, done, total, label):
            nonlocal first_archive_bytes
            hook(stage, done, total, label)
            if stage == "download" and label == "1/2":
                first_archive_bytes = done
            if stage == "download" and label == "2/2" and first_archive_bytes < done < total and not flowing.is_set():
                flowing.set()
                assert release.wait(15)
        return tick

    monkeypatch.setattr(lm, "_runtime_progress_hook", observed_hook)
    response = client.post(f"/api/local-models/{endpoint}", json={"backend": "cpu"} if endpoint == "runtime/install" else {})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    try:
        assert flowing.wait(5), "runtime did not enter PM's pinned archive transfer"
        paused = client.post("/api/local-models/download/pause", json={"job_id": job_id})
        assert paused.json()["paused"], paused.text
    finally:
        release.set()
    job = poll(client, job_id)
    assert job["status"] == "paused", job
    assert Facts(paths.facts_path()).get(package.name) is None
    assert calls == [] and not model.exists()
    assert job["can_resume"]
    assert list(paths.partials_root().glob("*.ranges"))

    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
    job = poll(client, job_id, until=("done", "error"))
    assert job["status"] == "done", job
    selected = pm.installed_package(package.name)
    assert selected is not None
    for name, body in files.items():
        assert (selected.path / name).read_bytes() == body
    assert Facts(paths.facts_path()).get(package.name)["artifacts"] == [p["sha256"] for p in pins]
    if endpoint == "quickstart":
        assert calls == ["server", "default"] and model.read_bytes() == RangeHandler.payloads["/model"]
    assert job_id not in lm._RUNNING
