"""Engine recovery crosses the actual archive installer and isolated PM worker."""
from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest

from pm import paths
from tests.hermes_cli.test_local_download_jobs import client as client, poll  # noqa: PLC0414 -- pytest fixture re-export
from tests.hermes_cli.test_local_quickstart_recovery import _routes
from tests.pm._fixtures import isolated_python as isolated_python, worker_toolchain  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._range_server import RangeHandler, dl_server as dl_server, url  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm.test_install_download_control import archive
from tests.pm.test_worker_registry import _load_file


@pytest.fixture
def engine_download(client, tmp_path, monkeypatch, isolated_python, dl_server):
    import hashlib
    import pm
    from hermes_cli.web_routers import local_models as lm

    installed = lm.binaries.installed_engine
    registry = importlib.import_module("pm.registry")
    monkeypatch.setattr(registry, "_packages", dict(registry._packages))
    lm, entries, plans, calls = _routes(monkeypatch, dl_server)
    monkeypatch.setattr(lm.binaries, "installed_engine", installed)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    source = tmp_path / "fixture_engine.py"
    source.write_text(
        "import os\nfrom pm import Package, register\n@register\nclass EnginePackage(Package):\n"
        "    name = 'recovery-fixture-engine'\n"
        "    def binary(self, entry, target): return entry / 'engine.dat'\n"
        "    def stage(self, store, staged, version, target):\n"
        "        (staged / 'worker.pid').write_text(str(os.getpid()))\n"
    )
    package = _load_file(source, monkeypatch, name="recovery_fixture_engine").EnginePackage()
    monkeypatch.setitem(registry._packages, package.name, package)
    monkeypatch.setitem(lm.binaries.BACKEND_PACKAGES, "cpu", package.name)
    content = b"engine fixture" * (1 << 19)
    payload = archive({"engine.dat": content})
    primary, mirror = (url(dl_server, path) for path in ("/engine.zip", "/mirror.zip"))
    RangeHandler.payloads.update({"/engine.zip": payload, "/mirror.zip": payload})
    lock = pm.Lockfile(paths.lockfile_path())
    lock.set_pin(package.name, "1", {"any": {"url": primary, "sha256": hashlib.sha256(payload).hexdigest()}})
    lock.save()
    manager = importlib.import_module("pm.client")
    monkeypatch.setattr(manager, "is_runtime", lambda: False)
    worker_toolchain(manager, monkeypatch, isolated_python,
        "from pm.downloader import Download, Source\nimport pm.artifact_mirror as mirror\n"
        f"mirror.pinned_source = lambda url, dest, digest: Source(url, dest, digest, ({mirror!r},))\n"
        "Download._wait_retry = lambda self, delay: None\n"
        "original = Download.__init__\n"
        "def serial(self, *args, **kwargs):\n    kwargs['connections'] = 1\n    original(self, *args, **kwargs)\n"
        "Download.__init__ = serial\n")
    monkeypatch.setattr(importlib.import_module("pm.install"), "ensure",
                        lambda *a, **kw: pytest.fail("installer ran in the app process"))
    state = {"outage": True, "dropped": False, "status": 503, "requests": []}
    original = RangeHandler.do_GET

    def respond(handler):
        if handler.path not in ("/engine.zip", "/mirror.zip") or not state["outage"]:
            return original(handler)
        bounds = handler.headers.get("Range")
        state["requests"].append((handler.path, bounds))
        if state["status"] == 503 and handler.path == "/engine.zip" and not state["dropped"]:
            if bounds == "bytes=0-0":
                return original(handler)
            state["dropped"] = True
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes 0-{len(payload)-1}/{len(payload)}")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload[:1 << 20])
            handler.wfile.flush()
            handler.close_connection = True
        else:
            handler.send_response(state["status"])
            handler.send_header("Content-Length", "0")
            handler.end_headers()

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    return SimpleNamespace(lm=lm, entries=entries, plans=plans, calls=calls, state=state, content=content)


def test_worker_preserves_install_error_and_transport_cause(engine_download):
    import pm
    from pm.downloader import DownloadTransportError

    with pytest.raises(pm.InstallError) as caught:
        pm.ensure("recovery-fixture-engine", explicit=True)
    assert caught.value.package == "recovery-fixture-engine"
    assert caught.value.remedy == "retry, or run `hermes pm doctor`"
    download = DownloadTransportError.find(caught.value)
    assert download is not None and download.recoverable
    assert download.status == 503
    assert [url.rsplit("/", 1)[-1] for url in download.urls] == ["engine.zip", "mirror.zip"]


@pytest.mark.parametrize("model_cached", [False, True])
@pytest.mark.parametrize("action", ["resume", "discard", "pause-discard", "permanent"])
def test_quickstart_recovers_real_engine_install(client, engine_download, monkeypatch, model_cached, action):
    fixture = engine_download
    lm, first, second = fixture.lm, *fixture.entries
    destination = fixture.plans[first.id][0][1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model_cached:
        destination.write_bytes(b"first")
    completed = destination.parent / "keep.gguf"
    completed.write_bytes(b"completed model")
    before = completed.stat().st_mtime_ns
    if action == "pause-discard":
        real_hook = lm._download_progress_hook

        def pause_after_progress(job):
            tick = real_hook(job)

            def observe(done, total, ranges):
                tick(done, total, ranges)
                if done:
                    lm._RUNNING[job["job_id"]]["pause"].set()
            return observe

        monkeypatch.setattr(lm, "_download_progress_hook", pause_after_progress)
    if action == "permanent":
        fixture.state["status"] = 403
    response = client.post("/api/local-models/quickstart", json={"model_id": first.id})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    parked = poll(client, job_id)
    if action == "pause-discard":
        assert parked["error"] is None
    else:
        assert "recovery-fixture-engine" in parked["error"]  # Package context survives the worker too.
    if action == "permanent":
        assert parked["status"] == "error" and not parked["can_resume"]
        assert len(fixture.state["requests"]) == 2  # Primary and mirror, with no retries.
        assert not lm._QUICKSTART_LOCK.locked()
        return
    assert parked["status"] == "paused" and parked["can_resume"] and parked["can_discard"], parked
    assert parked["done_bytes"] == 1 << 20
    assert len(fixture.state["requests"]) <= 12  # Both hosts exhaust a bounded retry budget.
    old_resume = lm._RUNNING[job_id]["resume"]
    fixture.state["outage"] = False
    if action.endswith("discard"):
        assert client.post("/api/local-models/download/discard", json={"job_id": job_id}).json()["discarded"]
        assert not list(paths.partials_root().glob("*.part"))
        assert not list(paths.partials_root().glob("*.ranges"))
        assert not old_resume()
        assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).status_code == 404
        if action == "pause-discard":
            monkeypatch.setattr(lm, "_download_progress_hook", real_hook)
        job_id = client.post("/api/local-models/quickstart", json={"model_id": second.id}).json()["job_id"]
    else:
        assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
    finished = poll(client, job_id, until=("done", "error"))
    assert finished["status"] == "done", finished
    engine = lm.binaries.installed_engine("cpu", allow_outdated=False)
    assert engine.binary.read_bytes() == fixture.content
    assert int((engine.binary.parent / "worker.pid").read_text()) != os.getpid()
    assert completed.read_bytes() == b"completed model" and completed.stat().st_mtime_ns == before
    assert not lm._QUICKSTART_LOCK.locked()
    if action == "resume":
        resumed = [start for path, start, _ in RangeHandler.ranges_seen if path == "/engine.zip"]
        assert resumed and all(start >= 1 << 20 for start in resumed)
