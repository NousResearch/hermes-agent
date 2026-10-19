"""A saved setup must retain its engine, model and configuration continuation."""
from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client, poll  # noqa: PLC0414
from tests.hermes_cli.test_local_engine_download_recovery import engine_download as engine_download  # noqa: PLC0414
from tests.pm._fixtures import isolated_python as isolated_python  # noqa: PLC0414
from tests.pm._range_server import RangeHandler, dl_server as dl_server  # noqa: PLC0414


@pytest.mark.parametrize("stage", ["engine", "model", "finish"])
@pytest.mark.parametrize("action", ["resume", "discard"])
def test_restart_retains_the_complete_setup(client, engine_download, monkeypatch, stage, action):
    fixture = engine_download
    lm = fixture.lm
    entry = fixture.entries[0]
    destination = fixture.plans[entry.id][0][1]
    if stage != "model":
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"first")
    fixture.state["outage"] = stage == "engine"
    original_get = RangeHandler.do_GET
    model_outage = {"active": stage == "model"}

    def respond(handler):
        if handler.path == "/0" and model_outage["active"]:
            handler.send_response(503)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        return original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    original_server = lm._ensure_server
    if stage == "finish":
        monkeypatch.setattr(lm, "_ensure_server", lambda *a, **kw: (_ for _ in ()).throw(lm.DownloadPaused("restart point")))
    response = client.post("/api/local-models/quickstart", json={"model_id": entry.id})
    assert response.status_code == 200, response.text
    original_id = response.json()["job_id"]
    parked = poll(client, original_id)
    assert parked["status"] == "paused", parked
    assert parked["setup_stage"] == stage

    # Lose every process-local callback and lock. Only disk state can restore setup.
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(restored) == 1, restored
    job = restored[0]
    assert job["job_id"] != original_id and job["kind"] == "quickstart"
    assert job["can_resume"] and job["can_discard"]
    assert job["setup_stage"] == stage
    if stage == "engine":
        assert job["done_bytes"] == 1 << 20
    if action == "discard":
        stale_resume = lm._RUNNING[job["job_id"]]["resume"]
        assert client.post("/api/local-models/download/discard", json={"job_id": job["job_id"]}).json()["discarded"]
        from pm.paths import partials_root

        assert not list(partials_root().glob("*.part"))
        assert not list(partials_root().glob("*.ranges"))
        assert not list(lm.download_plans._root().glob("*.json"))
        assert not lm._QUICKSTART_LOCK.locked() and not stale_resume()
        assert destination.exists() is (stage != "model")
        return
    fixture.state["outage"] = False
    model_outage["active"] = False
    if stage == "model":
        alternative = replace(entry.variants[0], files=(replace(entry.variants[0].files[0], path="different-build.gguf"),))
        changed = replace(entry, variants=(*entry.variants, alternative))
        monkeypatch.setattr(lm, "_quickstart_target", lambda body, budget: (changed, alternative))
        monkeypatch.setattr(lm, "_download_plan", lambda *args: pytest.fail("saved sources were recalculated"))
    monkeypatch.setattr(lm, "_ensure_server", original_server)
    assert client.post("/api/local-models/download/resume", json={"job_id": job["job_id"]}).json()["resumed"]
    finished = poll(client, job["job_id"], until=("done", "error"))
    assert finished["status"] == "done", finished
    assert destination.read_bytes() == b"first"
    assert fixture.calls[-2:] == [("server", entry.variants[0].model_id), ("default", entry.variants[0].model_id)]
    assert not lm._QUICKSTART_LOCK.locked()
    assert not list(lm.download_plans._root().glob("*.json"))


@pytest.mark.parametrize("launch_profile", [False, True])
def test_restored_setup_keeps_the_profile_that_requested_it(client, engine_download, monkeypatch, tmp_path, launch_profile):
    from hermes_cli.profiles import _get_profiles_root
    from hermes_constants import get_hermes_home

    fixture = engine_download
    lm = fixture.lm
    profile = _get_profiles_root() / "recovery-owner"
    assert profile.is_relative_to(tmp_path)
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: {}\n")
    if launch_profile:
        monkeypatch.setenv("HERMES_HOME", str(profile))
    entry = fixture.entries[0]
    fixture.state["outage"] = False
    destination = fixture.plans[entry.id][0][1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"first")
    owners = []
    gate = {"paused": True}

    def server(*args, **kwargs):
        if gate["paused"]:
            raise lm.DownloadPaused("restart point")
        owners.append(get_hermes_home())

    monkeypatch.setattr(lm, "_ensure_server", server)
    job_id = client.post("/api/local-models/quickstart", params={"profile": "current" if launch_profile else "recovery-owner"},
                         json={"model_id": entry.id}).json()["job_id"]
    assert poll(client, job_id)["status"] == "paused"
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    if launch_profile:
        monkeypatch.setenv("HERMES_HOME", str(profile.parent.parent))
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"][0]
    gate["paused"] = False
    client.post("/api/local-models/download/resume", json={"job_id": restored["job_id"]})
    finished = poll(client, restored["job_id"], until=("done", "error"))
    assert finished["status"] == "done", finished
    assert owners == [profile]


@pytest.mark.parametrize("stage", ["engine", "model"])
def test_automatic_setup_reprices_after_engine_install_and_keeps_the_choice_after_restart(
        client, engine_download, monkeypatch, stage):
    fixture = engine_download
    lm = fixture.lm
    before, after = fixture.entries

    def choose(body, budget):
        selected = (lm.catalog.catalog_by_id()[body.model_id] if body.model_id
                    else after if lm.binaries.installed_engine("cpu") else before)
        return selected, selected.variants[0]

    monkeypatch.setattr(lm, "_quickstart_target", choose)
    fixture.state["outage"] = stage == "engine"
    model_outage = {"active": stage == "model"}
    original_get = RangeHandler.do_GET

    def respond(handler):
        if handler.path in {"/0", "/1"} and model_outage["active"]:
            handler.send_response(503)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        return original_get(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", respond)
    response = client.post("/api/local-models/quickstart", json={})
    assert response.status_code == 200, response.text
    parked = poll(client, response.json()["job_id"])
    assert parked["status"] == "paused", parked
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", threading.Lock())
    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(restored) == 1, restored
    fixture.state["outage"] = model_outage["active"] = False
    if stage == "model":
        monkeypatch.setattr(lm, "_quickstart_target", lambda *args: pytest.fail("the chosen model was recalculated"))
    job_id = restored[0]["job_id"]
    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
    finished = poll(client, job_id, until=("done", "error"))
    assert finished["status"] == "done", finished
    assert (finished["model_id"], finished["target"]) == (after.id, after.display_name)
    assert fixture.plans[after.id][0][1].read_bytes() == b"other"
    assert not fixture.plans[before.id][0][1].exists()
    assert fixture.calls == [("server", after.variants[0].model_id), ("default", after.variants[0].model_id)]
    assert not list(lm.download_plans._root().glob("*.json"))


@pytest.mark.parametrize("action", ["resume", "discard"])
def test_restart_retains_a_standalone_engine_install(client, engine_download, monkeypatch, action):
    fixture = engine_download
    lm = fixture.lm
    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]
    parked = poll(client, job_id)
    assert parked["status"] == "paused" and parked["done_bytes"] == 1 << 20
    monkeypatch.setattr(lm, "_JOBS", {})
    monkeypatch.setattr(lm, "_RUNNING", {})
    lm._restore_download_plans()
    jobs = client.get("/api/local-models/jobs").json()["jobs"]
    assert len(jobs) == 1
    restored = jobs[0]
    assert restored["kind"] == "runtime-install" and restored["can_resume"] and restored["can_discard"]
    fixture.state["outage"] = False
    assert client.post(f"/api/local-models/download/{action}", json={"job_id": restored["job_id"]}).status_code == 200
    if action == "resume":
        finished = poll(client, restored["job_id"], until=("done", "error"))
        assert finished["status"] == "done", finished
        assert lm.binaries.installed_engine("cpu").binary.read_bytes() == fixture.content
    else:
        from pm.paths import partials_root

        assert not list(partials_root().glob("*.part"))
    assert fixture.calls == []
    assert not list(lm.download_plans._root().glob("*.json"))
