"""Consent/job boundaries use private homes and real harmless subprocesses."""
import contextvars
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"
load = runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"]
pytestmark = pytest.mark.linux_only


def test_prepare_is_readonly_and_consent_binds_owner_home_kind_and_revision(tmp_path, monkeypatch):
    flow = load("setup_flow")
    service = load("integration").RealmIntegration(tmp_path / "profile")
    service.bind(session_id="owner", runtime_session_id="runtime")
    service.bind(session_id="other", runtime_session_id="other-runtime")
    identity = {"runtime_session_id": "runtime", "stored_session_id": None}
    before = {p: p.read_bytes() for p in service.home.rglob("*") if p.is_file()}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("prepare spawned a process"))
    proposal = flow.prepare(service, "owner", "realm")
    assert proposal["action"] in {"repair", "install", "start"}
    assert proposal["kind"] == "realm" and isinstance(proposal["details"], list)
    assert before == {p: p.read_bytes() for p in service.home.rglob("*") if p.is_file()}
    assert proposal["consent"] != flow.prepare(service, "other", "realm")["consent"]
    other_profile = load("integration").RealmIntegration(tmp_path / "second")
    other_profile.bind(session_id="owner", runtime_session_id="runtime")
    assert proposal["consent"] != flow.prepare(other_profile, "owner", "realm")["consent"]
    with pytest.raises(ValueError, match="consent|proposal"):
        flow.start(service, "owner", "realm", {}, identity)
    with pytest.raises(ValueError, match="consent|proposal"):
        flow.start(service, "other", "realm", proposal["consent"], {"runtime_session_id": "other-runtime"})
    (service.home / "config.yaml").write_text("plugins:\n  realms:\n    size: 1280x720\n")
    with pytest.raises(ValueError, match="consent|proposal"):
        flow.start(service, "owner", "realm", proposal["consent"], identity)


def wait_job(flow, service, owner, job):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = flow.status(service, owner, job["id"])
        if result["state"] != "running":
            return result
        time.sleep(.02)
    pytest.fail("setup job did not finish")


def fixture_plan(service, owner, kind):
    from dataclasses import asdict

    return {"home": str(service.home), "owner": owner, "kind": kind, "revision": "test",
            "config": asdict(load("config").Config.load(service.home)),
            "ready": False, "action": "repair", "summary": "Repair fixture",
            "details": [], "packages": [], "blockers": [], "previous_kind": "realm"}


def test_background_job_real_child_owner_isolation_failure_and_context(tmp_path, monkeypatch):
    flow = load("setup_flow")
    service = load("integration").RealmIntegration(tmp_path / "profile")
    owner = service.bind(session_id="owner", runtime_session_id="runtime")
    service.bind(session_id="other", runtime_session_id="other-runtime")
    service.owners.set_mode(owner, "ask")
    identity = {"runtime_session_id": "runtime"}
    monkeypatch.setattr(flow, "build_plan", fixture_plan)
    witness = tmp_path / "child.json"
    marker = contextvars.ContextVar("setup-test-marker", default="absent")
    marker.set("copied")
    entered, release = threading.Event(), threading.Event()
    parent_env = dict(os.environ)

    def install(plan, phase, **kwargs):
        assert marker.get() == "copied"
        phase("verify")
        flow.run_child([sys.executable, "-c",
                        "import json,os,pathlib; pathlib.Path(os.environ['WITNESS']).write_text(json.dumps({'home':os.environ['HERMES_HOME'],'pid':os.getpid()}))"],
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": plan["home"], "WITNESS": str(witness)}, timeout=5)
        entered.set()
        assert release.wait(10)
        raise RuntimeError("sensitive-token-must-not-escape")

    monkeypatch.setattr(flow, "install_plan", install)
    monkeypatch.setattr(service, "activate_setup", lambda *a, **k: pytest.fail("failed install committed routing"))
    proposal = flow.prepare(service, owner, "realm")
    job = flow.start(service, owner, "realm", proposal["consent"], identity)
    assert entered.wait(10)
    try:
        with pytest.raises(ValueError, match="active"):
            flow.start(service, owner, "realm", proposal["consent"], identity)
        with pytest.raises(PermissionError):
            flow.status(service, "other", job["id"])
    finally:
        release.set()
    result = wait_job(flow, service, owner, job)
    assert result["state"] == "failed"
    assert "sensitive-token" not in json.dumps(result)
    assert service.owners.mode(owner, "realm") == "ask"
    assert os.environ == parent_env
    child = json.loads(witness.read_text())
    assert child["home"] == str(service.home) and child["pid"] != os.getpid()


def test_success_verifies_then_commits_in_backend_not_worker(tmp_path, monkeypatch):
    flow = load("setup_flow")
    service = load("integration").RealmIntegration(tmp_path / "profile")
    owner = service.bind(session_id="owner", runtime_session_id="runtime")
    service.owners.set_mode(owner, "ask")
    monkeypatch.setattr(flow, "build_plan", fixture_plan)
    monkeypatch.setattr(flow, "install_plan", lambda plan, phase, **kwargs: None)
    events = []
    monkeypatch.setattr(flow, "verify_ready", lambda home, kind: events.append("verify"))
    def activate(owner, kind, config, generation, identity):
        assert config == fixture_plan(service, owner, kind)["config"]
        assert generation == service.owners.setup_generation(owner)
        events.append((owner, kind, identity, os.getpid()))
    monkeypatch.setattr(service, "activate_setup", activate)
    proposal = flow.prepare(service, owner, "realm")
    result = wait_job(flow, service, owner, flow.start(service, owner, "realm", proposal["consent"], {"runtime_session_id": "runtime"}))
    assert result["state"] == "succeeded"
    assert events == ["verify", (owner, "realm", {"runtime_session_id": "runtime"}, os.getpid())]
    assert flow.status(service, owner, result["id"]) == result


def test_api_prepare_job_snapshot_and_ownership(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    flow = load("setup_flow")
    integration = load("integration")
    service = integration.RealmIntegration(tmp_path / "profile")
    monkeypatch.setenv("HERMES_HOME", str(service.home))
    service.bind(session_id="owner", runtime_session_id="runtime")
    service.bind(session_id="other", runtime_session_id="other-runtime")
    monkeypatch.setattr(integration, "get_integration", lambda: service)
    monkeypatch.setattr(flow, "build_plan", fixture_plan)
    monkeypatch.setattr(flow, "install_plan", lambda plan, phase, **kwargs: None)
    monkeypatch.setattr(flow, "verify_ready", lambda home, kind: None)
    monkeypatch.setattr(service, "activate_setup", lambda *a, **k: None)
    api = runpy.run_path(str(PLUGIN / "dashboard/plugin_api.py"))
    app = FastAPI()
    app.include_router(api["router"])
    with TestClient(app) as client:
        identity = {"runtime_session_id": "runtime", "kind": "realm"}
        proposal = client.post("/realms/setup/prepare", json=identity)
        assert proposal.status_code == 200
        assert client.post("/realms/setup/start", json=identity).status_code == 422
        response = client.post("/realms/setup/start", json={**identity, "consent": proposal.json()["consent"]})
        assert response.status_code == 200
        job = wait_job(flow, service, "owner", response.json())
        listing = client.get("/realms", params={"runtime_session_id": "runtime"}).json()
        assert listing["setup_job"] == job
        other = client.get("/realms", params={"runtime_session_id": "other-runtime"}).json()
        assert other.get("setup_job") is None
        assert client.get("/realms/setup/jobs/" + job["id"], params={"runtime_session_id": "other-runtime"}).status_code == 403
        assert client.post("/realms/setup/prepare", json={**identity, "stored_session_id": "unknown"}).status_code == 403


def test_package_allowlist_and_vm_preflight_never_invents_keys(tmp_path, monkeypatch):
    plan = load("setup_plan")
    assert plan.package_plan(["labwc", "Xwayland", "bwrap"], "arch") == ["bubblewrap", "labwc", "xorg-xwayland"]
    with pytest.raises(ValueError, match="Unsupported"):
        plan.package_plan(["labwc"], "debian")
    with pytest.raises(ValueError, match="Unsupported"):
        plan.package_plan(["untrusted; command"], "arch")
    monkeypatch.setenv("HOME", str(tmp_path))
    service = load("integration").RealmIntegration(tmp_path / "profile")
    service.bind(session_id="owner")
    proposal = plan.build_plan(service, "owner", "omarchy-vm")
    assert any("SSH" in reason for reason in proposal["blockers"])
    assert not (tmp_path / ".ssh").exists()


def test_base_install_uses_unique_runtime_and_failed_disk_is_not_ready(tmp_path, monkeypatch):
    from contextlib import nullcontext
    vm = load("vm_manager")
    config = load("config")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("ordinary test touched systemd"))
    monkeypatch.setattr(load("setup_process"), "vm_lifetime", lambda *args, **kwargs: nullcontext())
    # Exercise lifecycle with harmless executable fixture, never host systemd/QEMU.
    monkeypatch.setattr(vm, "_base_runtime", lambda generation: tmp_path / ("runtime-" + generation))
    seen = []
    def script(arguments, *, vm_home, unit, runtime, **kwargs):
        seen.append((unit, runtime))
        if arguments[0] == "install":
            Path(vm_home, "disk.qcow2").write_bytes(b"incomplete fixture")
            raise subprocess.CalledProcessError(1, ["fixture"])
    first = vm.VmManager(tmp_path / "first")
    second = vm.VmManager(tmp_path / "second")
    for manager in (first, second):
        monkeypatch.setattr(manager, "_run_script", script)
        monkeypatch.setattr(manager, "_force_stop", lambda unit: None)
        monkeypatch.setattr(manager, "_free_port", lambda taken: 2355)
        with pytest.raises(vm.VmError):
            manager.install_base()
        assert not manager.base_status()["present"]
    assert seen[0][0] != seen[1][0] and seen[0][1] != seen[1][1]
    assert all(not runtime.exists() for _, runtime in seen)


def test_child_timeout_kills_process_group_without_exposing_output(tmp_path):
    flow = load("setup_flow")
    started = time.monotonic()
    with pytest.raises(ValueError, match="timed out"):
        flow.run_child([sys.executable, "-c", "import time; print('secret'); time.sleep(30)"],
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, timeout=.2)
    assert time.monotonic() - started < 10


def test_jobs_reject_symlinks_and_unknown_ids(tmp_path):
    flow = load("setup_flow")
    service = load("integration").RealmIntegration(tmp_path / "profile")
    service.bind(session_id="owner")
    with pytest.raises(ValueError):
        flow.status(service, "owner", "../../config")
    with pytest.raises(FileNotFoundError):
        flow.status(service, "owner", "a" * 32)
    external = tmp_path / "external"
    external.mkdir()
    (service.home / "plugin-data").symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, PermissionError)):
        flow.prepare(service, "owner", "realm")
    assert list(external.iterdir()) == []
