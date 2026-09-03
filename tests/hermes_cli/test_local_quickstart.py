"""Quickstart route: one POST from nothing to a working local default.

Contract, not implementation: the route must (a) preflight-fail
synchronously when nothing fits, (b) report which legs the job will run
(runtime install / model download), skipping legs already satisfied,
and (c) run install -> download -> activate through the same code paths
the individual routes use. The slow legs are stubbed at their module
boundaries; the sequencing and job bookkeeping are real.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def _wait_job(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/local-models/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def test_quickstart_unknown_model_404s(client):
    r = client.post("/api/local-models/quickstart", json={"model_id": "no-such"})
    assert r.status_code == 404


def test_quickstart_refuses_when_nothing_fits(client, monkeypatch):
    """Preflight is synchronous: a machine no catalog entry fits gets a 409
    with guidance, not a doomed background job."""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.catalog.select_variant", lambda *a, **k: None)
    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 409
    assert "Local Models" in r.json()["detail"]


def test_quickstart_runs_all_three_legs(client, monkeypatch, tmp_path):
    """Fresh machine: install runtime -> download recommended -> activate.
    Each leg is asserted by its observable call, in order."""
    calls: list[str] = []

    # Leg 1: no runtime installed yet; install is the stubbed binaries call.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.installed_tags", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda tag, backend, progress=None: calls.append("install"))

    # Leg 2: nothing staged; the download writes the files the plan names.
    def _fake_download(url, dest, job, *, base_done=0, keep_totals=False):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"GGUF\x00")
        calls.append("download")

    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models.download_file", _fake_download)

    # Leg 3: activation — stub the server start and the model assignment.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda config, force=False: calls.append("server") or None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:1/v1", "api_key": "k"})
    from hermes_cli import web_deps

    monkeypatch.setattr(
        web_deps, "late",
        lambda name: (lambda *a, **k: calls.append("assign")))

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["needs_runtime"] is True
    assert body["needs_download"] is True
    assert body["download_bytes"] > 0

    job = _wait_job(client, body["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "quickstart"
    # Order is the contract: engine, weights, server, default.
    assert calls[0] == "install"
    assert "download" in calls
    assert calls.index("install") < calls.index("download") < calls.index("assign")

    # Durable effect: the runtime is enabled in config.
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["enabled"] is True


def test_quickstart_skips_satisfied_legs(client, monkeypatch):
    """Runtime present and model already staged: the response says so and
    the job goes straight to activation."""
    calls: list[str] = []

    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.installed_tags", lambda: ["b10362"])
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda tag, backend, progress=None: calls.append("install"))

    # Every catalog variant reads as staged.
    from hermes_cli.local_runtime.catalog import CATALOG

    all_ids = {v.model_id for e in CATALOG for v in e.variants}
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.staged_model_ids", lambda: all_ids)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models.download_file",
        lambda *a, **k: calls.append("download"))
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda config, force=False: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:1/v1", "api_key": "k"})
    from hermes_cli import web_deps

    monkeypatch.setattr(
        web_deps, "late",
        lambda name: (lambda *a, **k: calls.append("assign")))

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["needs_runtime"] is False
    assert body["needs_download"] is False
    assert body["download_bytes"] == 0

    job = _wait_job(client, body["job_id"])
    assert job["status"] == "done", job["error"]
    assert "install" not in calls and "download" not in calls
    assert calls == ["assign"] or calls[-1] == "assign"


@pytest.fixture
def quickstart_ready(monkeypatch):
    """Preflight passes without hardware or network: the runtime reads as
    installed and every entry's first variant is servable, so the POST
    reaches the single-flight lock instead of 409ing at fit/engine
    preflight on machines where nothing fits."""
    from hermes_cli.local_runtime.catalog import VariantChoice

    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.installed_tags", lambda: ["b10362"])
    monkeypatch.setattr(
        "hermes_cli.local_runtime.catalog.select_variant",
        lambda entry, budget: VariantChoice(variant=entry.variants[0],
                                            zero_spill=True,
                                            reason_key="best-fits"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._engine_too_old",
        lambda min_engine: False)


def test_quickstart_is_single_flight(client, quickstart_ready, monkeypatch):
    """A second quickstart while one runs must 409, not start a twin job
    (the job sequences installs, downloads, a server bounce, and a config
    write — two interleaved runs corrupt all four)."""
    import hermes_cli.web_routers.local_models as lm

    lm._QUICKSTART_LOCK.acquire()
    try:
        r = client.post("/api/local-models/quickstart", json={})
        assert r.status_code == 409
        assert "already running" in r.json()["detail"].lower()
    finally:
        lm._QUICKSTART_LOCK.release()
