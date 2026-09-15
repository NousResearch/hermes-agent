"""Quickstart route: one POST from nothing to a working local default.

Contract, not implementation: the route must (a) preflight-fail
synchronously when automatic setup has no recommendation, (b) report which legs the job will run
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
from hermes_cli.local_runtime.binaries import Engine


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


def test_quickstart_without_recommendation_requires_explicit_choice(client, monkeypatch):
    """One budget: automatic setup refuses; an explicit spilled choice reaches activation."""
    from hermes_cli.local_runtime.estimator import HardwareBudget
    import hermes_cli.web_routers.local_models as lm

    gib = 1 << 30
    budget = HardwareBudget(
        usable_vram_bytes=14 * gib, total_device_bytes=16 * gib,
        ram_available_bytes=64 * gib, uma=False,
    )
    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **kw: budget)
    monkeypatch.setattr(lm.catalog, "refresh_catalog_soon", lambda: None)
    monkeypatch.setattr(lm.binaries, "installed_tags", lambda: [lm.binaries.default_tag()])
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: set())
    config = lm.config_mod.load_config()
    config.setdefault("local_runtime", {})["backend"] = "cpu"
    config["local_runtime"]["enabled"] = False
    lm.config_mod.save_config(config)

    calls: list[tuple] = []

    def download(job, plan, label):
        calls.append(("download", label))

    class Server:
        def models(self):
            return [chosen["model_id"]]

    def start_server(config, force=False):
        calls.append(("server", config["local_runtime"]["enabled"], force))
        return Server()

    # Stub only slow external legs. Catalog, HTTP preflight, job sequencing,
    # runtime-enabled persistence and assignment dispatch remain real.
    monkeypatch.setattr(lm, "_run_download_plan", download)
    monkeypatch.setattr(lm.bootstrap, "ensure_local_runtime", start_server)
    monkeypatch.setattr(
        "hermes_cli.web_server_config._apply_model_assignment_sync",
        lambda *args: calls.append(("assign", *args)),
    )
    rows = client.get("/api/local-models/catalog").json()["models"]
    assert not any(row["recommended"] for row in rows)
    chosen = next(row for row in rows if row["id"] == "qwen3.8-27b")
    assert chosen["fits"] and chosen["spilled"]

    automatic = client.post("/api/local-models/quickstart", json={})
    assert automatic.status_code == 409
    assert "no automatic recommendation" in automatic.json()["detail"].lower()
    assert calls == []
    assert lm.config_mod.load_config()["local_runtime"]["enabled"] is False

    explicit = client.post("/api/local-models/quickstart", json={"model_id": chosen["id"]})
    assert explicit.status_code == 200, explicit.text
    result = explicit.json()
    assert result["model_id"] == chosen["id"]
    assert result["needs_download"] and not result["needs_runtime"]
    job = _wait_job(client, result["job_id"])
    assert job["status"] == "done", job["error"]
    assert calls == [
        ("download", chosen["display_name"]),
        ("server", True, True),
        ("assign", "main", "llamacpp", chosen["model_id"], "", "", ""),
    ]
    assert lm.config_mod.load_config()["local_runtime"]["enabled"] is True


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

    # Supply the same supported backend to preflight and the stubbed install;
    # host auto-detection may select CUDA without a published Linux archive.
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config.setdefault("local_runtime", {})["backend"] = "cpu"
    save_config(config)

    # Leg 1: no runtime installed yet; install is the stubbed binaries call.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.installed_engine", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_engine",
        lambda backend, **kwargs: calls.append("install"))

    # Leg 2: nothing staged; the download writes the files the plan names.
    def _fake_download(job, plan):
        for _url, dest, _size in plan:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"GGUF\x00")
        calls.append("download")

    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._download_job", _fake_download)

    # Leg 3: activation — stub the server start and the model assignment.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda config, force=False: calls.append("server") or None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:1/v1", "api_key": "k"})
    monkeypatch.setattr(
        "hermes_cli.web_server_config._apply_model_assignment_sync",
        lambda *a, **k: calls.append("assign"))

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
        "hermes_cli.local_runtime.binaries.installed_engine", lambda *args, **kwargs: Engine("cpu", "b10362", Path("unused")))
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_engine",
        lambda backend, **kwargs: calls.append("install"))

    # Every model file and companion is already present.
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.web_routers.local_models import _download_plan

    for entry in CATALOG:
        for variant in entry.variants:
            for _, dest, _ in _download_plan(entry, variant):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"downloaded fixture")
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._download_job",
        lambda *a, **k: calls.append("download"))
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda config, force=False: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:1/v1", "api_key": "k"})
    monkeypatch.setattr(
        "hermes_cli.web_server_config._apply_model_assignment_sync",
        lambda *a, **k: calls.append("assign"))

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
        "hermes_cli.local_runtime.binaries.installed_engine", lambda *args, **kwargs: Engine("cpu", "b10362", Path("unused")))
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


def test_assign_default_reaches_model_assignment(monkeypatch):
    """late() must resolve _apply_model_assignment_sync on web_server_config, the
    sibling that defines it. Only the leaf is stubbed; the default web_server lookup
    raised AttributeError at the quickstart's 'making it your default' step."""
    import hermes_cli.web_routers.local_models as lm

    seen: list[tuple] = []
    monkeypatch.setattr(
        "hermes_cli.web_server_config._apply_model_assignment_sync",
        lambda *a, **k: seen.append(a))
    lm._assign_default({}, "some-model")
    assert seen == [("main", "llamacpp", "some-model", "", "", "")]
