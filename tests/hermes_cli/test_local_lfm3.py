"""LFM3 follows the shared Desktop catalog/download/preset path, not a provider fork."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.local_runtime import binaries, catalog, hardware, presets
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.web_routers import local_models


def test_lfm3_desktop_download_resolves_the_advertised_official_build(monkeypatch):
    budget = HardwareBudget(8 << 30, 8 << 30, 8 << 30, uma=True)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: budget)
    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    monkeypatch.setattr(binaries, "installed_engine", lambda *a, **kw: binaries.Engine("cpu", "b20000", Path("unused")))
    monkeypatch.setattr(local_models, "_spawn_job", lambda *a, **kw: None)
    plans = []
    real_plan = local_models._download_plan

    def capture_plan(entry, variant):
        plan = real_plan(entry, variant)
        plans.extend(plan)
        return plan

    monkeypatch.setattr(local_models, "_download_plan", capture_plan)
    app = FastAPI()
    app.include_router(local_models.router)
    client = TestClient(app)

    response = client.post("/api/local-models/download", json={"model_id": "lfm3-1.7b"})

    assert response.status_code == 200
    row = next(row for row in client.get("/api/local-models/catalog").json()["models"]
               if row["id"] == "lfm3-1.7b")
    assert response.json()["model_id"] == row["model_id"]
    assert row["fits"] and row["start_window"] >= 65536
    assert not row["vision"] and not row["mtp"]
    assert len(plans) == 1
    url, dest, nbytes = plans[0]
    entry = catalog.catalog_by_id()["lfm3-1.7b"]
    asset = entry.variants[0].files[0]
    assert url == f"https://huggingface.co/{entry.repo}/resolve/{asset.revision}/{asset.path}"
    assert dest.name == row["model_id"] + ".gguf"
    assert nbytes == row["size_bytes"] == asset.size_bytes


def test_lfm3_presets_refuse_an_installed_engine_without_architecture_support(tmp_path):
    from tests.hermes_cli.test_local_runtime_gguf_shortconv import _write_shortconv

    entry = catalog.catalog_by_id()["lfm3-1.7b"]
    path = tmp_path / entry.variants[0].files[0].path
    _write_shortconv(path)
    budget = HardwareBudget(8 << 30, 8 << 30, 8 << 30, uma=True)

    refused = presets.preset_for_model(path, budget, set(), engine_tag="b11936")

    assert refused is not None and refused.refusal
    assert entry.min_engine in refused.refusal
    assert not refused.keys
    allowed = presets.preset_for_model(path, budget, set(), engine_tag="b20000")
    assert allowed is not None and not allowed.refusal
    assert allowed.window >= 65536
    assert allowed.keys is not None
    assert allowed.keys["repeat-penalty"] == "1.1"
