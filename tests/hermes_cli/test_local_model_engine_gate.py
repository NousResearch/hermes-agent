"""A day-zero model's exact variant must obey the same engine gate as its family."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.local_runtime import binaries, catalog
from hermes_cli.web_routers import local_models


@pytest.mark.parametrize("action, exact_variant", [
    ("download", False), ("download", True), ("activate", True), ("quickstart", False),
])
def test_model_actions_require_compatible_engine_for_family_and_variant(monkeypatch, action, exact_variant):
    entry = catalog.CatalogEntry(
        id="day0", display_name="Day 0", description="test fixture", repo="test/day0",
        variants=(catalog.QuantVariant("Q4_K_M", (catalog.AssetFile("Day0-Q4_K_M.gguf", 1024),)),),
        n_ctx_train=65536, full_layers=1, recurrent_layers=0, per_layer_f16=64,
        min_engine="b20000",
    )
    monkeypatch.setattr(catalog, "CATALOG", (entry,))
    monkeypatch.setattr(binaries, "installed_engine", lambda *a, **kw: binaries.Engine("cpu", "b10000", Path("unused")))
    monkeypatch.setattr(local_models, "_spawn_job", lambda *a, **kw: None)
    app = FastAPI()
    app.include_router(local_models.router)
    model_id = entry.variants[0].model_id if exact_variant else entry.id

    monkeypatch.setattr(local_models.bootstrap, "staged_model_ids", lambda: [entry.variants[0].model_id])
    response = TestClient(app).post(f"/api/local-models/{action}", json={"model_id": model_id})

    assert response.status_code == 409
    assert "b20000" in response.json()["detail"]
