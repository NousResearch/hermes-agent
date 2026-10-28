"""Staging a catalog model does not make an old managed engine support it."""

from pathlib import Path

import pytest

from hermes_cli import models, models_local, models_validate
from hermes_cli.local_runtime import binaries, bootstrap, catalog, growth


@pytest.mark.parametrize("base_url,engine_tag,accepted", [
    (None, "b10000", False),
    ("http://127.0.0.1:18434/v1", "b10000", False),
    (None, "b20000", True),
    ("http://127.0.0.1:9999/v1", "b10000", True),
])
def test_staged_model_engine_gate_does_not_reject_external_server(monkeypatch, base_url, engine_tag, accepted):
    entry = catalog.CatalogEntry(
        id="day0", display_name="Day 0", description="fixture", repo="test/day0",
        variants=(catalog.QuantVariant("Q4_K_M", (catalog.AssetFile("Day0-Q4_K_M.gguf", 1024),)),),
        n_ctx_train=65536, full_layers=1, recurrent_layers=0, per_layer_f16=64,
        min_engine="b20000",
    )
    model_id = entry.variants[0].model_id
    monkeypatch.setattr(catalog, "CATALOG", (entry,))
    monkeypatch.setattr(bootstrap, "staged_model_ids", lambda: [model_id])
    monkeypatch.setattr(binaries, "installed_engine", lambda *a, **kw: binaries.Engine("cpu", engine_tag, Path("unused")))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: url == "http://127.0.0.1:18434/v1")
    monkeypatch.setattr(models_local, "should_use_ollama_native_catalog", lambda *a, **kw: False)
    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **kw: [model_id.lower()])

    verdict = models_validate.validate_requested_model(model_id.lower(), "llamacpp", base_url=base_url)

    assert verdict["accepted"] is accepted
    assert verdict["persist"] is accepted
    if not accepted:
        assert entry.min_engine in verdict["message"]
