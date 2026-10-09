"""Shipped model-catalog manifest contracts after catalogue ownership moved lower."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from models.catalog_static import PREFERRED_SILENT_DEFAULT_MODEL


def test_shipped_manifest_labels_the_silent_default():
    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (repo_root / "website" / "static" / "api" / "model-catalog.json").read_text(
            encoding="utf-8"
        )
    )
    for provider in ("openrouter", "nous"):
        block = manifest["providers"][provider]
        labeled = [m["id"] for m in block["models"] if m.get("default")]
        assert labeled == [PREFERRED_SILENT_DEFAULT_MODEL]


def test_in_repo_lists_match_manifest():
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "website" / "static" / "api" / "model-catalog.json"
    if not manifest_path.exists():
        pytest.skip(f"manifest missing at {manifest_path}")

    script_path = repo_root / "scripts" / "build_model_catalog.py"
    spec = importlib.util.spec_from_file_location("_build_model_catalog", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    expected = mod.build_catalog()
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected.pop("updated_at", None)
    actual.pop("updated_at", None)
    assert actual == expected, (
        "website/static/api/model-catalog.json is out of sync with the in-repo "
        "catalogue; run scripts/build_model_catalog.py and commit the result"
    )
