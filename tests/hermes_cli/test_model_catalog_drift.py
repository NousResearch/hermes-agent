"""The curation drift report: what the curated lists and the live catalogs disagree about.

The report is a maintainer tool (``scripts/check_model_catalog_drift.py``). It reads the
curated OpenRouter list and the live ``/api/v1/models`` catalog and names both
directions of the disagreement:

* a curated id the live catalog does not publish — retired, or simply unpublished
  (OpenRouter does not publish every curated id, the stealth SKUs among them);
* a live, free, tool-capable model the curated list does not carry — invisible in
  the picker, which is the "the free ones never show up" symptom.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_drift_script():
    """The report lives in ``scripts/`` (not an importable package); load it by path."""
    path = REPO_ROOT / "scripts" / "check_model_catalog_drift.py"
    spec = importlib.util.spec_from_file_location("check_model_catalog_drift", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _live_entry(mid: str, *, prompt: str = "0", completion: str = "0", tools: bool = True) -> dict:
    return {
        "id": mid,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": ["tools"] if tools else ["max_tokens"],
    }


@pytest.fixture
def drift():
    return _load_drift_script()


def test_free_model_needs_both_price_fields_present_and_zero(drift):
    assert drift._is_free({"prompt": "0", "completion": "0"}) is True
    # A missing field is unknown pricing, not free pricing.
    assert drift._is_free({"prompt": "0"}) is False
    assert drift._is_free({"completion": "0"}) is False
    assert drift._is_free({"prompt": "0.000001", "completion": "0"}) is False
    assert drift._is_free(None) is False


def test_tool_support_is_required_and_absent_field_is_permissive(drift):
    assert drift._supports_tools({"supported_parameters": ["tools"]}) is True
    assert drift._supports_tools({"supported_parameters": ["max_tokens"]}) is False
    # Some OpenRouter-compatible gateways do not populate the field.
    assert drift._supports_tools({}) is True


def test_live_free_ids_skip_the_router_pseudo_model_and_toolless_entries(drift):
    items = [
        _live_entry("vendor/paid:free", prompt="0.001", completion="0.001"),
        _live_entry("vendor/free-model:free"),
        _live_entry("vendor/no-tools:free", tools=False),
        {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"},
         "supported_parameters": ["tools"]},
    ]

    assert drift._live_free_ids(items, exclude={"openrouter/free"}) == ["vendor/free-model:free"]


def test_a_free_suffix_does_not_make_a_priced_model_free(drift):
    """The suffix is a naming convention; the price decides."""
    items = [_live_entry("vendor/paid:free", prompt="0.001", completion="0.001")]

    assert drift._live_free_ids(items) == []


def test_drift_report_names_both_directions(drift, monkeypatch):
    live = {
        "data": [
            _live_entry("vendor/brand-new:free"),
            _live_entry("vendor/paid", prompt="0.001", completion="0.001"),
        ]
    }
    monkeypatch.setattr(drift, "_fetch_json", lambda url, timeout=drift.DEFAULT_TIMEOUT: live)
    monkeypatch.setattr(
        "hermes_cli.models_catalog_static.OPENROUTER_MODELS",
        [("vendor/retired:free", "free"), ("vendor/paid", "")],
    )

    report = drift.openrouter_drift()

    assert report["retired"] == ["vendor/retired:free"]
    assert report["uncurated_free"] == ["vendor/brand-new:free"]
    assert report["curated_count"] == 2
    assert report["live_free_count"] == 1


def test_a_curated_list_that_covers_the_live_free_tier_has_no_drift(drift, monkeypatch):
    live = {"data": [_live_entry("vendor/a:free"), _live_entry("vendor/b:free")]}
    monkeypatch.setattr(drift, "_fetch_json", lambda url, timeout=drift.DEFAULT_TIMEOUT: live)
    monkeypatch.setattr(
        "hermes_cli.models_catalog_static.OPENROUTER_MODELS",
        [("vendor/a:free", "free"), ("vendor/b:free", "free")],
    )

    report = drift.openrouter_drift()

    assert report["retired"] == []
    assert report["uncurated_free"] == []
