"""The doctor check for configured model ids the provider's catalog no longer lists.

A retired id fails at the first request with HTTP 404, and nothing else in the install
says so. The check reads the install's own provider catalog cache, so it costs no
network call and works offline.
"""
from __future__ import annotations

import json

import pytest

from hermes_cli.model_availability import (
    ModelRef,
    cached_catalogs,
    configured_models,
    unavailable_models,
)


def _catalogs(**providers: list[str]) -> dict[str, set[str]]:
    return {name: set(ids) for name, ids in providers.items()}


def test_every_model_slot_the_runtime_reads_is_collected():
    config = {
        "model": {
            "default": "vendor/a:free",
            "provider": "openrouter",
            "aliases": {"short": "vendor/b:free"},
        },
        "auxiliary": {
            "openrouter_model": "vendor/c:free",
            "compression": {"model": "vendor/d:free", "provider": "nous"},
            "vision": {"model": "", "provider": "auto"},
        },
        "moa": {
            "aggregator": {"model": "vendor/e:free", "provider": "openrouter"},
            "reference_models": [
                {"model": "vendor/f:free", "provider": "nous"},
                {"model": "vendor/g:free", "provider": "openrouter"},
            ],
        },
        "delegation": {"model": "vendor/h:free", "provider": "openrouter"},
    }

    found = {(ref.slot, ref.model_id, ref.provider) for ref in configured_models(config)}

    assert ("model.default", "vendor/a:free", "openrouter") in found
    assert ("model.aliases.short", "vendor/b:free", "openrouter") in found
    assert ("auxiliary.openrouter_model", "vendor/c:free", "openrouter") in found
    assert ("auxiliary.compression.model", "vendor/d:free", "nous") in found
    assert ("moa.aggregator.model", "vendor/e:free", "openrouter") in found
    assert ("moa.reference_models[0].model", "vendor/f:free", "nous") in found
    assert ("moa.reference_models[1].model", "vendor/g:free", "openrouter") in found
    assert ("delegation.model", "vendor/h:free", "openrouter") in found
    # An empty model and an "auto" provider carry no id to judge.
    assert not [ref for ref in found if ref[1] == ""]
    assert "auxiliary.vision.model" not in {slot for slot, _model, _provider in found}


def test_an_id_the_provider_no_longer_lists_is_reported():
    config = {
        "model": {"default": "vendor/gone:free", "provider": "openrouter"},
        "auxiliary": {"openrouter_model": "vendor/kept:free"},
    }
    catalogs = _catalogs(openrouter={"vendor/kept:free", "vendor/other:free"})

    unavailable = unavailable_models(config, catalogs)

    assert [(item.ref.slot, item.ref.model_id) for item in unavailable] == [
        ("model.default", "vendor/gone:free")
    ]
    assert "model.default" in unavailable[0].render()
    assert "openrouter" in unavailable[0].render()


def test_a_provider_without_a_cached_catalog_is_not_judged():
    """An unknown catalog is not evidence."""
    config = {"model": {"default": "vendor/anything", "provider": "someprovider"}}

    assert unavailable_models(config, _catalogs(openrouter={"vendor/other"})) == []


def test_every_slot_of_one_model_is_reported():
    config = {
        "model": {"default": "vendor/gone", "provider": "openrouter"},
        "auxiliary": {"openrouter_model": "vendor/gone"},
    }
    catalogs = _catalogs(openrouter={"vendor/other"})

    slots = {item.ref.slot for item in unavailable_models(config, catalogs)}

    assert slots == {"model.default", "auxiliary.openrouter_model"}


def test_the_cache_reader_returns_provider_model_sets(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "provider_models_cache.json").write_text(json.dumps({
        "openrouter": {"fp": "abc", "at": 1.0, "models": ["vendor/a", "vendor/b"]},
        "nous": {"fp": "def", "at": 1.0, "models": []},
        "broken": "not a row",
    }), encoding="utf-8")

    catalogs = cached_catalogs()

    assert catalogs == {"openrouter": {"vendor/a", "vendor/b"}}
    assert get_hermes_home() == home


def test_a_missing_or_broken_cache_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert cached_catalogs() == {}

    (tmp_path / "provider_models_cache.json").write_text("{not json", encoding="utf-8")
    assert cached_catalogs() == {}


def test_a_configured_id_that_is_listed_is_not_reported():
    config = {"model": {"default": "vendor/a:free", "provider": "openrouter"}}

    assert unavailable_models(config, _catalogs(openrouter={"vendor/a:free"})) == []


def test_the_model_ref_carries_the_provider_that_must_serve_it():
    ref = ModelRef(slot="model.default", model_id="vendor/a", provider="openrouter")

    assert ref.provider == "openrouter"
