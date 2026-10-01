"""Tests for gateway model-catalog application projection."""

from gateway import model_catalog_runtime
from models.catalog_manifest import catalog_settings


def test_refresh_catalogs_refreshes_manifest_and_picker_sources(monkeypatch, tmp_path):
    settings = catalog_settings({})
    monkeypatch.setattr(
        model_catalog_runtime,
        "_catalog_runtime",
        lambda: (settings, tmp_path / "model_catalog.json"),
    )
    seen = []
    monkeypatch.setattr(
        model_catalog_runtime,
        "refresh_manifest",
        lambda got_settings, path, *, user_agent: seen.append(
            ("manifest", got_settings, path, user_agent)
        )
        or True,
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.refresh_picker_catalog_sources",
        lambda: seen.append(("picker",)),
    )

    assert model_catalog_runtime.refresh_catalogs() is True
    assert seen[0][0] == "manifest"
    assert seen[0][1] is settings
    assert seen[1] == ("picker",)


def test_disabled_catalog_skips_all_refresh_work(monkeypatch, tmp_path):
    settings = catalog_settings({"enabled": False})
    monkeypatch.setattr(
        model_catalog_runtime,
        "_catalog_runtime",
        lambda: (settings, tmp_path / "model_catalog.json"),
    )
    seen = []
    monkeypatch.setattr(
        model_catalog_runtime,
        "refresh_manifest",
        lambda *_args, **_kwargs: seen.append("manifest") or True,
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.refresh_picker_catalog_sources",
        lambda: seen.append("picker"),
    )

    assert model_catalog_runtime.refresh_catalogs() is False
    assert seen == []


def test_refresh_interval_is_derived_from_lower_catalogue_settings(monkeypatch):
    settings = object()
    monkeypatch.setattr(
        model_catalog_runtime,
        "_catalog_runtime",
        lambda: (settings, None),
    )
    monkeypatch.setattr(
        model_catalog_runtime,
        "_refresh_interval_seconds",
        lambda got: 321 if got is settings else 0,
    )

    assert model_catalog_runtime.refresh_interval_seconds() == 321
