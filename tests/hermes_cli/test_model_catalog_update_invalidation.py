"""Regression coverage for checkout-triggered provider catalog invalidation."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import model_catalog

    importlib.reload(model_catalog)
    yield home
    model_catalog.reset_cache()


def test_missing_manifest_still_invalidates_every_profile_provider_cache(
    isolated_home, tmp_path
):
    from hermes_cli import model_catalog

    provider_cache = isolated_home / "provider_models_cache.json"
    provider_cache.write_text("{}", encoding="utf-8")
    sibling_cache = isolated_home / "profiles" / "work" / "provider_models_cache.json"
    sibling_cache.parent.mkdir(parents=True)
    sibling_cache.write_text("{}", encoding="utf-8")
    (sibling_cache.parent / "config.yaml").write_text("{}", encoding="utf-8")

    assert model_catalog.seed_cache_from_checkout(tmp_path / "missing-checkout") is False
    assert not provider_cache.exists()
    assert not sibling_cache.exists()


def test_default_root_failure_does_not_block_named_profile_invalidation(
    isolated_home, tmp_path
):
    from hermes_cli import model_catalog

    named_home = isolated_home / "profiles" / "work"
    named_home.mkdir(parents=True, exist_ok=True)
    named_cache = named_home / "provider_models_cache.json"
    named_cache.write_text("{}", encoding="utf-8")

    with (
        patch(
            "hermes_constants.get_default_hermes_root",
            side_effect=OSError("default root unavailable"),
        ),
        patch(
            "hermes_cli.profiles._iter_named_profile_dirs",
            return_value=[named_home],
        ),
    ):
        assert model_catalog.seed_cache_from_checkout(tmp_path / "missing-checkout") is False

    assert not named_cache.exists()
