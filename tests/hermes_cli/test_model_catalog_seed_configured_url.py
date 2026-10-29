"""The post-update catalog seed must not replace a configured manifest.

``hermes update`` seeds the model-catalog cache from the checkout's published manifest.
When the operator configured their own ``model_catalog.url``, that seed silently
replaces their manifest - and the default label it carries - with the published one
until the next fetch, so the picker's list and the silent default both change.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME + reset the module-level catalog cache."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import model_catalog

    importlib.reload(model_catalog)
    yield home
    model_catalog.reset_cache()


def _checkout(root: Path) -> Path:
    """A checkout whose published manifest labels a paid default."""
    published = root / "website" / "static" / "api" / "model-catalog.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps({
        "version": 1,
        "updated_at": "2026-10-22T00:00:00Z",
        "metadata": {"source": "checkout"},
        "providers": {
            "openrouter": {
                "metadata": {"display_name": "OpenRouter"},
                "models": [
                    {"id": "vendor/paid-default", "description": "default", "default": True},
                ],
            },
        },
    }), encoding="utf-8")
    return root


def _write_config(home: Path, url: str) -> None:
    (home / "config.yaml").write_text(
        f"model_catalog:\n  url: {url}\n", encoding="utf-8")


def _cached_default(home: Path) -> list[str]:
    cache = home / "cache" / "model_catalog.json"
    if not cache.exists():
        return []
    data = json.loads(cache.read_text(encoding="utf-8-sig"))
    return [m["id"] for m in data["providers"]["openrouter"]["models"] if m.get("default")]


def test_the_seed_skips_a_configured_manifest_url(isolated_home, tmp_path, monkeypatch):
    from hermes_cli import model_catalog

    checkout = _checkout(tmp_path / "checkout")
    _write_config(isolated_home, "file:///my-own/catalog.json")
    monkeypatch.setattr(model_catalog, "_load_catalog_config",
                        lambda: {"enabled": True, "url": "file:///my-own/catalog.json",
                                 "ttl_hours": 0.33, "providers": {}})

    assert model_catalog.seed_cache_from_checkout(checkout) is False
    assert _cached_default(isolated_home) == []


def test_the_seed_still_runs_with_the_default_url(isolated_home, tmp_path, monkeypatch):
    from hermes_cli import model_catalog

    checkout = _checkout(tmp_path / "checkout")
    monkeypatch.setattr(model_catalog, "_load_catalog_config",
                        lambda: {"enabled": True, "url": model_catalog.DEFAULT_CATALOG_URL,
                                 "ttl_hours": 0.33, "providers": {}})

    assert model_catalog.seed_cache_from_checkout(checkout) is True
    assert _cached_default(isolated_home) == ["vendor/paid-default"]


def test_a_config_error_does_not_break_the_seed(isolated_home, tmp_path, monkeypatch):
    """A config that cannot be read must not stop the update's seed."""
    from hermes_cli import model_catalog

    checkout = _checkout(tmp_path / "checkout")

    def _boom():
        raise RuntimeError("no config here")

    monkeypatch.setattr(model_catalog, "_load_catalog_config", _boom)

    assert model_catalog.seed_cache_from_checkout(checkout) is True
    assert _cached_default(isolated_home) == ["vendor/paid-default"]
