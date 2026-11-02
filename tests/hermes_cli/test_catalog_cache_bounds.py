"""Catalog caches in a long-lived process: one models.dev parse, no dead pricing catalogs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent import models_dev
import hermes_cli.models_catalog_static as static
import hermes_cli.models_pricing as pricing


def test_xai_floor_reuses_the_process_models_dev_copy_without_network(monkeypatch):
    catalog = {"xai": {"id": "xai", "models": {"grok-z": {}, "grok-a": {}}}}
    parses = []
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    monkeypatch.setattr(models_dev, "_load_disk_cache", lambda: parses.append(1) or catalog)
    monkeypatch.setattr(models_dev, "_disk_cache_age_seconds", lambda: 10.0)
    monkeypatch.setattr(models_dev, "_refresh_locked", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))

    ids = static._xai_curated_models()

    assert "grok-a" in ids and "grok-z" in ids
    assert models_dev.fetch_models_dev(allow_network=False) is catalog
    assert models_dev.fetch_models_dev() is catalog  # fresh copy: served from memory, no request
    assert len(parses) == 1


_LAZY_PROBE = r"""
import json, sys
from agent import models_dev
parses = []
real = models_dev._load_disk_cache
models_dev._load_disk_cache = lambda: parses.append(1) or real()
import hermes_cli.models  # the import every gateway/CLI process pays
from hermes_cli.models_catalog_static import _PROVIDER_MODELS
at_import = len(parses)
_PROVIDER_MODELS.get("nous")
after_other_row = len(parses)
copied = dict(_PROVIDER_MODELS)
json.dump({"at_import": at_import, "after_other_row": after_other_row, "copy_xai": copied["xai"],
           "row_xai_oauth": _PROVIDER_MODELS["xai-oauth"],
           "shared": models_dev.fetch_models_dev(allow_network=False) is models_dev._models_dev_cache,
           "total": len(parses)}, sys.stdout)
"""


def test_models_dev_is_parsed_on_first_xai_read_not_at_import(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "models_dev_cache.json").write_text(json.dumps(
        {"xai": {"id": "xai", "models": {"grok-disk-only": {}}}, "openai": {"id": "openai", "models": {"gpt-x": {}}}}))
    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HERMES_HOME=str(home), HOME=str(tmp_path), PYTHONPATH=str(repo))
    out = subprocess.run([sys.executable, "-c", _LAZY_PROBE], check=False, cwd=repo, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])

    assert got["at_import"] == 0 and got["after_other_row"] == 0  # idle process: no catalog parse
    assert "grok-disk-only" in got["copy_xai"] and "grok-disk-only" in got["row_xai_oauth"]
    assert got["shared"] and got["total"] == 1  # one parse, shared with every later reader


def test_pricing_cache_write_drops_catalogs_past_their_deadline(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(pricing.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(pricing, "_pricing_cache", {})
    monkeypatch.setattr(pricing, "_pricing_cache_retry_after", {})
    pricing._cache_catalog("https://n.example\x00auth:old", {"m": {"prompt": "1"}}, ttl_seconds=300)
    pricing._cache_catalog("https://static.example", {"m": {"prompt": "1"}})
    now["t"] = 500.0
    pricing._cache_catalog("https://n.example\x00auth:live", {"m": {"prompt": "2"}}, ttl_seconds=300)

    pricing._cache_catalog("https://n.example\x00auth:new", {"m": {"prompt": "3"}}, ttl_seconds=300)

    assert set(pricing._pricing_cache) == {"https://static.example", "https://n.example\x00auth:live", "https://n.example\x00auth:new"}
    assert "https://n.example\x00auth:old" not in pricing._pricing_cache_retry_after
    assert pricing._cached_catalog("https://n.example\x00auth:live") == {"m": {"prompt": "2"}}
