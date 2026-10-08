"""Featured is curation, not a trust tier; discovery retains catalog policy."""
import json
from urllib.parse import parse_qs, urlparse

import pytest
import hermes_yaml as yaml
from hermes_cli import plugin_catalog as catalog


def entry(name="featured-provider", **overrides):
    return {"name": name, "repo": f"https://github.com/example/{name}", "sha": "a" * 40,
            "description": "Memory provider", "maintainer": "example", "category": "memory",
            "featured": True, **overrides}


@pytest.mark.parametrize("value", [True, False, "true", 1, None, "absent"])
def test_featured_boolean_survives_parser_and_serialization(value):
    raw = entry(featured=value)
    if value == "absent":
        raw.pop("featured")
    parsed = catalog.entry_from_mapping(raw, "fixture")
    assert parsed is not None
    assert parsed.to_dict()["featured"] is (value is True)
    assert parsed.tier == "community"


def test_featured_live_cache_and_old_document_curated_fallback(tmp_path, monkeypatch):
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "provider.yaml").write_text(yaml.safe_dump(entry()))
    monkeypatch.setattr(catalog, "get_catalog_dir", lambda: root)
    monkeypatch.setattr(catalog, "in_tree_catalog_time", lambda: None)
    from hermes_constants import get_hermes_home
    cache = get_hermes_home() / "cache" / "plugin-catalog.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw = entry()
    for value in (True, False, "absent"):
        if value == "absent":
            raw.pop("featured")
        else:
            raw["featured"] = value
        cache.write_text(json.dumps({"entries": [raw], "removed": []}))
        assert catalog.load_catalog_live()[0].to_dict()["featured"] is (value is not False)


def test_marketplace_targets_memory_shelf():
    from hermes_cli.memory_catalog import MARKETPLACE_URL
    assert parse_qs(urlparse(MARKETPLACE_URL).query) == {"kind": ["memory"]}


def test_memory_discovery_filters_removals_compatibility_and_nonfeatured(tmp_path, monkeypatch):
    from hermes_cli.memory_catalog import featured_memory_entries
    from hermes_platform.host.facts import os_family
    root = tmp_path / "catalog"
    root.mkdir()
    other = "linux" if os_family() == "win32" else "windows"
    rows = [entry(), entry("plain", featured=False), entry("desktop", category="desktop"),
            entry("newer", requires_hermes=">=999.0.0"), entry("other-os", platforms=[other]),
            entry("blocked-name"), entry("blocked-repo")]
    for row in rows:
        (root / f"{row['name']}.yaml").write_text(yaml.safe_dump(row))
    (root / "removed.yaml").write_text(yaml.safe_dump({"removed": [
        {"name": "blocked-name"}, {"name": "alias", "repo": rows[-1]["repo"]}]}))
    monkeypatch.setattr(catalog, "get_catalog_dir", lambda: root)
    monkeypatch.setattr(catalog, "fetch_live_catalog", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.plugins_manifest.running_hermes_version", lambda: "1.0.0")
    assert [e.name for e in featured_memory_entries()] == ["featured-provider"]
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: (_ for _ in ()).throw(OSError("offline")))
    assert featured_memory_entries() == []
