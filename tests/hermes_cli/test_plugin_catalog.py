"""Plugin catalog contracts (hermes_cli/plugin_catalog.py): the in-tree seed is valid, bad entries are
skipped not raised, kill-list matching is name-or-repo, and the live catalog degrades to in-tree."""

from __future__ import annotations

import json
import os

import pytest

import hermes_yaml as yaml

from hermes_cli import plugin_catalog as pc

SHA = "38fe0fb53eff98d477f807432e965429e665ca33"


def _entry(name="good-plugin", **over):
    data = {"name": name, "repo": "https://github.com/owner/repo", "sha": SHA, "description": "d",
            "maintainer": "owner", "tier": "community", "capabilities": {"provides_tools": ["t1"]}}
    data.update(over)
    return data


def test_shipped_catalog_entries_are_all_valid_and_pinned():
    """Every file in plugin-catalog/ (minus removed.yaml) must parse — a dropped entry is a silent
    shipping regression the admission CI only catches on changed files."""
    root = pc.get_catalog_dir()
    files = [p for p in root.glob("*.yaml") if p.name != "removed.yaml"]
    entries = pc.load_catalog()
    assert len(entries) == len(files) >= 1
    assert all(pc._SHA_RE.match(e.sha) and e.repo.startswith("https://") for e in entries)
    assert all(e.install_identifier.startswith(e.repo) for e in entries)


def test_invalid_entries_are_skipped_not_raised(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_entry("ok", description="café")), encoding="utf-8-sig")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_entry("short-sha", sha="abc123")))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(_entry("http-repo", repo="http://x/y")))
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(_entry("Bad Name")))
    (tmp_path / "e.yaml").write_text("- not\n- a mapping\n")
    entries = pc.load_catalog(tmp_path)
    assert [e.name for e in entries] == ["ok"]
    assert entries[0].description == "café"


def test_find_removed_matches_name_or_normalized_repo(tmp_path):
    (tmp_path / "removed.yaml").write_text(yaml.safe_dump({"removed": [
        {"name": "evil", "repo": "https://github.com/x/evil.git", "reason": "malware", "date": "2026-01-01"}]}))
    assert pc.find_removed("evil", tmp_path).reason == "malware"
    assert pc.find_removed("https://github.com/x/EVIL/", tmp_path) is not None
    assert pc.find_removed("https://github.com/x/fine", tmp_path) is None


@pytest.mark.parametrize("stale", [False, True])
def test_live_catalog_falls_back_to_in_tree_and_unions_removals(tmp_path, monkeypatch, stale):
    """Network failure → in-tree entries; a cached live doc contributes entries AND removals."""
    import httpx

    requests = []
    get = httpx.get

    def record_get(*args, **kwargs):
        requests.append(args)
        return get(*args, **kwargs)

    monkeypatch.setattr(httpx, "get", record_get)
    cache = tmp_path / "cache" / "plugin-catalog.json"
    monkeypatch.setattr(pc, "_live_cache_path", lambda: cache)
    monkeypatch.setattr(pc, "LIVE_CATALOG_URL", "http://127.0.0.1:9/nope")  # unreachable
    assert [e.name for e in pc.load_catalog_live()] == [e.name for e in pc.load_catalog()]

    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"entries": [_entry("live-only", description="café")],
                                 "removed": [{"name": "pulled-live", "reason": "cve"}]},
                                ensure_ascii=False), encoding="utf-8-sig")
    if stale:
        os.utime(cache, (1, 1))
    requests.clear()
    entries = pc.load_catalog_live()
    assert [e.name for e in entries] == ["live-only"]
    assert entries[0].description == "café"
    assert bool(requests) is stale
    assert pc.find_removed("pulled-live").reason == "cve"
    cache.write_text("{", encoding="utf-8-sig")
    assert [e.name for e in pc.load_catalog_live()] == [e.name for e in pc.load_catalog()]
