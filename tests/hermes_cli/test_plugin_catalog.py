"""Plugin catalog contracts (hermes_cli/plugin_catalog.py): the in-tree seed is valid, bad entries are
skipped not raised, kill-list matching is name-or-repo, and the live catalog degrades to in-tree."""

from __future__ import annotations

import json

import yaml

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


def test_category_defaults_to_other_and_unknown_category_is_rejected(tmp_path):
    """``category`` is the browse shelf: absent → ``desktop``; a value outside CATALOG_CATEGORIES is an
    invalid entry (skipped with a warning) so a typo cannot create a phantom shelf on the site."""
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_entry("no-cat")))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_entry("mem", category="memory")))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(_entry("typo", category="memmory")))
    entries = {e.name: e for e in pc.load_catalog(tmp_path)}
    assert set(entries) == {"no-cat", "mem"}
    assert entries["no-cat"].category == "desktop" and entries["mem"].category == "memory"
    assert entries["mem"].to_dict()["category"] == "memory"


def test_version_and_image_are_cosmetic_and_offhost_images_are_dropped(tmp_path):
    """``version``/``image`` label the pin, they never gate it: a bad value is dropped with a warning and
    the entry survives; an image off GitHub is dropped because the Desktop browser must never fetch
    from third-party hosts."""
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_entry("labelled", version="1.4.0", image="https://raw.githubusercontent.com/owner/repo/38fe0fb53eff98d477f807432e965429e665ca33/banner.png")))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_entry("offhost", version="1.4.0 beta", image="https://evil.example/x.png")))
    entries = {e.name: e for e in pc.load_catalog(tmp_path)}
    assert set(entries) == {"labelled", "offhost"}
    assert entries["labelled"].version == "1.4.0" and entries["labelled"].image == "https://raw.githubusercontent.com/owner/repo/38fe0fb53eff98d477f807432e965429e665ca33/banner.png"
    assert entries["offhost"].version == "" and entries["offhost"].image == ""
    assert entries["labelled"].to_dict()["version"] == "1.4.0"


def test_invalid_entries_are_skipped_not_raised(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_entry("ok")))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_entry("short-sha", sha="abc123")))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(_entry("http-repo", repo="http://x/y")))
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(_entry("Bad Name")))
    (tmp_path / "e.yaml").write_text("- not\n- a mapping\n")
    assert [e.name for e in pc.load_catalog(tmp_path)] == ["ok"]


def test_find_removed_matches_name_or_normalized_repo(tmp_path):
    (tmp_path / "removed.yaml").write_text(yaml.safe_dump({"removed": [
        {"name": "evil", "repo": "https://github.com/x/evil.git", "reason": "malware", "date": "2026-01-01"}]}))
    assert pc.find_removed("evil", tmp_path).reason == "malware"
    assert pc.find_removed("https://github.com/x/EVIL/", tmp_path) is not None
    assert pc.find_removed("https://github.com/x/fine", tmp_path) is None


def test_live_catalog_falls_back_to_in_tree_and_unions_removals(tmp_path, monkeypatch):
    """Network failure → in-tree entries; a cached live doc contributes entries AND removals."""
    cache = tmp_path / "cache" / "plugin-catalog.json"
    monkeypatch.setattr(pc, "_live_cache_path", lambda: cache)
    monkeypatch.setattr(pc, "LIVE_CATALOG_URL", "http://127.0.0.1:9/nope")  # unreachable
    assert [e.name for e in pc.load_catalog_live()] == [e.name for e in pc.load_catalog()]

    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"entries": [_entry("live-only")],
                                 "removed": [{"name": "pulled-live", "reason": "cve"}]}))
    assert [e.name for e in pc.load_catalog_live()] == ["live-only"]
    assert pc.find_removed("pulled-live").reason == "cve"
