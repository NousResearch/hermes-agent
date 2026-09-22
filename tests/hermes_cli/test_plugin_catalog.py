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


def test_screenshots_and_readme_are_parsed_and_readme_defaults_on(tmp_path):
    shot = "https://raw.githubusercontent.com/owner/repo/38fe0fb53eff98d477f807432e965429e665ca33/docs/1.png"
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_entry("paged", screenshots=[shot, "https://evil.example/x.png"], readme=True)))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_entry("plain", readme=False)))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(_entry("bare")))
    entries = {e.name: e for e in pc.load_catalog(tmp_path)}
    assert entries["paged"].screenshots == [shot] and entries["paged"].readme is True
    assert entries["plain"].screenshots == [] and entries["plain"].readme is False
    assert entries["bare"].readme is True
    assert entries["paged"].to_dict()["screenshots"] == [shot] and entries["paged"].to_dict()["readme"] is True


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
    # Scheme/host/user spellings are not identity: every git way of naming the repo is blocked.
    for spelling in ("git@github.com:x/evil.git", "ssh://git@github.com/x/evil", "http://github.com/x/evil",
                     "https://www.github.com/x/evil/", "git://github.com/x/evil.git"):
        assert pc.find_removed(spelling, tmp_path) is not None, spelling
    assert pc.find_removed("git@gitlab.com:x/evil.git", tmp_path) is None  # different host stays distinct


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


def _fresh_cache(tmp_path, monkeypatch, doc: dict):
    cache = tmp_path / "cache" / "plugin-catalog.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc))
    monkeypatch.setattr(pc, "_live_cache_path", lambda: cache)
    monkeypatch.setattr(pc, "LIVE_CATALOG_URL", "http://127.0.0.1:9/nope")  # never reached / unreachable
    return cache


def test_newer_in_tree_pin_outranks_a_fresh_live_cache(tmp_path, monkeypatch):
    """Right after `hermes update` bumps an in-tree pin, a live cache fetched BEFORE the bump must not
    re-install the old sha: for the same entry the newer catalog wins (checkout catalog commit time vs
    the doc's generated_at). Live-only entries survive; an older checkout still defers to the doc."""
    old, new = SHA, "a" * 40
    _fresh_cache(tmp_path, monkeypatch, {"generated_at": "2026-09-22T10:00:00Z",
                                         "entries": [_entry("shared", sha=old), _entry("live-only")], "removed": []})
    monkeypatch.setattr(pc, "load_catalog", lambda catalog_dir=None: [pc.entry_from_mapping(_entry("shared", sha=new), "t")])
    bump_time = pc._live_generated_time({"generated_at": "2026-09-22T10:00:00Z"}) + 3600
    monkeypatch.setattr(pc, "in_tree_catalog_time", lambda: bump_time)
    by_name = {e.name: e for e in pc.load_catalog_live()}
    assert by_name["shared"].sha == new and "live-only" in by_name
    # Control: a checkout whose catalog predates the doc takes the live pin.
    monkeypatch.setattr(pc, "in_tree_catalog_time", lambda: bump_time - 7200)
    assert {e.name: e.sha for e in pc.load_catalog_live()}["shared"] == old


def test_live_cache_past_max_stale_age_stops_supplying_pins_but_keeps_removals(tmp_path, monkeypatch):
    """Offline for days: a 25-hour-old cache no longer outranks the in-tree catalog (its pins may be
    older than the checkout's), while its kill-list entries still block."""
    import os
    import time
    cache = _fresh_cache(tmp_path, monkeypatch, {"entries": [_entry("live-only")],
                                                 "removed": [{"name": "pulled-live", "reason": "cve"}]})
    stale = time.time() - pc.LIVE_CATALOG_MAX_STALE_SECONDS - 3600
    os.utime(cache, (stale, stale))
    pc._live_fetch_failed_until = 0.0
    assert [e.name for e in pc.load_catalog_live()] == [e.name for e in pc.load_catalog()]
    assert pc.find_removed("pulled-live").reason == "cve"
    assert pc.cached_removed_entries()[-1].name == "pulled-live"


def test_live_cache_write_never_truncates_the_previous_copy(tmp_path, monkeypatch):
    """The cache is replaced atomically: a reader racing the writer (or a write that dies half-way)
    sees the whole previous document, never a truncated one."""
    cache = _fresh_cache(tmp_path, monkeypatch, {"entries": [_entry("previous")], "removed": []})
    previous = cache.read_text()

    class _Resp:
        content = json.dumps({"entries": [_entry("fresh")], "removed": []}).encode()
        def raise_for_status(self): pass
        def json(self): return json.loads(self.content)

    import httpx
    import utils
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    pc._live_fetch_failed_until = 0.0

    def _dies(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "atomic_replace", _dies)
    pc.fetch_live_catalog(force=True)  # the failed write is swallowed like any other fetch failure
    assert cache.read_text() == previous
    assert not list(cache.parent.glob("*.tmp*"))  # no leftover temp file either
