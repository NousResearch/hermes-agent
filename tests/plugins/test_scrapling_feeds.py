"""Tests for scrapling-feeds plugin (no live network)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "scrapling-feeds"


def _load_module(name: str):
    pkg = "scrapling_feeds_test_pkg"
    if pkg not in sys.modules:
        import types

        package = types.ModuleType(pkg)
        package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
        sys.modules[pkg] = package
        for stem in ("feeds_catalog", "fetcher", "rss_parse", "gov_digest", "milspec_bridge"):
            if stem == "milspec_bridge":
                continue
            mod_name = f"{pkg}.{stem}"
            spec = importlib.util.spec_from_file_location(
                mod_name, PLUGIN_DIR / f"{stem}.py"
            )
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f"{pkg}.{name}"]


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test MOD</title>
    <item>
      <title>防衛省テスト報道</title>
      <link>https://www.mod.go.jp/j/press/example.html</link>
      <pubDate>Thu, 18 Jun 2026 08:00:00 GMT</pubDate>
      <description>テスト概要</description>
    </item>
  </channel>
</rss>
"""


def test_parse_rss_extracts_entry():
    rss_parse = _load_module("rss_parse")
    entries = rss_parse.parse_feed_xml(SAMPLE_RSS)
    assert len(entries) == 1
    assert entries[0]["title"] == "防衛省テスト報道"
    assert entries[0]["url"].startswith("https://www.mod.go.jp/")


def test_filter_entries_since():
    rss_parse = _load_module("rss_parse")
    entries = rss_parse.parse_feed_xml(SAMPLE_RSS)
    since = entries[0]["published_dt"] - timedelta(hours=1)
    filtered = rss_parse.filter_entries_since(entries, since=since, max_items=5)
    assert len(filtered) == 1


def test_feed_catalog_has_mod_and_cisa():
    feeds_catalog = _load_module("feeds_catalog")
    assert "mod_press" in feeds_catalog.GOV_FEEDS
    assert "cisa_advisories_all" in feeds_catalog.GOV_FEEDS
    assert feeds_catalog.GOV_FEEDS["mod_press"]["url"].endswith("news.xml")


def test_build_gov_feeds_block_markdown():
    wm_dir = Path(__file__).resolve().parents[2] / "plugins" / "worldmonitor-osint"
    spec = importlib.util.spec_from_file_location(
        "wm_milspec_test", wm_dir / "milspec_prose.py"
    )
    assert spec and spec.loader
    milspec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(milspec)
    enrichment = {
        "gov_feeds": {
            "success": True,
            "scrapling_available": False,
            "feeds_ok": 1,
            "feed_count": 1,
            "total_entries": 1,
            "window_hours": 24,
            "catalog_docs": {"mod": "https://www.mod.go.jp/j/rss/index.html"},
            "feeds": [
                {
                    "success": True,
                    "name": "防衛省 — お知らせ",
                    "agency": "防衛省",
                    "source_tier": "PRIMARY",
                    "entries": [
                        {
                            "title": "テスト",
                            "citation": "[出典: 防衛省] https://www.mod.go.jp/x",
                            "published_at": "2026-06-18T08:00:00+00:00",
                        }
                    ],
                }
            ],
        }
    }
    block = "\n".join(milspec.build_gov_feeds_block(enrichment))
    assert "GOVERNMENT FEEDS" in block
    assert "防衛省" in block
    assert "PRIMARY" in block
