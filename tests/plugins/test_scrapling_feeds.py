"""Tests for scrapling-feeds plugin (no live network)."""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import timedelta
from pathlib import Path

from tools.url_safety import SSRFProtectedTransport

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


def test_fetch_url_blocks_unsafe_target_before_backend(monkeypatch):
    fetcher = _load_module("fetcher")
    monkeypatch.setattr(fetcher, "resolve_safe_url_addresses", lambda *_a, **_k: ())
    monkeypatch.setattr(
        fetcher,
        "_fetch_httpx",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    result = fetcher.fetch_url("http://127.0.0.1/internal")

    assert result["success"] is False
    assert result["backend"] == "safety"


def test_scrapling_fetch_uses_safe_redirect_policy(monkeypatch):
    fetcher = _load_module("fetcher")
    captured = {}

    class _Fetcher:
        @staticmethod
        def get(url, **kwargs):
            captured.update({"url": url, **kwargs})
            return types.SimpleNamespace(
                status=200,
                body=b"<rss/>",
                url=url,
            )

    package = types.ModuleType("scrapling")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("scrapling.fetchers")
    module.Fetcher = _Fetcher
    monkeypatch.setitem(sys.modules, "scrapling", package)
    monkeypatch.setitem(sys.modules, "scrapling.fetchers", module)
    monkeypatch.setattr(
        fetcher,
        "resolve_safe_url_addresses",
        lambda *_a, **_k: ("93.184.216.34",),
    )

    result = fetcher._fetch_scrapling("https://www.mod.go.jp/feed.xml", timeout=5)

    assert result["success"] is True
    assert captured["follow_redirects"] == "safe"
    assert captured["max_redirects"] == 10


def test_httpx_fallback_uses_pinned_transport(monkeypatch):
    fetcher = _load_module("fetcher")
    captured = {}

    class _Response:
        status_code = 200
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b"<rss/>"

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, method, url):
            captured.update({"method": method, "url": url})
            return _Response()

    monkeypatch.setattr(fetcher.httpx, "Client", _Client)

    result = fetcher._fetch_httpx("https://www.cisa.gov/news.xml", timeout=5)

    assert result["success"] is True
    assert isinstance(captured["transport"], SSRFProtectedTransport)
    assert captured["transport"]._allow_private_urls is False


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
