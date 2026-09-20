#!/usr/bin/env python3
"""Tests for research_digest_preprocess.py — seen-cache dedup, RSS parsing, key helpers.

No network: fetch_url/fetch_json/run_cmd are monkeypatched.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_digest_preprocess as rdp


class TestSeenCache:
    def test_load_seen_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rdp, "SEEN_CACHE", tmp_path / "none.json")
        assert rdp.load_seen() == {}

    def test_load_seen_valid(self, tmp_path, monkeypatch):
        f = tmp_path / "seen.json"
        f.write_text(json.dumps({"items": {"a:1": {"first_seen": "2026-07-01"}}}))
        monkeypatch.setattr(rdp, "SEEN_CACHE", f)
        assert rdp.load_seen() == {"a:1": {"first_seen": "2026-07-01"}}

    def test_save_seen_prunes_old(self, tmp_path, monkeypatch):
        f = tmp_path / "seen.json"
        monkeypatch.setattr(rdp, "SEEN_CACHE", f)
        # Dates must be relative to now: the 30-day prune makes any hardcoded
        # fixture date stale once it ages past the cutoff (this test was red
        # from 2026-08-29 for exactly that reason).
        now = datetime.now()
        old = {"a:1": {"first_seen": (now - timedelta(days=60)).isoformat()}}
        fresh = {"b:2": {"first_seen": (now - timedelta(days=1)).isoformat()}}
        rdp.save_seen({**old, **fresh})
        data = json.loads(f.read_text())
        assert "a:1" not in data["items"]
        assert "b:2" in data["items"]


class TestKeysAndNew:
    def test_make_key(self):
        assert rdp.make_key("hn", "123") == "hn:123"

    def test_is_new_and_mark(self):
        seen = {}
        assert rdp.is_new("src", "id1", seen)
        rdp.mark_seen("src", "id1", seen, "Title")
        assert not rdp.is_new("src", "id1", seen)
        assert seen["src:id1"]["title"] == "Title"


class TestParseRss:
    def test_atom_entries(self):
        text = (
            "<feed><entry><title>Hello</title>"
            "<link href=\"https://example.com/1\"/></entry>"
            "<entry><title>World</title><link href=\"https://example.com/2\"/></entry></feed>"
        )
        items = rdp.parse_rss(text, "Test")
        assert len(items) == 2
        assert items[0]["title"] == "Hello"
        assert items[0]["url"] == "https://example.com/1"
        assert items[0]["source"] == "Test"

    def test_rss_items_fallback(self):
        text = (
            "<rss><channel><item><title>First</title><link>https://x.com/1</link></item></channel></rss>"
        )
        items = rdp.parse_rss(text, "RSS")
        assert len(items) == 1
        assert items[0]["title"] == "First"

    def test_empty(self):
        assert rdp.parse_rss("<feed></feed>", "Empty") == []


class TestMainOffline:
    def test_main_no_sources_returns_zero(self, monkeypatch, tmp_path):
        """All sources empty -> no crash, no candidates, exit 0."""
        monkeypatch.setattr(rdp, "fetch_json", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "fetch_url", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "run_cmd", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "OUTPUT_FILE", tmp_path / "candidates.json")
        monkeypatch.setattr(rdp, "SEEN_CACHE", tmp_path / "seen.json")

        assert rdp.main() == 0
        assert (tmp_path / "candidates.json").exists()

    def test_main_with_fake_hn(self, monkeypatch, tmp_path):
        """HN source yields a candidate with valid fields."""
        hn_hit = {
            "objectID": "999",
            "title": "AI coding agent tool",
            "points": 20,
            "created_at_i": 1700000000,
        }

        def fake_fetch_json(url, *a, **k):
            if "show_hn" in url:
                return {"hits": [hn_hit]}
            if "query=AI+LLM" in url or "AI%20LLM" in url:
                return {"hits": []}
            return None

        monkeypatch.setattr(rdp, "fetch_json", fake_fetch_json)
        monkeypatch.setattr(rdp, "fetch_url", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "run_cmd", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "OUTPUT_FILE", tmp_path / "candidates.json")
        monkeypatch.setattr(rdp, "SEEN_CACHE", tmp_path / "seen.json")

        rdp.main()
        data = json.loads((tmp_path / "candidates.json").read_text())
        assert data["total_candidates"] == 1
        assert data["candidates"][0]["source"] == "HN Show HN"

    def test_main_seen_cache_persisted(self, monkeypatch, tmp_path):
        hn_hit = {"objectID": "555", "title": "Fresh tool", "points": 30, "created_at_i": 1700000000}

        def fake_fetch_json(url, *a, **k):
            if "show_hn" in url:
                return {"hits": [hn_hit]}
            return {"hits": []}

        monkeypatch.setattr(rdp, "fetch_json", fake_fetch_json)
        monkeypatch.setattr(rdp, "fetch_url", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "run_cmd", lambda *a, **k: None)
        monkeypatch.setattr(rdp, "OUTPUT_FILE", tmp_path / "c.json")
        monkeypatch.setattr(rdp, "SEEN_CACHE", tmp_path / "seen.json")

        rdp.main()
        # second run must not re-emit the same item
        rdp.main()
        data = json.loads((tmp_path / "c.json").read_text())
        assert data["total_candidates"] == 0
