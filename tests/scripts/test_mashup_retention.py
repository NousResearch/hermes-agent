#!/usr/bin/env python3
"""Tests for mashup_retention.py — proposal archiving + blog queue retention.

Hermetic: operates on tmp_path fixtures, no real ~/.hermes or content_engine.
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mashup_retention as mr


@pytest.fixture
def env(tmp_path, monkeypatch):
    hermes = tmp_path / "hermes"
    blog = tmp_path / "blog"
    blog.mkdir()
    monkeypatch.setattr(mr, "HERMES_HOME", hermes)
    monkeypatch.setattr(mr, "PROPOSALS_DIR", hermes / "runbooks" / "proposals")
    monkeypatch.setattr(mr, "BLOG_DIR", blog)
    return hermes, blog


def _old_entry(slug="old-idea", days=120):
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return {"topic_id": f"research-{slug}", "title_hint": slug, "created_at": ts}


def _fresh_entry(slug="fresh-idea", days=1):
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return {"topic_id": f"research-{slug}", "title_hint": slug, "created_at": ts}


class TestArchiveProposals:
    def test_moves_old_only(self, env):
        hermes, _ = env
        props = mr.PROPOSALS_DIR
        props.mkdir(parents=True)
        old = props / "mashup-2026-05-01.html"
        fresh = props / "mashup-2026-07-31.html"
        old.write_text("old")
        fresh.write_text("fresh")
        # set old mtime 60 days back
        old_ts = datetime.now().timestamp() - 60 * 86400
        import os
        os.utime(old, (old_ts, old_ts))
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        moved = mr.archive_proposals(cutoff)
        assert moved == 1
        assert (props / "archive" / "mashup-2026-05-01.html").exists()
        assert fresh.exists()

    def test_no_dir_returns_zero(self, env):
        hermes, _ = env
        assert mr.archive_proposals(datetime.now(timezone.utc)) == 0


class TestArchiveBlogStream:
    def test_archives_old_keeps_fresh(self, env):
        hermes, blog = env
        f = blog / "ai.jsonl"
        f.write_text(
            json.dumps(_old_entry()) + "\n" + json.dumps(_fresh_entry()) + "\n",
            encoding="utf-8",
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        fresh, stale = mr.archive_blog_stream(f, cutoff)
        assert fresh == 1
        assert stale == 1
        assert (blog / "ai.archive.jsonl").exists()
        # active file has only fresh
        active = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        assert len(active) == 1
        assert active[0]["topic_id"] == "research-fresh-idea"

    def test_idempotent(self, env):
        hermes, blog = env
        f = blog / "ai.jsonl"
        f.write_text(json.dumps(_fresh_entry()) + "\n", encoding="utf-8")
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        f1, s1 = mr.archive_blog_stream(f, cutoff)
        f2, s2 = mr.archive_blog_stream(f, cutoff)
        assert (f1, s1) == (1, 0)
        assert (f2, s2) == (1, 0)  # no double-archive

    def test_malformed_line_archived_not_lost(self, env):
        hermes, blog = env
        f = blog / "builder.jsonl"
        f.write_text("not-json\n", encoding="utf-8")
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        fresh, stale = mr.archive_blog_stream(f, cutoff)
        assert fresh == 0
        assert stale == 1
        assert "not-json" in (blog / "builder.archive.jsonl").read_text()


class TestMain:
    def test_main_exits_zero(self, env, capsys, monkeypatch):
        hermes, blog = env
        (blog / "ai.jsonl").write_text(json.dumps(_fresh_entry()) + "\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["mashup_retention.py"])
        assert mr.main() == 0
        out = capsys.readouterr().out
        assert "ai.jsonl" in out
