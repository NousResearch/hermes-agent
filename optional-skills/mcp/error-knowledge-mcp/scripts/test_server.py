"""Tests for error-knowledge MCP server."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Point the server to a temp directory before importing
TEST_ROOT = Path(tempfile.mkdtemp(prefix="error_knowledge_test_"))
os.environ["ERROR_KNOWLEDGE_ROOT"] = str(TEST_ROOT)
os.environ["ERROR_KNOWLEDGE_AUTO_ARCHIVE"] = "20"
os.environ["ERROR_KNOWLEDGE_DEDUP_RATIO"] = "0.65"

# Force import after setting env
import importlib
import server as ek  # noqa: E402
importlib.reload(ek)  # reload to pick up env vars


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_index():
    """Clear the test root and index cache before each test."""
    import shutil
    for p in list(TEST_ROOT.rglob("*")):
        if p.is_dir() and p != TEST_ROOT:
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    for d in [ek.ROOT, ek.GENERIC, ek.BUSINESS]:
        d.mkdir(parents=True, exist_ok=True)
    ek._index_cache = None
    ek._index_mtime = 0.0
    ek._index_dirty = False


def _make_record(overrides: dict = None) -> dict:
    record = {
        "title": "Test error",
        "scope": "generic",
        "category": "null_pointer",
        "lang": "python",
        "project": "",
        "date": "2026-05-22",
        "reproduce_steps": "Call foo.bar when foo is None",
        "root_cause": "Missing None check before attribute access",
        "boundary": "Only affects async paths",
        "files": "src/foo.py",
        "fix_summary": "Add `if foo is not None:` guard",
    }
    if overrides:
        record.update(overrides)
    return record


def _write(record: dict) -> Path:
    fp = ek._resolve_path(record)
    fp.write_text(ek._format_markdown(record), encoding="utf-8")
    ek._invalidate_cache()
    return fp


# ── Slug ───────────────────────────────────────────────────────────────────


class TestSlug:
    def test_basic(self):
        assert ek._slug("Hello World") == "hello-world.md"

    def test_unicode(self):
        assert ek._slug("空指针异常") == "空指针异常.md"

    def test_special_chars(self):
        assert ek._slug("C# / .NET Core?") == "c-net-core.md"

    def test_truncation(self):
        long = "a" * 100
        assert len(ek._slug(long)) <= 83  # 80 + ".md"


# ── Parse ──────────────────────────────────────────────────────────────────


class TestParse:
    def test_valid_frontmatter(self):
        f = ek.ROOT / "test.md"
        f.write_text(
            "---\ntitle: Test\ncategory: logic\nroot_cause: Something\nfix_summary: Fix\n---\nBody text",
            encoding="utf-8",
        )
        result = ek._parse(f)
        assert result is not None
        assert result["title"] == "Test"
        assert result["_body"] == "Body text"

    def test_missing_title(self, tmp_path):
        f = tmp_path / "bad.md"
        f.write_text("---\ndate: 2026\n---\nNo title", encoding="utf-8")
        assert ek._parse(f) is None

    def test_no_frontmatter(self, tmp_path):
        f = tmp_path / "bad.md"
        f.write_text("Just text", encoding="utf-8")
        assert ek._parse(f) is None


# ── Dedup ──────────────────────────────────────────────────────────────────


class TestDedup:
    def test_exact_duplicate(self):
        _write(_make_record({"title": "Null check missing in async path"}))
        dup = ek._dedup_check(_make_record({"title": "Null check missing in async path"}))
        assert dup is not None

    def test_substring_duplicate(self):
        _write(_make_record({"title": "Null check missing in async path"}))
        dup = ek._dedup_check(_make_record({"title": "Null check missing"}))
        assert dup is not None

    def test_fuzzy_duplicate(self):
        _write(_make_record({"title": "Null check missing in async path"}))
        # SequenceMatcher ratio for these two should be > 0.65
        dup = ek._dedup_check(_make_record({"title": "Null check absent from async path"}))
        assert dup is not None

    def test_different_scope_no_duplicate(self):
        _write(_make_record({"title": "Some error", "scope": "generic", "lang": "python"}))
        dup = ek._dedup_check(_make_record({"title": "Some error", "scope": "business-specific", "project": "myapp"}))
        assert dup is None

    def test_different_lang_no_duplicate(self):
        _write(_make_record({"title": "Some error", "scope": "generic", "lang": "python"}))
        dup = ek._dedup_check(_make_record({"title": "Some error", "scope": "generic", "lang": "csharp"}))
        assert dup is None

    def test_unrelated_title_no_duplicate(self):
        _write(_make_record({"title": "Database connection timeout"}))
        dup = ek._dedup_check(_make_record({"title": "UI button not rendering"}))
        assert dup is None


# ── Resolve path ───────────────────────────────────────────────────────────


class TestResolvePath:
    def test_generic(self):
        fp = ek._resolve_path(_make_record({"lang": "csharp"}))
        assert "generic" in str(fp)
        assert "csharp" in str(fp)
        assert fp.suffix == ".md"

    def test_business_specific(self):
        fp = ek._resolve_path(_make_record({
            "scope": "business-specific",
            "project": "my-project",
        }))
        assert "business-specific" in str(fp)
        assert "my-project" in str(fp)


# ── Search ─────────────────────────────────────────────────────────────────


class TestSearch:
    def test_empty_db_returns_empty(self):
        results = ek._search_local(keywords="null")
        assert results == []

    def test_keyword_in_title(self):
        _write(_make_record({"title": "Null pointer in login flow"}))
        _write(_make_record({"title": "CSS layout broken"}))
        results = ek._search_local(keywords="null")
        assert len(results) == 1
        assert "Null pointer" in results[0].get("title", "")

    def test_keyword_in_root_cause(self):
        _write(_make_record({
            "title": "Login crash",
            "root_cause": "Missing authentication check",
        }))
        results = ek._search_local(keywords="authentication")
        assert len(results) == 1

    def test_filter_by_lang(self):
        _write(_make_record({"title": "Python error", "lang": "python"}))
        _write(_make_record({"title": "C# error", "lang": "csharp"}))
        results = ek._search_local(keywords="error", lang="python")
        assert len(results) == 1
        assert results[0].get("lang") == "python"

    def test_filter_by_scope(self):
        _write(_make_record({"title": "Generic error", "scope": "generic", "lang": "python"}))
        _write(_make_record({
            "title": "Biz error",
            "scope": "business-specific",
            "project": "myapp",
            "lang": "",
        }))
        results = ek._search_local(keywords="error", scope="generic")
        assert len(results) == 1
        assert results[0].get("scope") == "generic"

    def test_tfidf_ranking_title_higher_than_body(self):
        """Title match should rank higher than body-only match."""
        _write(_make_record({
            "title": "Something about tokens",
            "root_cause": "Other stuff",
        }))
        _write(_make_record({
            "title": "Random error",
            "root_cause": "The tokens issue is here",
        }))
        results = ek._search_local(keywords="tokens")
        assert len(results) == 2
        # Title match should be first (higher score)
        assert "tokens" in results[0].get("title", "").lower()

    def test_no_keywords_returns_all(self):
        _write(_make_record({"title": "Error A"}))
        _write(_make_record({"title": "Error B"}))
        results = ek._search_local(keywords="")
        assert len(results) == 2

    def test_limit(self):
        for i in range(5):
            _write(_make_record({"title": f"Error {i}"}))
        results = ek._search_local(keywords="Error", limit=3)
        assert len(results) == 3


# ── Format markdown ────────────────────────────────────────────────────────


class TestFormatMarkdown:
    def test_roundtrip(self):
        record = _make_record()
        md = ek._format_markdown(record)
        # Parse it back
        fp = ek.ROOT / "roundtrip.md"
        fp.write_text(md, encoding="utf-8")
        parsed = ek._parse(fp)
        assert parsed is not None
        assert parsed["title"] == record["title"]
        assert parsed["root_cause"] == record["root_cause"]

    def test_empty_fields_omitted(self):
        record = _make_record({"root_cause": "", "fix_summary": ""})
        md = ek._format_markdown(record)
        assert "root_cause" not in md
        assert "fix_summary" not in md


# ── Stats ──────────────────────────────────────────────────────────────────


class TestStats:
    def test_counts(self):
        _write(_make_record({"title": "E1", "scope": "generic", "lang": "python"}))
        _write(_make_record({"title": "E2", "scope": "generic", "lang": "python"}))
        _write(_make_record({"title": "E3", "scope": "business-specific", "project": "app"}))
        records = ek._load_index()
        assert len(records) == 3
        generic = sum(1 for r in records if r.get("scope") == "generic")
        biz = sum(1 for r in records if r.get("scope") == "business-specific")
        assert generic == 2
        assert biz == 1


# ── Auto archive ───────────────────────────────────────────────────────────


class TestAutoArchive:
    def test_moves_flat_files(self):
        # Write files at root level (not in subdirectories)
        for i in range(3):
            rec = _make_record({"title": f"Err {i}", "lang": "python"})
            # write directly to root, bypassing _resolve_path
            fp = ek.ROOT / f"err_{i}.md"
            fp.write_text(ek._format_markdown(rec), encoding="utf-8")

        ek._auto_archive()
        # After archiving, files should be in generic/python/
        archived = list((ek.GENERIC / "python").glob("*.md"))
        assert len(archived) == 3
