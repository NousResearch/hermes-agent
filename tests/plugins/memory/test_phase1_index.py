"""Tests for the Phase 1 Layer 5 index (SQLite + FTS5 over markdown).

Hermetic: HERMES_HOME redirected to tmp_path by the autouse fixture. We ALSO
monkeypatch ``hermes_constants.get_hermes_home`` to point at tmp_path so the
indexer's default path resolves inside the tempdir and never touches the real
~/.hermes. The FTS5 fallback is exercised by forcing ``_fts5_enabled = False``
via monkeypatch on the indexer module.
"""

from __future__ import annotations

import hashlib

import pytest

from hermes_cli.memory_index.indexer import MemoryIndex
from hermes_cli.memory_router.provenance import SearchResult


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "memories").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    return home


def _write(home, rel, text):
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_build_creates_index_db(hermes_home):
    _write(hermes_home, "memories/MEMORY.md", "# Memory\nJoe likes cats.\n")
    idx = MemoryIndex(hermes_home=hermes_home)
    n = idx.build(hermes_home)
    assert n > 0
    assert idx.db_path.exists()
    assert idx.available()


def test_search_finds_content(hermes_home):
    _write(
        hermes_home,
        "memories/MEMORY.md",
        "# Memory\nJoe's wife Tanya enjoys classical piano music.\n",
    )
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    results = idx.search("Tanya")
    assert results, "expected at least one result for 'Tanya'"
    r = results[0]
    assert isinstance(r, SearchResult)
    assert "MEMORY.md" in r.source_file
    assert r.retrieval_method in ("fts5", "sqlite-like")
    assert r.memory_layer
    assert "Tanya" in r.content


def test_search_does_not_interpret(hermes_home):
    _write(
        hermes_home,
        "memories/MEMORY.md",
        "The quarterly revenue was 4.2 million dollars in Q3.\n",
    )
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    results = idx.search("revenue")
    assert results
    # Raw content returned, not a summary/LLM output.
    assert "4.2 million" in results[0].content
    assert results[0].retrieval_method in ("fts5", "sqlite-like")


def test_rebuild_deterministic(hermes_home):
    text = (
        "---\ntitle: x\n---\n"
        "Alpha project started in 2024.\n\n"
        "Beta module handles billing logic.\n\n"
        "Gamma is the archival system.\n"
    )
    _write(hermes_home, "memories/MEMORY.md", text)
    _write(hermes_home, "HERMES_PROJECTS.md", "# Projects\nDelta is the gateway.\n")

    db_a = hermes_home / "memory" / "a.db"
    db_b = hermes_home / "memory" / "b.db"

    a = MemoryIndex(db_path=db_a, hermes_home=hermes_home)
    b = MemoryIndex(db_path=db_b, hermes_home=hermes_home)
    a.build(hermes_home)
    b.build(hermes_home)

    def fingerprint(db):
        import sqlite3

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT source_file, content FROM notes ORDER BY source_file, content"
        ).fetchall()
        conn.close()
        payload = repr(rows).encode("utf-8")
        return len(rows), hashlib.sha256(payload).hexdigest()

    na, ha = fingerprint(db_a)
    nb, hb = fingerprint(db_b)
    assert na == nb
    assert ha == hb


def test_fts5_fallback(monkeypatch, hermes_home):
    _write(
        hermes_home,
        "memories/MEMORY.md",
        "Tanya enjoys classical piano music on weekends.\n",
    )
    idx = MemoryIndex(hermes_home=hermes_home)
    # Force FTS5 to be treated as unavailable at create time.
    monkeypatch.setattr("hermes_cli.memory_index.indexer._FTS5_ENABLED", False)
    n = idx.build(hermes_home)
    assert n > 0
    results = idx.search("Tanya")
    assert results
    for r in results:
        assert r.retrieval_method == "sqlite-like"
        assert "Tanya" in r.content


def test_index_empty_when_no_sources(hermes_home):
    idx = MemoryIndex(hermes_home=hermes_home)
    n = idx.build(hermes_home)
    assert n == 0
    assert idx.search("anything") == []


def test_index_discovers_l1_and_special_files(hermes_home):
    _write(hermes_home, "SOUL.md", "I am Hermes, an AI agent.\n")
    _write(hermes_home, "memories/USER.md", "Joe is the user.\n")
    _write(hermes_home, "memories/MEMORY.md", "Fact: the sky is blue.\n")
    _write(hermes_home, "HERMES_PROJECTS.md", "Project Atlas is active.\n")
    _write(hermes_home, "HERMES_SESSION.md", "Last session fixed a bug.\n")
    idx = MemoryIndex(hermes_home=hermes_home)
    n = idx.build(hermes_home)
    assert n > 0
    # Each term hits a different layer; verify discovery covers all sources.
    l1 = {r.memory_layer for r in idx.search("Hermes")}
    projects = {r.memory_layer for r in idx.search("Atlas")}
    assert "L1-identity" in l1
    assert "L5-projects" in projects
