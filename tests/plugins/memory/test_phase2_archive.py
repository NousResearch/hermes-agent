"""Tests for Phase 2 — L3 conversation archive indexing (in-place, no migration).

Hermetic: HERMES_HOME redirected to tmp_path. We write synthetic session JSONL
files under <home>/sessions/ and assert the indexer discovers, chunks, tags,
and enriches them — without mutating the raw files. No summarization, no
extraction, no embeddings: the indexed content equals the raw line content.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.memory_index.indexer import MemoryIndex
from hermes_cli.memory_router.provenance import SearchResult


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "memories").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Patch BOTH bindings: the module attribute and the name indexer.py bound
    # at import time (`from hermes_constants import get_hermes_home`). The
    # Router's IndexCapability builds a default MemoryIndex() that resolves via
    # the latter, so only patching the module attribute leaves it pointing at
    # the real HERMES_HOME (non-hermetic). We also redirect the capability's
    # MemoryIndex so the Router exercises the temp home end-to-end.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    import hermes_cli.memory_index.indexer as _idx
    import hermes_cli.memory_index.capability as _cap

    monkeypatch.setattr(_idx, "get_hermes_home", lambda: home)
    monkeypatch.setattr(
        _cap,
        "MemoryIndex",
        lambda *a, **k: _idx.MemoryIndex(*a, **{kk: vv for kk, vv in k.items() if kk != "hermes_home"}, hermes_home=home),
    )
    yield home


def _write_jsonl(home, rel, events):
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return p


def _session_events(topic_word):
    return [
        {"role": "session_meta", "tools": [], "model": "x"},  # skipped
        {"role": "user", "content": f"Let's discuss the {topic_word} design.", "ts": "2026-07-01T10:00:00Z"},
        {"role": "assistant", "content": f"The {topic_word} protocol uses FTS5.", "ts": "2026-07-01T10:00:05Z"},
        {"role": "system", "content": f"System note about {topic_word}.", "ts": "2026-07-01T10:00:10Z"},
        {"role": "tool", "content": f"tool output for {topic_word}", "ts": "2026-07-01T10:00:15Z"},
    ]


def test_archive_discovery_tags_L3(hermes_home):
    _write_jsonl(hermes_home, "sessions/20260701_aaa.jsonl", _session_events("widget"))
    idx = MemoryIndex(hermes_home=hermes_home)
    n = idx.build(hermes_home)
    assert n > 0
    layers = {r.memory_layer for r in idx.search("widget")}
    assert "L3-archive" in layers


def test_archive_one_row_per_chat_line(hermes_home):
    evs = _session_events("gizmo")
    p = _write_jsonl(hermes_home, "sessions/20260702_bbb.jsonl", evs)
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    # session_meta skipped (1) -> 4 chat lines should become 4 rows.
    import sqlite3

    conn = sqlite3.connect(str(idx.db_path))
    count = conn.execute(
        "SELECT count(*) FROM notes WHERE memory_layer='L3-archive'"
    ).fetchone()[0]
    conn.close()
    assert count == 4, f"expected 4 chat rows, got {count}"
    # Raw file line count unchanged (5 lines: 1 meta + 4 chat).
    assert len(p.read_text().splitlines()) == 5


def test_archive_session_meta_skipped(hermes_home):
    evs = _session_events("thing")
    _write_jsonl(hermes_home, "sessions/20260703_ccc.jsonl", evs)
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    # 'session_meta' is not a retrievable role; 'tools' content must not appear.
    res = idx.search("tools")
    assert all(r.memory_layer != "L3-archive" or "tools" not in r.content for r in res)


def test_archive_provenance_enrichment(hermes_home):
    _write_jsonl(hermes_home, "sessions/20260704_ddd.jsonl", _session_events("alpha"))
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    res = idx.search("alpha")
    arc = [r for r in res if r.memory_layer == "L3-archive"]
    assert arc, "expected an L3 archive hit for 'alpha'"
    r = arc[0]
    assert isinstance(r.extra, dict)
    assert r.extra.get("session_id") == "20260704_ddd"
    assert r.extra.get("role") in ("user", "assistant", "system", "tool")
    assert "chunk_index" in r.extra
    # event_ts preserved from the line when present.
    assert r.extra.get("event_ts") is not None
    # Optional fields may be present or absent; if present they are strings.
    for k in ("project_context", "hermes_version", "git_commit", "working_directory"):
        if k in r.extra:
            assert isinstance(r.extra[k], str)


def test_historical_intent_includes_archive(hermes_home):
    # HISTORICAL is the catch-all intent; L3 rows share the unified notes
    # table, so a broad historical search must surface archive hits.
    from hermes_cli.memory_router.router import MemoryRouter
    from hermes_cli.memory_index.indexer import MemoryIndex

    _write_jsonl(hermes_home, "sessions/20260705_eee.jsonl", _session_events("omega"))
    # Build the index the router will read (same temp home).
    MemoryIndex(hermes_home=hermes_home).build(hermes_home)
    router = MemoryRouter()
    rr = router.search("omega")
    assert rr.ok
    assert any(r.memory_layer == "L3-archive" for r in rr.results)


def test_archive_no_content_interpretation(hermes_home):
    _write_jsonl(
        hermes_home,
        "sessions/20260706_fff.jsonl",
        [{"role": "user", "content": "Revenue was 4.2 million in Q3.", "ts": "2026-07-06T09:00:00Z"}],
    )
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    r = idx.search("Revenue")[0]
    assert "4.2 million" in r.content  # raw content, not a summary


def test_archive_fts5_fallback(monkeypatch, hermes_home):
    _write_jsonl(hermes_home, "sessions/20260707_ggg.jsonl", _session_events("delta"))
    monkeypatch.setattr("hermes_cli.memory_index.indexer._FTS5_ENABLED", False)
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    res = idx.search("delta")
    assert res
    for r in res:
        assert r.retrieval_method == "sqlite-like"
        assert "delta" in r.content.lower()


def test_archive_deterministic(hermes_home):
    import hashlib
    import sqlite3

    evs = _session_events("kappa")
    _write_jsonl(hermes_home, "sessions/20260708_hhh.jsonl", evs)

    db_a = hermes_home / "memory" / "a.db"
    db_b = hermes_home / "memory" / "b.db"
    a = MemoryIndex(db_path=db_a, hermes_home=hermes_home)
    b = MemoryIndex(db_path=db_b, hermes_home=hermes_home)
    a.build(hermes_home)
    b.build(hermes_home)

    def fp(db):
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT source_file, content FROM notes WHERE memory_layer='L3-archive' "
            "ORDER BY source_file, content"
        ).fetchall()
        conn.close()
        return len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()

    na, ha = fp(db_a)
    nb, hb = fp(db_b)
    assert na == nb and ha == hb


def test_raw_archive_not_mutated(hermes_home):
    evs = _session_events("zeta")
    p = _write_jsonl(hermes_home, "sessions/20260709_iii.jsonl", evs)
    before = p.read_bytes()
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    after = p.read_bytes()
    assert before == after, "indexing must not mutate the raw archive file"


def test_archive_and_markdown_coexist(hermes_home):
    _write_jsonl(hermes_home, "sessions/20260710_jjj.jsonl", _session_events("neutron"))
    (hermes_home / "memories" / "MEMORY.md").write_text(
        "# Memory\nNeutron stars are dense.\n", encoding="utf-8"
    )
    idx = MemoryIndex(hermes_home=hermes_home)
    idx.build(hermes_home)
    layers = {r.memory_layer for r in idx.search("neutron")}
    assert "L3-archive" in layers
    assert "L1-identity" in layers
