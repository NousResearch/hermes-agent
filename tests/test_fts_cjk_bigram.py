"""Tests for the messages_fts_cjk CJK-bigram index (salvaged from PR #65544).

Builds the loadable tokenizer from native/fts5_cjk/fts5_cjk.c on the fly;
skips when no C toolchain / extension loading is available.
"""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_state import FTS_CJK_STALE_KEY, SessionDB

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "native" / "fts5_cjk" / "fts5_cjk.c"
VENDOR = REPO / "native" / "fts5_cjk" / "vendor"


@pytest.fixture(scope="session")
def cjk_so(tmp_path_factory):
    if shutil.which("gcc") is None or not SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    out = tmp_path_factory.mktemp("fts5cjk") / "libfts5_cjk.so"
    try:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", f"-I{VENDOR}", str(SRC),
             "-o", str(out)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(f"tokenizer build failed: {e.stderr[:200]}")
    # Loadability probe (extension loading may be disabled in this build).
    probe = sqlite3.connect(":memory:")
    try:
        probe.enable_load_extension(True)
        probe.load_extension(str(out))
    except Exception as e:
        pytest.skip(f"extension loading unavailable: {e}")
    finally:
        probe.close()
    return out


@pytest.fixture()
def db(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d = SessionDB(db_path=tmp_path / "state.db")
    assert d._fts_cjk_loaded, "tokenizer must load on the writer connection"
    assert d._fts_cjk_available, "fresh DB must be born with the cjk index"
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="웅기가 shared default 프로필을 요청했다")
    d.append_message("s1", role="assistant", content="일본 MCP 후보 우선순위 정리했습니다")
    d.append_message("s1", role="user", content="graphiti daemon looks healthy")
    d.append_message("s1", role="tool", content="일본 tool output blob", tool_name="terminal")
    yield d
    d.close()


def test_two_char_korean_hits_cjk_index(db):
    rows = db.search_messages("웅기", limit=10)
    assert rows and "웅기" in rows[0]["snippet"]
    rows = db.search_messages("일본", limit=10)
    assert rows


def test_mixed_and_ascii_queries(db):
    assert db.search_messages("graphiti", limit=10)
    assert db.search_messages('"shared default" AND 웅기', limit=10)
    assert db.search_messages("우선순위", limit=10)


def test_no_false_positive_across_words(db):
    # 기가/했다 exist inside runs; a bigram crossing a word boundary must not.
    assert db.search_messages("다프로", limit=10) == []


def test_lone_single_cjk_char_routes_like(db):
    # 1-char CJK terms keep LIKE substring semantics (bigram index only
    # holds unigrams for isolated chars). "가" appears inside 웅기가.
    assert db._describe_search_path("가") == "like_scan"
    rows = db.search_messages("가", limit=10)
    assert rows, "LIKE fallback must still find substring matches"


def test_tool_role_filter_routes_like(db):
    # Tool rows are excluded from the cjk index; role_filter=['tool'] CJK
    # queries take the LIKE route and still find tool output.
    rows = db.search_messages("일본", role_filter=["tool"], limit=10)
    assert rows and all(r["role"] == "tool" for r in rows)


def test_triggers_mirror_updates_and_deletes(db):
    db.append_message("s1", role="user", content="자바스크립트 리팩토링")
    assert db.search_messages("리팩토링", limit=10)
    with db._lock:
        db._conn.execute(
            "UPDATE messages SET content = '파이썬 리라이트' WHERE content LIKE '%리팩토링%'"
        )
        db._conn.commit()
    assert db.search_messages("리팩토링", limit=10) == []
    assert db.search_messages("리라이트", limit=10)
    with db._lock:
        db._conn.execute("DELETE FROM messages WHERE content = '파이썬 리라이트'")
        db._conn.commit()
    assert db.search_messages("리라이트", limit=10) == []


def test_rewound_rows_hidden_from_cjk_search(db):
    db.append_message("s1", role="user", content="되돌리기 대상 메시지")
    assert db.search_messages("되돌리기", limit=10)
    with db._lock:
        db._conn.execute(
            "UPDATE messages SET active = 0, compacted = 0 "
            "WHERE content LIKE '%되돌리기%'"
        )
        db._conn.commit()
    assert db.search_messages("되돌리기", limit=10) == []
    assert db.search_messages("되돌리기", include_inactive=True, limit=10)


def test_config_toggle_disables_cjk(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    monkeypatch.setenv("HERMES_CJK_FTS", "0")
    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert not d._fts_cjk_loaded
        assert not d._fts_cjk_available
        # No cjk objects created at all.
        with d._lock:
            row = d._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_cjk'"
            ).fetchone()
        assert row is None
    finally:
        d.close()


def test_no_extension_no_cjk_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "nonexistent.so"))
    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert not d._fts_cjk_loaded
        assert not d._fts_cjk_available
        d.create_session(session_id="s1", source="cli", model="m")
        d.append_message("s1", role="user", content="일본 MCP 정리")
        # Trigram/LIKE routing still answers.
        assert d.search_messages("일본", limit=10)
    finally:
        d.close()


def test_tokenizer_loss_self_heals_and_optimize_rebuilds(cjk_so, tmp_path, monkeypatch):
    """Full stale lifecycle: capable open → tokenizer-less open (drops
    triggers, breadcrumbs) → rows written in the gap → capable open again
    (index NOT served) → optimize-storage rebuilds → search complete."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "state.db"

    d1 = SessionDB(db_path=db_path)
    assert d1._fts_cjk_available
    d1.create_session(session_id="s1", source="cli", model="m")
    d1.append_message("s1", role="user", content="첫번째 메시지")
    d1.close()

    # Tokenizer-less open: triggers dropped, breadcrumb set, writes fine.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "gone.so"))
    d2 = SessionDB(db_path=db_path)
    assert not d2._fts_cjk_loaded
    assert not d2._fts_cjk_available
    d2.append_message("s1", role="user", content="틈새에 쓰인 메시지")
    assert d2.get_meta(FTS_CJK_STALE_KEY) == "1"
    with d2._lock:
        trigs = d2._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE 'messages_fts_cjk%'"
        ).fetchone()[0]
    assert trigs == 0
    # Search still answers via trigram/LIKE.
    assert d2.search_messages("틈새", limit=10)
    d2.close()

    # Capable open again: stale index must NOT be served.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d3 = SessionDB(db_path=db_path)
    assert d3._fts_cjk_loaded
    assert not d3._fts_cjk_available, "stale index must not serve reads"
    assert d3.fts_optimize_available(), "optimize must offer the rebuild"
    # Search still complete through the legacy routes meanwhile.
    assert d3.search_messages("틈새", limit=10)

    result = d3.optimize_fts_storage(vacuum=False)
    assert result["ok"]
    assert d3._fts_cjk_available
    assert d3.get_meta(FTS_CJK_STALE_KEY) is None
    assert d3.fts_cjk_rebuild_status() is None
    # Both the pre-gap and in-gap rows are now searchable via the index.
    assert d3._describe_search_path("틈새") == "fts_cjk"
    assert d3.search_messages("첫번째", limit=10)
    assert d3.search_messages("틈새", limit=10)
    d3.close()


def test_existing_v23_db_gains_cjk_via_optimize(cjk_so, tmp_path, monkeypatch):
    """A v23 DB created BEFORE the extension existed: next capable open
    creates the index with backfill markers; optimize-storage backfills."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "absent.so"))
    db_path = tmp_path / "state.db"
    d1 = SessionDB(db_path=db_path)
    d1.create_session(session_id="s1", source="cli", model="m")
    for i in range(10):
        d1.append_message("s1", role="user", content=f"기존 메시지 {i}")
    d1.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d2 = SessionDB(db_path=db_path)
    assert d2._fts_cjk_loaded
    # Backfill pending — index not served yet, old rows not indexed.
    assert not d2._fts_cjk_available
    st = d2.fts_cjk_rebuild_status()
    assert st is not None and st["pending"]
    assert d2.fts_optimize_available()
    # NEW rows are indexed live by the id-gated triggers even mid-backfill.
    d2.append_message("s1", role="user", content="새로운 메시지")
    # Search answers via legacy routes meanwhile.
    assert d2.search_messages("기존", limit=10)

    result = d2.optimize_fts_storage(vacuum=False)
    assert result["ok"]
    assert d2._fts_cjk_available
    assert d2.fts_cjk_rebuild_status() is None
    assert d2._describe_search_path("기존") == "fts_cjk"
    rows = d2.search_messages("기존", limit=20)
    assert len(rows) == 10
    assert d2.search_messages("새로운", limit=10)
    d2.close()


def test_legacy_v22_optimize_lands_on_cjk(cjk_so, tmp_path, monkeypatch):
    """A legacy inline-FTS (pre-v23) DB optimized on a tokenizer-capable
    host comes out with BOTH the v23 external-content layout AND a complete
    cjk index in the same run."""
    import time as _time

    from hermes_state import SCHEMA_SQL

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "state.db"

    # Hand-build a genuine legacy inline DB (single-column messages_fts).
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.executescript("""
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;
        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, COALESCE(new.content,''));
        END;
    """)
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (10)")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
        (_time.time(),),
    )
    for role, content in (
        ("user", "레거시 일본 메시지"),
        ("assistant", "legacy english reply"),
        ("tool", "레거시 tool output"),
    ):
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content) "
            "VALUES ('s1', ?, ?, ?)",
            (_time.time(), role, content),
        )
    conn.commit()
    conn.close()

    d = SessionDB(db_path=db_path)
    try:
        assert d.fts_optimize_available()
        # Legacy DB: cjk index deliberately not created at open (the legacy
        # branch of _init_schema doesn't touch v23 surfaces).
        assert not d._fts_cjk_available

        result = d.optimize_fts_storage(vacuum=False)
        assert result["ok"]
        assert d._fts_cjk_available
        assert d.fts_cjk_rebuild_status() is None
        assert d._describe_search_path("일본") == "fts_cjk"
        assert d.search_messages("일본", limit=10)
        assert d.search_messages("legacy english", limit=10)
        with d._lock:
            idx = d._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_cjk"
            ).fetchone()[0]
            non_tool = d._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
            ).fetchone()[0]
        assert idx == non_tool
    finally:
        d.close()


def test_pure_latin_embedded_in_cjk_recovered_via_cjk_index(db):
    """#54242 residual: a pure-Latin query for a token embedded in CJK text
    (no whitespace) misses on unicode61; with the cjk index available the
    zero-result fallback recovers it as an exact ranked token match."""
    db.append_message("s1", role="user", content="修改youer服务端的계획")
    rows = db.search_messages("youer", limit=10)
    assert rows and "youer" in rows[0]["snippet"]
    # Short tokens (<3 chars, no trigram) are also recoverable via cjk.
    db.append_message("s1", role="user", content="에러코드ab확인")
    rows = db.search_messages("ab", limit=10)
    assert rows


def test_fresh_db_index_counts_exclude_tool_rows(db):
    with db._lock:
        idx = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts_cjk"
        ).fetchone()[0]
        non_tool = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
        ).fetchone()[0]
    assert idx == non_tool


def test_integrity_after_lifecycle(db):
    db.append_message("s1", role="user", content="무결성 검사")
    with db._lock:
        db._conn.execute(
            "INSERT INTO messages_fts_cjk(messages_fts_cjk) "
            "VALUES('integrity-check')"
        )
