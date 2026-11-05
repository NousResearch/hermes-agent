"""Tests for the messages_fts_cjk CJK-bigram index (salvaged from PR #65544).

Builds the loadable tokenizer from native/fts5_cjk/fts5_cjk.c on the fly;
skips when no C toolchain / extension loading is available.
"""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import FTS_CJK_STALE_KEY
from persistence.sqlite.admin._maintenance import _FtsStorageOptimizer
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import _describe_search_path, _rebuild_status, search_messages
from persistence.sqlite.conversation._store.sessions import create_session

REPO = Path(__file__).resolve().parent.parent.parent.parent
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
    d = StoreHolder(db_path=tmp_path / "state.db")
    assert d._fts_cjk_loaded, "tokenizer must load on the writer connection"
    assert d._fts_cjk_available, "fresh DB must be born with the cjk index"
    create_session(d, session_id="s1", source="cli", model="m")
    append_message(d, "s1", role="user", content="웅기가 shared default 프로필을 요청했다")
    append_message(d, "s1", role="assistant", content="일본 MCP 후보 우선순위 정리했습니다")
    append_message(d, "s1", role="user", content="graphiti daemon looks healthy")
    append_message(d, "s1", role="tool", content="일본 tool output blob", tool_name="terminal")
    yield d
    d.close()


def test_two_char_korean_hits_cjk_index(db):
    rows = search_messages(db, "웅기", limit=10)
    assert rows and "웅기" in rows[0]["snippet"]
    rows = search_messages(db, "일본", limit=10)
    assert rows


def test_mixed_and_ascii_queries(db):
    assert search_messages(db, "graphiti", limit=10)
    assert search_messages(db, '"shared default" AND 웅기', limit=10)
    assert search_messages(db, "우선순위", limit=10)




def test_lone_single_cjk_char_routes_like(db):
    # 1-char CJK terms keep LIKE substring semantics (bigram index only
    # holds unigrams for isolated chars). "가" appears inside 웅기가.
    assert _describe_search_path(db, "가") == "like_scan"
    rows = search_messages(db, "가", limit=10)
    assert rows, "LIKE fallback must still find substring matches"








def test_config_toggle_disables_cjk(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    monkeypatch.setenv("HERMES_CJK_FTS", "0")
    d = StoreHolder(db_path=tmp_path / "state.db")
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






def test_existing_v23_db_gains_cjk_via_optimize(cjk_so, tmp_path, monkeypatch):
    """A v23 DB created BEFORE the extension existed: next capable open
    creates the index with backfill markers; optimize-storage backfills."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "absent.so"))
    db_path = tmp_path / "state.db"
    d1 = StoreHolder(db_path=db_path)
    create_session(d1, session_id="s1", source="cli", model="m")
    for i in range(10):
        append_message(d1, "s1", role="user", content=f"기존 메시지 {i}")
    d1.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d2 = StoreHolder(db_path=db_path)
    assert d2._fts_cjk_loaded
    # Backfill pending — index not served yet, old rows not indexed.
    assert not d2._fts_cjk_available
    st = _rebuild_status(d2, "fts_cjk_rebuild")
    assert st is not None and st["pending"]
    assert _FtsStorageOptimizer(d2).fts_optimize_available()
    # NEW rows are indexed live by the id-gated triggers even mid-backfill.
    append_message(d2, "s1", role="user", content="새로운 메시지")
    # Search answers via legacy routes meanwhile.
    assert search_messages(d2, "기존", limit=10)

    result = _FtsStorageOptimizer(d2).optimize_fts_storage(vacuum=False)
    assert result["ok"]
    assert d2._fts_cjk_available
    assert _rebuild_status(d2, "fts_cjk_rebuild") is None
    assert _describe_search_path(d2, "기존") == "fts_cjk"
    rows = search_messages(d2, "기존", limit=20)
    assert len(rows) == 10
    assert search_messages(d2, "새로운", limit=10)
    d2.close()


def test_legacy_v22_optimize_lands_on_cjk(cjk_so, tmp_path, monkeypatch):
    """A legacy inline-FTS (pre-v23) DB optimized on a tokenizer-capable
    host comes out with BOTH the v23 external-content layout AND a complete
    cjk index in the same run."""
    import time as _time

    from persistence.sqlite._schema_ddl import SCHEMA_SQL

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

    d = StoreHolder(db_path=db_path)
    try:
        assert _FtsStorageOptimizer(d).fts_optimize_available()
        # Legacy DB: cjk index deliberately not created at open (the legacy
        # branch of _init_schema doesn't touch v23 surfaces).
        assert not d._fts_cjk_available

        result = _FtsStorageOptimizer(d).optimize_fts_storage(vacuum=False)
        assert result["ok"]
        assert d._fts_cjk_available
        assert _rebuild_status(d, "fts_cjk_rebuild") is None
        assert _describe_search_path(d, "일본") == "fts_cjk"
        assert search_messages(d, "일본", limit=10)
        assert search_messages(d, "legacy english", limit=10)
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


def test_optimize_demote_leaves_established_cjk_index_intact(cjk_so, tmp_path, monkeypatch):
    """#103647: a legacy inline-FTS DB that ALSO carries an established (backfilled,
    trigger-live) messages_fts_cjk index must demote without touching the cjk family.
    The demote enumeration matches 'messages_fts_%', which sweeps the cjk vtable and
    its shadow tables into the fts_v22_trash_* renames; renaming them breaks the
    vtable constructor chain ('vtable constructor failed: messages_fts_cjk') and
    aborts optimize-storage on every CJK-enabled host before any space is reclaimed."""
    import time as _time

    from persistence.sqlite._schema_ddl import SCHEMA_SQL
    from persistence.sqlite._schema_ddl import FTS_CJK_TABLE_SQL, FTS_CJK_TRIGGER_SQL

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "state.db"

    # Hand-build the coexistence shape: legacy inline FTS + a live cjk index.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
        conn.load_extension(str(cjk_so))
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
        # Established cjk index: tables + triggers, no backfill markers (complete).
        conn.executescript(FTS_CJK_TABLE_SQL)
        conn.executescript(FTS_CJK_TRIGGER_SQL)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (10)")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
            (_time.time(),),
        )
        for role, content in (
            ("user", "레거시 일본 메시지"),
            ("assistant", "legacy english reply"),
        ):
            conn.execute(
                "INSERT INTO messages (session_id, timestamp, role, content) "
                "VALUES ('s1', ?, ?, ?)",
                (_time.time(), role, content),
            )
        conn.commit()
    finally:
        conn.close()

    d = StoreHolder(db_path=db_path)
    try:
        assert _FtsStorageOptimizer(d).fts_optimize_available(), "legacy inline layout must be eligible"
        assert d._fts_cjk_loaded
        with d._lock:
            cjk_triggers = d._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'messages_fts_cjk_%'"
            ).fetchone()[0]
        assert cjk_triggers == 3, "cjk index is established and trigger-live"
        result = _FtsStorageOptimizer(d).optimize_fts_storage(vacuum=False)
        assert result["ok"]
        # The cjk index survives demote untouched: same tables, same service path.
        assert d._fts_cjk_available
        assert _rebuild_status(d, "fts_cjk_rebuild") is None
        assert _describe_search_path(d, "일본") == "fts_cjk"
        assert search_messages(d, "일본", limit=10)
        with d._lock:
            trash_cjk = d._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE name LIKE 'fts_v22_trash_messages_fts_cjk%'"
            ).fetchone()[0]
        assert trash_cjk == 0, "no cjk table may be renamed into the trash family"
    finally:
        d.close()


def test_pure_latin_embedded_in_cjk_recovered_via_cjk_index(db):
    """#54242 residual: a pure-Latin query for a token embedded in CJK text
    (no whitespace) misses on unicode61; with the cjk index available the
    zero-result fallback recovers it as an exact ranked token match."""
    append_message(db, "s1", role="user", content="修改youer服务端的계획")
    rows = search_messages(db, "youer", limit=10)
    assert rows and "youer" in rows[0]["snippet"]
    # Short tokens (<3 chars, no trigram) are also recoverable via cjk.
    append_message(db, "s1", role="user", content="에러코드ab확인")
    rows = search_messages(db, "ab", limit=10)
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
    append_message(db, "s1", role="user", content="무결성 검사")
    with db._lock:
        db._conn.execute(
            "INSERT INTO messages_fts_cjk(messages_fts_cjk) "
            "VALUES('integrity-check')"
        )
