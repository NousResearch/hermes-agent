"""Tests for the messages_fts_v2 CJK-bigram index (patch fts5-cjk-bigram-index).

Builds the loadable tokenizer from native/fts5_cjk/fts5_cjk.c on the fly;
skips when no C toolchain / sqlite3ext.h is available.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_state
from hermes_state import FTS_V2_SQL, SessionDB

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "native" / "fts5_cjk" / "fts5_cjk.c"


@pytest.fixture(scope="session")
def cjk_so(tmp_path_factory):
    if shutil.which("gcc") is None or not SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    out = tmp_path_factory.mktemp("fts5cjk") / "libfts5_cjk.so"
    try:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", str(SRC), "-o", str(out)],
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
    monkeypatch.setenv("HERMES_FTS_V2_READ", "1")
    d = SessionDB(db_path=tmp_path / "state.db")
    assert d._fts_cjk_loaded, "tokenizer must load on the writer connection"
    # migration step: table + triggers (idempotent DDL from the module)
    with d._lock:
        d._conn.executescript(FTS_V2_SQL)
    d._fts_v2_ready = d._probe_fts_v2()
    assert d._fts_v2_ready
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="웅기가 shared default 프로필을 요청했다")
    d.append_message("s1", role="assistant", content="일본 MCP 후보 우선순위 정리했습니다")
    d.append_message("s1", role="user", content="graphiti daemon looks healthy")
    yield d
    d.close()


def test_two_char_korean_hits_v2(db):
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


def test_lone_single_cjk_char_routes_legacy(db, monkeypatch):
    called = {"v2": 0}
    orig = db._query_fts_v2

    def spy(*a, **k):
        called["v2"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(db, "_query_fts_v2", spy)
    db.search_messages("가", limit=10)
    assert called["v2"] == 0, "1-char CJK query must stay on the legacy path"


def test_read_flag_off_uses_legacy(db, monkeypatch):
    monkeypatch.setenv("HERMES_FTS_V2_READ", "0")
    # trigram (>=3 chars) still answers on the legacy path
    assert db.search_messages("우선순위", limit=10)


def test_triggers_mirror_updates_and_deletes(db):
    db.append_message("s1", role="user", content="자바스크립트 리팩토링")
    assert db.search_messages("리팩토링", limit=10)
    with db._lock:
        db._conn.execute(
            "UPDATE messages SET content = '파이썬 리라이트' WHERE content LIKE '%리팩토링%'"
        )
    assert db.search_messages("리팩토링", limit=10) == []
    assert db.search_messages("리라이트", limit=10)
    with db._lock:
        db._conn.execute("DELETE FROM messages WHERE content = '파이썬 리라이트'")
    assert db.search_messages("리라이트", limit=10) == []


def test_flag_only_update_does_not_fire_v2_trigger(db):
    """active-flag flips must not re-tokenize (AFTER UPDATE OF scoping)."""
    rows = db.search_messages("웅기", limit=10)
    mid = rows[0]["id"]
    with db._lock:
        before = db._conn.execute(
            "SELECT count(*) FROM messages_fts_v2 WHERE rowid = ?", (mid,)
        ).fetchone()[0]
        db._conn.execute("UPDATE messages SET active = 1 WHERE id = ?", (mid,))
        after = db._conn.execute(
            "SELECT count(*) FROM messages_fts_v2 WHERE rowid = ?", (mid,)
        ).fetchone()[0]
    assert before == after == 1


def test_self_heal_drops_triggers_without_tokenizer(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d = SessionDB(db_path=tmp_path / "heal.db")
    with d._lock:
        d._conn.executescript(FTS_V2_SQL)
    d.close()

    # Reopen WITHOUT the extension: triggers must be dropped, writes must work.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "missing.so"))
    d2 = SessionDB(db_path=tmp_path / "heal.db")
    try:
        assert not d2._fts_cjk_loaded
        assert not d2._fts_v2_ready
        with d2._lock:
            trig = d2._conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'messages_fts_v2%'"
            ).fetchone()[0]
        assert trig == 0
        d2.create_session(session_id="w", source="cli", model="m")
        d2.append_message("w", role="user", content="writes still work")
    finally:
        d2.close()


def test_fresh_db_is_v2_native(cjk_so, tmp_path, monkeypatch):
    """A fresh DB with a loadable tokenizer runs on v2 alone (no v1 tables)."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    monkeypatch.delenv("HERMES_FTS_V2_READ", raising=False)
    d = SessionDB(db_path=tmp_path / "plain.db")
    try:
        assert d._fts_v2_ready  # empty DB → ready marker set at init
        assert not d._fts_v1_present
        with d._lock:
            v1 = d._conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name IN "
                "('messages_fts','messages_fts_trigram')"
            ).fetchone()[0]
        assert v1 == 0
        d.create_session(session_id="p", source="cli", model="m")
        d.append_message("p", role="user", content="일본 MCP 정리")
        assert d.search_messages("일본 MCP", limit=5)  # default-on v2
        # With no v1 fallback, the off-flag must NOT blind search.
        monkeypatch.setenv("HERMES_FTS_V2_READ", "0")
        assert d.search_messages("일본 MCP", limit=5)
    finally:
        d.close()


def test_no_tokenizer_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "missing.so"))
    d = SessionDB(db_path=tmp_path / "legacy.db")
    try:
        assert not d._fts_v2_ready
        d.create_session(session_id="p", source="cli", model="m")
        d.append_message("p", role="user", content="일본 MCP 정리")
        assert d.search_messages("일본 MCP", limit=5)  # legacy trigram/LIKE
    finally:
        d.close()


def _run_migrate(db_path, so_path):
    """Run scripts/fts_v2_migrate.py against db_path with the tokenizer."""
    env = dict(os.environ, HERMES_FTS5_CJK_SO=str(so_path))
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fts_v2_migrate.py"),
         "--db", str(db_path)],
        capture_output=True, text=True, env=env,
    )


def test_trigger_drop_invalidates_marker_until_verified_backfill(
    cjk_so, tmp_path, monkeypatch
):
    """Rows written while a tokenizer-less process had dropped the triggers
    must never be silently missing from a served v2 index: the drop durably
    invalidates the ready marker, an extension-capable reopen must NOT serve
    v2 until a verified backfill completes, and afterwards the gap rows are
    indexed."""
    db_path = tmp_path / "gap.db"
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d = SessionDB(db_path=db_path)
    assert d._fts_v2_ready
    d.create_session(session_id="g", source="cli", model="m")
    d.append_message("g", role="user", content="갭 이전 메시지")
    d.close()

    # Tokenizer-less open: triggers dropped → marker durably invalidated.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "missing.so"))
    d2 = SessionDB(db_path=db_path)
    d2.append_message("g", role="user", content="갭 도중 메시지")  # the gap write
    with d2._lock:
        marker = d2._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state.FTS_V2_READY_KEY,),
        ).fetchone()[0]
    assert marker == hermes_state.FTS_V2_NEEDS_BACKFILL
    d2.close()

    # Extension-capable reopen: triggers come back, but the index is missing
    # the gap row — v2 must not serve; the legacy route still answers.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d3 = SessionDB(db_path=db_path)
    try:
        assert d3._fts_cjk_loaded
        assert not d3._fts_v2_ready
        with d3._lock:
            in_v2 = d3._conn.execute(
                "SELECT count(*) FROM messages_fts_v2 "
                "WHERE messages_fts_v2 MATCH '도중'"
            ).fetchone()[0]
        assert in_v2 == 0  # the gap row really is absent from the index
        assert d3.search_messages("도중", limit=5)  # legacy still answers
    finally:
        d3.close()

    # Verified backfill restores v2 with no missing rows.
    res = _run_migrate(db_path, cjk_so)
    assert res.returncode == 0, res.stdout + res.stderr
    d4 = SessionDB(db_path=db_path)
    try:
        assert d4._fts_v2_ready
        rows = d4.search_messages("도중", limit=5)
        assert rows and "도중" in rows[0]["snippet"]
        assert d4.search_messages("이전", limit=5)
    finally:
        d4.close()


def test_repair_rebuilds_corrupt_v2_index(cjk_so, tmp_path, monkeypatch):
    """Corrupt v2 shadow blocks → repair_state_db_schema rebuilds in place;
    the index keeps serving with all rows and writes work again."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "corrupt.db"
    d = SessionDB(db_path=db_path)
    d.create_session(session_id="c", source="cli", model="m")
    d.append_message("c", role="user", content="복구 대상 메시지")
    d.close()
    assert hermes_state._db_opens_cleanly(db_path) is None  # healthy before

    # Corrupt the segment leaves but keep the structure record (id 10)
    # intact, so the vtable still connects and the in-place FTS5 'rebuild'
    # path applies (a corrupt structure record fails the constructor and
    # escalates to drop_fts_rebuild — same as v1).
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.execute(
        "UPDATE messages_fts_v2_data SET block = X'DEADBEEFDEADBEEF' "
        "WHERE id > 10"
    )
    raw.close()
    assert hermes_state._db_opens_cleanly(db_path) is not None

    report = hermes_state.repair_state_db_schema(db_path, backup=False)
    assert report["repaired"], report
    assert report["strategy"] == "rebuild_fts"

    d2 = SessionDB(db_path=db_path)
    try:
        assert d2._fts_v2_ready  # in-place rebuild keeps the ready marker
        assert d2.search_messages("복구", limit=5)
        d2.append_message("c", role="user", content="복구 이후 쓰기")
        assert d2.search_messages("쓰기", limit=5)
    finally:
        d2.close()


def test_repair_recovers_dropped_v2_table(cjk_so, tmp_path, monkeypatch):
    """An absent v2 table with dangling triggers fails every message write:
    the health probe must flag it, repair must recover, and the recreated
    index must stay out of the read path until a verified backfill."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "absent.db"
    d = SessionDB(db_path=db_path)
    d.create_session(session_id="a", source="cli", model="m")
    d.append_message("a", role="user", content="테이블 소실 메시지")
    d.close()

    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.enable_load_extension(True)
    raw.load_extension(str(cjk_so))
    raw.enable_load_extension(False)
    raw.execute("DROP TABLE messages_fts_v2")  # triggers left dangling
    raw.close()
    assert hermes_state._db_opens_cleanly(db_path) is not None

    report = hermes_state.repair_state_db_schema(db_path, backup=False)
    assert report["repaired"], report
    assert hermes_state._db_opens_cleanly(db_path) is None

    # Reopen recreates v2 on a populated DB → empty index must not serve.
    d2 = SessionDB(db_path=db_path)
    try:
        assert not d2._fts_v2_ready
        d2.append_message("a", role="user", content="복원 이후 메시지")  # writes work
        assert d2.search_messages("소실", limit=5)  # legacy route answers
    finally:
        d2.close()

    res = _run_migrate(db_path, cjk_so)
    assert res.returncode == 0, res.stdout + res.stderr
    d3 = SessionDB(db_path=db_path)
    try:
        assert d3._fts_v2_ready
        assert d3.search_messages("소실", limit=5)
        assert d3.search_messages("복원", limit=5)
    finally:
        d3.close()


def test_optimize_and_rebuild_cover_v2(db):
    """The _FTS_TABLES-driven maintenance must cover the serving v2 index on
    a v2-only DB (fresh, or v1 retired by scripts/fts_v1_drop.py)."""
    assert db.optimize_fts() == 1  # v1/trigram absent, v2 merged
    assert db.rebuild_fts() == 1
    rows = db.search_messages("웅기", limit=5)
    assert rows and "웅기" in rows[0]["snippet"]  # still serving after rebuild


def test_partial_backfill_not_served(cjk_so, tmp_path, monkeypatch):
    """v2 triggers without the ready marker must not serve reads."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "missing.so"))
    d = SessionDB(db_path=tmp_path / "partial.db")
    d.create_session(session_id="p", source="cli", model="m")
    d.append_message("p", role="user", content="백필 전 메시지")
    d.close()
    # Reopen with the tokenizer: init ensures v2 table+triggers, but the DB
    # is non-empty and unmigrated → no marker → probe must refuse.
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d2 = SessionDB(db_path=tmp_path / "partial.db")
    try:
        assert d2._fts_cjk_loaded
        assert not d2._fts_v2_ready
        assert d2.search_messages("백필", limit=5)  # legacy still answers
    finally:
        d2.close()
