"""#103840: a ``sqlite3 .recover`` restore re-emits FTS5 shadow tables as ordinary tables
but cannot re-emit the ``CREATE VIRTUAL TABLE`` row. The next SessionDB open then failed
in ``_ensure_fts_schema`` with "fts5: error creating shadow table messages_fts_data: table
already exists". Only families whose vtable row is absent may be repaired; a healthy family's
shadows must survive untouched.
"""

import sqlite3
import threading
from contextlib import contextmanager

import hermes_state
import hermes_state_schema
from hermes_state import SessionDB
from hermes_state_common import FTS_STALE_KEY, fts_rebuild_admission


def _orphan_family(db_path, family: str) -> None:
    """Emulate the ``.recover`` residue for one family: vtable row gone, shadows kept."""
    raw = sqlite3.connect(db_path)
    raw.isolation_level = None
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute(
        "DELETE FROM sqlite_master WHERE name = ? AND sql LIKE 'CREATE VIRTUAL TABLE%'", (family,),
    )
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.close()


def _fts_master_rows(db_path, prefix: str) -> list:
    raw = sqlite3.connect(db_path)
    try:
        return raw.execute(
            "SELECT rowid, type, name FROM sqlite_master WHERE name LIKE ? ESCAPE '\\' ORDER BY rowid",
            (prefix.replace("_", "\\_") + "%",),
        ).fetchall()
    finally:
        raw.close()


def test_orphaned_base_family_is_repaired_and_healthy_trigram_untouched(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    for i in range(3):
        db.append_message("s1", "user", f"recovered orphan {i}")
    db.close()

    _orphan_family(db_path, "messages_fts")
    trigram_before = _fts_master_rows(db_path, "messages_fts_trigram")
    assert trigram_before, "fixture needs a live trigram family"
    raw = sqlite3.connect(db_path)
    orphan_shadows = raw.execute(
        "SELECT count(*) FROM sqlite_master WHERE name IN ('messages_fts_data', 'messages_fts_config')"
    ).fetchone()[0]
    raw.close()
    assert orphan_shadows == 2, "fixture must leave the base shadows behind"

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened._fts_enabled is True
        with reopened._lock:
            hits = reopened._conn.execute(
                "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'orphan'"
            ).fetchone()[0]
        assert hits == 3, "recreated index must be rebuilt from the canonical messages table"
    finally:
        reopened.close()
    # Same rowids: the healthy family was neither dropped nor recreated.
    assert _fts_master_rows(db_path, "messages_fts_trigram") == trigram_before


def test_concurrent_orphan_repair_does_not_drop_recreated_fts_family(
    tmp_path, monkeypatch
):
    """A stale orphan snapshot must not drop a family recreated by another opener."""
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("s1", source="cli", model="m")
    seed.append_message("s1", "user", "concurrent orphan repair")
    seed.close()
    _orphan_family(db_path, "messages_fts")

    selected = threading.Event()
    release = threading.Event()
    state = threading.local()

    class CoordinatedCursor(sqlite3.Cursor):
        _selected_base_orphans = False

        def execute(self, sql, parameters=()):
            self._selected_base_orphans = (
                getattr(state, "opener", None) == "stale"
                and "messages_fts_data" in parameters
                and (
                    "SELECT name FROM sqlite_master" in sql
                    or "SELECT 1 FROM sqlite_master AS shadow" in sql
                )
            )
            return super().execute(sql, parameters)

        def fetchall(self):
            rows = super().fetchall()
            if self._selected_base_orphans:
                selected.set()
                with fts_rebuild_admission(db_path, timeout_seconds=0) as admitted:
                    snapshot_is_unguarded = admitted
                if snapshot_is_unguarded:
                    assert release.wait(timeout=10), "competing opener did not finish"
            return rows

    class CoordinatedConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            kwargs.setdefault("factory", CoordinatedCursor)
            return super().cursor(*args, **kwargs)

    real_connect = hermes_state._connect_tracked_db

    def coordinated_connect(path, tracking_path=None, **kwargs):
        kwargs["factory"] = CoordinatedConnection
        return real_connect(path, tracking_path=tracking_path, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", coordinated_connect)

    opened = {}
    errors = {}

    def open_db(name):
        state.opener = name
        try:
            opened[name] = SessionDB(db_path=db_path)
        except BaseException as exc:
            errors[name] = exc
        finally:
            if name == "fresh":
                release.set()

    stale_thread = threading.Thread(target=open_db, args=("stale",), daemon=True)
    fresh_thread = threading.Thread(target=open_db, args=("fresh",), daemon=True)
    fresh_started = False
    stale_thread.start()
    try:
        assert selected.wait(timeout=10), "stale opener did not snapshot orphan shadows"
        fresh_thread.start()
        fresh_started = True
        fresh_thread.join(timeout=15)
        stale_thread.join(timeout=15)
        assert not fresh_thread.is_alive(), "competing opener did not terminate"
        assert not stale_thread.is_alive(), "stale opener did not terminate"
        assert errors == {}

        db = opened["fresh"]
        with db._lock:
            hits = db._conn.execute(
                "SELECT count(*) FROM messages_fts "
                "WHERE messages_fts MATCH 'concurrent'"
            ).fetchone()[0]
        assert hits == 1
    finally:
        release.set()
        stale_thread.join(timeout=5)
        if fresh_started:
            fresh_thread.join(timeout=5)
        for db in opened.values():
            db.close()


def _seed_cjk_orphan_with_trigger(db_path):
    seed = SessionDB(db_path=db_path)
    seed.close()

    raw = sqlite3.connect(db_path)
    raw.executescript(
        "CREATE TABLE messages_fts_cjk_data(id INTEGER PRIMARY KEY);"
        "CREATE TRIGGER messages_fts_cjk_insert AFTER INSERT ON messages BEGIN "
        "INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls) "
        "VALUES (new.id, new.content, new.tool_name, new.tool_calls); END;"
    )
    raw.close()


def _assert_cjk_trigger_detached_and_append_survives(db_path):
    reopened = SessionDB(db_path=db_path)
    try:
        trigger = reopened._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'messages_fts_cjk_insert'"
        ).fetchone()
        assert trigger is None
        reopened.create_session("s1", source="cli", model="m")
        reopened.append_message("s1", "user", "canonical write survives")
    finally:
        reopened.close()


def test_cjk_orphan_without_tokenizer_drops_broken_sync_triggers(
    tmp_path, monkeypatch
):
    """An unavailable optional tokenizer must not leave a trigger targeting no table."""
    db_path = tmp_path / "state.db"
    _seed_cjk_orphan_with_trigger(db_path)
    monkeypatch.setattr(hermes_state, "load_fts5_cjk_extension", lambda _conn: False)

    _assert_cjk_trigger_detached_and_append_survives(db_path)


def test_cjk_orphan_repair_deferral_drops_broken_sync_triggers(tmp_path, monkeypatch):
    """Lost admission must leave canonical writes independent of a missing CJK table."""
    db_path = tmp_path / "state.db"
    _seed_cjk_orphan_with_trigger(db_path)

    @contextmanager
    def deny_admission(*_args, **_kwargs):
        yield False

    monkeypatch.setattr(hermes_state_schema, "fts_rebuild_admission", deny_admission)
    _assert_cjk_trigger_detached_and_append_survives(db_path)


def test_trigram_orphan_without_tokenizer_drops_broken_sync_triggers(
    tmp_path, monkeypatch
):
    """A missing optional tokenizer must detach triggers for the absent trigram table."""
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.close()
    _orphan_family(db_path, "messages_fts_trigram")

    real_ensure = SessionDB._ensure_fts_schema

    def ensure_without_trigram(self, cursor, table_name, ddl):
        if table_name == "messages_fts_trigram":
            return False
        return real_ensure(self, cursor, table_name, ddl)

    monkeypatch.setattr(SessionDB, "_ensure_fts_schema", ensure_without_trigram)
    reopened = SessionDB(db_path=db_path)
    try:
        triggers = reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'messages_fts_trigram_%'"
        ).fetchall()
        assert triggers == []
        reopened.create_session("s1", source="cli", model="m")
        reopened.append_message("s1", "user", "canonical write survives")
    finally:
        reopened.close()


def test_stale_recovery_keeps_admission_through_cjk_ensure(tmp_path, monkeypatch):
    """CJK recreation must not run after releasing the orphan-repair authority."""
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.close()
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO state_meta(key, value) VALUES (?, '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'",
        (FTS_STALE_KEY,),
    )
    raw.execute("CREATE TABLE IF NOT EXISTS messages_fts_cjk_data(id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()

    entered = threading.Event()
    original_ensure = SessionDB._ensure_fts_cjk_schema

    def ensure_while_admitted(self, cursor):
        with fts_rebuild_admission(db_path, timeout_seconds=0) as admitted:
            assert not admitted, "CJK ensure ran after rebuild admission was released"
        entered.set()
        return original_ensure(self, cursor)

    monkeypatch.setattr(SessionDB, "_ensure_fts_cjk_schema", ensure_while_admitted)
    reopened = SessionDB(db_path=db_path)
    try:
        assert entered.is_set()
    finally:
        reopened.close()


def test_base_and_trigram_orphans_without_tokenizer_detach_trigram(
    tmp_path, monkeypatch
):
    """Base repair must fail closed when an orphaned trigram sibling cannot return."""
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.close()
    _orphan_family(db_path, "messages_fts")
    _orphan_family(db_path, "messages_fts_trigram")

    real_ensure = SessionDB._ensure_fts_schema

    def ensure_without_trigram(self, cursor, table_name, ddl):
        if table_name == "messages_fts_trigram":
            return False
        return real_ensure(self, cursor, table_name, ddl)

    monkeypatch.setattr(SessionDB, "_ensure_fts_schema", ensure_without_trigram)
    reopened = SessionDB(db_path=db_path)
    try:
        triggers = reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'messages_fts_trigram_%'"
        ).fetchall()
        assert triggers == []
        reopened.create_session("s1", source="cli", model="m")
        reopened.append_message("s1", "user", "canonical write survives")
    finally:
        reopened.close()


def test_base_and_cjk_orphans_without_tokenizer_detach_cjk(tmp_path, monkeypatch):
    """Base repair must fail closed when an orphaned CJK sibling cannot return."""
    db_path = tmp_path / "state.db"
    _seed_cjk_orphan_with_trigger(db_path)
    _orphan_family(db_path, "messages_fts")
    monkeypatch.setattr(hermes_state, "load_fts5_cjk_extension", lambda _conn: False)

    _assert_cjk_trigger_detached_and_append_survives(db_path)
