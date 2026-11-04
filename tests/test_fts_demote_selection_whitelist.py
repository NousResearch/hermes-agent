"""Legacy-FTS demotion must select tables by the exact fts5 shadow whitelist.

The demotion enumeration used an unescaped LIKE prefix: with ESCAPE declared
but ``_`` left unescaped, the underscore stayed a single-character wildcard
and the prefix was wider than the five fts5 shadow suffixes, so unrelated
look-alike tables were renamed into the fts_v22_trash_* family and dropped by
the teardown drain — silent data loss (#109748).
"""

import sqlite3
import time

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


def _legacy_db_with_lookalikes(db_path):
    """Hand-build a legacy inline-FTS layout that also carries two unrelated
    look-alike tables the old LIKE prefix would sweep in."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript("""
            DROP TABLE IF EXISTS messages_fts;
            DROP TABLE IF EXISTS messages_fts_trigram;
            DROP VIEW IF EXISTS messages_fts_trigram_src;
            CREATE VIRTUAL TABLE messages_fts USING fts5(content);
            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, COALESCE(new.content,''));
            END;
            -- Unrelated look-alikes: ``messagesXfts_probe`` only matches the old
            -- pattern because the unescaped ``_`` acts as a wildcard; ``messages_fts_other``
            -- matches because the prefix is wider than the five shadow suffixes.
            CREATE TABLE messagesXfts_probe (id INTEGER PRIMARY KEY, note TEXT);
            CREATE TABLE messages_fts_other (id INTEGER PRIMARY KEY, note TEXT);
        """)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (10)")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
            (time.time(),),
        )
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content) "
            "VALUES ('s1', ?, 'user', 'legacy inline message')",
            (time.time(),),
        )
        conn.execute("INSERT INTO messagesXfts_probe (note) VALUES ('probe-data-keep')")
        conn.execute("INSERT INTO messages_fts_other (note) VALUES ('other-data-keep')")
        conn.commit()
    finally:
        conn.close()


def test_optimize_demote_drops_only_exact_shadow_names(tmp_path):
    db_path = tmp_path / "state.db"
    _legacy_db_with_lookalikes(db_path)

    db = SessionDB(db_path=db_path)
    try:
        assert db.fts_optimize_available(), "legacy inline layout must be eligible"
        result = db.optimize_fts_storage(vacuum=False)
        assert result["ok"]
        # The real legacy shadow family is demoted and the v23 index serves search.
        assert db.search_messages("legacy inline message")
        # The unrelated look-alikes survive untouched, data intact.
        with db._lock:
            names = {
                row[0] for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert "messagesXfts_probe" in names, "wildcard-match table must not be demoted"
            assert "messages_fts_other" in names, "over-broad-prefix table must not be demoted"
            assert [r[0] for r in db._conn.execute(
                "SELECT note FROM messagesXfts_probe"
            ).fetchall()] == ["probe-data-keep"]
            assert [r[0] for r in db._conn.execute(
                "SELECT note FROM messages_fts_other"
            ).fetchall()] == ["other-data-keep"]
            trash = [
                row[0] for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\'"
                )
            ]
        assert not trash, f"unexpected tables left in the trash family: {trash}"
    finally:
        db.close()
