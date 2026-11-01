#!/usr/bin/env python3
"""
genie_rebuild_fts.py — Rebuild FTS indexes on a no-FTS state.db.
Run inside a test container after restoring a compressed backup.

Usage:
    python3 genie_rebuild_fts.py [--db /path/to/state.db] [--verify]
"""

import sqlite3, sys, os, time

DB_PATH = "/root/.hermes/state.db"

TRIGGER_SQL = """
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;
CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

def rebuild_fts(db_path, verify=True):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_size_limit=0")

    # Drop old FTS if present
    conn.executescript("""
        DROP TRIGGER IF EXISTS messages_fts_insert;
        DROP TRIGGER IF EXISTS messages_fts_delete;
        DROP TRIGGER IF EXISTS messages_fts_update;
        DROP TRIGGER IF EXISTS messages_fts_trigram_insert;
        DROP TRIGGER IF EXISTS messages_fts_trigram_delete;
        DROP TRIGGER IF EXISTS messages_fts_trigram_update;
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
    """)

    # Create FTS5 tables
    conn.executescript("""
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, content='messages', content_rowid='id', tokenize='trigram'
        );
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
            content, content='messages', content_rowid='id', tokenize='trigram'
        );
    """)

    # Repopulate
    t0 = time.time()
    conn.execute("""
        INSERT INTO messages_fts(rowid, content)
        SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '')
        FROM messages
    """)
    conn.execute("""
        INSERT INTO messages_fts_trigram(rowid, content)
        SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '')
        FROM messages
    """)
    conn.commit()
    build_time = time.time() - t0

    # Triggers
    conn.executescript(TRIGGER_SQL)

    if verify:
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        tri_count = conn.execute("SELECT COUNT(*) FROM messages_fts_trigram").fetchone()[0]
        assert fts_count == msg_count, f"FTS mismatch: {fts_count} != {msg_count}"
        assert tri_count == msg_count, f"Trigram mismatch: {tri_count} != {msg_count}"
        print(f"  Verified: {msg_count:,} messages indexed")

    conn.close()
    return build_time

if __name__ == "__main__":
    db_path = DB_PATH
    if "--db" in sys.argv:
        db_path = sys.argv[sys.argv.index("--db") + 1]

    print(f"Rebuilding FTS on {db_path} ({os.path.getsize(db_path)/1e9:.2f} GB)")
    t = rebuild_fts(db_path)
    print(f"  Done in {t:.1f}s")
    print(f"  Final size: {os.path.getsize(db_path)/1e9:.2f} GB")
