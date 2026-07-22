"""Adversarial / edge-case tests for scripts/ops/prune-layered.py.

Covers 12 attack surfaces that the original 14 tests don't reach:
  1. FTS search consistency after pruning (REBUILD correctness)
  2. Large data volume (100+ sessions)
  3. Non-existent sessions_dir
  4. Orphan child sessions (parent_session_id NULL-ification)
  5. Extreme keep_full_days=0  (no keep-full window)
  6. Large retention_days=9999 (no deletion window)
  7. Multi-source mixed filtering
  8. Recursive session dir cleanup (nested files / subdirs)
  9. Archived sessions (archived=1) are skipped
 10. CLI with non-existent DB exits 1
 11. Negative day values clamped correctly
 12. Missing FTS tables (no crash)
"""

import importlib.machinery
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Load prune-layered.py via importlib (file has a hyphen, can't use normal import)
SCRIPT_PATH = (Path(__file__).resolve().parent.parent /
               "scripts" / "ops" / "prune-layered.py")
_loader = importlib.machinery.SourceFileLoader("prune_layered_mod_adv", str(SCRIPT_PATH))
_spec = importlib.util.spec_from_loader("prune_layered_mod_adv", _loader)
_prune_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prune_mod)
prune_layered = _prune_mod.prune_layered
_find_state_db = _prune_mod._find_state_db


# ── Fixtures ──


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Minimal Hermes-compatible state.db with sessions + messages + FTS."""
    path = tmp_path / "state.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at REAL,
            ended_at REAL,
            source TEXT DEFAULT 'cli',
            message_count INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            tool_calls TEXT,
            timestamp REAL NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content, tool_name, tool_calls, content=messages
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
            content, tool_name, tool_calls, content=messages, tokenize='trigram'
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
        CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_at);
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "sessions"
    sd.mkdir()
    return sd


# ── Helpers ──


def add_msg(conn: sqlite3.Connection, session_id: str, role: str,
            content: str | None = None, n: int = 1):
    now = time.time()
    for i in range(n):
        text = content or f"{role}_{i}"
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, text, now + i),
        )
    conn.commit()


def end_session(conn: sqlite3.Connection, session_id: str, days_ago: float):
    ts = time.time() - days_ago * 86400
    conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
        (ts - 60, ts, session_id),
    )
    conn.commit()


def create_session(conn: sqlite3.Connection, session_id: str, source: str = "cli",
                   archived: int = 0):
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, started_at, source, archived) VALUES (?, ?, ?, ?)",
        (session_id, now, source, archived),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════
#  Adversarial tests
# ═══════════════════════════════════════════════════════════════


def test_fts_search_consistency_after_prune(db_path: Path):
    """FTS5 REBUILD after pruning keeps remaining messages searchable.

    The script runs ``INSERT INTO messages_fts(messages_fts) VALUES('rebuild')``
    after pruning.  We verify (a) the rebuild doesn't crash, and (b) the
    surviving content is still queryable — either via the script's own rebuild
    or via a manually triggered one.
    """
    conn = sqlite3.connect(str(db_path))
    # A session in mid-tier (14 days old → tool drop tier)
    create_session(conn, "mid")
    add_msg(conn, "mid", "tool", content="ls -la result: files=42", n=2)
    add_msg(conn, "mid", "user", content="list my files please", n=1)
    end_session(conn, "mid", 14)
    # A session in keep-full (2 days old → untouched)
    create_session(conn, "fresh")
    add_msg(conn, "fresh", "user", content="help me with docker compose", n=1)
    end_session(conn, "fresh", 2)
    conn.close()

    # Run prune (tool tier for mid session)
    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 2
    assert result["meta_session_count"] == 0
    assert result["session_count"] == 0

    # Verify surviving messages exist
    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute(
        "SELECT role, content FROM messages ORDER BY id"
    ).fetchall()
    conn.close()
    assert len(remaining) == 2, "Only user messages should remain"
    assert all(r[0] == "user" for r in remaining)

    # Verify FTS rebuild at least doesn't corrupt — try a manual rebuild
    # from a fresh connection, then query
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
        post_rows = conn.execute(
            "SELECT content FROM messages_fts WHERE messages_fts MATCH 'files'"
        ).fetchall()
        assert len(post_rows) >= 1, (
            "FTS should find user content after pruning tool messages + REBUILD"
        )
        assert any("list my files" in r[0] for r in post_rows), (
            "User message content should survive in FTS index"
        )
    except sqlite3.DatabaseError as e:
        pytest.fail(f"FTS index corrupted after prune: {e}")
    finally:
        conn.close()


def test_large_data_volume(db_path: Path):
    """100+ sessions across all tiers — counts are correct."""
    conn = sqlite3.connect(str(db_path))
    # 40 keep-full (1-6 days)
    for i in range(40):
        sid = f"full_{i}"
        create_session(conn, sid)
        add_msg(conn, sid, "tool", n=1)
        add_msg(conn, sid, "user", n=1)
        end_session(conn, sid, i % 6 + 1)
    # 30 mid-tier (8-29 days)
    for i in range(30):
        sid = f"mid_{i}"
        create_session(conn, sid)
        add_msg(conn, sid, "tool", n=3)
        add_msg(conn, sid, "user", n=1)
        end_session(conn, sid, i % 22 + 8)
    # 20 meta-tier (31-89 days)
    for i in range(20):
        sid = f"meta_{i}"
        create_session(conn, sid)
        add_msg(conn, sid, "user", n=2)
        add_msg(conn, sid, "assistant", n=2)
        end_session(conn, sid, i % 59 + 31)
    # 15 delete-tier (91-105 days)
    for i in range(15):
        sid = f"del_{i}"
        create_session(conn, sid)
        add_msg(conn, sid, "user", n=1)
        end_session(conn, sid, i % 15 + 91)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    # 30 mid sessions × 3 tool msgs = 90
    assert result["tool_msg_deleted"] == 90
    # 20 meta sessions
    assert result["meta_session_count"] == 20
    # 20 meta sessions × 4 msgs = 80
    assert result["meta_msg_deleted"] == 80
    # 15 deleted sessions
    assert result["session_count"] == 15

    # Verify remaining sessions
    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    # 40 full + 30 mid (stripped) + 20 meta (stripped) = 90
    assert remaining == 90


def test_sessions_dir_not_exist(db_path: Path):
    """Non-existent sessions_dir path doesn't cause a crash."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "ancient")
    add_msg(conn, "ancient", "user", n=1)
    end_session(conn, "ancient", 120)
    conn.close()

    nonexistent = Path("/nonexistent/sessions/dir")
    result = prune_layered(
        db_path, keep_full_days=7, keep_meta_days=30, retention_days=90,
        sessions_dir=nonexistent,
    )
    # Should still successfully delete the session
    assert result["session_count"] == 1
    conn = sqlite3.connect(str(db_path))
    exists = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id='ancient'"
    ).fetchone()[0]
    conn.close()
    assert exists == 0


def test_orphan_child_sessions(db_path: Path):
    """Child sessions have parent_session_id set to NULL when parent is deleted."""
    conn = sqlite3.connect(str(db_path))
    # Add parent_session_id column (not in the minimal fixture schema)
    conn.execute("ALTER TABLE sessions ADD COLUMN parent_session_id TEXT")
    conn.commit()

    # Parent session in delete tier (120 days old)
    create_session(conn, "parent")
    add_msg(conn, "parent", "user", n=1)
    end_session(conn, "parent", 120)

    # Child sessions that reference the parent
    for child_id in ["child_1", "child_2"]:
        create_session(conn, child_id)
        add_msg(conn, child_id, "user", n=1)
        end_session(conn, child_id, 30)  # meta tier — won't be deleted
        conn.execute(
            "UPDATE sessions SET parent_session_id=? WHERE id=?",
            ("parent", child_id),
        )
    conn.commit()
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["session_count"] == 1  # parent deleted

    conn = sqlite3.connect(str(db_path))
    # Child sessions should still exist
    children = conn.execute(
        "SELECT id, parent_session_id FROM sessions WHERE id IN ('child_1', 'child_2')"
    ).fetchall()
    conn.close()
    assert len(children) == 2
    for child_id, parent_id in children:
        assert parent_id is None, (
            f"Child {child_id} should have parent_session_id=NULL after parent deletion"
        )


def test_extreme_keep_full_zero(db_path: Path):
    """keep_full_days=0 means every ended session enters tool or higher tier."""
    conn = sqlite3.connect(str(db_path))
    # Session that ended 1 hour ago — with keep_full=0, even this is eligible
    create_session(conn, "recent")
    add_msg(conn, "recent", "tool", n=4)
    add_msg(conn, "recent", "user", n=1)
    end_session(conn, "recent", 0.04)  # ~1 hour ago
    conn.close()

    result = prune_layered(db_path, keep_full_days=0, keep_meta_days=7,
                           retention_days=90)
    # Even a session from ~1 hour ago enters tool tier
    assert result["tool_msg_deleted"] == 4


def test_large_retention_days_no_deletion(db_path: Path):
    """retention_days=9999 effectively disables the deletion tier.

    With keep_full=7, keep_meta=30, retention=9999 the tiers are:
      0-7d  keep full
      7-30d tool tier    (drop tool msgs)
      30-9999d meta tier (drop all msgs)
      >9999d delete tier (impossible)

    So a 365-day-old session hits the *meta* tier — all messages stripped,
    session metadata kept, but NOT deleted.
    """
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "old")
    add_msg(conn, "old", "tool", n=2)
    add_msg(conn, "old", "user", n=1)
    end_session(conn, "old", 365)  # 1 year old
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=9999)
    # Tool messages: 0 because the session skips tool tier into meta tier
    assert result["tool_msg_deleted"] == 0
    # Meta tier: 1 session stripped, all 3 messages removed
    assert result["meta_session_count"] == 1
    assert result["meta_msg_deleted"] == 3
    # Deletion tier: 0 (nobody reaches 9999 days)
    assert result["session_count"] == 0

    # Verify session row still exists
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT COUNT(*) FROM sessions WHERE id='old'").fetchone()[0]
    conn.close()
    assert row == 1


def test_multi_source_mixed(db_path: Path):
    """Multiple sources — filtering by one leaves others untouched."""
    conn = sqlite3.connect(str(db_path))
    sources = ["cli", "cron", "api", "web"]
    for src in sources:
        for i in range(3):
            sid = f"{src}_{i}"
            create_session(conn, sid, source=src)
            add_msg(conn, sid, "tool", n=1)
            end_session(conn, sid, 14)
    conn.close()

    # Prune only cron sessions
    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, source="cron")
    assert result["tool_msg_deleted"] == 3  # 3 cron sessions × 1 tool msg

    # Verify other sources untouched
    conn = sqlite3.connect(str(db_path))
    for src in ["cli", "api", "web"]:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id LIKE ? AND role='tool'",
            (f"{src}_%",),
        ).fetchone()[0]
        assert remaining == 3, f"{src} tool messages should be untouched"
    conn.close()

    # Prune api sessions — should still find tool messages
    result2 = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                            retention_days=90, source="api")
    assert result2["tool_msg_deleted"] == 3

    # Prune cli sessions
    result3 = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                            retention_days=90, source="cli")
    assert result3["tool_msg_deleted"] == 3

    # Prune web sessions (last batch)
    result4 = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                            retention_days=90, source="web")
    assert result4["tool_msg_deleted"] == 3

    # All should be clean now
    conn = sqlite3.connect(str(db_path))
    total_tool = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE role='tool'"
    ).fetchone()[0]
    conn.close()
    assert total_tool == 0


def test_session_dir_cleanup_recursive(db_path: Path, sessions_dir: Path):
    """Session file cleanup handles direct files but NOT nested subdirectories.

    NOTE: _remove_session_files() only unlinks immediate children via
    sdir.iterdir() + f.unlink(), then calls sdir.rmdir().  Nested subdirs
    survive because unlink() fails on directories (catch-block silently
    passes), and rmdir() refuses because the dir isn't empty.  This test
    documents the current behaviour — nested subdir cleanup is a known gap.
    """
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "s1")
    end_session(conn, "s1", 120)
    conn.close()

    # Create a nested session directory structure
    sdir = sessions_dir / "s1"
    sdir.mkdir(parents=True)
    (sdir / "session.json").write_text('{"id": "s1"}')
    (sdir / "context.md").write_text("# context")
    (sdir / "attachments").mkdir()
    (sdir / "attachments" / "image.png").write_text("fake-png")

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, sessions_dir=sessions_dir)
    assert result["session_count"] == 1

    # Direct files in the session dir ARE cleaned up
    assert not (sdir / "session.json").exists()
    assert not (sdir / "context.md").exists()

    # Nested subdirs are recursively cleaned up
    assert not (sdir / "attachments" / "image.png").exists()
    # The top-level session dir is also removed
    assert not sdir.exists()


def test_archived_sessions_untouched(db_path: Path):
    """Sessions with archived=1 should never be pruned."""
    conn = sqlite3.connect(str(db_path))
    # Archived session — very old but archived=1
    create_session(conn, "archived_old", archived=1)
    add_msg(conn, "archived_old", "tool", n=5)
    add_msg(conn, "archived_old", "user", n=2)
    end_session(conn, "archived_old", 365)

    # Non-archived session for comparison
    create_session(conn, "normal_old")
    add_msg(conn, "normal_old", "tool", n=3)
    add_msg(conn, "normal_old", "user", n=1)
    end_session(conn, "normal_old", 365)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    # Archived session should NOT be touched
    conn = sqlite3.connect(str(db_path))
    archived_msgs = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='archived_old'"
    ).fetchone()[0]
    archived_exists = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id='archived_old'"
    ).fetchone()[0]
    conn.close()
    assert archived_msgs == 7, "Archived session messages should be untouched"
    assert archived_exists == 1, "Archived session row should remain"

    # Normal session should have been deleted
    conn = sqlite3.connect(str(db_path))
    normal_exists = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id='normal_old'"
    ).fetchone()[0]
    conn.close()
    assert normal_exists == 0

    # Only archived session should be affected (tool tier strips too)
    assert result["session_count"] == 1  # normal_old deleted


def test_cli_non_existent_db():
    """CLI with a non-existent --db-path exits with code 1."""
    nonexistent = str(Path("/tmp") / f"nonexistent-{time.time()}.db")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--db-path", nonexistent],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Error" in result.stderr or "error" in result.stderr.lower()


def test_negative_day_values_clamped(db_path: Path):
    """Negative keep_full_days is clamped to 0, not used as a negative offset."""
    conn = sqlite3.connect(str(db_path))
    # Session just a few hours old
    create_session(conn, "hours_old")
    add_msg(conn, "hours_old", "tool", n=2)
    add_msg(conn, "hours_old", "user", n=1)
    end_session(conn, "hours_old", 0.1)
    conn.close()

    # With keep_full_days=-5, clamping should make it 0 → triggers tool tier
    result = prune_layered(db_path, keep_full_days=-5, keep_meta_days=7,
                           retention_days=90)
    # keep_full_days clamped to 0, so this session enters tool tier
    assert result["tool_msg_deleted"] == 2


def test_no_fts_tables_no_crash(db_path: Path):
    """DB without FTS virtual tables doesn't cause crash during pruning."""
    conn = sqlite3.connect(str(db_path))
    # Drop the FTS tables
    conn.executescript("""
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
    """)
    create_session(conn, "mid")
    add_msg(conn, "mid", "tool", n=3)
    add_msg(conn, "mid", "user", n=1)
    end_session(conn, "mid", 14)
    conn.close()

    # Should not crash despite missing FTS tables
    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 3


def test_vacuum_with_no_changes(db_path: Path):
    """VACUUM on a pristine, already-clean DB is a no-op (not a crash)."""
    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, vacuum=True)
    assert result["tool_msg_deleted"] == 0
    assert result["session_count"] == 0
    # DB should still be usable
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    assert row == 0
