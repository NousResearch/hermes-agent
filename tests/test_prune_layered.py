"""Tests for scripts/ops/prune-layered.py — standalone tiered session pruning."""

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
_loader = importlib.machinery.SourceFileLoader("prune_layered_mod", str(SCRIPT_PATH))
_spec = importlib.util.spec_from_loader("prune_layered_mod", _loader)
_prune_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prune_mod)
prune_layered = _prune_mod.prune_layered
_find_state_db = _prune_mod._find_state_db


# ── Fixtures ──


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a minimal Hermes-compatible state.db with sessions + messages tables."""
    path = tmp_path / "state.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    # Minimal schema (only what prune_layered needs)
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


def add_msg(conn: sqlite3.Connection, session_id: str, role: str, n: int = 1):
    now = time.time()
    for i in range(n):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, f"{role}_{i}", now + i),
        )
    conn.commit()


def end_session(conn: sqlite3.Connection, session_id: str, days_ago: float):
    ts = time.time() - days_ago * 86400
    conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
        (ts - 60, ts, session_id),
    )
    conn.commit()


def create_session(conn: sqlite3.Connection, session_id: str, source: str = "cli"):
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, started_at, source) VALUES (?, ?, ?)",
        (session_id, now, source),
    )
    conn.commit()


# ── Tests ──


def test_dry_run_returns_counts(db_path: Path):
    """Dry-run returns per-tier session counts and message previews."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "s1")
    add_msg(conn, "s1", "tool", 3)
    add_msg(conn, "s1", "user", 1)
    end_session(conn, "s1", 14)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, dry_run=True)
    assert result["_dry_run"] is True
    assert result["tool_sessions"] == 1
    assert result["tool_msg_preview"] == 3  # preview of deletable tool msgs


def test_keep_full_tier_untouched(db_path: Path):
    """Sessions younger than keep_full_days are not included in candidates."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "fresh")
    add_msg(conn, "fresh", "tool", 2)
    end_session(conn, "fresh", 2)  # 2 days old < 7
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 0


def test_tool_messages_dropped_in_mid_tier(db_path: Path):
    """7-30 day sessions have tool messages removed, user messages kept."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "mid")
    add_msg(conn, "mid", "tool", 3)
    add_msg(conn, "mid", "user", 1)
    end_session(conn, "mid", 14)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 3

    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute(
        "SELECT role, COUNT(*) FROM messages WHERE session_id='mid' GROUP BY role"
    ).fetchall()
    conn.close()
    roles = {r[0]: r[1] for r in remaining}
    assert roles.get("user") == 1
    assert roles.get("tool") is None


def test_meta_tier_strips_all_messages(db_path: Path):
    """30-90 day sessions have all messages removed, session row kept."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "old")
    add_msg(conn, "old", "user", 2)
    add_msg(conn, "old", "assistant", 2)
    end_session(conn, "old", 45)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["meta_session_count"] == 1
    assert result["meta_msg_deleted"] == 4

    conn = sqlite3.connect(str(db_path))
    msg_count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old'"
    ).fetchone()[0]
    sess_exists = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id='old'"
    ).fetchone()[0]
    conn.close()
    assert msg_count == 0
    assert sess_exists == 1


def test_delete_tier_removes_session(db_path: Path, sessions_dir: Path):
    """Sessions past retention_days are deleted entirely, files cleaned up."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "ancient")
    add_msg(conn, "ancient", "user", 1)
    end_session(conn, "ancient", 120)
    conn.close()

    # Create a session file to verify cleanup
    sdir = sessions_dir / "ancient"
    sdir.mkdir()
    (sdir / "session.json").write_text("{}")

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, sessions_dir=sessions_dir)
    assert result["session_count"] == 1

    conn = sqlite3.connect(str(db_path))
    exists = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id='ancient'"
    ).fetchone()[0]
    conn.close()
    assert exists == 0
    # Session file should have been cleaned up
    assert not sdir.exists()


def test_source_filter(db_path: Path):
    """--source cron only prunes cron sessions, leaves CLI sessions."""
    conn = sqlite3.connect(str(db_path))
    for sid, src in [("cron_session", "cron"), ("cli_session", "cli")]:
        create_session(conn, sid, source=src)
        add_msg(conn, sid, "tool", 1)
        end_session(conn, sid, 14)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90, source="cron")
    assert result["tool_msg_deleted"] == 1  # only cron session's tool msg

    conn = sqlite3.connect(str(db_path))
    cli_tools = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='cli_session' AND role='tool'"
    ).fetchone()[0]
    conn.close()
    assert cli_tools == 1  # cli session untouched


def test_idempotent_second_run(db_path: Path):
    """Running twice deletes nothing the second time."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "s1")
    add_msg(conn, "s1", "tool", 2)
    end_session(conn, "s1", 14)
    conn.close()

    r1 = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                       retention_days=90)
    assert r1["tool_msg_deleted"] == 2

    r2 = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                       retention_days=90)
    assert r2["tool_msg_deleted"] == 0
    assert r2["meta_session_count"] == 0
    assert r2["session_count"] == 0


def test_empty_db_no_crash(db_path: Path):
    """An empty database produces a no-op result, not a crash."""
    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 0
    assert result["meta_session_count"] == 0
    assert result["session_count"] == 0


def test_active_sessions_untouched(db_path: Path):
    """Sessions without ended_at (still active) are never pruned."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "active")
    add_msg(conn, "active", "tool", 5)
    # No end_session call — session has no ended_at
    conn.close()

    result = prune_layered(db_path, keep_full_days=1, keep_meta_days=7,
                           retention_days=30)
    assert result["tool_msg_deleted"] == 0


def test_vacuum_does_not_crash(db_path: Path):
    """--vacuum flag runs without error."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "s1")
    add_msg(conn, "s1", "tool", 1)
    end_session(conn, "s1", 3)  # 3d old → tool tier (keep_full=1, keep_meta=7)
    conn.close()

    result = prune_layered(db_path, keep_full_days=1, keep_meta_days=7,
                           retention_days=30, vacuum=True)
    assert result["tool_msg_deleted"] == 1


def test_cli_help_exits_zero():
    """The script's --help flag exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Tiered session retention" in result.stdout


def test_cli_dry_run_json(db_path: Path):
    """CLI --dry-run --json produces valid JSON output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--db-path", str(db_path), "--dry-run", "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["tool_msg_deleted"] == 0
    assert data["_dry_run"] is True


def test_mixed_tiers(db_path: Path):
    """Sessions at different ages each handled by their correct tier."""
    conn = sqlite3.connect(str(db_path))
    for sid, days, tool_n, user_n in [
        ("young", 3, 2, 1),     # keep full
        ("mid", 14, 3, 1),      # tool tier
        ("old", 45, 1, 2),      # meta tier
        ("ancient", 120, 1, 1),  # delete tier
    ]:
        create_session(conn, sid)
        add_msg(conn, sid, "tool", tool_n)
        add_msg(conn, sid, "user", user_n)
        end_session(conn, sid, days)
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=30,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 3  # mid's tool msgs
    assert result["meta_session_count"] == 1  # old
    assert result["session_count"] == 1  # ancient


def test_zero_width_tiers(db_path: Path):
    """When keep_full == keep_meta, the tool tier is correctly skipped."""
    conn = sqlite3.connect(str(db_path))
    create_session(conn, "z")
    add_msg(conn, "z", "tool", 2)
    end_session(conn, "z", 5)  # 5 days old < keep_full=7 — fully kept
    conn.close()

    result = prune_layered(db_path, keep_full_days=7, keep_meta_days=7,
                           retention_days=90)
    assert result["tool_msg_deleted"] == 0  # no tool tier
