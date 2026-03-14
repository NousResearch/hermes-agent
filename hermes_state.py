#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "state.db"

SCHEMA_VERSION = 5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    title TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    timestamp REAL NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT,
    result_summary TEXT,
    duration_ms INTEGER,
    is_error INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()

    def _init_schema(self):
        """Create tables and FTS if they don't exist, run migrations."""
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # Check schema version and run migrations
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            if current_version < 2:
                # v2: add finish_reason column to messages
                try:
                    cursor.execute("ALTER TABLE messages ADD COLUMN finish_reason TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 2")
            if current_version < 3:
                # v3: add title column to sessions
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                cursor.execute("UPDATE schema_version SET version = 3")
            if current_version < 4:
                # v4: add unique index on title (NULLs allowed, only non-NULL must be unique)
                try:
                    cursor.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                        "ON sessions(title) WHERE title IS NOT NULL"
                    )
                except sqlite3.OperationalError:
                    pass  # Index already exists
                cursor.execute("UPDATE schema_version SET version = 4")
            if current_version < 5:
                # v5: add audit_log table for full tool-call history
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT REFERENCES sessions(id),
                        timestamp REAL NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments TEXT,
                        result_summary TEXT,
                        duration_ms INTEGER,
                        is_error INTEGER DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
                    CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool_name);
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
                """)
                cursor.execute("UPDATE schema_version SET version = 5")

        # Unique title index — always ensure it exists (safe to run after migrations
        # since the title column is guaranteed to exist at this point)
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
                "ON sessions(title) WHERE title IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass  # Index already exists

        # FTS5 setup (separate because CREATE VIRTUAL TABLE can't be in executescript with IF NOT EXISTS reliably)
        try:
            cursor.execute("SELECT * FROM messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)

        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
    ) -> str:
        """Create a new session record. Returns the session_id."""
        self._conn.execute(
            """INSERT INTO sessions (id, source, user_id, model, model_config,
               system_prompt, parent_session_id, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                source,
                user_id,
                model,
                json.dumps(model_config) if model_config else None,
                system_prompt,
                parent_session_id,
                time.time(),
            ),
        )
        self._conn.commit()
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended."""
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
            (time.time(), end_reason, session_id),
        )
        self._conn.commit()

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        self._conn.execute(
            "UPDATE sessions SET system_prompt = ? WHERE id = ?",
            (system_prompt, session_id),
        )
        self._conn.commit()

    def update_token_counts(
        self, session_id: str, input_tokens: int = 0, output_tokens: int = 0,
        model: str = None,
    ) -> None:
        """Increment token counters and backfill model if not already set."""
        self._conn.execute(
            """UPDATE sessions SET
               input_tokens = input_tokens + ?,
               output_tokens = output_tokens + ?,
               model = COALESCE(model, ?)
               WHERE id = ?""",
            (input_tokens, output_tokens, model, session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        title = self.sanitize_title(title)
        if title:
            # Check uniqueness (allow the same session to keep its own title)
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE title = ? AND id != ?",
                (title, session_id),
            )
            conflict = cursor.fetchone()
            if conflict:
                raise ValueError(
                    f"Title '{title}' is already in use by session {conflict['id']}"
                )
        cursor = self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        cursor = self._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        return row["title"] if row else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE title = ?", (title,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cursor = self._conn.execute(
            "SELECT id, title, started_at FROM sessions "
            "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
            (f"{escaped} #%",),
        )
        numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cursor = self._conn.execute(
            "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
            (base, f"{escaped} #%"),
        )
        existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def list_sessions_rich(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.
        """
        source_clause = "WHERE s.source = ?" if source else ""
        query = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            {source_clause}
            ORDER BY s.started_at DESC
            LIMIT ? OFFSET ?
        """
        params = (source, limit, offset) if source else (limit, offset)
        cursor = self._conn.execute(query, params)
        sessions = []
        for row in cursor.fetchall():
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            sessions.append(s)

        return sessions

    # =========================================================================
    # Message storage
    # =========================================================================

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).
        """
        cursor = self._conn.execute(
            """INSERT INTO messages (session_id, role, content, tool_call_id,
               tool_calls, tool_name, timestamp, token_count, finish_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                role,
                content,
                tool_call_id,
                json.dumps(tool_calls) if tool_calls else None,
                tool_name,
                time.time(),
                token_count,
                finish_reason,
            ),
        )
        msg_id = cursor.lastrowid

        # Update counters
        # Count actual tool calls from the tool_calls list (not from tool responses).
        # A single assistant message can contain multiple parallel tool calls.
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1
        if num_tool_calls > 0:
            self._conn.execute(
                """UPDATE sessions SET message_count = message_count + 1,
                   tool_call_count = tool_call_count + ? WHERE id = ?""",
                (num_tool_calls, session_id),
            )
        else:
            self._conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )

        self._conn.commit()
        return msg_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by timestamp."""
        cursor = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(msg)
        return result

    def get_messages_as_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.
        """
        cursor = self._conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, tool_name "
            "FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        )
        messages = []
        for row in cursor.fetchall():
            msg = {"role": row["role"], "content": row["content"]}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            messages.append(msg)
        return messages

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}`` and bare boolean operators (``AND``, ``OR``,
        ``NOT``) have special meaning.  Passing raw user input directly to
        MATCH can cause ``sqlite3.OperationalError``.

        Strategy: strip characters that are only meaningful as FTS5 operators
        and would otherwise cause syntax errors.  This preserves normal keyword
        search while preventing crashes on inputs like ``C++``, ``"unterminated``,
        or ``hello AND``.
        """
        # Remove FTS5-special characters that are not useful in keyword search
        sanitized = re.sub(r'[+{}()"^]', " ", query)
        # Collapse repeated * (e.g. "***") into a single one, and remove
        # leading * (prefix-only matching requires at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
        # Remove dangling boolean operators at start/end that would cause
        # syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())
        return sanitized.strip()

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).
        """
        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        if source_filter is None:
            source_filter = ["cli", "telegram", "discord", "whatsapp", "slack"]

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]

        source_placeholders = ",".join("?" for _ in source_filter)
        where_clauses.append(f"s.source IN ({source_placeholders})")
        params.extend(source_filter)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """

        try:
            cursor = self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            # FTS5 query syntax error despite sanitization — return empty
            return []
        matches = [dict(row) for row in cursor.fetchall()]

        # Add surrounding context (1 message before + after each match)
        for match in matches:
            try:
                ctx_cursor = self._conn.execute(
                    """SELECT role, content FROM messages
                       WHERE session_id = ? AND id >= ? - 1 AND id <= ? + 1
                       ORDER BY id""",
                    (match["session_id"], match["id"], match["id"]),
                )
                context_msgs = [
                    {"role": r["role"], "content": (r["content"] or "")[:200]}
                    for r in ctx_cursor.fetchall()
                ]
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

            # Remove full content from result (snippet is enough, saves tokens)
            match.pop("content", None)

        return matches

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source."""
        if source:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE source = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (source, limit, offset),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(self, source: str = None) -> int:
        """Count sessions, optionally filtered by source."""
        if source:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE source = ?", (source,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
        return cursor.fetchone()[0]

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        if session_id:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
            (session_id,),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages and audit log entries. Returns True if found."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
        )
        if cursor.fetchone()[0] == 0:
            return False
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM audit_log WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return True

    # =========================================================================
    # Audit log
    # =========================================================================

    # Keywords whose matching argument keys are redacted before storage.
    _SENSITIVE_KEY_FRAGMENTS = frozenset({
        "password", "secret", "token", "api_key", "apikey",
        "private_key", "secret_key", "auth_token", "auth_key",
        "credential",
    })

    @classmethod
    def _redact_arguments(cls, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *arguments* with sensitive values replaced.

        A key is considered sensitive when any of the ``_SENSITIVE_KEY_FRAGMENTS``
        appears as a substring of the lower-cased key name.
        """
        redacted: Dict[str, Any] = {}
        for k, v in arguments.items():
            if any(frag in k.lower() for frag in cls._SENSITIVE_KEY_FRAGMENTS):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = v
        return redacted

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        result_summary: str = None,
        duration_ms: int = None,
        is_error: bool = False,
    ) -> int:
        """Record a single tool-call execution in the audit log.

        Sensitive argument values (passwords, tokens, keys, etc.) are
        automatically redacted before storage.  Returns the new row ID.
        """
        safe_args = self._redact_arguments(arguments) if arguments else None
        capped_summary = (result_summary or "")[:500] or None
        cursor = self._conn.execute(
            """INSERT INTO audit_log
               (session_id, timestamp, tool_name, arguments,
                result_summary, duration_ms, is_error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                time.time(),
                tool_name,
                json.dumps(safe_args, ensure_ascii=False) if safe_args else None,
                capped_summary,
                duration_ms,
                1 if is_error else 0,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def query_audit_log(
        self,
        session_id: str = None,
        tool_name: str = None,
        since: float = None,
        until: float = None,
        errors_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query the audit log with optional filters.

        Args:
            session_id: Filter to entries for this session.
            tool_name:  Filter to entries for this tool (exact match).
            since:      Unix timestamp lower bound (inclusive).
            until:      Unix timestamp upper bound (inclusive).
            errors_only: When True only return entries where is_error = 1.
            limit:      Max rows to return.
            offset:     Pagination offset.

        Returns list of dicts with keys: id, session_id, timestamp, tool_name,
        arguments (parsed dict or None), result_summary, duration_ms, is_error.
        """
        where: List[str] = []
        params: List[Any] = []

        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if tool_name:
            where.append("tool_name = ?")
            params.append(tool_name)
        if since is not None:
            where.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            where.append("timestamp <= ?")
            params.append(until)
        if errors_only:
            where.append("is_error = 1")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, offset])

        cursor = self._conn.execute(
            f"SELECT * FROM audit_log {where_sql} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = []
        for row in cursor.fetchall():
            entry = dict(row)
            if entry.get("arguments"):
                try:
                    entry["arguments"] = json.loads(entry["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass
            rows.append(entry)
        return rows

    def audit_log_stats(
        self,
        session_id: str = None,
        since: float = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return tool-call frequency counts, sorted by most-called.

        Returns a list of dicts with keys: tool_name, call_count, error_count,
        avg_duration_ms.  Optionally scoped to a single session or time window.
        """
        where: List[str] = []
        params: List[Any] = []

        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if since is not None:
            where.append("timestamp >= ?")
            params.append(since)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)

        cursor = self._conn.execute(
            f"""SELECT tool_name,
                       COUNT(*) AS call_count,
                       SUM(is_error) AS error_count,
                       CAST(AVG(duration_ms) AS INTEGER) AS avg_duration_ms
                FROM audit_log {where_sql}
                GROUP BY tool_name
                ORDER BY call_count DESC
                LIMIT ?""",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

    def prune_sessions(self, older_than_days: int = 90, source: str = None) -> int:
        """
        Delete sessions older than N days. Returns count of deleted sessions.
        Only prunes ended sessions (not active ones).
        """
        import time as _time
        cutoff = _time.time() - (older_than_days * 86400)

        if source:
            cursor = self._conn.execute(
                """SELECT id FROM sessions
                   WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?""",
                (cutoff, source),
            )
        else:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                (cutoff,),
            )
        session_ids = [row["id"] for row in cursor.fetchall()]

        for sid in session_ids:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            self._conn.execute("DELETE FROM audit_log WHERE session_id = ?", (sid,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))

        self._conn.commit()
        return len(session_ids)

    def import_sessions(
        self,
        sessions_data: List[Dict[str, Any]],
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Import sessions from a list of dicts (one per JSONL line from export).

        Each dict must have the same shape as export_session() output:
        session-level fields + a "messages" list.

        Args:
            sessions_data: List of session dicts (with nested "messages").
            overwrite: If True, delete existing sessions with the same ID
                       before importing. If False (default), skip duplicates.

        Returns:
            Dict with keys: imported, skipped, errors, messages_imported.
        """
        imported = 0
        skipped = 0
        errors = []
        messages_imported = 0

        # Collect all session IDs currently in the DB for parent_session_id
        # validation and duplicate detection.
        existing_ids: set = set()
        cursor = self._conn.execute("SELECT id FROM sessions")
        for row in cursor:
            existing_ids.add(row["id"])

        # IDs that will exist after import (existing + newly imported)
        all_ids = set(existing_ids)
        for entry in sessions_data:
            sid = entry.get("id")
            if sid:
                all_ids.add(sid)

        for line_num, entry in enumerate(sessions_data, 1):
            sid = entry.get("id")
            if not sid:
                errors.append(f"Line {line_num}: missing session 'id' field")
                continue

            source = entry.get("source")
            if not source:
                errors.append(f"Line {line_num} ({sid}): missing 'source' field")
                continue

            started_at = entry.get("started_at")
            if started_at is None:
                errors.append(f"Line {line_num} ({sid}): missing 'started_at' field")
                continue

            # Duplicate handling
            if sid in existing_ids:
                if overwrite:
                    self._conn.execute(
                        "DELETE FROM audit_log WHERE session_id = ?", (sid,)
                    )
                    self._conn.execute(
                        "DELETE FROM messages WHERE session_id = ?", (sid,)
                    )
                    self._conn.execute(
                        "DELETE FROM sessions WHERE id = ?", (sid,)
                    )
                else:
                    skipped += 1
                    continue

            # parent_session_id: set to NULL if parent not in DB and not
            # being imported in this batch
            parent_id = entry.get("parent_session_id")
            if parent_id and parent_id not in all_ids:
                parent_id = None

            # model_config is already a JSON string from the export
            # (dict(row) returns the raw TEXT column). Don't double-encode.
            model_config = entry.get("model_config")

            # title: on UNIQUE conflict, set to NULL rather than failing
            title = entry.get("title")

            try:
                self._conn.execute(
                    """INSERT INTO sessions
                       (id, source, user_id, model, model_config,
                        system_prompt, parent_session_id, started_at,
                        ended_at, end_reason, message_count,
                        tool_call_count, input_tokens, output_tokens, title)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sid,
                        source,
                        entry.get("user_id"),
                        entry.get("model"),
                        model_config,
                        entry.get("system_prompt"),
                        parent_id,
                        started_at,
                        entry.get("ended_at"),
                        entry.get("end_reason"),
                        entry.get("message_count", 0),
                        entry.get("tool_call_count", 0),
                        entry.get("input_tokens", 0),
                        entry.get("output_tokens", 0),
                        title,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                # Most likely a duplicate title — retry with NULL title
                if "title" in str(exc).lower():
                    try:
                        self._conn.execute(
                            """INSERT INTO sessions
                               (id, source, user_id, model, model_config,
                                system_prompt, parent_session_id, started_at,
                                ended_at, end_reason, message_count,
                                tool_call_count, input_tokens, output_tokens,
                                title)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                       ?, ?, ?)""",
                            (
                                sid,
                                source,
                                entry.get("user_id"),
                                entry.get("model"),
                                model_config,
                                entry.get("system_prompt"),
                                parent_id,
                                started_at,
                                entry.get("ended_at"),
                                entry.get("end_reason"),
                                entry.get("message_count", 0),
                                entry.get("tool_call_count", 0),
                                entry.get("input_tokens", 0),
                                entry.get("output_tokens", 0),
                                None,
                            ),
                        )
                    except sqlite3.IntegrityError:
                        errors.append(
                            f"Line {line_num} ({sid}): duplicate session ID"
                        )
                        continue
                else:
                    errors.append(f"Line {line_num} ({sid}): {exc}")
                    continue

            # Insert messages — raw INSERT to preserve exported counters.
            # tool_calls was json.loads()'d during export, must re-serialize.
            messages = entry.get("messages") or []
            for msg in messages:
                tool_calls_raw = msg.get("tool_calls")
                if tool_calls_raw is not None:
                    if not isinstance(tool_calls_raw, str):
                        tool_calls_raw = json.dumps(tool_calls_raw)
                # else: already None

                self._conn.execute(
                    """INSERT INTO messages
                       (session_id, role, content, tool_call_id,
                        tool_calls, tool_name, timestamp, token_count,
                        finish_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sid,
                        msg.get("role", "user"),
                        msg.get("content"),
                        msg.get("tool_call_id"),
                        tool_calls_raw,
                        msg.get("tool_name"),
                        msg.get("timestamp", started_at),
                        msg.get("token_count"),
                        msg.get("finish_reason"),
                    ),
                )
                messages_imported += 1

            existing_ids.add(sid)
            imported += 1

        self._conn.commit()

        # Rebuild FTS5 index to ensure consistency after bulk insert
        try:
            self._conn.execute(
                "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # FTS5 not available or already consistent

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "messages_imported": messages_imported,
        }
