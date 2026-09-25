"""Per-thread response mute for the Slack adapter, toggled by a principal's emoji reaction.

A configured reaction (default ``:mute:``) placed by an allowed user on any message in a thread
marks that thread as muted: the adapter keeps ingesting the thread (auth, hydration, transcript
row) but does not start an agent turn there. Removing the reaction unmutes.

State is a *reactor set* per ``(channel_id, thread_ts)``: several principals may add the same
reaction, and the thread stays muted until the last of them removes theirs. Every toggle is an
``INSERT OR IGNORE`` / ``DELETE`` so a repeated add or a removal for an unknown reactor is a no-op
in both directions.

Persisted in a dedicated SQLite file (default ``$HERMES_HOME/slack_reaction_mute.db``) so a
gateway restart does not come back unmuted. Deliberately NOT a table in ``state.db``: that file
carries the versioned session schema and this feature must stay a plugin-local concern.

Modes (``slack.reaction_mute_mode`` / ``SLACK_REACTION_MUTE_MODE``):

* ``off`` (default): reactions are not tracked and nothing is suppressed.
* ``log-only``: toggles are tracked and every message that *would* be suppressed is recorded
  (``mute_events.kind = 'would_suppress'``) and logged, but the agent still responds.
* ``enforce``: suppressed messages are recorded (``kind = 'suppressed'``), appended to the thread
  session's transcript, and never reach the agent.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_LOG_ONLY = "log-only"
MODE_ENFORCE = "enforce"
MODES = (MODE_OFF, MODE_LOG_ONLY, MODE_ENFORCE)

DEFAULT_MUTE_EMOJI = "mute"
DEFAULT_DB_NAME = "slack_reaction_mute.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_mutes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id  TEXT NOT NULL,
    thread_ts   TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    UNIQUE (channel_id, thread_ts)
);
CREATE TABLE IF NOT EXISTS thread_mute_reactors (
    channel_id  TEXT NOT NULL,
    thread_ts   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    added_at    REAL NOT NULL,
    UNIQUE (channel_id, thread_ts, user_id)
);
CREATE TABLE IF NOT EXISTS mute_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    mode        TEXT,
    channel_id  TEXT,
    thread_ts   TEXT,
    user_id     TEXT,
    message_ts  TEXT,
    reaction    TEXT,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_mute_events_thread ON mute_events (channel_id, thread_ts, id);
"""


def normalize_mode(raw: Any) -> str:
    """Coerce a config/env value to one of :data:`MODES`. Unknown values are ``off`` (fail closed:
    a typo must never silently enforce)."""
    if raw is None:
        return MODE_OFF
    if isinstance(raw, bool):
        return MODE_ENFORCE if raw else MODE_OFF
    text = str(raw).strip().lower().replace("_", "-").replace(" ", "-")
    if text in {"enforce", "enforced", "on", "true", "1", "yes"}:
        return MODE_ENFORCE
    if text in {"log-only", "logonly", "log", "dry-run", "dryrun", "observe", "shadow"}:
        return MODE_LOG_ONLY
    return MODE_OFF


def normalize_emoji(raw: Any) -> str:
    """``:mute:`` / ``mute`` / ``MUTE`` → ``mute``; empty → the default."""
    text = str(raw or "").strip().strip(":").lower()
    return text or DEFAULT_MUTE_EMOJI


class ReactionMuteStore:
    """SQLite-backed reactor set per thread plus an append-only audit log.

    All methods are synchronous and cheap (single-row statements); the adapter calls them from
    the event loop. A process-wide lock serializes access to the shared connection.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # ── connection ──────────────────────────────────────────────────────────

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ── toggles ─────────────────────────────────────────────────────────────

    def add_reactor(self, channel_id: str, thread_ts: str, user_id: str) -> Tuple[bool, bool]:
        """Record ``user_id``'s mute reaction on the thread.

        Returns ``(muted_now, changed)``: ``changed`` is False when this reactor was already
        recorded (idempotent re-add)."""
        now = time.time()
        with self._lock:
            conn = self._connection()
            with conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO thread_mute_reactors (channel_id, thread_ts, user_id, added_at) "
                    "VALUES (?, ?, ?, ?)", (channel_id, thread_ts, user_id, now))
                changed = cur.rowcount == 1
                conn.execute(
                    "INSERT INTO thread_mutes (channel_id, thread_ts, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (channel_id, thread_ts) DO UPDATE SET updated_at = excluded.updated_at",
                    (channel_id, thread_ts, now, now))
            return True, changed

    def remove_reactor(self, channel_id: str, thread_ts: str, user_id: str) -> Tuple[bool, bool]:
        """Forget ``user_id``'s mute reaction. The thread unmutes only when no reactor remains.

        Returns ``(muted_now, changed)``; ``changed`` is False for a reactor that was never
        recorded (idempotent removal, e.g. an event missed while the gateway was down)."""
        with self._lock:
            conn = self._connection()
            with conn:
                cur = conn.execute(
                    "DELETE FROM thread_mute_reactors WHERE channel_id = ? AND thread_ts = ? AND user_id = ?",
                    (channel_id, thread_ts, user_id))
                changed = cur.rowcount == 1
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM thread_mute_reactors WHERE channel_id = ? AND thread_ts = ?",
                    (channel_id, thread_ts)).fetchone()[0]
                if remaining == 0:
                    conn.execute(
                        "DELETE FROM thread_mutes WHERE channel_id = ? AND thread_ts = ?",
                        (channel_id, thread_ts))
                else:
                    conn.execute(
                        "UPDATE thread_mutes SET updated_at = ? WHERE channel_id = ? AND thread_ts = ?",
                        (time.time(), channel_id, thread_ts))
            return remaining > 0, changed

    # ── queries ─────────────────────────────────────────────────────────────

    def is_muted(self, channel_id: str, thread_ts: str) -> bool:
        with self._lock:
            row = self._connection().execute(
                "SELECT 1 FROM thread_mutes WHERE channel_id = ? AND thread_ts = ? LIMIT 1",
                (channel_id, thread_ts)).fetchone()
            return row is not None

    def reactors(self, channel_id: str, thread_ts: str) -> Set[str]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT user_id FROM thread_mute_reactors WHERE channel_id = ? AND thread_ts = ?",
                (channel_id, thread_ts)).fetchall()
            return {r["user_id"] for r in rows}

    def list_mutes(self) -> List[Dict[str, Any]]:
        """Every muted thread with its reactor set (operator/debug helper)."""
        with self._lock:
            conn = self._connection()
            mutes = conn.execute(
                "SELECT channel_id, thread_ts, created_at, updated_at FROM thread_mutes "
                "ORDER BY updated_at DESC").fetchall()
            out: List[Dict[str, Any]] = []
            for m in mutes:
                reactors = conn.execute(
                    "SELECT user_id FROM thread_mute_reactors WHERE channel_id = ? AND thread_ts = ? "
                    "ORDER BY added_at", (m["channel_id"], m["thread_ts"])).fetchall()
                out.append({
                    "channel_id": m["channel_id"], "thread_ts": m["thread_ts"],
                    "created_at": m["created_at"], "updated_at": m["updated_at"],
                    "reactors": [r["user_id"] for r in reactors]})
            return out

    # ── audit ───────────────────────────────────────────────────────────────

    def record_event(
        self, kind: str, *, mode: Optional[str] = None, channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None, user_id: Optional[str] = None,
        message_ts: Optional[str] = None, reaction: Optional[str] = None,
        detail: Optional[str] = None) -> None:
        """Append an audit row. Kinds: ``muted``, ``unmuted``, ``noop_add``, ``noop_remove``,
        ``ignored_non_principal``, ``would_suppress``, ``suppressed``, ``command_bypass``.
        Never raises: the audit trail must not break message handling."""
        try:
            with self._lock:
                conn = self._connection()
                with conn:
                    conn.execute(
                        "INSERT INTO mute_events (ts, kind, mode, channel_id, thread_ts, user_id, "
                        "message_ts, reaction, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (time.time(), kind, mode, channel_id, thread_ts, user_id, message_ts,
                         reaction, detail))
        except Exception:
            logger.debug("[Slack] reaction_mute audit write failed", exc_info=True)

    def events(self, *, limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connection()
            if kind:
                rows = conn.execute(
                    "SELECT * FROM mute_events WHERE kind = ? ORDER BY id DESC LIMIT ?",
                    (kind, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM mute_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
