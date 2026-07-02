"""
SQLite-backed session-orchestration registry.

Architecture — single-writer discipline
----------------------------------------
The cron watcher is the **sole mutator** of registry rows and derived counters.
All other callers (webhook-adopt, Discord-drive) MUST NOT write the
``session_orchestration`` table directly.  Instead they append intent rows to
``session_orchestration_queue``; the cron watcher drains that queue and applies
the intents in a single serialised transaction.

This eliminates the lost-update class: there is only one writer to the
registry, so no compare-and-swap or optimistic-lock scheme is needed for
counter updates.  Counter bumps use atomic SQL expressions
(``SET col = col + 1``) rather than read-modify-write in Python, so a
process crash mid-increment cannot leave a stale value.

``repo`` — canonical stable key
---------------------------------
``repo`` is a 12-character hex prefix of the SHA-256 of the *normalised*
repository identity string:

  * If the working directory has a ``git remote`` named "origin", the
    normalised remote URL is used (stripped of trailing ``.git``, lowercased,
    scheme removed so ``https://github.com/foo/bar`` and
    ``git@github.com:foo/bar`` both hash to the same value).
  * If there is no remote, the absolute ``workdir`` path is used as-is.

The normalisation is deterministic across machines for the same remote, so
two checkouts of the same repository share the same ``repo`` key, while
different repositories (different remote URLs) always differ.

Call :func:`canonical_repo_id` to compute the key before inserting.

Lock TTL
---------
``lock_ts`` stores the wallclock time (ISO-8601 UTC string from
``datetime('now')`` in SQLite) at which the lock was acquired.
``acquire_lock`` refuses to grant the lock if a non-expired row exists.
Callers pass ``ttl_seconds`` (default: 5× cron interval = 300 s) and
the lock is auto-reclaimed via the ``lock_ts`` column.

Queue intent kinds
------------------
  ``adopt``      — register a session spawned externally (webhook path).
  ``drive``      — queue a message to relay to the agent's tmux session.
  ``update``     — set arbitrary fields on an existing row (cron applies them).
  ``terminate``  — kill the tmux process and mark the row terminal (watcher applies).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from session_orchestration.types import SessionLifecycle

logger = logging.getLogger(__name__)
_UNSET = object()

# ---------------------------------------------------------------------------
# Default busy-timeout: how long SQLite waits for a write lock before
# raising OperationalError("database is locked").  5 s is generous; the
# cron watcher holds the lock for milliseconds.
# ---------------------------------------------------------------------------
_BUSY_TIMEOUT_MS = 5_000

# ---------------------------------------------------------------------------
# DDL — added to state.db via hermes_state._init_schema()
# ---------------------------------------------------------------------------

SESSION_ORCHESTRATION_DDL = """
CREATE TABLE IF NOT EXISTS session_orchestration (
    task_id                TEXT PRIMARY KEY,
    agent                  TEXT NOT NULL,
    tmux_session           TEXT,
    project                TEXT,
    discord_thread_id      TEXT,
    hermes_session_key     TEXT,
    workdir                TEXT,
    state                  TEXT NOT NULL DEFAULT 'RUNNING',
    idle_ticks             INTEGER NOT NULL DEFAULT 0,
    nudge_count            INTEGER NOT NULL DEFAULT 0,
    last_pane_hash         TEXT,
    last_output_ts         REAL,
    heartbeat_counter      INTEGER NOT NULL DEFAULT 0,
    status_message_id      TEXT,
    feed_message_id        TEXT,
    lock_holder            TEXT,
    lock_ts                TEXT,
    source                 TEXT NOT NULL DEFAULT 'spawn',
    run_id                 TEXT,
    repo                   TEXT,
    marker_offset          INTEGER NOT NULL DEFAULT 0,
    terminated_at          REAL,
    attention_since        REAL,
    last_renudge_at        REAL,
    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at             TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, repo)
);

CREATE INDEX IF NOT EXISTS idx_so_state
    ON session_orchestration(state);
CREATE INDEX IF NOT EXISTS idx_so_run_repo
    ON session_orchestration(run_id, repo);

CREATE TABLE IF NOT EXISTS session_orchestration_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT,
    run_id     TEXT,
    repo       TEXT,
    intent     TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_soq_task
    ON session_orchestration_queue(task_id);

CREATE TABLE IF NOT EXISTS session_orchestration_attention_items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id               TEXT NOT NULL,
    reason                TEXT NOT NULL,
    state                 TEXT NOT NULL DEFAULT 'unresolved',
    priority              INTEGER NOT NULL DEFAULT 0,
    detail                TEXT,
    opened_at             TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at           TEXT,
    resolution_reason     TEXT,
    FOREIGN KEY(task_id) REFERENCES session_orchestration(task_id) ON DELETE CASCADE,
    CHECK (
        (
            state = 'unresolved'
            AND resolved_at IS NULL
            AND resolution_reason IS NULL
        )
        OR
        (
            state = 'resolved'
            AND resolved_at IS NOT NULL
            AND resolution_reason IS NOT NULL
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_soai_unresolved_task_reason
    ON session_orchestration_attention_items(task_id, reason)
    WHERE resolved_at IS NULL;
DROP INDEX IF EXISTS idx_soai_unresolved_lookup;
CREATE INDEX IF NOT EXISTS idx_soai_unresolved_lookup
    ON session_orchestration_attention_items(state, priority DESC, opened_at)
    WHERE resolved_at IS NULL;

CREATE TABLE IF NOT EXISTS session_orchestration_projection (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id            TEXT NOT NULL,
    projection_name       TEXT NOT NULL,
    message_id            TEXT,
    content_hash          TEXT,
    payload               TEXT NOT NULL DEFAULT '{}',
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(channel_id, projection_name)
);
DROP INDEX IF EXISTS idx_sop_channel_projection;

"""

# ---------------------------------------------------------------------------
# Canonical repo-id derivation
# ---------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[a-z+]+://", re.IGNORECASE)
_GIT_SCP_RE = re.compile(r"^[^/@]+@[^:]+:", re.IGNORECASE)


def _normalise_remote_url(url: str) -> str:
    """Strip scheme, credentials, and trailing .git; lowercase."""
    url = url.strip()
    # ssh://git@github.com/foo/bar.git  →  github.com/foo/bar
    url = _SCHEME_RE.sub("", url)
    # git@github.com:foo/bar.git  →  github.com/foo/bar
    url = _GIT_SCP_RE.sub(lambda m: m.group(0).split("@", 1)[1].replace(":", "/"), url)
    # Remove credentials (user:pass@host → host)
    if "@" in url:
        url = url.split("@", 1)[1]
    # Drop trailing .git
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower().rstrip("/")


def canonical_repo_id(workdir: Optional[str] = None, remote_url: Optional[str] = None) -> str:
    """Return a 12-char hex SHA-256 prefix that stably identifies a repo.

    Parameters
    ----------
    workdir:
        Absolute path to the working directory.  Used as fallback when
        ``remote_url`` is absent.
    remote_url:
        The git remote URL (e.g. from ``git remote get-url origin``).
        When present, the normalised URL is hashed so two checkouts of
        the same remote produce the same id.

    Returns a 12-hex-character string (48 bits of collision resistance —
    adequate for a registry with O(hundreds) of concurrent sessions).
    """
    if remote_url:
        key = _normalise_remote_url(remote_url)
    elif workdir:
        key = str(Path(workdir).resolve())
    else:
        key = f"unknown-{time.time()}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return digest[:12]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SessionOrchestrationRegistry:
    """SQLite-backed registry for managed external-agent sessions.

    Callers obtain a connection via the existing hermes_state machinery or
    pass a ``db_path`` directly.  The constructor sets ``busy_timeout`` on
    every connection, regardless of origin.

    Thread / process safety
    -----------------------
    All writes use ``BEGIN IMMEDIATE`` so SQLite serialises concurrent writers
    at the WAL write-lock level.  Counter updates are atomic SQL expressions
    (never read-modify-write in Python) so the cron watcher can bump counters
    without a separate SELECT.

    The cron watcher is the sole caller of :meth:`upsert`, :meth:`drain_intents`,
    :meth:`acquire_lock`, and :meth:`release_lock`.  All other callers use only
    :meth:`enqueue_intent` (append-only, safe from any thread/process) and
    :meth:`get` / :meth:`list` (read-only).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        busy_timeout_ms: int = _BUSY_TIMEOUT_MS,
    ) -> None:
        if db_path is None:
            # Import lazily to avoid a circular dep at module load time.
            from hermes_state import DEFAULT_DB_PATH
            db_path = DEFAULT_DB_PATH

        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._ensure_schema()
        self._migrate_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with busy_timeout and WAL-compatible settings."""
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,  # we manage transactions explicitly
        )
        conn.row_factory = sqlite3.Row
        # busy_timeout makes SQLite spin-wait (with exponential back-off)
        # rather than immediately returning SQLITE_BUSY.  This is the
        # primary guard against lost writes under concurrent readers.
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        """Create session_orchestration tables if they don't exist yet.

        Uses ``IF NOT EXISTS`` throughout, so this is idempotent and safe
        to call on every startup even when the tables were created by an
        earlier version.
        """
        conn = self._connect()
        try:
            conn.executescript(SESSION_ORCHESTRATION_DDL)
        finally:
            conn.close()

    def _migrate_schema(self) -> None:
        """Add new columns to existing tables (idempotent ALTER TABLE).

        Called after ``_ensure_schema`` on every startup.  Each ALTER TABLE
        is wrapped in a try/except so repeated calls on an already-migrated
        DB are silent no-ops (SQLite raises OperationalError when the column
        already exists).
        """
        def _do(conn: sqlite3.Connection) -> None:
            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN marker_offset INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN discord_user_id TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN last_question TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN terminated_at REAL"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN feed_message_id TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN attention_since REAL"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN last_renudge_at REAL"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            # Answerable needs-input: the ordered option labels (JSON array) and
            # the input kind ("menu" | "prompt" | "") extracted from the pane on
            # WAITING_USER. omp emits no markers, so the watcher populates these
            # by pane-text parsing (see session_orchestration/menu_parse.py).
            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN last_options TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

            try:
                conn.execute(
                    "ALTER TABLE session_orchestration "
                    "ADD COLUMN last_input_kind TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — idempotent

        self._write(_do)

    def _write(self, fn, conn: Optional[sqlite3.Connection] = None) -> Any:
        """Execute *fn(conn)* inside a BEGIN IMMEDIATE transaction.

        Closes the connection when ``conn`` is None (i.e. we opened it).
        Returns whatever *fn* returns.
        """
        owned = conn is None
        if owned:
            conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
                conn.execute("COMMIT")
                return result
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        finally:
            if owned:
                conn.close()

    # ------------------------------------------------------------------
    # Public API — read paths (any caller)
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return a registry row as a dict, or None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM session_orchestration WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list(
        self,
        *,
        state: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all (or filtered) registry rows as dicts."""
        clauses: List[str] = []
        params: List[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM session_orchestration {where}",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_unresolved_attention_items(self) -> List[Dict[str, Any]]:
        """Return currently unresolved attention items in render-stable order."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT *
                  FROM session_orchestration_attention_items
                 WHERE state = 'unresolved'
                   AND resolved_at IS NULL
                 ORDER BY priority DESC, opened_at ASC, id ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_projection(
        self,
        channel_id: str,
        projection_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Return projection metadata for a channel/projection pair, if present."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT *
                  FROM session_orchestration_projection
                 WHERE channel_id = ?
                   AND projection_name = ?
                """,
                (channel_id, projection_name),
            ).fetchone()
            if row is None:
                return None
            projection = dict(row)
            projection["payload"] = json.loads(projection["payload"] or "{}")
            return projection
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API — watcher-owned attention/projection mutations
    # ------------------------------------------------------------------

    def open_attention_item(
        self,
        task_id: str,
        reason: str,
        *,
        priority: int = 0,
        detail: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Open or refresh an unresolved attention item for ``task_id``/``reason``.

        Watcher-owned API: external actors must not use this as an authority
        for session state.  Reopening the same unresolved reason refreshes only
        attention lifecycle fields instead of inserting a duplicate.
        """
        def _do(conn: sqlite3.Connection) -> Dict[str, Any]:
            existing = conn.execute(
                """
                SELECT id
                  FROM session_orchestration_attention_items
                 WHERE task_id = ?
                   AND reason = ?
                   AND resolved_at IS NULL
                """,
                (task_id, reason),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO session_orchestration_attention_items (
                        task_id, reason, state, priority, detail
                    ) VALUES (?, ?, 'unresolved', ?, ?)
                    """,
                    (task_id, reason, priority, detail),
                )
                item_id = cursor.lastrowid
            else:
                item_id = existing["id"]
                conn.execute(
                    """
                    UPDATE session_orchestration_attention_items
                       SET priority = ?,
                           detail = ?,
                           updated_at = datetime('now')
                     WHERE id = ?
                    """,
                    (priority, detail, item_id),
                )

            row = conn.execute(
                "SELECT * FROM session_orchestration_attention_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            return dict(row)

        return self._write(_do)

    def update_attention_item(
        self,
        task_id: str,
        reason: str,
        *,
        priority: Optional[int] = None,
        detail: Any = _UNSET,
    ) -> Optional[Dict[str, Any]]:
        """Refresh lifecycle fields on an unresolved attention item."""
        set_parts: List[str] = ["updated_at = datetime('now')"]
        values: List[Any] = []
        if priority is not None:
            set_parts.append("priority = ?")
            values.append(priority)
        if detail is not _UNSET:
            set_parts.append("detail = ?")
            values.append(detail)

        def _do(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
            update_values = [*values, task_id, reason]
            conn.execute(
                f"""
                UPDATE session_orchestration_attention_items
                   SET {', '.join(set_parts)}
                 WHERE task_id = ?
                   AND reason = ?
                   AND resolved_at IS NULL
                """,
                update_values,
            )
            row = conn.execute(
                """
                SELECT *
                  FROM session_orchestration_attention_items
                 WHERE task_id = ?
                   AND reason = ?
                   AND resolved_at IS NULL
                """,
                (task_id, reason),
            ).fetchone()
            return dict(row) if row else None

        return self._write(_do)

    def resolve_attention_item(
        self,
        task_id: str,
        reason: str,
        resolution_reason: str,
    ) -> bool:
        """Resolve the currently unresolved attention item for one reason."""
        def _do(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """
                UPDATE session_orchestration_attention_items
                   SET state = 'resolved',
                       resolved_at = datetime('now'),
                       resolution_reason = ?,
                       updated_at = datetime('now')
                 WHERE task_id = ?
                   AND reason = ?
                   AND resolved_at IS NULL
                """,
                (resolution_reason, task_id, reason),
            )
            return cursor.rowcount > 0

        return bool(self._write(_do))

    def upsert_projection(
        self,
        channel_id: str,
        projection_name: str,
        *,
        message_id: Optional[str] | object = _UNSET,
        content_hash: Optional[str] | object = _UNSET,
        payload: Optional[Dict[str, Any]] | object = _UNSET,
    ) -> Dict[str, Any]:
        """Insert or update watcher-owned projection metadata.

        Omitted optional fields preserve existing values on update; passing
        ``None`` explicitly clears nullable fields and resets ``payload`` to
        ``{}``.
        """

        def _do(conn: sqlite3.Connection) -> Dict[str, Any]:
            existing = conn.execute(
                """
                SELECT id
                  FROM session_orchestration_projection
                 WHERE channel_id = ?
                   AND projection_name = ?
                """,
                (channel_id, projection_name),
            ).fetchone()
            if existing is None:
                insert_message_id = None if message_id is _UNSET else message_id
                insert_content_hash = None if content_hash is _UNSET else content_hash
                insert_payload = {} if payload is _UNSET else (payload or {})
                cursor = conn.execute(
                    """
                    INSERT INTO session_orchestration_projection (
                        channel_id, projection_name, message_id, content_hash, payload
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        channel_id,
                        projection_name,
                        insert_message_id,
                        insert_content_hash,
                        json.dumps(insert_payload, sort_keys=True),
                    ),
                )
                projection_id = cursor.lastrowid
            else:
                projection_id = existing["id"]
                set_parts: List[str] = ["updated_at = datetime('now')"]
                values: List[Any] = []
                if message_id is not _UNSET:
                    set_parts.append("message_id = ?")
                    values.append(message_id)
                if content_hash is not _UNSET:
                    set_parts.append("content_hash = ?")
                    values.append(content_hash)
                if payload is not _UNSET:
                    set_parts.append("payload = ?")
                    values.append(json.dumps(payload or {}, sort_keys=True))
                values.append(projection_id)
                conn.execute(
                    f"""
                    UPDATE session_orchestration_projection
                       SET {', '.join(set_parts)}
                     WHERE id = ?
                    """,
                    values,
                )

            row = conn.execute(
                "SELECT * FROM session_orchestration_projection WHERE id = ?",
                (projection_id,),
            ).fetchone()
            projection = dict(row)
            projection["payload"] = json.loads(projection["payload"] or "{}")
            return projection

        return self._write(_do)

    # ------------------------------------------------------------------
    # Public API — core cron-watcher-only session mutations
    # ------------------------------------------------------------------

    def upsert(
        self,
        task_id: str,
        *,
        agent: str,
        run_id: Optional[str] = None,
        repo: Optional[str] = None,
        source: str = "spawn",
        **fields: Any,
    ) -> None:
        """Insert or update a registry row.

        **Cron watcher only.** All other callers must use :meth:`enqueue_intent`.

        On conflict (same ``task_id``) the row is updated in-place.
        On conflict on the ``UNIQUE(run_id, repo)`` constraint (a duplicate
        adopt/spawn for the same logical repository + run), the existing row
        wins (INSERT OR IGNORE semantics) and an INFO log is emitted.

        Counter columns (``idle_ticks``, ``nudge_count``, ``heartbeat_counter``)
        are *not* set here unless explicitly passed — use the atomic increment
        helpers or ``_apply_intent`` for those.
        """
        # Determine which columns to set
        all_fields: Dict[str, Any] = {
            "agent": agent,
            "run_id": run_id,
            "repo": repo,
            "source": source,
            **fields,
        }

        def _do(conn: sqlite3.Connection) -> None:
            # Try INSERT first; on task_id conflict, UPDATE.
            # On (run_id, repo) conflict, log and skip.
            existing = conn.execute(
                "SELECT task_id FROM session_orchestration WHERE task_id = ?",
                (task_id,),
            ).fetchone()

            if existing is None:
                # Check for (run_id, repo) duplicate before INSERT
                if run_id is not None and repo is not None:
                    dup = conn.execute(
                        "SELECT task_id FROM session_orchestration "
                        "WHERE run_id = ? AND repo = ? AND task_id != ?",
                        (run_id, repo, task_id),
                    ).fetchone()
                    if dup is not None:
                        logger.info(
                            "registry.upsert: duplicate (run_id=%s, repo=%s) "
                            "already exists as task_id=%s — skipping insert of %s",
                            run_id,
                            repo,
                            dup[0],
                            task_id,
                        )
                        return

                col_names = ["task_id"] + list(all_fields.keys())
                placeholders = ", ".join("?" * len(col_names))
                col_clause = ", ".join(col_names)
                values = [task_id] + list(all_fields.values())
                conn.execute(
                    f"INSERT INTO session_orchestration ({col_clause}) "
                    f"VALUES ({placeholders})",
                    values,
                )
            else:
                # UPDATE — only set non-None fields; always bump updated_at.
                set_parts: List[str] = ["updated_at = datetime('now')"]
                set_vals: List[Any] = []
                for k, v in all_fields.items():
                    if v is not None:
                        set_parts.append(f"{k} = ?")
                        set_vals.append(v)
                if len(set_parts) > 1:
                    set_vals.append(task_id)
                    conn.execute(
                        f"UPDATE session_orchestration SET {', '.join(set_parts)} "
                        f"WHERE task_id = ?",
                        set_vals,
                    )

        self._write(_do)

    def increment_counter(self, task_id: str, column: str, by: int = 1) -> None:
        """Atomically increment a counter column.

        **Cron watcher only.**

        Uses ``SET col = col + ?`` (atomic SQL expression) — never
        read-modify-write in Python.

        Parameters
        ----------
        task_id:
            Registry row to update.
        column:
            One of ``heartbeat_counter``, ``idle_ticks``, ``nudge_count``.
        by:
            Increment amount (default 1).
        """
        _ALLOWED_COUNTERS = frozenset(
            {"heartbeat_counter", "idle_ticks", "nudge_count"}
        )
        if column not in _ALLOWED_COUNTERS:
            raise ValueError(
                f"increment_counter: column {column!r} not in allowed set "
                f"{_ALLOWED_COUNTERS}"
            )

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE session_orchestration "
                f"SET {column} = {column} + ?, updated_at = datetime('now') "
                f"WHERE task_id = ?",
                (by, task_id),
            )

        self._write(_do)

    def drain_intents(self) -> List[Dict[str, Any]]:
        """Return and delete all pending queue rows (oldest first).

        **Cron watcher only.**

        The cron watcher calls this once per tick, applies each intent
        via :meth:`_apply_intent`, and then commits.  The DELETE and the
        application of each intent happen inside a single
        ``BEGIN IMMEDIATE`` transaction so no intent is lost on crash.

        Returns the list of intent dicts (already removed from the queue).
        """
        def _do(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM session_orchestration_queue ORDER BY id ASC"
            ).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                conn.execute(
                    "DELETE FROM session_orchestration_queue WHERE id IN "
                    f"({','.join('?' * len(ids))})",
                    ids,
                )
            return [dict(r) for r in rows]

        return self._write(_do) or []

    def _apply_intent(self, intent: Dict[str, Any]) -> None:
        """Apply a single drained intent to the registry (cron watcher only).

        Intent kinds:

        ``adopt``
            Upsert a new row (webhook-adopt path).  ``payload`` must contain
            ``agent``, ``run_id``, ``repo``, and optionally other columns.

        ``drive``
            Store a pending drive payload on the row (column ``drive_payload``
            is not in the schema by default; store in ``state_meta`` or as a
            future column).  For now this just updates ``updated_at`` so the
            watcher knows something is pending.

        ``update``
            Set arbitrary ``payload`` columns on the named ``task_id``.

        ``terminate``
            Kill the tmux process and mark the session terminal.  The full
            application (``adapter.terminate()`` + state update) is the
            watcher's job (level 2).  Payload shape: ``{"restart": bool}``.
            Recognised here to avoid an unknown-kind warning in the log; the
            watcher is the actual consumer.
        """
        kind = intent.get("intent", "")
        import json as _json
        payload: Dict[str, Any] = _json.loads(intent.get("payload", "{}"))

        if kind == "adopt":
            task_id = intent.get("task_id") or payload.get("task_id")
            if not task_id:
                logger.warning("registry._apply_intent: adopt intent missing task_id")
                return
            # Pop all explicitly-named upsert kwargs so they don't collide
            # with positional args when the caller includes them in payload.
            payload.pop("task_id", None)
            agent = payload.pop("agent", "unknown")
            run_id = payload.pop("run_id", None)
            repo = payload.pop("repo", None)
            self.upsert(
                task_id,
                agent=agent,
                run_id=run_id,
                repo=repo,
                source="adopt",
                **payload,
            )

        elif kind == "update":
            task_id = intent.get("task_id") or payload.get("task_id")
            if not task_id:
                logger.warning("registry._apply_intent: update intent missing task_id")
                return
            payload.pop("task_id", None)
            if payload:
                def _do(conn: sqlite3.Connection) -> None:
                    set_parts = ["updated_at = datetime('now')"]
                    vals: List[Any] = []
                    for k, v in payload.items():
                        set_parts.append(f"{k} = ?")
                        vals.append(v)
                    vals.append(task_id)
                    conn.execute(
                        f"UPDATE session_orchestration SET {', '.join(set_parts)} "
                        f"WHERE task_id = ?",
                        vals,
                    )
                self._write(_do)

        elif kind == "drive":
            # Drive intents are consumed by the relay module; the registry
            # records them as pending so the watcher knows a drive is queued.
            # We just touch updated_at here; the relay reads from the queue
            # directly before it is drained.
            task_id = intent.get("task_id") or payload.get("task_id")
            if task_id:
                def _do(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        "UPDATE session_orchestration "
                        "SET updated_at = datetime('now') WHERE task_id = ?",
                        (task_id,),
                    )
                self._write(_do)

        elif kind == "terminate":
            task_id = intent.get("task_id") or payload.get("task_id")
            if not task_id:
                logger.warning(
                    "registry._apply_intent: terminate intent missing task_id"
                )
                return
            restart = payload.get("restart", False)
            existing = self.get(task_id)
            if existing is None:
                logger.warning(
                    "registry._apply_intent: terminate for unknown task_id=%s", task_id
                )
                return
            terminal_state = (
                SessionLifecycle.ERROR.value if restart else SessionLifecycle.DONE.value
            )
            self.upsert(
                task_id,
                agent=existing.get("agent", "unknown"),
                run_id=existing.get("run_id"),
                repo=existing.get("repo"),
                source=existing.get("source", "spawn"),
                state=terminal_state,
                terminated_at=time.time(),
            )

        else:
            logger.warning("registry._apply_intent: unknown intent kind %r", kind)

    def gc_terminal_rows(self, *, now: float, max_age_seconds: int) -> int:
        """Delete terminal rows whose ``terminated_at`` stamp is old enough.

        A row is eligible for GC when ALL of:

        - ``state`` is in the terminal set (``DONE`` or ``ERROR``).
        - ``terminated_at IS NOT NULL`` (rows without a stamp are never GC-ed
          so legacy pre-stamp rows are preserved indefinitely).
        - ``terminated_at < now - max_age_seconds`` (the row is older than the
          configured retention window).

        The DELETE runs inside a single ``BEGIN IMMEDIATE`` transaction (the
        watcher is the sole mutator, so this is safe within the tick).

        Parameters
        ----------
        now:
            Current epoch timestamp (seconds).  Pass ``time.time()`` in
            production; inject a fixed value in tests for determinism.
        max_age_seconds:
            Retention window in seconds.  Rows with a ``terminated_at`` older
            than ``now - max_age_seconds`` are deleted.

        Returns
        -------
        int
            Number of rows deleted.
        """
        cutoff = now - max_age_seconds
        terminal_states = (
            SessionLifecycle.DONE.value,
            SessionLifecycle.ERROR.value,
        )

        def _do(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "DELETE FROM session_orchestration "
                "WHERE state IN (?, ?) "
                "AND terminated_at IS NOT NULL "
                "AND terminated_at < ?",
                (*terminal_states, cutoff),
            )
            return cur.rowcount

        return self._write(_do) or 0

    def set_feed_message_id(
        self,
        task_id: str,
        feed_message_id: Optional[str],
    ) -> None:
        """Persist the Discord feed-channel message id for *task_id*.

        **Cron watcher only.**  Called by ``_on_turn_change`` after a feed
        POST/PATCH/DELETE to keep the registry in sync with the live board.

        Parameters
        ----------
        task_id:
            Registry row to update.
        feed_message_id:
            The Discord message id of the live feed-board entry, or ``None``
            to clear it (used after a DELETE when the session terminates).
        """
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE session_orchestration "
                "SET feed_message_id = ?, updated_at = datetime('now') "
                "WHERE task_id = ?",
                (feed_message_id, task_id),
            )

        self._write(_do)

    def set_attention_stamps(
        self,
        task_id: str,
        attention_since: Optional[float],
        last_renudge_at: Optional[float],
    ) -> None:
        """Persist attention_since and last_renudge_at for *task_id*.

        **Cron watcher only.**  Pass ``None`` to clear either column to NULL.
        Called by ``_process_row`` when a session enters or leaves an attention
        state (WAITING_USER / PAUSED_HANDOFF) and when a re-nudge fires.

        Parameters
        ----------
        task_id:
            Registry row to update.
        attention_since:
            Epoch timestamp when the row entered the current attention state,
            or ``None`` to clear (row is leaving attention).
        last_renudge_at:
            Epoch timestamp of the most recent re-nudge, or ``None`` to reset
            the debounce marker (row just entered attention, no nudge yet).
        """
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE session_orchestration "
                "SET attention_since = ?, last_renudge_at = ?, "
                "    updated_at = datetime('now') "
                "WHERE task_id = ?",
                (attention_since, last_renudge_at, task_id),
            )

        self._write(_do)

    # ------------------------------------------------------------------
    # Public API — queue write (any caller, append-only)
    # ------------------------------------------------------------------

    def enqueue_intent(
        self,
        intent: str,
        *,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
        repo: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an intent to the queue for the cron watcher to drain.

        Safe to call from any thread or process (webhook-adopt, Discord-drive).
        Does NOT write to ``session_orchestration`` directly.

        Parameters
        ----------
        intent:
            One of ``"adopt"``, ``"drive"``, ``"update"``, ``"terminate"``.
        task_id:
            Target registry row (required for ``drive``/``update``; optional
            for ``adopt`` when the task_id is inside ``payload``).
        run_id, repo:
            Correlation keys for ``adopt`` intents where the ``task_id`` is
            not yet known.
        payload:
            Arbitrary JSON-serialisable dict passed to the cron drainer.
        """
        import json as _json
        payload_str = _json.dumps(payload or {})

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO session_orchestration_queue "
                "(task_id, run_id, repo, intent, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, run_id, repo, intent, payload_str),
            )

        self._write(_do)

    def enqueue_terminate(self, task_id: str, *, restart: bool = False) -> None:
        """Queue a terminate intent for ``task_id``.

        Safe to call from any thread or process.  The cron watcher drains
        the intent and applies it (kill tmux + mark terminal) at its next tick.

        Parameters
        ----------
        task_id:
            Target registry row to terminate.
        restart:
            When ``True`` the watcher will re-spawn the session after killing
            it (``/so-restart`` path).  When ``False`` the session is stopped
            permanently (``/so-stop`` path).
        """
        self.enqueue_intent(
            "terminate",
            task_id=task_id,
            payload={"restart": restart},
        )

    # ------------------------------------------------------------------
    # Per-session lock (cron watcher + relay)
    # ------------------------------------------------------------------

    def acquire_lock(
        self,
        task_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Atomically acquire the per-session lock for ``task_id``.

        Returns True on success.  Returns False if a non-expired lock is
        already held by a different holder.

        Expired locks (``lock_ts`` + ``ttl_seconds`` < wallclock) are
        auto-reclaimed: the lock is taken by the new holder.

        The TTL is checked using the Python wallclock (``time.time()``) so
        stale-reclaim works even when the DB is on a host with a different
        clock — only the acquiring process's clock matters.

        ``lock_ts`` is stored as a float (seconds since epoch) rather than an
        ISO-8601 string to avoid timezone-parsing ambiguity (SQLite's
        ``datetime('now')`` returns a naive UTC string that ``fromisoformat``
        would silently interpret as local time).

        Parameters
        ----------
        task_id:
            Registry row to lock.
        holder:
            Opaque identifier of the locking entity (e.g. ``"cron:pid:1234"``).
        ttl_seconds:
            Lock expiry in seconds (default 300 = 5× a 60 s cron interval).
        """
        now_epoch = time.time()
        expiry_epoch = now_epoch + ttl_seconds

        def _do(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT lock_holder, lock_ts FROM session_orchestration WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False

            current_holder = row["lock_holder"] if isinstance(row, sqlite3.Row) else row[0]
            current_lock_ts = row["lock_ts"] if isinstance(row, sqlite3.Row) else row[1]

            # Already held by us?
            if current_holder == holder:
                # Refresh expiry
                conn.execute(
                    "UPDATE session_orchestration SET lock_ts = ? WHERE task_id = ?",
                    (str(expiry_epoch), task_id),
                )
                return True

            # Held by someone else — check expiry (lock_ts stores expiry epoch)
            if current_holder is not None and current_lock_ts is not None:
                try:
                    lock_expiry = float(current_lock_ts)
                except (ValueError, TypeError):
                    lock_expiry = 0.0
                if lock_expiry > now_epoch:
                    # Lock is still fresh — deny
                    return False

            # Either no lock or expired lock — claim it
            conn.execute(
                "UPDATE session_orchestration "
                "SET lock_holder = ?, lock_ts = ?, updated_at = datetime('now') "
                "WHERE task_id = ?",
                (holder, str(expiry_epoch), task_id),
            )
            return True

        try:
            return bool(self._write(_do))
        except sqlite3.Error as exc:
            logger.warning("acquire_lock(%s) failed: %s", task_id, exc)
            return False

    def release_lock(self, task_id: str, holder: str) -> None:
        """Release the per-session lock iff *holder* is the current owner.

        Idempotent — no-op if the lock is already expired/reclaimed or was
        never held by ``holder``.
        """
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE session_orchestration "
                "SET lock_holder = NULL, lock_ts = NULL, "
                "    updated_at = datetime('now') "
                "WHERE task_id = ? AND lock_holder = ?",
                (task_id, holder),
            )

        try:
            self._write(_do)
        except sqlite3.Error as exc:
            logger.warning("release_lock(%s) failed: %s", task_id, exc)
