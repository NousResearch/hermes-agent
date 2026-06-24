"""Ágora dashboard plugin — backend API routes.

Mounted at ``/api/plugins/agora/`` by the dashboard plugin system.

Ágora is a thin social/deliberation layer on top of Hermes Kanban. It keeps
its own SQLite database for channels, threads, messages, agent presence and
decisions, and only *links* to Kanban tasks via ``task_id``. It does not
duplicate the Kanban work engine.

Live updates arrive via the ``/events`` WebSocket and the ``GET /events``
long-poll endpoint, both tailing the append-only ``agora_events`` table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status as http_status,
)
from pydantic import BaseModel, Field

from hermes_cli.plugins import PluginContext
from hermes_cli import kanban_db as _kanban_db
from hermes_constants import get_default_hermes_root

log = logging.getLogger(__name__)

try:
    from hermes_cli import kanban_db as _kanban_db
except Exception as exc:  # pragma: no cover - Kanban may not be installed/enabled
    _kanban_db = None
    log.debug("Ágora could not import kanban_db: %s", exc)

router = APIRouter()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DEFAULT_CHANNELS: list[dict[str, str]] = [
    {
        "slug": "praca",
        "name": "Praça",
        "description": "Conversa geral entre agentes e humanos.",
    },
    {
        "slug": "planejamento",
        "name": "Planejamento",
        "description": "Discussões sobre próximos passos e estratégia.",
    },
    {
        "slug": "decisoes",
        "name": "Decisões",
        "description": "Propostas e decisões formais.",
    },
    {
        "slug": "incidentes",
        "name": "Incidentes",
        "description": "Bloqueios, erros e ações de recuperação.",
    },
    {
        "slug": "profarma",
        "name": "Profarma",
        "description": "Tópicos relacionados ao workspace profarma.dev/Aura.",
    },
]

MAX_CHANNEL_SLUG_LENGTH = 64

# Profiles that are not agents and must never appear in agora_agent_status.
# ``human`` is the canonical human author profile used by messages and
# notifications; it has no Hermes process and therefore must never be
# promoted to an agent presence row. Extend this set if other pseudo-profiles
# (e.g. ``kanban`` or ``system``) start showing up in the agent store.
RESERVED_AGENT_PROFILES: frozenset[str] = frozenset({"human"})


def _is_reserved_agent_profile(profile: str) -> bool:
    return profile.strip().lower() in RESERVED_AGENT_PROFILES


class ChannelSlugError(HTTPException):
    """Raised when a channel slug is invalid."""


class ChannelNameError(HTTPException):
    """Raised when a channel name is invalid."""


def _validate_channel_slug(raw: str) -> str:
    """Normalize and validate a channel slug for admin creation.

    Rules (ordered):

    * Must be non-empty after stripping whitespace.
    * NFKD-decomposed and stripped of combining diacritics so accents are
      removed (``são-paulo`` becomes ``sao-paulo``).
    * Must be plain ASCII after accent folding — Cyrillic, CJK or other
      non-Latin scripts are rejected.
    * Must match ``[a-z0-9_-]{1,64}``.
    * Must not start or end with ``-`` or ``_``.

    Returns the normalized slug. Raises :class:`ChannelSlugError` with a
    clear ``detail`` message otherwise.
    """
    if raw is None:
        raise ChannelSlugError(status_code=400, detail="slug is required")

    slug = raw.strip().lower()
    if not slug:
        raise ChannelSlugError(status_code=400, detail="slug is required")

    # Fold common Latin accents: decompose and drop combining marks.
    slug = "".join(
        ch for ch in unicodedata.normalize("NFKD", slug)
        if not unicodedata.combining(ch)
    )

    try:
        slug.encode("ascii")
    except UnicodeEncodeError:
        raise ChannelSlugError(
            status_code=400,
            detail="slug must be ASCII after removing accents",
        )

    if not re.fullmatch(rf"[a-z0-9_-]{{1,{MAX_CHANNEL_SLUG_LENGTH}}}", slug):
        raise ChannelSlugError(
            status_code=400,
            detail=(
                "slug must be 1-64 lowercase ASCII letters, digits, "
                "hyphens or underscores"
            ),
        )

    if slug[0] in "-_" or slug[-1] in "-_":
        raise ChannelSlugError(
            status_code=400,
            detail="slug must not start or end with '-' or '_'",
        )

    return slug


def _validate_channel_name(raw: str) -> str:
    """Normalize and validate a channel name."""
    if raw is None:
        raise ChannelNameError(status_code=400, detail="name is required")
    name = raw.strip()
    if not name:
        raise ChannelNameError(status_code=400, detail="name is required")
    return name


def _insert_channel(
    slug: str,
    name: str,
    description: Optional[str] = None,
) -> dict[str, Any]:
    """Insert a channel into the shared Ágora DB and emit a ``created`` event.

    Raises :class:`HTTPException` 409 if the slug already exists.
    """
    with _connect() as conn:
        try:
            now = int(time.time())
            cur = conn.execute(
                "INSERT INTO agora_channels (slug, name, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (slug, name, (description or "").strip(), now),
            )
            _emit_event(
                conn,
                "channel",
                str(cur.lastrowid),
                "created",
                {"slug": slug, "name": name},
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            log.debug("Channel insert conflict for slug %r: %s", slug, exc)
            raise HTTPException(
                status_code=409,
                detail=f"channel slug '{slug}' already exists",
            )
        row = conn.execute(
            "SELECT * FROM agora_channels WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _channel_dict(row)



@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Open the Ágora SQLite DB, initialising it if needed."""
    _init_db()
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _db_path() -> Path:
    return get_default_hermes_root() / "agora.db"


_db_init_path: Optional[Path] = None


def _init_db() -> None:
    """Create the schema and seed default channels. Idempotent.

    Re-initialises when the resolved DB path changes (e.g. ``HERMES_HOME``
    switches between shared root and a profile directory in tests or
    long-lived processes), preventing stale schema/channel state.
    """
    global _db_init_path
    current = _db_path().resolve()
    if _db_init_path == current:
        return

    _db_path().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path()))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agora_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agora_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                linked_task_id TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at INTEGER NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES agora_channels(id)
            );

            CREATE INDEX IF NOT EXISTS idx_threads_channel ON agora_threads(channel_id);
            CREATE INDEX IF NOT EXISTS idx_threads_task ON agora_threads(linked_task_id);

            CREATE TABLE IF NOT EXISTS agora_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                thread_id INTEGER,
                author_type TEXT NOT NULL,
                author_profile TEXT,
                body TEXT NOT NULL,
                linked_task_id TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES agora_channels(id),
                FOREIGN KEY (thread_id) REFERENCES agora_threads(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel ON agora_messages(channel_id);
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON agora_messages(thread_id);
            CREATE INDEX IF NOT EXISTS idx_messages_task ON agora_messages(linked_task_id);

            CREATE TABLE IF NOT EXISTS agora_agent_status (
                profile TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                current_task_id TEXT,
                current_step TEXT,
                status_text TEXT,
                last_heartbeat_at INTEGER NOT NULL,
                pid INTEGER,
                run_id INTEGER,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS agora_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                proposal TEXT NOT NULL,
                decision TEXT NOT NULL,
                rationale TEXT,
                decided_by TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES agora_threads(id)
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_thread ON agora_decisions(thread_id);

            CREATE TABLE IF NOT EXISTS agora_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_entity ON agora_events(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_events_created ON agora_events(created_at);

            CREATE TABLE IF NOT EXISTS agora_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                body_snippet TEXT,
                author_profile TEXT,
                read_at INTEGER,
                ack_at INTEGER,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (message_id) REFERENCES agora_messages(id),
                FOREIGN KEY (channel_id) REFERENCES agora_channels(id)
            );

            CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON agora_notifications(recipient);
            CREATE INDEX IF NOT EXISTS idx_notifications_recipient_created ON agora_notifications(recipient, created_at);
            CREATE INDEX IF NOT EXISTS idx_notifications_recipient_read ON agora_notifications(recipient, read_at);

            CREATE TABLE IF NOT EXISTS agora_migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                source_table TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                target_id INTEGER,
                migrated_at INTEGER NOT NULL,
                UNIQUE(source_path, source_table, source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_migration_log_source ON agora_migration_log(source_path, source_table, source_id);
            """
        )

        existing = {row[0] for row in conn.execute("SELECT slug FROM agora_channels")}
        now = int(time.time())
        for ch in DEFAULT_CHANNELS:
            if ch["slug"] not in existing:
                conn.execute(
                    "INSERT INTO agora_channels (slug, name, description, created_at) VALUES (?, ?, ?, ?)",
                    (ch["slug"], ch["name"], ch["description"], now),
                )

        # Remove any stale rows for pseudo-profiles (human, system, etc.) that
        # may have been written before the reserved-profile guard existed.
        if RESERVED_AGENT_PROFILES:
            placeholders = ",".join("?" * len(RESERVED_AGENT_PROFILES))
            conn.execute(
                f"DELETE FROM agora_agent_status WHERE profile IN ({placeholders})",
                tuple(sorted(RESERVED_AGENT_PROFILES)),
            )

        # On every fresh start, fold any legacy "emptyname" channel into #praca
        # so the orphan can never reappear in the channel list.
        _cleanup_emptyname_channel(conn)

        conn.commit()
        _db_init_path = current
    finally:
        conn.close()


def _cleanup_emptyname_channel(conn: sqlite3.Connection) -> dict[str, Any]:
    """Migrate any legacy ``emptyname`` channel into the default ``praca`` channel.

    Idempotent: no-op if there is no ``emptyname`` channel. Messages and threads
    are reassigned to ``praca`` so nothing is lost.
    """
    target_slug = "praca"
    source = conn.execute(
        "SELECT id FROM agora_channels WHERE slug = ?", ("emptyname",)
    ).fetchone()
    if source is None:
        return {
            "ok": True,
            "moved_messages": 0,
            "moved_threads": 0,
            "target_slug": target_slug,
        }

    target = conn.execute(
        "SELECT id FROM agora_channels WHERE slug = ?", (target_slug,)
    ).fetchone()
    if target is None:
        return {
            "ok": True,
            "moved_messages": 0,
            "moved_threads": 0,
            "target_slug": target_slug,
        }

    source_id = source["id"]
    target_id = target["id"]

    conn.execute(
        "UPDATE agora_threads SET channel_id = ? WHERE channel_id = ?",
        (target_id, source_id),
    )
    moved_threads = conn.execute("SELECT changes()").fetchone()[0]

    conn.execute(
        "UPDATE agora_messages SET channel_id = ? WHERE channel_id = ?",
        (target_id, source_id),
    )
    moved_messages = conn.execute("SELECT changes()").fetchone()[0]

    conn.execute("DELETE FROM agora_channels WHERE id = ?", (source_id,))

    return {
        "ok": True,
        "moved_messages": moved_messages,
        "moved_threads": moved_threads,
        "target_slug": target_slug,
    }


def migrate_profile_agora_dbs(*, dry_run: bool = False) -> dict[str, Any]:
    """Merge legacy per-profile ``agora.db`` files into the shared root DB.

    Before t_a685c879 the plugin wrote to ``<root>/profiles/<profile>/agora.db``.
    This function discovers those leftover files, backs them up, and copies
    their data into the canonical ``<root>/agora.db``.  Idempotent: each
    source row is tracked by path/table/id and only migrated once.
    """
    shared_root = get_default_hermes_root()
    shared_db_path = shared_root / "agora.db"
    profile_root = shared_root / "profiles"
    if not profile_root.exists():
        return {"ok": True, "migrated": [], "skipped": [], "note": "no profiles directory"}

    report: dict[str, Any] = {
        "ok": True,
        "shared_db": str(shared_db_path),
        "dry_run": dry_run,
        "migrated": [],
        "skipped": [],
        "errors": [],
    }

    for profile_db_path in sorted(profile_root.glob("*/agora.db")):
        profile_name = profile_db_path.parent.name
        if profile_db_path.stat().st_size == 0:
            report["skipped"].append({"profile": profile_name, "reason": "empty-file"})
            continue

        source_resolved = profile_db_path.resolve()
        if source_resolved == shared_db_path.resolve():
            report["skipped"].append({"profile": profile_name, "reason": "same-as-shared"})
            continue

        try:
            result = _migrate_single_profile_db(
                profile_db_path=profile_db_path,
                profile_name=profile_name,
                dry_run=dry_run,
            )
            report["migrated"].append(result)
        except Exception as exc:
            log.exception("Failed to migrate %s", profile_db_path)
            report["errors"].append(
                {"profile": profile_name, "path": str(profile_db_path), "error": str(exc)}
            )

    report["ok"] = not report["errors"]
    return report


def _migrate_single_profile_db(
    *, profile_db_path: Path, profile_name: str, dry_run: bool
) -> dict[str, Any]:
    """Migrate one profile DB into the current shared DB."""
    target_path = _db_path().resolve()
    source_path = profile_db_path.resolve()

    if dry_run:
        return {"profile": profile_name, "dry_run": True, "source": str(source_path)}

    # Backup before touching anything.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = source_path.parent / f"agora.db.{timestamp}.bak"
    shutil.copy2(source_path, backup_path)

    # Ensure shared schema (and migration log) exists.
    _init_db()

    target_conn = sqlite3.connect(str(target_path))
    target_conn.row_factory = sqlite3.Row
    source_conn = sqlite3.connect(str(source_path))
    source_conn.row_factory = sqlite3.Row

    try:
        _ensure_migration_log_table(target_conn)
        now = int(time.time())

        def _log_migration(source_table: str, source_id: int, target_id: int) -> None:
            target_conn.execute(
                """INSERT OR REPLACE INTO agora_migration_log
                   (source_path, source_table, source_id, target_id, migrated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(source_path), source_table, source_id, target_id, now),
            )

        def _already_migrated(source_table: str, source_id: int) -> Optional[int]:
            row = target_conn.execute(
                """SELECT target_id FROM agora_migration_log
                   WHERE source_path = ? AND source_table = ? AND source_id = ?""",
                (str(source_path), source_table, source_id),
            ).fetchone()
            return row["target_id"] if row else None

        # Channels: map by slug, creating missing ones.
        channel_map: dict[int, int] = {}
        for row in source_conn.execute(
            "SELECT id, slug, name, description, created_at FROM agora_channels"
        ):
            source_id = row["id"]
            target_id = _already_migrated("agora_channels", source_id)
            if target_id is None:
                existing = target_conn.execute(
                    "SELECT id FROM agora_channels WHERE slug = ?", (row["slug"],)
                ).fetchone()
                if existing:
                    target_id = existing["id"]
                else:
                    cur = target_conn.execute(
                        "INSERT INTO agora_channels (slug, name, description, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (row["slug"], row["name"], row["description"], row["created_at"]),
                    )
                    target_id = _rowid(cur)
                _log_migration("agora_channels", source_id, target_id)
            channel_map[source_id] = target_id

        # Threads: map source id to target id via migration log.
        thread_map: dict[int, int] = {}
        for row in source_conn.execute(
            "SELECT id, channel_id, title, linked_task_id, status, created_at FROM agora_threads"
        ):
            source_id = row["id"]
            target_id = _already_migrated("agora_threads", source_id)
            if target_id is None:
                target_channel_id = channel_map.get(row["channel_id"])
                if target_channel_id is None:
                    continue
                cur = target_conn.execute(
                    "INSERT INTO agora_threads (channel_id, title, linked_task_id, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        target_channel_id,
                        row["title"],
                        row["linked_task_id"],
                        row["status"],
                        row["created_at"],
                    ),
                )
                target_id = _rowid(cur)
                _log_migration("agora_threads", source_id, target_id)
            thread_map[source_id] = target_id

        # Messages.
        for row in source_conn.execute(
            "SELECT id, channel_id, thread_id, author_type, author_profile, body, "
            "linked_task_id, created_at FROM agora_messages"
        ):
            source_id = row["id"]
            if _already_migrated("agora_messages", source_id) is not None:
                continue
            target_channel_id = channel_map.get(row["channel_id"])
            if target_channel_id is None:
                continue
            target_thread_id = thread_map.get(row["thread_id"]) if row["thread_id"] else None
            cur = target_conn.execute(
                "INSERT INTO agora_messages (channel_id, thread_id, author_type, author_profile, "
                "body, linked_task_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    target_channel_id,
                    target_thread_id,
                    row["author_type"],
                    row["author_profile"],
                    row["body"],
                    row["linked_task_id"],
                    row["created_at"],
                ),
            )
            _log_migration("agora_messages", source_id, _rowid(cur))

        # Agent status: keyed by profile, keep newest heartbeat.
        for row in source_conn.execute(
            "SELECT profile, state, current_task_id, current_step, status_text, "
            "last_heartbeat_at, pid, run_id, metadata_json FROM agora_agent_status"
        ):
            existing = target_conn.execute(
                "SELECT last_heartbeat_at FROM agora_agent_status WHERE profile = ?",
                (row["profile"],),
            ).fetchone()
            if existing and existing["last_heartbeat_at"] >= row["last_heartbeat_at"]:
                continue
            target_conn.execute(
                """INSERT OR REPLACE INTO agora_agent_status
                   (profile, state, current_task_id, current_step, status_text,
                    last_heartbeat_at, pid, run_id, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["profile"],
                    row["state"],
                    row["current_task_id"],
                    row["current_step"],
                    row["status_text"],
                    row["last_heartbeat_at"],
                    row["pid"],
                    row["run_id"],
                    row["metadata_json"],
                ),
            )

        # Decisions: map thread_id.
        for row in source_conn.execute(
            "SELECT id, thread_id, proposal, decision, rationale, decided_by, created_at "
            "FROM agora_decisions"
        ):
            source_id = row["id"]
            if _already_migrated("agora_decisions", source_id) is not None:
                continue
            target_thread_id = thread_map.get(row["thread_id"]) if row["thread_id"] else None
            if target_thread_id is None:
                continue
            cur = target_conn.execute(
                "INSERT INTO agora_decisions (thread_id, proposal, decision, rationale, "
                "decided_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    target_thread_id,
                    row["proposal"],
                    row["decision"],
                    row["rationale"],
                    row["decided_by"],
                    row["created_at"],
                ),
            )
            _log_migration("agora_decisions", source_id, _rowid(cur))

        # Events: append-only.
        for row in source_conn.execute(
            "SELECT id, entity_type, entity_id, event_type, payload, created_at FROM agora_events"
        ):
            source_id = row["id"]
            if _already_migrated("agora_events", source_id) is not None:
                continue
            cur = target_conn.execute(
                "INSERT INTO agora_events (entity_type, entity_id, event_type, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (row["entity_type"], row["entity_id"], row["event_type"], row["payload"], row["created_at"]),
            )
            _log_migration("agora_events", source_id, _rowid(cur))

        # Notifications: map message_id and channel_id.
        for row in source_conn.execute(
            "SELECT id, recipient, message_id, channel_id, body_snippet, author_profile, "
            "read_at, ack_at, created_at FROM agora_notifications"
        ):
            source_id = row["id"]
            if _already_migrated("agora_notifications", source_id) is not None:
                continue
            msg_migrated = target_conn.execute(
                """SELECT target_id FROM agora_migration_log
                   WHERE source_path = ? AND source_table = ? AND source_id = ?""",
                (str(source_path), "agora_messages", row["message_id"]),
            ).fetchone()
            if not msg_migrated:
                continue
            target_message_id = msg_migrated["target_id"]
            target_channel_id = channel_map.get(row["channel_id"])
            if target_channel_id is None:
                continue
            cur = target_conn.execute(
                "INSERT INTO agora_notifications (recipient, message_id, channel_id, body_snippet, "
                "author_profile, read_at, ack_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["recipient"],
                    target_message_id,
                    target_channel_id,
                    row["body_snippet"],
                    row["author_profile"],
                    row["read_at"],
                    row["ack_at"],
                    row["created_at"],
                ),
            )
            _log_migration("agora_notifications", source_id, _rowid(cur))

        target_conn.commit()

        # After successful migration, rename source to avoid reprocessing.
        migrated_path = source_path.with_suffix(".db.migrated")
        source_path.rename(migrated_path)

        return {
            "profile": profile_name,
            "backup": str(backup_path),
            "migrated_to": str(migrated_path),
        }
    finally:
        source_conn.close()
        target_conn.close()


def _ensure_migration_log_table(conn: sqlite3.Connection) -> None:
    """Ensure the migration log table exists (defensive for pre-existing DBs)."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agora_migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            target_id INTEGER,
            migrated_at INTEGER NOT NULL,
            UNIQUE(source_path, source_table, source_id)
        );
        CREATE INDEX IF NOT EXISTS idx_migration_log_source ON agora_migration_log(source_path, source_table, source_id);
        """
    )


def _rowid(cur: sqlite3.Cursor) -> int:
    """Return ``cur.lastrowid`` or raise if the insert produced no row id."""
    rid = cur.lastrowid
    if rid is None:
        raise RuntimeError("INSERT did not return a row id")
    return rid


def _emit_event(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Append an event to the audit/live feed table."""
    conn.execute(
        "INSERT INTO agora_events (entity_type, entity_id, event_type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            entity_type,
            str(entity_id),
            event_type,
            json.dumps(payload, default=str),
            int(time.time()),
        ),
    )


# ---------------------------------------------------------------------------
# Auth helper — WebSocket only
# ---------------------------------------------------------------------------


def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Delegate WebSocket auth to the dashboard's canonical gate.

    Falls back to accepting in test contexts where ``hermes_cli.web_server``
    is not importable, keeping the tail loop testable.
    """
    try:
        from hermes_cli import web_server as _ws
    except Exception:
        return True
    return bool(_ws._ws_auth_ok(ws))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _channel_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "slug": row["slug"],
        "name": row["name"],
        "description": row["description"],
        "created_at": row["created_at"],
    }


def _thread_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "channel_id": row["channel_id"],
        "title": row["title"],
        "linked_task_id": row["linked_task_id"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "channel_id": row["channel_id"],
        "thread_id": row["thread_id"],
        "author_type": row["author_type"],
        "author_profile": row["author_profile"],
        "body": row["body"],
        "linked_task_id": row["linked_task_id"],
        "created_at": row["created_at"],
    }


def _notification_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "recipient": row["recipient"],
        "message_id": row["message_id"],
        "channel_id": row["channel_id"],
        "body_snippet": row["body_snippet"],
        "author_profile": row["author_profile"],
        "read_at": row["read_at"],
        "ack_at": row["ack_at"],
        "created_at": row["created_at"],
    }


_MENTION_RE = re.compile(r"@([a-zA-Z0-9_-]+)", re.UNICODE)
_BROADCAST_HANDLES = frozenset({"all", "todos"})

_PROFILE_SKILLS = {
    "agent-techlead": "hermes-agent,multi-agent-telemetry,kanban-orchestrator,systematic-debugging",
    "agent-backend": "hermes-agent,multi-agent-telemetry,test-driven-development,systematic-debugging",
    "agent-frontend": "hermes-agent,multi-agent-telemetry,chrome-dom-automation,systematic-debugging",
    "agent-qa": "dogfood,chrome-dom-automation,systematic-debugging,multi-agent-telemetry,hermes-agent",
}

# Channel where kanban task completion reports are posted.  The message
# always @-mentions ``agent-techlead`` and creates a mailbox notification,
# so the channel choice is mainly about where the public record lives.
_AGORA_COMPLETION_NOTIFY_CHANNEL: str = (
    os.environ.get("AGORA_COMPLETION_NOTIFY_CHANNEL", "planejamento").strip()
    or "planejamento"
)

# Channel where kanban blocked / handoff messages are posted.
_AGORA_HANDOFF_NOTIFY_CHANNEL: str = (
    os.environ.get("AGORA_HANDOFF_NOTIFY_CHANNEL", "praca").strip()
    or "praca"
)

# Known handoff targets used to @-mention the right agent when a reason or
# summary does not already include an explicit handle.
_AGORA_KNOWN_HANDOFF_PROFILES: list[str] = [
    "agent-frontend",
    "agent-backend",
    "agent-qa",
    "agent-techlead",
]


def _resolve_handoff_mention(reason_or_summary: Optional[str]) -> str:
    """Pick the profile that should be @-mentioned for a handoff.

    1. If the text already contains an explicit @mention of a known profile,
       return the first one so the worker's intent is preserved verbatim.
    2. Otherwise infer from keywords (frontend/backend/qa/tech-lead).
    3. Fall back to ``agent-techlead`` when no clear owner is found.
    """
    text = (reason_or_summary or "").lower()
    # Explicit @mentions win.
    seen = set()
    for match in _MENTION_RE.finditer(reason_or_summary or ""):
        handle = match.group(1)
        if handle in _AGORA_KNOWN_HANDOFF_PROFILES and handle not in seen:
            seen.add(handle)
            return handle
    # Keyword inference.
    if "frontend" in text or "front-end" in text:
        return "agent-frontend"
    if "backend" in text or "back-end" in text:
        return "agent-backend"
    if "qa" in text or "quality" in text or "test" in text:
        return "agent-qa"
    if "tech-lead" in text or "tech lead" in text or "lead" in text:
        return "agent-techlead"
    return "agent-techlead"


def _format_blocked_handoff(
    *,
    task_id: str,
    title: str,
    assignee: Optional[str],
    reason: Optional[str],
    next_profile: str,
) -> str:
    """Build the human-readable blocked handoff posted to Ágora."""
    lines = [
        f"@{next_profile} tarefa bloqueada aguardando ação:",
        "",
        f"**{title}** ({task_id})",
        f"Assignee: {assignee or 'não atribuído'}",
    ]
    if reason:
        lines.extend(["", f"Motivo / próximo passo: {reason}"])
    else:
        lines.extend(["", "⚠️ Nenhum motivo de bloqueio foi fornecido."])
    return "\n".join(lines)


def _post_system_message_to_channel(
    *,
    channel_slug: str,
    body: str,
    task_id: str,
    event_origin: str,
) -> Optional[int]:
    """Insert a system message from ``kanban`` into an Ágora channel.

    Returns the inserted message id, or ``None`` if the channel does not exist.
    Adds the message, an audit event, and any mailbox notifications implied by
    @mentions in the body.
    """
    with _connect() as conn:
        channel = conn.execute(
            "SELECT id FROM agora_channels WHERE slug = ?", (channel_slug,)
        ).fetchone()
        if channel is None:
            log.warning("Ágora channel %r not found", channel_slug)
            return None
        now = int(time.time())
        cur = conn.execute(
            "INSERT INTO agora_messages "
            "(channel_id, thread_id, author_type, author_profile, body, linked_task_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (channel["id"], None, "system", "kanban", body, task_id, now),
        )
        message_id = cur.lastrowid
        if message_id is None:
            log.warning("Ágora INSERT did not return a message id for task %s", task_id)
            return None
        _emit_event(
            conn,
            "message",
            str(message_id),
            "created",
            {
                "channel_id": channel["id"],
                "channel_slug": channel_slug,
                "thread_id": None,
                "author_type": "system",
                "author_profile": "kanban",
                "linked_task_id": task_id,
                "event_origin": event_origin,
            },
        )
        _create_notifications(
            conn,
            message_id=message_id,
            channel_id=channel["id"],
            channel_slug=channel_slug,
            body=body,
            author_profile="kanban",
        )
        conn.commit()
        return message_id


def _on_kanban_task_blocked(
    *,
    task_id: str,
    title: Optional[str] = None,
    assignee: Optional[str] = None,
    reason: Optional[str] = None,
    blocked_at: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Post a blocked handoff to Ágora and notify the next owner.

    Registered as a ``kanban_db`` blocked callback so the notification fires
    from the existing blocked event without polling. The callback is defensive:
    if the hook caller does not supply task metadata, it is loaded from Kanban.
    """
    try:
        if not _task_exists_in_any_board(task_id):
            log.warning(
                "Ágora blocked notification dropped: task %s does not exist "
                "in any Kanban board (synthetic/test block?)",
                task_id,
            )
            return

        # Backfill metadata when the hook caller only supplies the task id.
        if title is None or assignee is None:
            from hermes_cli import kanban_db as kb

            try:
                conn = kb.connect()
                task = kb.get_task(conn, task_id)
                conn.close()
            except Exception:
                task = None
            if task:
                if hasattr(task, "get"):
                    title = title or task.get("title") or "(sem título)"
                    assignee = assignee if assignee is not None else task.get("assignee")
                else:
                    title = title or getattr(task, "title", None) or "(sem título)"
                    assignee = assignee if assignee is not None else getattr(task, "assignee", None)
            else:
                title = title or "(sem título)"
                assignee = assignee if assignee is not None else None

        next_profile = _resolve_handoff_mention(reason)
        body = _format_blocked_handoff(
            task_id=task_id,
            title=title,
            assignee=assignee,
            reason=reason,
            next_profile=next_profile,
        )
        _post_system_message_to_channel(
            channel_slug=_AGORA_HANDOFF_NOTIFY_CHANNEL,
            body=body,
            task_id=task_id,
            event_origin="kanban_task_blocked",
        )
    except Exception:
        log.exception("Failed to notify Ágora of blocked task %s", task_id)


def _format_delivery_report(
    *,
    task_id: str,
    title: str,
    assignee: Optional[str],
    result: Optional[str],
    summary: Optional[str],
    metadata: Optional[dict],
    verified_cards: Optional[list[str]] = None,
) -> str:
    """Build the human-readable completion report posted to Ágora."""
    lines = [
        "@agent-techlead entrega concluída:",
        "",
        f"**{title}** ({task_id})",
        f"Assignee: {assignee or 'não atribuído'}",
        "Status: done",
    ]
    report = (summary or result or "").strip()
    if report:
        lines.extend(["", "Relatório de entrega:", report])
    else:
        lines.extend(["", "⚠️ Nenhum relatório de entrega foi fornecido."])

    if verified_cards:
        lines.extend(["", "Cards criados:"])
        lines.extend(f"- {c}" for c in verified_cards)

    artifacts: list[str] = []
    if isinstance(metadata, dict):
        raw = metadata.get("artifacts")
        if isinstance(raw, (list, tuple)):
            artifacts = [
                str(a).strip() for a in raw if isinstance(a, str) and str(a).strip()
            ]
    if artifacts:
        lines.extend(["", "Artifacts:"])
        lines.extend(f"- {a}" for a in artifacts)

    return "\n".join(lines)


def _task_exists_in_any_board(task_id: str) -> bool:
    """Verify ``task_id`` exists in at least one Kanban board.

    Completion callbacks can be invoked from tests, ephemeral processes, or
    board contexts where the Ágora DB happens to be shared with production.
    Before publishing a delivery report, confirm the referenced task is real
    by checking every known board (active board first)."""
    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return False

    # Try the active/default board first; this is the common path.
    try:
        conn = kb.connect()
        try:
            if kb.get_task(conn, task_id) is not None:
                return True
        finally:
            conn.close()
    except Exception:
        pass

    # Fallback: scan every discovered board. Completion callbacks do not
    # receive board context, so a task completed on a non-default board must
    # still be accepted.
    try:
        boards = kb.list_boards(include_archived=True)
    except Exception:
        boards = []
    seen: set[str] = set()
    for board_meta in boards:
        slug = board_meta.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        try:
            conn = kb.connect(board=slug)
            try:
                if kb.get_task(conn, task_id) is not None:
                    return True
            finally:
                conn.close()
        except Exception:
            continue
    return False


def _on_kanban_task_completed(
    *,
    task_id: str,
    title: Optional[str] = None,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    completed_at: Optional[int] = None,
    verified_cards: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
    """Post a delivery report to Ágora and notify agent-techlead.

    Registered as a ``kanban_db`` completion callback so the notification
    fires from the existing completion event without polling. The callback is
    defensive: if the hook caller does not supply task metadata, it is loaded
    from Kanban.
    """
    try:
        if not _task_exists_in_any_board(task_id):
            log.warning(
                "Ágora completion notification dropped: task %s does not exist "
                "in any Kanban board (synthetic/test completion?)",
                task_id,
            )
            return

        # Backfill metadata when the hook caller only supplies the task id.
        if title is None or assignee is None or summary is None:
            from hermes_cli import kanban_db as kb

            try:
                conn = kb.connect()
                task = kb.get_task(conn, task_id)
                conn.close()
            except Exception:
                task = None
            if task:
                title = title or task.title or "(sem título)"
                assignee = assignee if assignee is not None else task.assignee
                summary = summary or task.result or "(sem resumo)"
            else:
                title = title or "(sem título)"
                assignee = assignee if assignee is not None else None
                summary = summary or "(sem resumo)"

        channel_slug = _AGORA_COMPLETION_NOTIFY_CHANNEL
        body = _format_delivery_report(
            task_id=task_id,
            title=title,
            assignee=assignee,
            result=result,
            summary=summary,
            metadata=metadata,
            verified_cards=verified_cards,
        )
        with _connect() as conn:
            channel = conn.execute(
                "SELECT id FROM agora_channels WHERE slug = ?", (channel_slug,)
            ).fetchone()
            if channel is None:
                log.warning(
                    "Ágora completion notify channel %r not found", channel_slug
                )
                return

            now = int(time.time())
            cur = conn.execute(
                "INSERT INTO agora_messages "
                "(channel_id, thread_id, author_type, author_profile, body, linked_task_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (channel["id"], None, "system", "kanban", body, task_id, now),
            )
            message_id = cur.lastrowid
            if message_id is None:
                log.warning(
                    "Ágora INSERT did not return a message id for task %s", task_id
                )
                return
            _emit_event(
                conn,
                "message",
                str(message_id),
                "created",
                {
                    "channel_id": channel["id"],
                    "channel_slug": channel_slug,
                    "thread_id": None,
                    "author_type": "system",
                    "author_profile": "kanban",
                    "linked_task_id": task_id,
                    "event_origin": "kanban_task_completed",
                },
            )
            _create_notifications(
                conn,
                message_id=message_id,
                channel_id=channel["id"],
                channel_slug=channel_slug,
                body=body,
                author_profile="kanban",
            )
            conn.commit()
    except Exception:
        log.exception("Failed to notify Ágora of completed task %s", task_id)


def _tmux_wake_enabled() -> bool:
    """Return whether local tmux wake-up delivery is enabled."""
    val = os.environ.get("AGORA_TMUX_WAKE_ENABLED", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _tmux_has_session(session: str) -> bool:
    try:
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _ensure_visible_agent_terminal(profile: str, *, allow_known: bool = True) -> dict[str, Any]:
    """Ensure a profile has a tmux session.

    Existing sessions are always reused. New sessions are auto-created for any
    profile passed from the dashboard's agent list when ``allow_known`` is True,
    using the skills map if available, otherwise falling back to the generic
    hermes-agent skill set. Arbitrary @handles in chat still receive mailbox
    entries without spawning new agents via the mention path (which also uses
    ``allow_known=True`` but only after permission checks).
    """
    if shutil.which("tmux") is None:
        return {"ok": False, "profile": profile, "reason": "tmux-not-found"}

    session = profile
    created = False
    if not _tmux_has_session(session):
        skills = _PROFILE_SKILLS.get(profile) if allow_known else None
        if not skills:
            skills = "hermes-agent"
        cmd = (
            "cd /home/felipi/.hermes/hermes-agent; "
            f"exec hermes -p {profile} chat --cli --skills {skills}"
        )
        proc = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "-x", "160", "-y", "50", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "profile": profile,
                "session": session,
                "reason": "tmux-new-session-failed",
                "stderr": proc.stderr.strip()[:300],
            }
        created = True

    return {"ok": True, "profile": profile, "session": session, "created": created}


def _tmux_session_has_clients(session: str) -> bool:
    """Return True if ``session`` already has one or more attached clients."""
    try:
        proc = subprocess.run(
            ["tmux", "list-clients", "-t", session, "-F", "#{client_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0 and proc.stdout.strip() != ""
    except Exception:
        return False


def _open_agent_tmux_terminal(profile: str) -> dict[str, Any]:
    """Open or focus a human-visible terminal attached to an agent tmux session."""
    ensure = _ensure_visible_agent_terminal(profile)
    if not ensure.get("ok"):
        return ensure
    session = ensure["session"]

    # If the session already has a client, treat the click as "focus" and do
    # not spawn another window that would duplicate the agent view.
    if _tmux_session_has_clients(session):
        return {**ensure, "opened": False, "focused": True, "command": None}

    attach_cmd = f"tmux attach -t {session}"
    candidates: list[list[str]] = []
    if sys.platform.startswith("win"):
        return {**ensure, "opened": False, "reason": "unsupported-platform"}
    if sys.platform == "darwin":
        escaped = attach_cmd.replace("\\", "\\\\").replace('"', '\\"')
        candidates.append([
            "osascript",
            "-e",
            'tell application "Terminal" to do script "' + escaped + '"',
        ])
    else:
        terminal_specs = [
            ("ptyxis", ["ptyxis", "--new-window", "--title", f"Ágora {profile}", "--", "bash", "-lc", attach_cmd]),
            ("gnome-terminal", ["gnome-terminal", "--", "bash", "-lc", attach_cmd]),
            ("x-terminal-emulator", ["x-terminal-emulator", "-e", "bash", "-lc", attach_cmd]),
            ("konsole", ["konsole", "-e", "bash", "-lc", attach_cmd]),
            ("xfce4-terminal", ["xfce4-terminal", "-e", f"bash -lc '{attach_cmd}'"]),
        ]
        candidates.extend(argv for binary, argv in terminal_specs if shutil.which(binary))
    for argv in candidates:
        try:
            subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=not sys.platform.startswith("win"),
            )
            return {**ensure, "opened": True, "command": argv[0]}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {**ensure, "opened": False, "reason": "no-terminal", "error": locals().get("last_error")}


def _tmux_send_message(profile: str, message: str) -> dict[str, Any]:
    ensure = _ensure_visible_agent_terminal(profile)
    if not ensure.get("ok"):
        return ensure
    session = ensure["session"]
    safe_message = " ".join(message.split())
    proc = subprocess.run(
        ["tmux", "send-keys", "-t", session, "-l", safe_message],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        return {
            **ensure,
            "delivered": False,
            "reason": "send-keys-failed",
            "stderr": proc.stderr.strip()[:300],
        }
    proc_enter = subprocess.run(
        ["tmux", "send-keys", "-t", session, "Enter"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )
    if proc_enter.returncode != 0:
        return {
            **ensure,
            "delivered": False,
            "reason": "enter-failed",
            "stderr": proc_enter.stderr.strip()[:300],
        }
    return {**ensure, "delivered": True}


def _resolve_profile_pid(profile: str) -> Optional[int]:
    """Return the PID of a visible Hermes process for ``profile``, if any.

    Discovery order:

    1. Active tmux pane for ``profile`` session — returns the pane leader PID,
       which is a real local process even when the foreground command is a shell.
    2. Running ``hermes -p <profile> chat`` process via ``psutil``.
    3. Running ``hermes -p <profile>`` process via ``pgrep``.

    Falls back to ``None`` when no tmux session or Hermes process is found so
    callers can decide whether to store a missing PID.
    """
    if shutil.which("tmux"):
        try:
            proc = subprocess.run(
                ["tmux", "list-panes", "-t", profile, "-F", "#{pane_active} #{pane_pid}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == "1":
                        try:
                            pid = int(parts[1])
                            if pid > 0:
                                return pid
                        except ValueError:
                            continue
        except Exception:
            pass

    try:
        import psutil

        candidates: list[tuple[int, str]] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            cmdline = p.info["cmdline"] or []
            cmd = " ".join(cmdline)
            if "hermes" in cmd and f"-p {profile}" in cmd:
                candidates.append((p.info["pid"], cmd))
        # Prefer the interactive chat process; any Hermes process is acceptable.
        for pid, cmd in candidates:
            if "chat" in cmd:
                return pid
        if candidates:
            return candidates[0][0]
    except Exception:
        pass

    if shutil.which("pgrep"):
        try:
            proc = subprocess.run(
                ["pgrep", "-a", "-f", f"hermes -p {profile}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
                for line in lines:
                    if "chat" in line:
                        try:
                            return int(line.split(None, 1)[0])
                        except ValueError:
                            continue
                if lines:
                    try:
                        return int(lines[0].split(None, 1)[0])
                    except ValueError:
                        pass
        except Exception:
            pass

    return None


def _agent_state(
    conn: sqlite3.Connection,
    profile: str,
    *,
    active_profiles: Optional[set[str]] = None,
) -> Optional[str]:
    row = conn.execute(
        "SELECT state FROM agora_agent_status WHERE profile = ?",
        (profile,),
    ).fetchone()
    if row is not None:
        state = str(row["state"] or "").strip().lower() or None
    else:
        state = None
    if state == "working":
        return state
    if active_profiles is None:
        active_profiles = _kanban_active_profiles()
    if profile in active_profiles:
        return "working"
    return state


def _kanban_active_profiles() -> set[str]:
    """Return profiles that have at least one active Kanban worker.

    Falls back to the empty set when Kanban is unavailable so Ágora keeps
    working without the Kanban plugin installed.
    """
    if _kanban_db is None:
        return set()
    try:
        # init_db is idempotent; connect resolves the active board via env/config.
        _kanban_db.init_db()
        with _kanban_db.connect() as kb_conn:
            rows = kb_conn.execute(
                """
                SELECT r.profile
                FROM task_runs r
                JOIN tasks t ON t.id = r.task_id
                WHERE r.ended_at IS NULL
                  AND r.worker_pid IS NOT NULL
                  AND t.status = 'running'
                """,
            ).fetchall()
            return {str(row["profile"]) for row in rows if row["profile"]}
    except Exception as exc:
        log.warning("Ágora could not read Kanban active workers: %s", exc)
        return set()


def _wrap_tmux_delivery_for_agent_state(state: Optional[str], message: str) -> tuple[str, str]:
    """Return (mode, message) for delivering a mention to a worker tmux.

    Idle workers can receive the mention as a normal prompt. Non-idle workers
    must not be interrupted by a plain Enter-submitted message, so the line is
    delivered as ``/steer``. If the worker becomes idle before the CLI processes
    the line, Hermes safely degrades /steer to next-turn queue semantics.
    """
    if state == "idle":
        return "prompt", message
    return "steer", f"/steer {message}"


def _deliver_mentions_to_tmux(
    conn: sqlite3.Connection,
    recipients: set[str],
    *,
    message_id: int,
    channel_slug: str,
    body: str,
    author_profile: Optional[str],
) -> list[dict[str, Any]]:
    """Actively deliver mentions to local visible agent terminals."""
    if not _tmux_wake_enabled():
        return []
    sender = author_profile or "human"
    wake_text = (
        f"Ágora mention in #{channel_slug} from {sender} (message {message_id}): {body} "
        "Respond in Ágora/Kanban if action is needed."
    )
    deliveries = []
    active_profiles = _kanban_active_profiles()
    for recipient in sorted(recipients):
        state = _agent_state(conn, recipient, active_profiles=active_profiles)
        delivery_mode, delivery_text = _wrap_tmux_delivery_for_agent_state(state, wake_text)
        delivery = _tmux_send_message(recipient, delivery_text)
        delivery["delivery_mode"] = delivery_mode
        delivery["agent_state"] = state
        deliveries.append(delivery)
        _emit_event(
            conn,
            "wake_delivery",
            recipient,
            "delivered" if delivery.get("delivered") else "failed",
            {
                "recipient": recipient,
                "message_id": message_id,
                "channel_slug": channel_slug,
                "delivery_mode": delivery_mode,
                "agent_state": state,
                **delivery,
            },
        )
    return deliveries


def _quoted_char_indexes(body: str) -> set[int]:
    """Return character positions that are inside simple quoted/code spans."""
    quoted: set[int] = set()
    quote: Optional[str] = None
    start: Optional[int] = None
    escaped = False
    for idx, ch in enumerate(body):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is None:
            if ch in {"'", '"', "`"}:
                quote = ch
                start = idx
            continue
        if ch == quote:
            if start is not None:
                quoted.update(range(start, idx + 1))
            quote = None
            start = None
    return quoted


def _extract_mentions(body: str) -> list[str]:
    """Return unique mentions in the order they first appear.

    Mentions inside simple quotes or inline-code backticks are treated as
    examples/literal text, not as routing directives.
    """
    quoted_indexes = _quoted_char_indexes(body)
    seen = set()
    mentions = []
    for match in _MENTION_RE.finditer(body):
        if match.start() in quoted_indexes:
            continue
        handle = match.group(1)
        if handle not in seen:
            seen.add(handle)
            mentions.append(handle)
    return mentions


def _valid_recipient_profiles(conn: sqlite3.Connection) -> set[str]:
    """Return the set of profiles allowed to receive mailbox notifications.

    Includes the canonical Ágora handoff profiles plus any profile that has
    ever checked in via the agent status endpoint. This filters out literal
    placeholders such as ``@perfil`` or ``@fallback`` used in system/docs
    text, while still allowing real mentions of known or active profiles.
    """
    valid: set[str] = set(_AGORA_KNOWN_HANDOFF_PROFILES)
    rows = conn.execute("SELECT DISTINCT profile FROM agora_agent_status").fetchall()
    valid.update(r["profile"] for r in rows)
    return valid


def _resolve_recipients(conn: sqlite3.Connection, mentions: list[str]) -> set[str]:
    """Resolve mentions to concrete recipients.

    * @all and @todos expand to every profile currently present in
      ``agora_agent_status``.
    * Individual handles are kept only when they name a known Ágora handoff
      profile or a profile that has checked in via agent status. This avoids
      creating notifications for literal/documentation placeholders such as
      ``@perfil`` or ``@fallback``.
    """
    recipients: set[str] = set()
    broadcast = False
    valid_profiles = {p.lower() for p in _valid_recipient_profiles(conn)}
    for handle in mentions:
        lower_handle = handle.lower()
        if lower_handle in _BROADCAST_HANDLES:
            broadcast = True
        elif lower_handle in valid_profiles:
            recipients.add(handle)
    if broadcast:
        rows = conn.execute(
            "SELECT DISTINCT profile FROM agora_agent_status"
        ).fetchall()
        recipients.update(r["profile"] for r in rows)
    return recipients


def _create_notifications(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    channel_slug: str,
    body: str,
    author_profile: Optional[str],
) -> list[int]:
    """Persist notifications for any mentions in the message body.

    Emits a ``notification.created`` event per recipient so live clients can
    update unread counts without polling.
    """
    mentions = _extract_mentions(body)
    if not mentions:
        return []
    recipients = _resolve_recipients(conn, mentions)
    # Authors should never be notified (or woken) by their own mentions,
    # including @all / @todos broadcasts and self-mentions.
    recipients.discard(author_profile)
    if not recipients:
        return []

    now = int(time.time())
    snippet = (body[:120] + "…") if len(body) > 120 else body
    ids = []
    ids_by_recipient: dict[str, list[int]] = {}
    for recipient in sorted(recipients):
        cur = conn.execute(
            "INSERT INTO agora_notifications "
            "(recipient, message_id, channel_id, body_snippet, author_profile, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (recipient, message_id, channel_id, snippet, author_profile, now),
        )
        if cur.lastrowid is None:
            raise RuntimeError("failed to create Ágora notification")
        notification_id = cur.lastrowid
        ids.append(notification_id)
        ids_by_recipient.setdefault(recipient, []).append(notification_id)
        _emit_event(
            conn,
            "notification",
            str(notification_id),
            "created",
            {
                "recipient": recipient,
                "message_id": message_id,
                "channel_id": channel_id,
                "body_snippet": snippet,
                "author_profile": author_profile,
            },
        )
    deliveries = _deliver_mentions_to_tmux(
        conn,
        recipients,
        message_id=message_id,
        channel_slug=channel_slug,
        body=body,
        author_profile=author_profile,
    )
    for delivery in deliveries:
        if not delivery.get("delivered"):
            continue
        recipient = str(delivery.get("profile") or delivery.get("recipient") or "").strip()
        if not recipient:
            continue
        for notification_id in ids_by_recipient.get(recipient, []):
            # Active tmux delivery means the local agent terminal received the
            # mention immediately. Mark it read/acked so dashboard bells show
            # pending attention, not already-delivered wake-ups.
            conn.execute(
                """
                UPDATE agora_notifications
                SET read_at = COALESCE(read_at, ?),
                    ack_at = COALESCE(ack_at, ?)
                WHERE id = ?
                """,
                (now, now, notification_id),
            )
            _emit_event(
                conn,
                "notification",
                str(notification_id),
                "read",
                {"recipient": recipient, "message_id": message_id, "auto_ack": True},
            )
    return ids


def _agent_status_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "profile": row["profile"],
        "state": row["state"],
        "current_task_id": row["current_task_id"],
        "current_step": row["current_step"],
        "status_text": row["status_text"],
        "last_heartbeat_at": row["last_heartbeat_at"],
        "pid": row["pid"],
        "run_id": row["run_id"],
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None,
    }


def _decision_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "proposal": row["proposal"],
        "decision": row["decision"],
        "rationale": row["rationale"],
        "decided_by": row["decided_by"],
        "created_at": row["created_at"],
    }


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "entity_type": row["entity_type"],
        "entity_id": row["entity_id"],
        "event_type": row["event_type"],
        "payload": json.loads(row["payload"]) if row["payload"] else None,
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@router.get("/channels")
def list_channels():
    """List all channels, including defaults."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM agora_channels ORDER BY id").fetchall()
        return {"channels": [_channel_dict(r) for r in rows]}


class CreateChannelBody(BaseModel):
    slug: str
    name: str
    description: Optional[str] = None


@router.post("/channels")
def create_channel(payload: CreateChannelBody):
    """Create a new channel. Slug must be unique and lowercase-ASCII.

    This endpoint remains available for backwards compatibility. New admin
    UI should prefer ``POST /admin/channels`` for stricter validation and
    clearer semantics.

    Request body::

        {
          "slug": "meu-canal",
          "name": "Meu Canal",
          "description": "Opcional"
        }

    Response::

        {"channel": {"id": 6, "slug": "meu-canal", ...}}
    """
    slug = _validate_channel_slug(payload.slug)
    name = _validate_channel_name(payload.name)
    channel = _insert_channel(slug, name, payload.description)
    return {"channel": channel}


@router.post("/admin/channels", status_code=http_status.HTTP_201_CREATED)
def admin_create_channel(payload: CreateChannelBody):
    """Admin endpoint to create an Ágora channel.

    Validated for the future settings/admin view. The slug is normalized
    (lowercase, accents removed) and restricted to ASCII letters, digits,
    hyphens and underscores.

    Request body::

        {
          "slug": "novo-canal",
          "name": "Novo Canal",
          "description": "Descrição opcional."
        }

    Response (201 Created)::

        {"channel": {"id": 6, "slug": "novo-canal", "name": "Novo Canal", ...}}

    Errors:

    * ``400`` — slug invalid or name empty.
    * ``409`` — slug already exists.
    """
    slug = _validate_channel_slug(payload.slug)
    name = _validate_channel_name(payload.name)
    channel = _insert_channel(slug, name, payload.description)
    return {"channel": channel}


@router.post("/channels/cleanup-emptyname")
def cleanup_emptyname_channel():
    """Migrate any legacy 'emptyname' channel into the default 'praca' channel.

    Idempotent: no-op if there is no 'emptyname' channel. Messages and threads
    are reassigned to 'praca' so nothing is lost.
    """
    with _connect() as conn:
        result = _cleanup_emptyname_channel(conn)
        conn.commit()
    return result


@router.get("/channels/{slug}")
def get_channel(slug: str):
    """Return a single channel by slug."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM agora_channels WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"channel '{slug}' not found")
        return {"channel": _channel_dict(row)}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@router.get("/channels/{slug}/messages")
def list_channel_messages(
    slug: str,
    thread_id: Optional[int] = Query(None),
    since_id: Optional[int] = Query(None),
    before_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Return messages for a channel, optionally filtered by thread.

    ``since_id`` returns only messages with id > value, useful for polling.
    ``before_id`` returns only messages with id < value, useful for pagination
    of older messages.
    """
    if since_id is not None and before_id is not None:
        raise HTTPException(
            status_code=400,
            detail="cannot use both since_id and before_id"
        )

    with _connect() as conn:
        channel = conn.execute(
            "SELECT id FROM agora_channels WHERE slug = ?", (slug,)
        ).fetchone()
        if channel is None:
            raise HTTPException(status_code=404, detail=f"channel '{slug}' not found")

        conditions = ["channel_id = ?"]
        params: list[Any] = [channel["id"]]
        if thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(thread_id)

        # ``since_id`` polls forward chronologically; ``before_id`` walks
        # backward for older pages. The default initial load returns the
        # newest messages first so the UI can show the latest N immediately.
        if since_id is not None:
            conditions.append("id > ?")
            params.append(since_id)
            order_by = "created_at ASC, id ASC"
        elif before_id is not None:
            conditions.append("id < ?")
            params.append(before_id)
            order_by = "created_at DESC, id DESC"
        else:
            order_by = "created_at DESC, id DESC"

        where = " AND ".join(conditions)
        # Fetch one extra row to determine if there are older/newer pages without
        # requiring a follow-up empty request when the total is an exact multiple
        # of the page size.
        rows = conn.execute(
            f"SELECT * FROM agora_messages WHERE {where} ORDER BY {order_by} LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        return {
            "messages": [_message_dict(r) for r in rows[:limit]],
            "has_more": has_more,
        }


class CreateMessageBody(BaseModel):
    body: str
    author_type: str = "human"
    author_profile: Optional[str] = None
    thread_id: Optional[int] = None
    linked_task_id: Optional[str] = None


@router.post("/channels/{slug}/messages")
def create_channel_message(slug: str, payload: CreateMessageBody):
    """Post a message to a channel. Optionally inside a thread or linked to a Kanban task."""
    body = payload.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="body is required")

    author_type = (payload.author_type or "human").strip().lower()
    if author_type not in ("human", "agent", "system"):
        raise HTTPException(
            status_code=400, detail="author_type must be human|agent|system"
        )

    # Human messages sent from the dashboard often omit author_profile. The
    # notifications table stores only author_profile, so a null value renders
    # as "sistema" in the mailbox; default to a non-null label.
    author_profile = (payload.author_profile or "").strip() or None
    if author_profile is None and author_type == "human":
        author_profile = "human"

    with _connect() as conn:
        channel = conn.execute(
            "SELECT id FROM agora_channels WHERE slug = ?", (slug,)
        ).fetchone()
        if channel is None:
            raise HTTPException(status_code=404, detail=f"channel '{slug}' not found")

        if payload.thread_id is not None:
            thread = conn.execute(
                "SELECT id FROM agora_threads WHERE id = ? AND channel_id = ?",
                (payload.thread_id, channel["id"]),
            ).fetchone()
            if thread is None:
                raise HTTPException(
                    status_code=404, detail="thread not found in this channel"
                )

        now = int(time.time())
        cur = conn.execute(
            "INSERT INTO agora_messages (channel_id, thread_id, author_type, author_profile, body, linked_task_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                channel["id"],
                payload.thread_id,
                author_type,
                author_profile,
                body,
                payload.linked_task_id,
                now,
            ),
        )
        _emit_event(
            conn,
            "message",
            str(cur.lastrowid),
            "created",
            {
                "channel_id": channel["id"],
                "channel_slug": slug,
                "thread_id": payload.thread_id,
                "author_type": author_type,
                "author_profile": author_profile,
            },
        )
        _create_notifications(
            conn,
            message_id=cur.lastrowid,
            channel_id=channel["id"],
            channel_slug=slug,
            body=body,
            author_profile=author_profile,
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM agora_messages WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return {"message": _message_dict(row)}


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@router.get("/threads")
def list_threads(channel_id: Optional[int] = Query(None)):
    """List threads, optionally filtered by channel."""
    with _connect() as conn:
        conditions: list[str] = []
        params: list[Any] = []
        if channel_id is not None:
            conditions.append("channel_id = ?")
            params.append(channel_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = conn.execute(
            f"SELECT * FROM agora_threads {where} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
        return {"threads": [_thread_dict(r) for r in rows]}


class CreateThreadBody(BaseModel):
    channel_id: int
    title: str
    linked_task_id: Optional[str] = None


@router.post("/threads")
def create_thread(payload: CreateThreadBody):
    """Create a discussion thread inside a channel."""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    with _connect() as conn:
        channel = conn.execute(
            "SELECT id FROM agora_channels WHERE id = ?", (payload.channel_id,)
        ).fetchone()
        if channel is None:
            raise HTTPException(status_code=404, detail="channel not found")

        now = int(time.time())
        cur = conn.execute(
            "INSERT INTO agora_threads (channel_id, title, linked_task_id, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (payload.channel_id, title, payload.linked_task_id, "open", now),
        )
        _emit_event(
            conn,
            "thread",
            str(cur.lastrowid),
            "created",
            {
                "channel_id": payload.channel_id,
                "title": title,
                "linked_task_id": payload.linked_task_id,
            },
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agora_threads WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return {"thread": _thread_dict(row)}


@router.get("/threads/{thread_id}")
def get_thread(thread_id: int):
    """Return thread details plus its messages."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM agora_threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="thread not found")
        messages = conn.execute(
            "SELECT * FROM agora_messages WHERE thread_id = ? ORDER BY created_at ASC, id ASC",
            (thread_id,),
        ).fetchall()
        return {
            "thread": _thread_dict(row),
            "messages": [_message_dict(r) for r in messages],
        }


class UpdateThreadBody(BaseModel):
    status: Optional[str] = None
    linked_task_id: Optional[str] = None


@router.patch("/threads/{thread_id}")
def update_thread(thread_id: int, payload: UpdateThreadBody):
    """Update thread status or linked task."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM agora_threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="thread not found")

        sets, params = [], []
        if payload.status is not None:
            if payload.status not in ("open", "closed", "archived"):
                raise HTTPException(
                    status_code=400, detail="status must be open|closed|archived"
                )
            sets.append("status = ?")
            params.append(payload.status)
        if payload.linked_task_id is not None:
            sets.append("linked_task_id = ?")
            params.append(payload.linked_task_id)
        if not sets:
            return {"thread": _thread_dict(row)}

        params.append(thread_id)
        conn.execute(f"UPDATE agora_threads SET {', '.join(sets)} WHERE id = ?", params)
        _emit_event(
            conn,
            "thread",
            str(thread_id),
            "updated",
            {"status": payload.status, "linked_task_id": payload.linked_task_id},
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agora_threads WHERE id = ?", (thread_id,)
        ).fetchone()
        return {"thread": _thread_dict(row)}


# ---------------------------------------------------------------------------
# Agent status / presence
# ---------------------------------------------------------------------------


@router.get("/agents/status")
def list_agent_status():
    """Return current semantic status for every known agent profile.

    This is Ágora's own presence store. It can be enriched read-only with
    Kanban active workers on the frontend, but the backend keeps only the
    semantic state submitted by agents or humans.
    """
    with _connect() as conn:
        if RESERVED_AGENT_PROFILES:
            placeholders = ",".join("?" * len(RESERVED_AGENT_PROFILES))
            rows = conn.execute(
                f"""
                SELECT * FROM agora_agent_status
                WHERE profile NOT IN ({placeholders})
                ORDER BY last_heartbeat_at DESC
                """,
                tuple(sorted(RESERVED_AGENT_PROFILES)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agora_agent_status ORDER BY last_heartbeat_at DESC"
            ).fetchall()
        return {"agents": [_agent_status_dict(r) for r in rows]}


class AgentStatusBody(BaseModel):
    state: str
    current_task_id: Optional[str] = None
    current_step: Optional[str] = None
    status_text: Optional[str] = None
    pid: Optional[int] = None
    run_id: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("/agents/status/{profile}")
def upsert_agent_status(profile: str, payload: AgentStatusBody):
    """Report or update the semantic status of an agent profile."""
    profile = profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")

    state = (payload.state or "").strip().lower()
    if not state:
        raise HTTPException(status_code=400, detail="state is required")
    if state not in (
        "idle",
        "deliberating",
        "working",
        "reviewing",
        "waiting-human",
        "blocked",
        "error",
    ):
        raise HTTPException(
            status_code=400,
            detail="state must be idle|deliberating|working|reviewing|waiting-human|blocked|error",
        )

    # Pseudo-profiles such as ``human`` are valid message authors, not agents.
    if _is_reserved_agent_profile(profile):
        raise HTTPException(
            status_code=400,
            detail=f"profile '{profile}' is reserved and cannot be reported as an agent",
        )

    # If the caller did not supply a PID, discover the local Hermes/tmux process
    # for the profile so the dashboard card can expose a clickable PID button.
    pid = payload.pid
    if pid is None:
        pid = _resolve_profile_pid(profile)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO agora_agent_status
            (profile, state, current_task_id, current_step, status_text, last_heartbeat_at, pid, run_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile) DO UPDATE SET
                state = excluded.state,
                current_task_id = excluded.current_task_id,
                current_step = excluded.current_step,
                status_text = excluded.status_text,
                last_heartbeat_at = excluded.last_heartbeat_at,
                pid = excluded.pid,
                run_id = excluded.run_id,
                metadata_json = excluded.metadata_json
            """,
            (
                profile,
                state,
                payload.current_task_id,
                payload.current_step,
                payload.status_text,
                int(time.time()),
                pid,
                payload.run_id,
                json.dumps(payload.metadata) if payload.metadata else None,
            ),
        )
        _emit_event(
            conn,
            "agent_status",
            profile,
            "updated",
            {"state": state, "current_task_id": payload.current_task_id},
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agora_agent_status WHERE profile = ?", (profile,)
        ).fetchone()
        return {"agent": _agent_status_dict(row)}


@router.get("/agents/status/{profile}")
def get_agent_status(profile: str):
    """Return status for a single agent profile."""
    profile = profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    if _is_reserved_agent_profile(profile):
        raise HTTPException(status_code=404, detail=f"agent '{profile}' not found")
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM agora_agent_status WHERE profile = ?", (profile,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"agent '{profile}' not found")
        return {"agent": _agent_status_dict(row)}


@router.post("/agents/{profile}/open-terminal")
def open_agent_terminal(profile: str):
    """Open a human-visible terminal attached to the agent tmux session."""
    profile = profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    result = _open_agent_tmux_terminal(profile)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    if not (result.get("opened") or result.get("focused")):
        raise HTTPException(status_code=503, detail=result)
    return {"ok": True, "terminal": result}


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@router.get("/decisions")
def list_decisions(thread_id: Optional[int] = Query(None)):
    """List decisions, optionally filtered by thread."""
    with _connect() as conn:
        if thread_id is not None:
            rows = conn.execute(
                "SELECT * FROM agora_decisions WHERE thread_id = ? ORDER BY created_at DESC",
                (thread_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agora_decisions ORDER BY created_at DESC"
            ).fetchall()
        return {"decisions": [_decision_dict(r) for r in rows]}


class CreateDecisionBody(BaseModel):
    thread_id: int
    proposal: str
    decision: str
    rationale: Optional[str] = None
    decided_by: Optional[str] = None


@router.post("/decisions")
def create_decision(payload: CreateDecisionBody):
    """Record a formal decision/proposal outcome inside a thread."""
    proposal = payload.proposal.strip()
    decision = payload.decision.strip()
    if not proposal or not decision:
        raise HTTPException(
            status_code=400, detail="proposal and decision are required"
        )

    with _connect() as conn:
        thread = conn.execute(
            "SELECT id FROM agora_threads WHERE id = ?", (payload.thread_id,)
        ).fetchone()
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")

        now = int(time.time())
        cur = conn.execute(
            "INSERT INTO agora_decisions (thread_id, proposal, decision, rationale, decided_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                payload.thread_id,
                proposal,
                decision,
                payload.rationale,
                payload.decided_by,
                now,
            ),
        )
        _emit_event(
            conn,
            "decision",
            str(cur.lastrowid),
            "created",
            {
                "thread_id": payload.thread_id,
                "decision": decision,
                "decided_by": payload.decided_by,
            },
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agora_decisions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return {"decision": _decision_dict(row)}


# ---------------------------------------------------------------------------
# Events — long polling fallback for clients without WebSocket
# ---------------------------------------------------------------------------


@router.get("/events")
def list_events(
    since_id: Optional[int] = Query(0), limit: int = Query(200, ge=1, le=500)
):
    """Return append-only events newer than ``since_id``.

    This is the easiest way for a frontend to catch live changes when it
    cannot maintain a WebSocket.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM agora_events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (since_id or 0, limit),
        ).fetchall()
        return {
            "events": [_event_dict(r) for r in rows],
            "cursor": (rows[-1]["id"] if rows else since_id or 0),
        }


# ---------------------------------------------------------------------------
# WebSocket: /events?since=<event_id>
# ---------------------------------------------------------------------------

_EVENT_POLL_SECONDS = 0.3


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    """Push new events to the client as they are appended."""
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    try:
        since_raw = ws.query_params.get("since", "0")
        try:
            cursor = int(since_raw)
        except ValueError:
            cursor = 0

        def _fetch_new(cursor_val: int) -> tuple[int, list[dict[str, Any]]]:
            with _connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM agora_events WHERE id > ? ORDER BY id ASC LIMIT 200",
                    (cursor_val,),
                ).fetchall()
                out = []
                new_cursor = cursor_val
                for r in rows:
                    out.append(_event_dict(r))
                    new_cursor = r["id"]
                return new_cursor, out

        while True:
            cursor, events = await asyncio.to_thread(_fetch_new, cursor)
            if events:
                await ws.send_json({"events": events, "cursor": cursor})
            await asyncio.sleep(_EVENT_POLL_SECONDS)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return
    except Exception as exc:
        log.warning("Agora event stream error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mentions / mailbox
# ---------------------------------------------------------------------------


@router.get("/notifications")
def list_notifications(
    recipient: str,
    unread_only: bool = Query(False),
    since_id: Optional[int] = Query(None),
    before_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Return a recipient's notifications, newest first.

    ``since_id`` returns only notifications with id > value, useful for polling.
    ``before_id`` returns only notifications with id < value, useful for
    pagination of older notifications.
    """
    if since_id is not None and before_id is not None:
        raise HTTPException(
            status_code=400,
            detail="cannot use both since_id and before_id"
        )

    with _connect() as conn:
        conditions = ["recipient = ?"]
        params: list[Any] = [recipient]
        if unread_only:
            conditions.append("read_at IS NULL")
        if since_id is not None:
            conditions.append("id > ?")
            params.append(since_id)
        if before_id is not None:
            conditions.append("id < ?")
            params.append(before_id)
        where = " AND ".join(conditions)
        # Fetch one extra row to detect the end of pagination without an empty
        # follow-up request when the total is an exact multiple of the page size.
        rows = conn.execute(
            f"SELECT * FROM agora_notifications WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        return {
            "notifications": [_notification_dict(r) for r in rows[:limit]],
            "has_more": has_more,
        }


@router.get("/notifications/count")
def count_notifications(recipient: str):
    """Return total/unread counts for a recipient."""
    with _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM agora_notifications WHERE recipient = ?",
            (recipient,),
        ).fetchone()[0]
        unread = conn.execute(
            "SELECT COUNT(*) FROM agora_notifications WHERE recipient = ? AND read_at IS NULL",
            (recipient,),
        ).fetchone()[0]
        return {"recipient": recipient, "total": total, "unread": unread}


@router.post("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int):
    """Mark a single notification as read."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM agora_notifications WHERE id = ?", (notification_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="notification not found")
        now = int(time.time())
        conn.execute(
            "UPDATE agora_notifications SET read_at = ? WHERE id = ?",
            (now, notification_id),
        )
        _emit_event(
            conn,
            "notification",
            str(notification_id),
            "read",
            {"recipient": row["recipient"], "message_id": row["message_id"]},
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agora_notifications WHERE id = ?", (notification_id,)
        ).fetchone()
        return {"notification": _notification_dict(row)}


@router.post("/notifications/read-all")
def mark_all_notifications_read(recipient: str):
    """Mark every notification for a recipient as read."""
    with _connect() as conn:
        now = int(time.time())
        conn.execute(
            "UPDATE agora_notifications SET read_at = ? WHERE recipient = ? AND read_at IS NULL",
            (now, recipient),
        )
        _emit_event(
            conn, "notification", recipient, "read_all", {"recipient": recipient}
        )
        conn.commit()
        return {"recipient": recipient, "ok": True}

def register(ctx: PluginContext) -> None:
    """Plugin registration entry point for Ágora Dashboard."""
    ctx.register_hook("kanban_task_completed", _on_kanban_task_completed)
    ctx.register_hook("kanban_task_blocked", _on_kanban_task_blocked)
