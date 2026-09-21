"""X experience inbox — deterministic capture store for Sahil's X manager.

Sahil drops anything worth remembering (a take, a build experience, a repo,
a weird agent behaviour) into the X manager channel as ``/x <note>``. This
module stores each capture once (case-insensitive text dedupe), keeps a
``used`` flag so the thesis incubator does not re-consume it, and exposes
the small query surface the incubator and the weekly mix report need.

Fail-closed: an empty capture is rejected; a duplicate is a no-op returning
the existing id. There is no publish path here — captures are raw material
for downstream approval-only draft lanes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from config import DB_PATH

CAPTURE_TABLE = "x_inbox_captures"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CAPTURE_TABLE} (
    id          TEXT PRIMARY KEY,
    text_key    TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'user',
    channel_id  TEXT NOT NULL DEFAULT '',
    author_id   TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_x_inbox_used ON {CAPTURE_TABLE}(used);
CREATE INDEX IF NOT EXISTS idx_x_inbox_created ON {CAPTURE_TABLE}(created_at);
"""


def _new_id() -> str:
    return f"xin_{uuid.uuid4().hex[:10]}"


def _text_key(text: str) -> str:
    """Stable dedupe key: casefolded, whitespace-collapsed text hash."""
    norm = " ".join(text.casefold().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def init_db() -> None:
    """Idempotently create the inbox table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def add_capture(
    text: str,
    *,
    source: str = "user",
    channel: str = "",
    author_id: str = "",
) -> str:
    """Persist one capture. Returns the capture id.

    Raises ``ValueError`` for an empty capture. Duplicates (same
    case-insensitive normalized text) are no-ops that return the existing id.
    """
    clean = (text or "").strip()
    if not clean:
        raise ValueError("capture has no text")
    init_db()
    key = _text_key(clean)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            f"SELECT id FROM {CAPTURE_TABLE} WHERE text_key = ?", (key,)
        ).fetchone()
        if row:
            return row[0]
        # Insert with the key column (schema below) so dedupe is a plain index hit.
        cap_id = _new_id()
        conn.execute(
            f"""
            INSERT INTO {CAPTURE_TABLE}
              (id, text_key, text, source, channel_id, author_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cap_id,
                key,
                clean,
                source,
                channel,
                author_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cap_id
    finally:
        conn.close()


def list_captures(*, limit: int = 100, used: bool | None = None) -> list[dict]:
    """List captures newest-first, optionally filtered by ``used``."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE used = ?" if used is not None else ""
        params: tuple = (int(used),) if used is not None else ()
        rows = conn.execute(
            f"""
            SELECT id, text, source, channel_id, author_id, created_at, used
            FROM {CAPTURE_TABLE}
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params + (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_used(ids: list[str]) -> int:
    """Mark captures as consumed by the thesis incubator. Returns rows changed."""
    if not ids:
        return 0
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"UPDATE {CAPTURE_TABLE} SET used = 1 WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def counts() -> dict:
    """Total captures and unused captures, for the weekly mix report."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM {CAPTURE_TABLE}").fetchone()[0]
        used = conn.execute(
            f"SELECT COUNT(*) FROM {CAPTURE_TABLE} WHERE used = 1"
        ).fetchone()[0]
        return {"total": total, "used": used, "unused": total - used}
    finally:
        conn.close()
