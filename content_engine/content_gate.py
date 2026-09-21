"""G03 content gate — approval-gated enforcement for the content engine.

The gate is the code-level chokepoint that prevents auto-delivery of any
generated content. It owns two surfaces:

1. **Draft approval state** (delegates to ``drafts`` table, status column):
   ``gate_publish(draft_id)`` returns True only when status == 'approved'.

2. **Pending approval items** (new table ``content_gate_pending``): for
   non-draft content like X-scout suggestions, the gate stages them as
   ``approved:false`` rows and sends an approval card to Discord. Publishing
   proceeds only after an explicit ``!approve <id>`` flips the row.

Dry-run support: when env ``CONTENT_GATE_DRY_RUN=1`` is set, no Discord posts
are made and no rows are mutated — calls log intent and return synthetic ids.

This module is the single source of truth for "is this allowed to publish".
``publish_to_postiz.py`` and any future publish path MUST call
``gate_publish()`` before queueing to Postiz or any external platform.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Optional

from config import DB_PATH

# ── Schema ────────────────────────────────────────────────────────────────

PENDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS content_gate_pending (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,           -- 'x_scout', 'repurpose', 'manual', ...
    brand         TEXT NOT NULL,
    platform      TEXT NOT NULL,
    title         TEXT,
    body_text     TEXT NOT NULL,
    media_path    TEXT,
    external_url  TEXT,                    -- e.g. the source tweet URL for x_scout
    approved      INTEGER NOT NULL DEFAULT 0,   -- 0 = pending, 1 = approved, -1 = rejected
    created_at    TEXT NOT NULL,
    decided_at    TEXT,
    decided_by    TEXT,
    decision_note TEXT,
    metadata      TEXT                     -- JSON blob for source-specific fields
);

CREATE INDEX IF NOT EXISTS idx_gate_pending_approved ON content_gate_pending(approved);
CREATE INDEX IF NOT EXISTS idx_gate_pending_source ON content_gate_pending(source);
CREATE INDEX IF NOT EXISTS idx_gate_pending_created ON content_gate_pending(created_at);
"""


def init_gate_db() -> None:
    """Idempotently create the pending table alongside the main drafts DB."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(PENDING_SCHEMA)
    conn.commit()
    conn.close()


# ── Dry-run helper ────────────────────────────────────────────────────────

def _dry_run() -> bool:
    return os.environ.get("CONTENT_GATE_DRY_RUN", "0") == "1"


# ── Draft-side gate (delegates to drafts table) ───────────────────────────

def check_approval(draft_id: str) -> bool:
    """Return True iff the draft row exists and status == 'approved'.

    Safe against missing table (fresh DB) or any sqlite error — returns False.
    """
    from database import get_draft

    try:
        d = get_draft(draft_id)
    except Exception:
        return False
    if not d:
        return False
    return d.get("status") == "approved"


def gate_publish(draft_id: str) -> bool:
    """Chokepoint for any publish path. Returns True only when publishing is
    permitted — i.e. the draft has been explicitly approved.

    Dry-run mode: always returns False (publish is never permitted in dry-run).
    This is safe-by-default: a dry-run that "succeeds" would defeat the gate.
    """
    if _dry_run():
        return False
    return check_approval(draft_id)


# ── Pending-item gate (non-draft content) ─────────────────────────────────

def _new_pending_id(source: str) -> str:
    return f"gate_{source[:4]}_{uuid.uuid4().hex[:10]}"


def register_for_approval(
    body_text: str,
    brand: str,
    platform: str,
    source: str,
    title: Optional[str] = None,
    media_path: Optional[str] = None,
    external_url: Optional[str] = None,
    metadata: Optional[dict] = None,
    send_card: bool = True,
    channel_id: Optional[str] = None,
) -> str:
    """Stage a non-draft content item as ``approved:0`` and (optionally) send
    an approval card to Discord. Returns the pending-item id.

    In dry-run mode: skips the DB write and the Discord post; returns a
    synthetic id so callers can still exercise the call shape.
    """
    pending_id = _new_pending_id(source)

    if _dry_run():
        print(f"[content_gate] dry-run: would register {pending_id} ({source}, {brand}/{platform})")
        return pending_id

    init_gate_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        INSERT INTO content_gate_pending
          (id, source, brand, platform, title, body_text, media_path,
           external_url, approved, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            pending_id,
            source,
            brand,
            platform,
            title,
            body_text,
            media_path,
            external_url,
            datetime.now(UTC).isoformat(),
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()

    if send_card:
        _send_approval_card(
            pending_id=pending_id,
            brand=brand,
            platform=platform,
            body_text=body_text,
            title=title,
            external_url=external_url,
            media_path=media_path,
            channel_id=channel_id,
        )

    return pending_id


def process_approval(
    pending_id: str,
    action: str,
    decided_by: Optional[str] = None,
    note: Optional[str] = None,
) -> bool:
    """Apply a decision to a pending item.

    ``action`` must be one of ``"approve"``, ``"reject"``, ``"amend"``.
    ``"amend"`` records the decision but leaves ``approved=0`` (caller must
    re-register the amended item or update body_text explicitly).

    Returns True iff the row was found and updated.
    """
    if action not in ("approve", "reject", "amend"):
        raise ValueError(f"unknown approval action: {action!r}")

    if _dry_run():
        print(f"[content_gate] dry-run: would {action} {pending_id}")
        return True

    new_state = {"approve": 1, "reject": -1, "amend": 0}[action]

    init_gate_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        """
        UPDATE content_gate_pending
           SET approved = ?, decided_at = ?, decided_by = ?, decision_note = ?
         WHERE id = ?
        """,
        (
            new_state,
            datetime.now(UTC).isoformat(),
            decided_by,
            note,
            pending_id,
        ),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def get_pending(source: Optional[str] = None) -> list[dict]:
    """List all pending (``approved=0``) items, optionally filtered by source."""
    init_gate_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    if source:
        rows = conn.execute(
            "SELECT * FROM content_gate_pending WHERE approved = 0 AND source = ? "
            "ORDER BY created_at ASC",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM content_gate_pending WHERE approved = 0 "
            "ORDER BY created_at ASC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_pending_approved(pending_id: str) -> bool:
    """Return True iff the pending row exists and approved == 1."""
    init_gate_db()
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT approved FROM content_gate_pending WHERE id = ?", (pending_id,)
    ).fetchone()
    conn.close()
    return bool(row and row[0] == 1)


# ── Discord approval card ─────────────────────────────────────────────────

_DEFAULT_CARD_CHANNEL = os.getenv("DISCORD_CONTENT_CHANNEL_ID", "1507448580649123900")


def _send_approval_card(
    pending_id: str,
    brand: str,
    platform: str,
    body_text: str,
    title: Optional[str],
    external_url: Optional[str],
    media_path: Optional[str],
    channel_id: Optional[str] = None,
) -> bool:
    """Send a single approval card for a pending item. Reuses the
    ``discord_digest._post`` helper so we stay on the same bot token and
    rate-limit behaviour. Returns True on successful post."""
    from discord_digest import _post  # local import to avoid cycle at module load

    target = channel_id or _DEFAULT_CARD_CHANNEL
    stamp = datetime.now(UTC).strftime("%d/%m/%y %H:%M UTC")
    lines = [
        f"🛂 **Approval needed** · {stamp}",
        f"`{pending_id}` · `{brand}` / `{platform}`",
    ]
    if title:
        lines.append(f"**{title}**")
    if external_url:
        lines.append(external_url)
    body = (body_text or "").strip()
    if body:
        snippet = body if len(body) <= 600 else body[:597] + "..."
        lines.append("")
        lines.append(snippet)
    lines.append("")
    lines.append(f"Reply `!approve {pending_id}` to publish · `!reject {pending_id}` to drop")

    return _post(target, "\n".join(lines), file_path=media_path)
