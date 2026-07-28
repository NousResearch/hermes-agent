"""Flush pending messages to disk before shutdown to prevent data loss.

When FTS5 index corruption prevents ``INSERT INTO messages``, the gateway
accumulates messages in ``_pending_messages`` (memory-only).  On shutdown,
``.clear()`` discards the only surviving copy — permanent user data loss.

This module provides two hooks:

1. ``flush_pending_to_file()`` — called BEFORE ``_pending_messages.clear()``
   during shutdown.  Serialises any non-empty pending slots to a JSON file
   under ``~/.hermes/pending_messages/``.

2. ``recover_pending_to_db()`` — called AFTER ``runner.start()`` on startup.
   Reads flush files, inserts messages directly into state.db, then deletes
   the flush file on success.

See issue #72680 for the full incident report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FLUSH_DIR = Path.home() / ".hermes" / "pending_messages"


def _get_flush_dir() -> Path:
    _FLUSH_DIR.mkdir(parents=True, exist_ok=True)
    return _FLUSH_DIR


def flush_pending_to_file(
    pending: Dict[str, Any],
    *,
    reason: str = "shutdown",
) -> int:
    """Serialise non-empty ``_pending_messages`` slots to disk.

    Parameters
    ----------
    pending:
        The adapter or runner ``_pending_messages`` dict.  Values may be
        ``MessageEvent`` objects (adapter) or plain strings (runner).
    reason:
        Logged context (``shutdown``, ``restart``, etc.).

    Returns
    -------
    int
        Number of sessions flushed.
    """
    if not pending:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0

    for session_key, value in list(pending.items()):
        if value is None:
            continue
        try:
            serialised = _serialise_value(value)
            if serialised is None:
                continue
            safe_key = session_key.replace("/", "_").replace("\\", "_")
            path = flush_dir / f"{safe_key}_{ts}.json"
            path.write_text(
                json.dumps({
                    "session_key": session_key,
                    "reason": reason,
                    "ts": ts,
                    "data": serialised,
                }, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            flushed += 1
        except Exception as exc:
            logger.debug(
                "Failed to flush pending message for %s: %s",
                session_key, exc,
            )

    if flushed:
        logger.info(
            "Flushed %d pending message(s) to %s (reason=%s)",
            flushed, flush_dir, reason,
        )
    return flushed


def _serialise_value(value: Any) -> Optional[dict]:
    """Convert a pending message value to a JSON-serialisable dict."""
    # MessageEvent objects have a .text attribute and other fields
    if hasattr(value, "text"):
        result: Dict[str, Any] = {"text": getattr(value, "text", "")}
        # Preserve additional fields if present
        for attr in ("session_id", "platform", "sender_id", "sender_name",
                      "reply_to", "media", "raw_event"):
            val = getattr(value, attr, None)
            if val is not None:
                try:
                    json.dumps(val)
                    result[attr] = val
                except (TypeError, ValueError):
                    result[attr] = str(val)
        return result
    # Plain string (runner-level _pending_messages)
    if isinstance(value, str):
        return {"text": value}
    # Dict — try direct serialisation
    if isinstance(value, dict):
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"text": str(value)}
    return {"text": str(value)}


def recover_pending_to_db(
    db_path: Optional[Path] = None,
) -> int:
    """Recover flushed pending messages into state.db.

    Reads all ``*.json`` files from the flush directory, inserts messages
    into the ``messages`` table, and deletes the file on success.

    Parameters
    ----------
    db_path:
        Path to state.db.  Defaults to ``~/.hermes/state.db``.

    Returns
    -------
    int
        Number of messages recovered.
    """
    import sqlite3

    flush_dir = _get_flush_dir()
    flush_files = sorted(flush_dir.glob("*.json"))
    if not flush_files:
        return 0

    if db_path is None:
        db_path = Path.home() / ".hermes" / "state.db"
    if not db_path.exists():
        logger.warning("state.db not found at %s — skipping recovery", db_path)
        return 0

    recovered = 0
    for path in flush_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            session_key = payload.get("session_key", "")
            data = payload.get("data", {})
            text = data.get("text", "")
            if not text or not session_key:
                path.unlink(missing_ok=True)
                continue

            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "INSERT INTO messages (session_key, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_key, "user", text, payload.get("ts", int(time.time()))),
                )
                conn.commit()
                recovered += 1
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "Failed to recover pending message for %s: %s",
                    session_key, exc,
                )
                # Re-save so next startup can retry
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("Failed to read flush file %s: %s", path, exc)
            path.unlink(missing_ok=True)

    if recovered:
        logger.info(
            "Recovered %d pending message(s) from shutdown flush", recovered,
        )
    return recovered
