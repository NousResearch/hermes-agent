"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 index corruption prevents ``INSERT INTO messages``, the gateway
accumulates messages in ``_pending_messages`` (memory-only) and the live
``agent._session_messages`` cannot be flushed via ``_flush_messages_to_session_db``.
On shutdown, ``.clear()`` discards the only surviving copy — permanent user data loss.

This module provides three hooks:

1. ``flush_pending_to_file()`` — called BEFORE ``_pending_messages.clear()``
   during shutdown.  Serialises any non-empty pending slots to a JSON file
   under ``<hermes_home>/pending_messages/``.

2. ``recover_pending_to_db()`` — called AFTER ``runner.start()`` on startup.
   Reads flush files, inserts messages into state.db via ``SessionDB.append_message``
   (so FTS indexing, session metadata, and display_kind are handled correctly),
   then deletes the flush file on success.

3. ``flush_agent_history_to_file()`` — called from ``_finalize_shutdown_agents``
   when ``_flush_messages_to_session_db`` raises.  Dumps the live
   ``agent._session_messages`` to the same atomic JSON recovery directory.

See issue #72680 for the full incident report.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home

    flush_dir = get_hermes_home() / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on platforms that support directory fsync."""
    if os.name != "posix":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_payload(flush_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one private, uniquely named recovery payload.

    Returns the path of the published payload file.
    """
    from utils import atomic_json_write

    file_id = uuid.uuid4().hex
    final_path = flush_dir / f"pending-{file_id}.json"
    atomic_json_write(
        final_path,
        payload,
        mode=0o600,
        default=str,
    )

    try:
        _fsync_directory(flush_dir)
    except OSError as exc:
        # The atomically published file is still the only recovery copy.
        # Keep it even if this filesystem cannot persist directory entries.
        logger.debug("Failed to fsync pending-message directory: %s", exc)
    return final_path


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
            _write_payload(
                flush_dir,
                {
                    "session_key": session_key,
                    "reason": reason,
                    "ts": ts,
                    "data": serialised,
                },
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


# Reason tag for transcript messages dropped by the in-memory pending cap
# during live operation (#78182). These payloads carry the full transcript
# message dict so they can be replayed verbatim once the DB recovers.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"


def spool_dropped_transcript_message(
    session_id: str,
    message: Dict[str, Any],
) -> Optional[Path]:
    """Spool a transcript message evicted by the runtime pending cap.

    Uses the same on-disk pending spool as :func:`flush_pending_to_file`
    (one atomic JSON payload per message under
    ``<hermes_home>/pending_messages/``), so a runtime cap rotation no
    longer silently discards user data while the process stays up
    (#78182).

    Returns the written spool path, or ``None`` when spooling failed —
    callers must degrade to the previous drop-and-log behaviour.
    """
    try:
        flush_dir = _get_flush_dir()
        return _write_payload(
            flush_dir,
            {
                "session_key": session_id,
                "reason": TRANSCRIPT_CAP_DROP_REASON,
                "ts": int(time.time()),
                "seq": next(_TRANSCRIPT_SPOOL_SEQ),
                "data": {
                    "session_id": session_id,
                    "message": message,
                },
            },
        )
    except Exception as exc:
        logger.debug(
            "Failed to spool cap-dropped transcript message for %s: %s",
            session_id, exc,
        )
        return None


# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()


def drain_transcript_spool(session_id: str, replay) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*.

    ``replay(message_dict)`` is invoked for each spooled message in drop
    order; the spool file is deleted only after a successful replay.  On
    the first replay failure the drain stops and remaining files are kept
    for the next attempt (the DB is likely still unhealthy).

    Returns ``(replayed, remaining)`` — messages replayed and spool files
    left behind for a later retry.
    """
    try:
        flush_dir = _get_flush_dir()
        candidates = list(flush_dir.glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0

    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON:
            continue
        if payload.get("session_key") != session_id:
            continue
        message = (payload.get("data") or {}).get("message")
        if not isinstance(message, dict):
            logger.warning(
                "Removing structurally invalid transcript spool file %s", path,
            )
            path.unlink(missing_ok=True)
            continue
        entries.append(
            (payload.get("ts", 0), payload.get("seq", 0), path.name, path, message)
        )

    replayed = 0
    ordered = sorted(entries, key=lambda e: e[:3])
    remaining = 0
    for idx, (_ts, _seq, _name, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            logger.warning(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s",
                path, session_id, exc,
            )
            remaining = len(ordered) - idx
            break
        path.unlink(missing_ok=True)
        replayed += 1

    if replayed:
        logger.info(
            "Replayed %d spooled transcript message(s) for %s after DB recovery",
            replayed, session_id,
        )
    return replayed, remaining


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
    session_db=None,
) -> int:
    """Recover flushed pending messages into state.db via SessionDB.

    Reads all ``*.json`` files from the flush directory, inserts messages
    using ``SessionDB.append_message`` (so FTS indexing, session metadata
    updates, and all required columns are handled correctly), and deletes
    the flush file on success.

    Parameters
    ----------
    session_db:
        An existing ``SessionDB`` instance.  If ``None``, a new one is
        opened on the default ``state.db`` path.

    Returns
    -------
    int
        Number of messages recovered.
    """
    flush_dir = _get_flush_dir()
    flush_files = sorted(flush_dir.glob("*.json"))
    if not flush_files:
        return 0

    # Use the provided SessionDB or open one on the default path.
    own_db = False
    if session_db is None:
        from hermes_state import SessionDB
        session_db = SessionDB()
        own_db = True

    recovered = 0
    for path in flush_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Agent-history snapshots use a different schema (reason +
            # messages list) and are meant for manual operator recovery,
            # not automatic DB insertion. Skip them silently.
            if payload.get("reason") == "shutdown-with-unpersisted-agent-history":
                continue
            # Cap-dropped transcript payloads carry the full message dict
            # keyed by session_id — replay directly (#78182). This handles
            # spool files that were never drained before a restart.
            if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
                data = payload.get("data", {}) or {}
                spooled_sid = data.get("session_id", "")
                message = data.get("message")
                if not spooled_sid or not isinstance(message, dict):
                    logger.warning(
                        "Cannot recover structurally invalid transcript spool "
                        "file %s; preserved for manual inspection",
                        path,
                    )
                    continue
                session_db.append_message(
                    session_id=spooled_sid,
                    role=message.get("role", "unknown"),
                    content=message.get("content") or "",
                    timestamp=message.get("timestamp") or payload.get("ts"),
                )
                recovered += 1
                path.unlink(missing_ok=True)
                continue
            session_key = payload.get("session_key", "")
            data = payload.get("data", {})
            text = data.get("text", "")
            if not text or not session_key:
                logger.warning(
                    "Cannot recover structurally invalid pending message from %s; "
                    "the flush file has been preserved",
                    path,
                )
                continue

            # The session_key is a gateway routing key (e.g.
            # "agent:main:telegram:supergroup:...").  We need the actual
            # session_id (e.g. "20260728_120000_abc123") to append a
            # message row.  Try the session_id field from the serialised
            # data first; fall back to scanning sessions for a matching
            # session_key in the source column.
            session_id = data.get("session_id", "")

            if not session_id:
                # Try to extract from the session_key itself — gateway
                # session keys contain the session_id as the last segment
                # in some formats, but that's not guaranteed.  Log and
                # skip if we can't resolve it.
                logger.warning(
                    "Cannot recover pending message for %s: no session_id "
                    "in flush file and session_key-to-id resolution is not "
                    "available at this recovery stage. The message text is "
                    "preserved in %s",
                    session_key, path,
                )
                continue

            session_db.append_message(
                session_id=session_id,
                role="user",
                content=text,
                timestamp=payload.get("ts", int(time.time())),
            )
            recovered += 1
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(
                "Failed to recover pending message from %s: %s",
                path, exc,
            )
            # Leave the file for next startup retry.

    if own_db:
        try:
            session_db.close()
        except Exception:
            pass

    if recovered:
        logger.info(
            "Recovered %d pending message(s) from shutdown flush", recovered,
        )
    return recovered


def flush_agent_history_to_file(
    session_id: Optional[str],
    history: list,
) -> None:
    """Best-effort dump of an agent's in-memory transcript before teardown.

    Used when ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite
    index corruption, #72680): the live ``agent._session_messages`` could
    not be written to disk, and a plain debug log would lose it permanently
    when the process exits. Serialize to an atomic JSON file outside the
    broken DB so an operator can salvage the conversation after repairing
    state.db.

    Failures are swallowed — shutdown must never block on a best-effort
    backup.
    """
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        snapshot = []
        for _m in history:
            try:
                snapshot.append(
                    _m if isinstance(_m, (dict, list, str, int, float, bool, type(None)))
                    else str(_m)
                )
            except Exception:
                continue
        _write_payload(
            flush_dir,
            {
                "reason": "shutdown-with-unpersisted-agent-history",
                "issue": "#72680",
                "session_id": session_id,
                "count": len(snapshot),
                "messages": snapshot,
            },
        )
        logger.warning(
            "Preserved %d in-memory message(s) for session %s "
            "(possible FTS corruption — recover after repairing state.db)",
            len(snapshot),
            session_id,
        )
    except Exception as _e:
        logger.warning(
            "Agent-history shutdown preservation failed for session %s: %s",
            session_id, _e,
        )
