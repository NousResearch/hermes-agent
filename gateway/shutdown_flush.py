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
from typing import Any, Dict, List, Optional

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


def flush_interrupt_messages_to_file(
    interrupts: Dict[str, Any],
    *,
    reason: str = "shutdown",
) -> int:
    """Serialise pending ``_interrupt_message`` texts to disk.

    Sibling of :func:`flush_pending_to_file` for the busy-mode interrupt
    path: when a follow-up message arrives while the agent is busy, the
    gateway acks ("⚡ Interrupting current task...") and stores the text on
    the running agent (``agent._interrupt_message``), waiting for the agent
    loop to reach its next checkpoint and replay it as a new turn.  A turn
    blocked on a long-running tool (or on an internal wait primitive) may
    never reach that checkpoint before shutdown — and that field is
    in-memory only, so the user's message vanished with the process even
    though the ack promised delivery (#lost-interrupt-text).

    Each entry is written in the SAME payload shape as a pending-message
    flush so ``recover_pending_to_db`` replays them unchanged.  Values may
    be plain strings or objects exposing ``.text`` (mirrors
    :func:`_serialise_value`).  Slash-command texts are skipped: the
    recovery path discards them anyway (commands must never be replayed as
    agent input), and a stale command replay after restart is worse than a
    dropped one.

    Parameters
    ----------
    interrupts:
        Mapping of ``session_key`` → interrupt text (str) or event-like
        object with a ``.text`` attribute.  ``None`` / empty values are
        skipped.
    reason:
        Logged context (``shutdown``, ``restart``, etc.).

    Returns
    -------
    int
        Number of interrupt texts flushed.
    """
    if not interrupts:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0

    for session_key, value in list(interrupts.items()):
        if value is None or not session_key:
            continue
        try:
            if hasattr(value, "text"):
                text = getattr(value, "text", "") or ""
            else:
                text = str(value)
            text = text.strip()
            # Commands are never replayed as user input (see
            # recover_pending_to_db's caller-side safety net); do not flush.
            if not text or text.startswith("/"):
                continue
            _write_payload(
                flush_dir,
                {
                    "session_key": session_key,
                    "reason": reason,
                    "ts": ts,
                    "data": {"text": text, "origin": "interrupt_message"},
                },
            )
            flushed += 1
        except Exception as exc:
            logger.debug(
                "Failed to flush interrupt message for %s: %s",
                session_key, exc,
            )

    if flushed:
        logger.info(
            "Flushed %d pending interrupt message(s) to %s (reason=%s)",
            flushed, flush_dir, reason,
        )
    return flushed


def flush_overflow_to_file(
    overflow_by_session: Dict[str, Any],
    *,
    reason: str = "shutdown",
) -> int:
    """Serialise the FIFO overflow tails (``queued_events``) to disk.

    Sibling of :func:`flush_pending_to_file` for the second half of the
    gateway FIFO (#99882): the adapter slot holds the queue head, and the
    per-session ``SessionState.conversation.queued_events`` list holds the
    tail.  Shutdown flushed only the slot, so every follow-up parked in
    overflow at restart time vanished with the process.  Each overflow
    event is written as its own payload in the same shape as a slot flush
    so ``recover_pending_to_db`` replays them unchanged; a ``seq`` field
    preserves arrival order within a session.

    Returns the number of events flushed.
    """
    if not overflow_by_session:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0

    for session_key, events in list(overflow_by_session.items()):
        if not session_key or not events:
            continue
        for seq, value in enumerate(list(events)):
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
                        "seq": seq,
                        "data": serialised,
                    },
                )
                flushed += 1
            except Exception as exc:
                logger.debug(
                    "Failed to flush overflow message for %s: %s",
                    session_key, exc,
                )

    if flushed:
        logger.info(
            "Flushed %d queued overflow message(s) to %s (reason=%s)",
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


def _session_key_alias_variants(session_key: str) -> List[str]:
    """Return phone/LID alias variants of a WhatsApp gateway session key.

    Gateway session keys end with the WhatsApp identity segment — the DM
    chat_id, or the group participant id when per-user group sessions are
    enabled. The bridge may materialise the SAME account under either a
    phone-format id or a linked-device LID, and the pending flush may
    carry a different shape than the one persisted on the session row
    (live case: flush ``agent:<ns>:whatsapp:dm:55...`` vs row key
    ``agent:<ns>:whatsapp:dm:56...@lid-era``). ``expand_whatsapp_aliases``
    deterministically returns every alias reachable through the bridge's
    ``lid-mapping-*.json`` files, and returns the input unchanged for
    non-WhatsApp or non-numeric tails (group JIDs, other platforms), so
    the variant list degrades to just the original when expansion does
    not apply.  Only the trailing segment is touched (``rpartition``).
    """
    raw = str(session_key or "")
    variants = [raw]
    if ":whatsapp:" not in raw:
        return variants
    try:
        from gateway.whatsapp_identity import expand_whatsapp_aliases

        prefix, _, tail = raw.rpartition(":")
        if not prefix:
            return variants
        expanded = sorted(expand_whatsapp_aliases(tail))
        variants.extend(f"{prefix}:{v}" for v in expanded if f"{prefix}:{v}" != raw)
    except Exception:
        pass
    return variants


def _candidate_session_dbs(session_key: str, ambient_db):
    """Yield (SessionDB, owned) candidates that may hold ``session_key``.

    Under multiplex, gateway session keys encode the owning profile as the
    ``agent:<namespace>:`` prefix and that profile's transcript rows live in
    ``profiles/<name>/state.db`` — NOT in the root store the recovery worker
    opens by default.  Yield the profile DB first (the authoritative store
    for the key), then the ambient DB as a legacy fallback (single-profile
    installs and keys predating the namespace split).
    """
    seen_paths = set()
    profile = None
    try:
        parts = str(session_key).split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            namespace = parts[1] or "main"
            profile = None if namespace == "main" else namespace
    except Exception:
        profile = None

    if profile:
        try:
            from hermes_cli.profiles import get_profile_dir, profile_exists

            if profile_exists(profile):
                db_path = Path(get_profile_dir(profile)) / "state.db"
                if db_path.exists() and str(db_path) not in seen_paths:
                    seen_paths.add(str(db_path))
                    from hermes_state import SessionDB

                    yield SessionDB(db_path=db_path), True
        except Exception as exc:
            logger.debug(
                "Could not open profile DB for %r (key %r): %s",
                profile, session_key, exc,
            )

    if ambient_db is not None:
        yield ambient_db, False


def _resolve_session_id_for_key(session_db, session_key: str):
    """Resolve a gateway session_key to its live session row.

    Tries every phone/LID alias variant of the key (see
    :func:`_session_key_alias_variants`) against the key's owning profile
    DB first, then the ambient store.  ``source`` is derived from the key's
    platform segment, matching how gateway turns stamp session rows
    (``Platform.<name>.value``, e.g. ``whatsapp`` → source ``whatsapp``).

    Returns ``(session_id, db_to_write)``: the session_id string and the
    SessionDB instance the message MUST be appended to.  Under multiplex the
    authoritative store for an ``agent:<namespace>:`` key is
    ``profiles/<namespace>/state.db`` — appending to the ambient root store
    instead violates the sessions FK and strands the message (#multiplex-
    recovery-split).  ``db_to_write`` is the ambient ``session_db`` when the
    row was found there, and a caller-owned profile DB otherwise (the caller
    releases it via :func:`hermes_state.release_or_close` after the append).
    Returns ``("", None)`` when nothing matches.
    """
    if not session_key:
        return "", None
    # ``session_db`` may legitimately be None: the resolver opens the key's
    # profile store itself via _candidate_session_dbs (the ambient DB is
    # only a fallback candidate).

    platform = ""
    try:
        parts = str(session_key).split(":")
        if len(parts) >= 3:
            platform = parts[2].strip().lower()
    except Exception:
        platform = ""

    for key_variant in _session_key_alias_variants(session_key):
        for db, owned in _candidate_session_dbs(session_key, session_db):
            try:
                row = db.find_latest_gateway_session_for_peer(
                    source=platform, session_key=key_variant,
                )
            except Exception as exc:
                logger.debug(
                    "session_key lookup failed for %r: %s", key_variant, exc,
                )
                row = None
            if row and row.get("id"):
                return str(row["id"]), db
            if owned:
                try:
                    from hermes_state import release_or_close

                    release_or_close(db)
                except Exception:
                    pass
    return "", None


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
        from hermes_state import get_shared_session_db
        session_db = get_shared_session_db()
        own_db = True

    def _close_owned_db() -> None:
        if not own_db:
            return
        try:
            from hermes_state import release_or_close
            release_or_close(session_db)
        except Exception:
            pass

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
            target_db = session_db

            if not session_id:
                session_id, target_db = _resolve_session_id_for_key(
                    session_db, session_key,
                )

            if not session_id or target_db is None:
                # Try to extract from the session_key itself — gateway
                # session keys contain the session_id as the last segment
                # in some formats, but that's not guaranteed.  Log and
                # skip if we can't resolve it.
                logger.warning(
                    "Cannot recover pending message for %s: no session_id "
                    "in flush file and session_key-to-id resolution failed. "
                    "The message text is preserved in %s",
                    session_key, path,
                )
                continue

            # ``target_db`` is the store that holds the resolved session row:
            # the ambient DB when the key belongs to it, else the owning
            # profile's DB (multiplex).  Appending elsewhere violates the
            # sessions FK and strands the message.
            _profile_db_owned = target_db is not session_db
            try:
                target_db.append_message(
                    session_id=session_id,
                    role="user",
                    content=text,
                    timestamp=payload.get("ts", int(time.time())),
                )
            except BaseException:
                if _profile_db_owned:
                    try:
                        from hermes_state import release_or_close

                        release_or_close(target_db)
                    except Exception:
                        pass
                raise
            recovered += 1
            path.unlink(missing_ok=True)
            if _profile_db_owned:
                try:
                    from hermes_state import release_or_close

                    release_or_close(target_db)
                except Exception:
                    pass
        except BaseException:
            # Shutdown cancellation/interrupt must not strand an owned DB.
            _close_owned_db()
            raise
        except Exception as exc:
            logger.warning(
                "Failed to recover pending message from %s: %s",
                path, exc,
            )
            # Leave the file for next startup retry.

    _close_owned_db()

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
