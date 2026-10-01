"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 corruption blocks ``INSERT INTO messages``, ``_pending_messages`` and the live
``agent._session_messages`` are the only surviving copies; shutdown ``.clear()`` would drop them.
All hooks write atomic JSON payloads under ``<hermes_home>/pending_messages/``:
``flush_pending_to_file`` / ``flush_overflow_to_file`` (queue head / FIFO tail, before clear),
``recover_pending_to_db`` (after ``runner.start()``; ordinary user turns use
``SessionDB.append_message`` and transcript envelopes use the canonical idempotent batch writer;
files are deleted only after success), ``flush_agent_history_to_file`` (DB flush raised),
``spool_dropped_transcript_message`` / ``drain_transcript_spool``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Reason tag for transcript messages dropped by the in-memory pending cap during live
# operation. Payloads carry the full transcript message dict for verbatim replay.
# See #78182.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"
AGENT_HISTORY_REASON = "shutdown-with-unpersisted-agent-history"
# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home
    flush_dir = get_hermes_home() / "pending_messages"
    from hermes_constants import assert_named_profile_home_live
    assert_named_profile_home_live(flush_dir)
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _write_payload(flush_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one private, uniquely named recovery payload; return its path."""
    from utils import atomic_json_write
    final_path = flush_dir / f"pending-{uuid.uuid4().hex}.json"
    atomic_json_write(final_path, payload, mode=0o600, default=str)
    if os.name == "posix":
        # Persist the directory entry too; keep the published file (the only recovery copy) even if
        # fsync fails.
        try:
            directory_fd = os.open(flush_dir, os.O_RDONLY)
        except OSError as exc:
            logger.debug("Failed to fsync pending-message directory: %s", exc)
        else:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                logger.debug("Failed to fsync pending-message directory: %s", exc)
            finally:
                os.close(directory_fd)
    return final_path


def _flush_value(flush_dir: Path, kind: str, session_key: str, value: Any, **extra: Any) -> bool:
    """Serialise and write one pending value; return True when a payload was written."""
    try:
        serialised = _serialise_value(value)
        if serialised is None:
            return False
        _write_payload(flush_dir, {"session_key": session_key, **extra, "data": serialised})
        return True
    except Exception as exc:
        logger.debug("Failed to flush %s message for %s: %s", kind, session_key, exc)
        return False


def flush_pending_to_file(pending: Dict[str, Any], *, reason: str = "shutdown") -> int:
    """Serialise non-empty ``_pending_messages`` slots (``MessageEvent`` or str); return count."""
    if not pending:
        return 0
    flush_dir, ts, flushed = _get_flush_dir(), int(time.time()), 0
    for session_key, value in list(pending.items()):
        if value is not None:
            flushed += _flush_value(flush_dir, "pending", session_key, value, reason=reason, ts=ts)
    if flushed:
        logger.info("Flushed %d pending message(s) to %s (reason=%s)", flushed, flush_dir, reason)
    return flushed


def flush_overflow_to_file(overflow_by_session: Dict[str, Any], *, reason: str = "shutdown") -> int:
    """Serialise the FIFO overflow tails (``queued_events``) to disk; return events flushed.

    The adapter slot holds the queue head and ``SessionState.conversation.queued_events`` the
    tail; both must survive restart. Each event is its own payload in the slot-flush shape so
    ``recover_pending_to_db`` replays them unchanged; ``seq`` preserves arrival order per session.
    """
    if not overflow_by_session:
        return 0
    flush_dir, ts, flushed = _get_flush_dir(), int(time.time()), 0
    for session_key, events in list(overflow_by_session.items()):
        if not session_key or not events:
            continue
        for seq, value in enumerate(list(events)):
            if value is not None:
                flushed += _flush_value(flush_dir, "overflow", session_key, value, reason=reason,
                                        ts=ts, seq=seq)
    if flushed:
        logger.info("Flushed %d queued overflow message(s) to %s (reason=%s)", flushed, flush_dir,
                    reason)
    return flushed


def spool_dropped_transcript_message(session_id: str, message: Dict[str, Any]) -> Optional[Path]:
    """Spool a cap-evicted transcript message; ``None`` on failure (callers degrade to drop+log).

    Uses the same on-disk pending spool as :func:`flush_pending_to_file` (one atomic JSON payload per
    message under ``<hermes_home>/pending_messages/``), so a runtime cap rotation no longer silently
    discards user data while the process stays up (#78182).
    """
    try:
        from hermes_state_transcript_codec import encode_session_transcript
        transcript = encode_session_transcript([message])
        return _write_payload(_get_flush_dir(), {
            "session_key": session_id, "reason": TRANSCRIPT_CAP_DROP_REASON, "ts": int(time.time()),
            "seq": next(_TRANSCRIPT_SPOOL_SEQ),
            # ``message`` keeps the legacy live-drain/older-build shape byte-compatible; current
            # restart recovery prefers the UID-stamped versioned envelope.
            "data": {"session_id": session_id, "message": dict(message), "transcript": transcript},
        })
    except Exception as exc:
        logger.debug("Failed to spool cap-dropped transcript message for %s: %s", session_id, exc)
        return None


def drain_transcript_spool(session_id: str, replay, *, db_known_failing: bool = False) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*; return ``(replayed,
    remaining)``. ``replay(message_dict)`` runs per message in drop order; a spool file is deleted
    only after its replay succeeds. The first failure stops the drain (the DB is likely still
    unhealthy) and keeps the rest for retry. With ``db_known_failing`` (the caller's last write
    already failed and is being logged/escalated) a replay failure is expected and logs at DEBUG,
    so a stalled session does not add one WARNING per append on top of its ERROR (#114266).
    """
    try:
        candidates = list(_get_flush_dir().glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0
    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        # A parseable non-object file (scalar/list) cannot be attributed to any session: skip it
        # like unparseable JSON instead of letting ``.get`` abort the whole drain.
        if (not isinstance(payload, dict)
                or payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON
                or payload.get("session_key") != session_id):
            continue
        message = (payload.get("data") or {}).get("message")
        if not isinstance(message, dict):
            logger.warning("Removing structurally invalid transcript spool file %s", path)
            path.unlink(missing_ok=True)
            continue
        entries.append((payload.get("ts", 0), payload.get("seq", 0), path.name, path, message))
    ordered, replayed, remaining = sorted(entries, key=lambda e: e[:3]), 0, 0
    for idx, (_ts, _seq, _name, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            (logger.debug if db_known_failing else logger.warning)(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s", path, session_id, exc)
            remaining = len(ordered) - idx
            break
        path.unlink(missing_ok=True)
        replayed += 1
    if replayed:
        logger.info("Replayed %d spooled transcript message(s) for %s after DB recovery", replayed,
                    session_id)
    return replayed, remaining


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _serialise_value(value: Any) -> Optional[dict]:
    """Convert a pending message value to a JSON-serialisable dict."""
    if hasattr(value, "text"):  # MessageEvent-like object
        result: Dict[str, Any] = {"text": getattr(value, "text", "")}
        for attr in ("session_id", "platform", "sender_id", "sender_name", "reply_to", "media",
                     "raw_event"):
            val = getattr(value, attr, None)
            if val is not None:
                result[attr] = val if _json_safe(val) else str(val)
        return result
    if isinstance(value, str):  # runner-level _pending_messages
        return {"text": value}
    if isinstance(value, dict) and _json_safe(value):
        return value
    return {"text": str(value)}


def _spool_order_number(value: Any) -> float | int:
    """A total numeric sort key for untrusted JSON ordering fields."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return 0


def _decode_transcript_payload(payload: Dict[str, Any]) -> tuple[str, list, bool]:
    """One transcript spool payload -> ``(session_id, messages, upgraded_legacy)``."""
    reason = payload.get("reason")
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError("transcript spool data must be an object")
    session_id = data.get("session_id") or payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("transcript spool session_id is required")

    from hermes_state_transcript_codec import decode_session_transcript, encode_session_transcript
    envelope = data.get("transcript")
    if envelope is not None:
        messages = decode_session_transcript(envelope)
        upgraded = False
        if reason == TRANSCRIPT_CAP_DROP_REASON:
            for message in messages:
                if message.get("timestamp") is None and payload.get("ts") is not None:
                    message["timestamp"] = payload["ts"]
                    upgraded = True
            if upgraded:
                data["transcript"] = {**envelope, "messages": messages}
                payload["data"] = data
        return session_id, messages, upgraded
    if reason == AGENT_HISTORY_REASON:
        legacy_messages = payload.get("messages")
    else:
        legacy_message = data.get("message")
        if isinstance(legacy_message, dict) and legacy_message.get("timestamp") is None:
            legacy_message = {**legacy_message, "timestamp": payload.get("ts")}
        legacy_messages = [legacy_message] if isinstance(legacy_message, dict) else None
    if not isinstance(legacy_messages, list):
        raise ValueError("legacy transcript messages are missing")
    envelope = encode_session_transcript(legacy_messages)
    data["transcript"] = envelope
    data["session_id"] = session_id
    payload["data"] = data
    # Keep the legacy projection stable too, so rolling back to an older Hermes build does not mint
    # different recovery identities or lose operator-readable history.
    if reason == AGENT_HISTORY_REASON:
        payload["messages"] = envelope["messages"]
        payload["count"] = len(envelope["messages"])
    else:
        data["message"] = envelope["messages"][0]
    return session_id, envelope["messages"], True


def _append_recovered_transcript(session_db, session_id: str, messages: list) -> int:
    """Use SessionDB's canonical idempotent writer; retain the legacy fake/store seam."""
    writer = getattr(type(session_db), "append_recovered_messages_batch", None)
    if callable(writer):
        inserted = writer(session_db, session_id, messages)
        if isinstance(inserted, bool) or not isinstance(inserted, int):
            raise RuntimeError("session store returned an invalid transcript recovery count")
        return inserted

    # Backward-compatible duck-typed stores used by embedding callers predate the batch codec.
    # Production SessionDB always takes the branch above.
    for message in messages:
        timestamp = message.get("timestamp")
        session_db.append_message(
            session_id=session_id,
            role=message.get("role", "unknown"),
            content=message.get("content") or "",
            timestamp=timestamp,
        )
    return len(messages)


def recover_pending_to_db(session_db=None, *, session_resolver=None) -> int:
    """Replay shutdown spool files, deleting each only after its canonical write succeeds.

    Transcript envelopes are decoded and appended once per session, in their persisted ``(ts,
    seq, filename)`` order. That keeps assistant/tool adjacency in one transaction and lets the
    SessionDB codec restore every canonical field. Other pending-message payloads retain their
    historical one-file-at-a-time path.
    """
    flush_files = list(_get_flush_dir().glob("*.json"))
    if not flush_files:
        return 0

    parsed: list[tuple[Path, Dict[str, Any]]] = []
    for path in flush_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            parsed.append((path, payload))
        except Exception as exc:
            logger.warning("Failed to recover pending message from %s: %s", path, exc)
    parsed.sort(key=lambda item: (
        _spool_order_number(item[1].get("ts")),
        _spool_order_number(item[1].get("seq")),
        item[0].name,
    ))

    own_db = session_db is None
    if own_db:
        from hermes_state_registry import acquire
        session_db = acquire()
    recovered = 0
    try:
        transcript_groups: Dict[str, list[tuple[Path, list]]] = {}
        blocked_transcript_sessions: set[str] = set()
        regular: list[tuple[Path, Dict[str, Any]]] = []
        for path, payload in parsed:
            if payload.get("reason") not in {TRANSCRIPT_CAP_DROP_REASON, AGENT_HISTORY_REASON}:
                regular.append((path, payload))
                continue
            try:
                session_id, messages, upgraded = _decode_transcript_payload(payload)
                if upgraded:
                    # Publish stable message UIDs before touching SQLite. A crash after commit but
                    # before unlink can then prove the retry is already present instead of duplicating it.
                    from utils import atomic_json_write
                    atomic_json_write(path, payload, mode=0o600, default=str)
                transcript_groups.setdefault(session_id, []).append((path, messages))
            except Exception as exc:
                raw_data = payload.get("data") or {}
                blocked_session = (
                    raw_data.get("session_id") if isinstance(raw_data, dict) else None
                ) or payload.get("session_id")
                if isinstance(blocked_session, str) and blocked_session:
                    blocked_transcript_sessions.add(blocked_session)
                logger.warning("Failed to decode transcript recovery payload %s; preserving it: %s", path, exc)

        for session_id, entries in transcript_groups.items():
            if session_id in blocked_transcript_sessions:
                logger.warning(
                    "Holding back %d transcript spool file(s) for %s because another payload is invalid",
                    len(entries), session_id)
                continue
            messages = [message for _path, batch in entries for message in batch]
            try:
                from agent.message_metadata import message_uid_or_none
                unique_messages: Dict[str, dict] = {}
                for message in messages:
                    uid = message_uid_or_none(message)
                    if uid is None:
                        raise ValueError("recovery message_uid is required")
                    previous = unique_messages.get(uid)
                    if previous is not None and previous != message:
                        raise ValueError(f"conflicting recovery copies for message_uid: {uid}")
                    unique_messages.setdefault(uid, message)
                recovered += _append_recovered_transcript(session_db, session_id, list(unique_messages.values()))
            except Exception as exc:
                logger.warning(
                    "Failed to recover transcript for %s; preserving %d spool file(s): %s",
                    session_id, len(entries), exc)
                continue
            for path, _messages in entries:
                path.unlink(missing_ok=True)

        for path, payload in regular:
            # One rejected ordinary payload must not poison every later boot; its file stays put.
            try:
                inserted = _recover_one_payload(
                    session_db, path, payload, session_resolver=session_resolver)
                if inserted is not None:
                    recovered += inserted
                    path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to recover pending message from %s: %s", path, exc)
    finally:
        if own_db:  # shutdown cancellation/interrupt must not strand an owned DB
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close
                release_or_close(session_db)
    if recovered:
        logger.info("Recovered %d pending message(s) from shutdown flush", recovered)
    return recovered


def _recover_one_payload(session_db, path: Path, payload: Dict[str, Any], *,
                         session_resolver=None) -> Optional[int]:
    """Append one ordinary pending-user payload; ``None`` preserves an invalid file."""
    session_key, data = payload.get("session_key", ""), payload.get("data", {})
    text = data.get("text", "")
    if not text or not session_key:
        logger.warning("Cannot recover structurally invalid pending message from %s; "
                       "the flush file has been preserved", path)
        return None
    # session_key is a gateway routing key (e.g. "agent:main:telegram:..."); appending a row
    # needs the real session_id, which real payloads lack — the resolver supplies it together with
    # the store owning the key. ``session_db`` (the owned default) serves only payloads that already
    # carry a session_id; a resolver-resolved payload goes to the resolver's db alone, never the
    # ambient root store (a None db from the resolver is not a fallback signal — it is "preserve").
    session_id, target_db = data.get("session_id", ""), session_db
    if not session_id and session_resolver is not None:
        try:
            resolved = session_resolver(session_key, not_after=payload.get("ts"))
        except Exception as exc:
            logger.debug("Session key->id resolution failed for %s: %s", session_key, exc)
            resolved = None
        if resolved and resolved[1] is not None:
            session_id, target_db = resolved
    if not session_id:
        logger.warning("Cannot recover pending message for %s: no session_id in flush file and "
                       "session_key-to-id resolution failed. "
                       "The message text is preserved in %s", session_key, path)
        return None
    target_db.append_message(session_id=session_id, role="user", content=text,
                             timestamp=payload.get("ts", int(time.time())))
    return 1


def flush_agent_history_to_file(session_id: Optional[str], history: list) -> None:
    """Best-effort spool of an agent's in-memory transcript before teardown. Used when
    ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite corruption): the versioned envelope
    sits outside the broken DB and is replayed through SessionDB on the next healthy startup.
    Failures are swallowed — shutdown must never block on a best-effort backup."""
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        from hermes_state_transcript_codec import encode_session_transcript
        transcript = encode_session_transcript(history)
        snapshot = transcript["messages"]
        _write_payload(flush_dir, {
            "reason": AGENT_HISTORY_REASON, "issue": "#72680",
            "session_id": session_id, "count": len(snapshot), "messages": snapshot,
            "data": {"session_id": session_id, "transcript": transcript},
        })
        logger.warning("Preserved %d in-memory message(s) for session %s "
                       "(possible FTS corruption — recover after repairing state.db)",
                       len(snapshot), session_id)
    except Exception as _e:
        logger.warning("Agent-history shutdown preservation failed for session %s: %s", session_id,
                       _e)
