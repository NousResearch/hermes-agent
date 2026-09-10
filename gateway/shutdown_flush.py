"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 corruption blocks ``INSERT INTO messages``, ``_pending_messages`` and the live
``agent._session_messages`` are the only surviving copies; shutdown ``.clear()`` would drop them.
All hooks write atomic JSON payloads under ``<hermes_home>/pending_messages/``:
``flush_pending_to_file`` / ``flush_overflow_to_file`` (queue head / FIFO tail, before clear),
``recover_pending_to_db`` (after ``runner.start()``; replays via ``SessionDB.append_message``,
deletes each file on success), ``flush_agent_history_to_file`` (DB flush raised),
``spool_dropped_transcript_message`` / ``drain_transcript_spool``.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Reason tag for transcript messages dropped by the in-memory pending cap during live
# operation. Payloads carry the full transcript message dict for verbatim replay.
# See #78182.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"
# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()

DURABLE_INBOUND_REASON = "durable_inbound"
_DURABLE_INBOUND_ID_KEY = "_hermes_durable_inbound_id"
_DURABLE_INBOUND_PATH_KEY = "_hermes_durable_inbound_path"
_DURABLE_INBOUND_LOCK = threading.Lock()


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home
    flush_dir = get_hermes_home() / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _write_payload(
    flush_dir: Path,
    payload: Dict[str, Any],
    *,
    file_name: str | None = None,
) -> Path:
    """Atomically write one private, uniquely named recovery payload; return its path."""
    from utils import atomic_json_write
    final_path = flush_dir / (file_name or f"pending-{uuid.uuid4().hex}.json")
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


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-safe value without losing a user turn to rich metadata."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _durable_inbound_id(session_key: str, event: Any) -> str:
    """Give one inbound event a stable, private spool identity.

    Platform message/update ids are stable across a restart.  Some adapters do
    not expose either, so keep one generated id on the in-memory event before
    its first acknowledgement.
    """
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        existing = str(metadata.get(_DURABLE_INBOUND_ID_KEY) or "").strip()
        if existing:
            return existing
    source = getattr(event, "source", None)
    identity = "|".join(
        str(value or "")
        for value in (
            session_key,
            getattr(getattr(source, "platform", None), "value", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            getattr(event, "message_id", None),
            getattr(event, "platform_update_id", None),
            getattr(event, "timestamp", None),
        )
    )
    value = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    if isinstance(metadata, dict):
        metadata[_DURABLE_INBOUND_ID_KEY] = value
    return value


def durable_inbound_obligation_id(event: Any) -> str:
    """Return the already-persisted inbound identity, never a new one.

    ``record_durable_inbound_event`` is the sole owner that creates this id.
    Consumers may use it to derive the matching outbound delivery identity,
    but must not invent a second inbound identity after dispatch begins.
    """
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get(_DURABLE_INBOUND_ID_KEY) or "").strip()


def _next_durable_inbound_sequence(flush_dir: Path) -> int:
    """Allocate the next persisted FIFO sequence from surviving records.

    Wall time can move backwards after NTP/RTC correction, so ``time_ns`` is
    not an arrival ordering authority.  A gateway owns this private spool
    exclusively; under that process contract, scanning its small bounded
    pending set gives a restart-stable monotonic sequence without another
    queue or database.
    """
    highest = 0
    for path in flush_dir.glob("inbound-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("reason") != DURABLE_INBOUND_REASON:
                continue
            highest = max(highest, int(payload.get("arrival_sequence", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return highest + 1


def _has_durable_outbound_transfer(session_key: str, inbound_obligation_id: str) -> bool:
    """Whether an existing delivery-ledger row owns this inbound turn.

    The delivery ledger is the only outbound owner.  Its stable-id helper is
    deliberately reused here so a crash after its durable write but before
    inbound unlink never causes the agent to regenerate a second reply.
    """
    if not session_key or not inbound_obligation_id:
        return False
    try:
        from gateway.delivery_ledger import (
            inbound_transfer_state,
            ledger_enabled,
        )

        if not ledger_enabled():
            return False

        # Exhausted delivery is still transferred, not permission to rerun tools.
        return inbound_transfer_state(session_key, inbound_obligation_id) is not None
    except Exception:
        logger.debug("Could not resolve durable outbound transfer", exc_info=True)
        return False


def _serialise_inbound_event(event: Any) -> Optional[dict]:
    """Serialise the normalized event shape required to replay it safely."""
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    chat_id = getattr(source, "chat_id", None)
    if not platform or not chat_id:
        return None
    source_fields = (
        "chat_id", "chat_name", "chat_type", "user_id", "user_name",
        "thread_id", "chat_topic", "user_id_alt", "chat_id_alt", "is_bot",
        "scope_id", "guild_id", "parent_chat_id", "message_id",
        "role_authorized", "profile", "auto_thread_created",
        "auto_thread_initial_name", "prospective_thread_id",
    )
    event_fields = (
        "input_members",
        "text", "user_id", "user_name", "message_id", "platform_update_id",
        "media_urls", "media_types", "media_text_inlined", "reply_to_message_id",
        "reply_to_text", "reply_to_author_id", "reply_to_author_name",
        "reply_to_is_own_message", "prompt_response", "auto_skill",
        "channel_prompt", "channel_context", "internal", "allow_gateway_control",
    )
    return {
        "source": {
            "platform": platform,
            **{
                field: _json_safe_value(getattr(source, field, None))
                for field in source_fields
            },
        },
        "message_type": getattr(getattr(event, "message_type", None), "value", "text"),
        "timestamp": getattr(getattr(event, "timestamp", None), "isoformat", lambda: None)(),
        "metadata": _json_safe_value(getattr(event, "metadata", {}) or {}),
        **{
            field: _json_safe_value(getattr(event, field, None))
            for field in event_fields
        },
    }


def record_durable_inbound_event(session_key: str, event: Any) -> bool:
    """Persist an accepted queued inbound event before its acknowledgement.

    Returning ``False`` means the caller must not claim the message was queued:
    the only authoritative copy would still be volatile memory.
    """
    try:
        data = _serialise_inbound_event(event)
        if data is None:
            return False
        obligation_id = _durable_inbound_id(session_key, event)
        flush_dir = _get_flush_dir()
        # A hash is an identity, not an arrival sequence.  Nor is wall time:
        # NTP/RTC correction can move it backwards.  The durable FIFO sequence
        # survives restart and the id remains the dedupe key.
        with _DURABLE_INBOUND_LOCK:
            matches = list(flush_dir.glob(f"inbound-*-{obligation_id}.json"))
            legacy_path = flush_dir / f"inbound-{obligation_id}.json"
            if legacy_path.exists():
                matches.append(legacy_path)
            if matches:
                existing_path = sorted(matches)[0]
                metadata = getattr(event, "metadata", None)
                if isinstance(metadata, dict):
                    metadata[_DURABLE_INBOUND_PATH_KEY] = existing_path.name
                return True
            arrival_sequence = _next_durable_inbound_sequence(flush_dir)
            final_path = flush_dir / (
                f"inbound-{arrival_sequence:020d}-{obligation_id}.json"
            )
            _write_payload(
                flush_dir,
                {
                    "session_key": session_key,
                    "reason": DURABLE_INBOUND_REASON,
                    "obligation_id": obligation_id,
                    "arrival_sequence": arrival_sequence,
                    "ts": int(time.time()),
                    "data": data,
                },
                file_name=final_path.name,
            )
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            metadata[_DURABLE_INBOUND_PATH_KEY] = final_path.name
        return True
    except Exception as exc:
        logger.warning("Unable to durably queue inbound event for %s: %s", session_key, exc)
        return False


def _deserialise_inbound_event(data: Any) -> Any:
    """Rebuild a normalized MessageEvent from a durable inbound payload."""
    if not isinstance(data, dict):
        raise ValueError("inbound payload data is not an object")
    source_data = data.get("source")
    if not isinstance(source_data, dict):
        raise ValueError("inbound payload has no source")
    from datetime import datetime
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    platform = Platform(str(source_data.get("platform") or ""))
    source_kwargs = {key: value for key, value in source_data.items() if key != "platform"}
    source = SessionSource(platform=platform, **source_kwargs)
    raw_timestamp = data.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(raw_timestamp) if raw_timestamp else datetime.now()
    except (TypeError, ValueError):
        timestamp = datetime.now()
    message_type = MessageType(str(data.get("message_type") or "text"))
    event_kwargs = {
        key: data.get(key)
        for key in (
            "input_members",
            "text", "user_id", "user_name", "message_id", "platform_update_id",
            "media_urls", "media_types", "media_text_inlined", "reply_to_message_id",
            "reply_to_text", "reply_to_author_id", "reply_to_author_name",
            "reply_to_is_own_message", "prompt_response", "auto_skill",
            "channel_prompt", "channel_context", "internal", "allow_gateway_control",
        )
        if key in data
    }
    event_kwargs["metadata"] = data.get("metadata") or {}
    return MessageEvent(source=source, message_type=message_type, timestamp=timestamp, **event_kwargs)


def recover_durable_inbound_events() -> list[Any]:
    """Load durable queued input without acknowledging it yet.

    Dispatching to an adapter merely transfers an event to RAM.  The record
    stays until a durable outbound response obligation exists (or that response
    is confirmed delivered when the ledger is disabled).  Invalid payloads
    stay on disk for an explicit operator failure rather than being silently
    lost.
    """
    try:
        candidates = sorted(_get_flush_dir().glob("inbound-*.json"))
    except Exception as exc:
        logger.warning("Cannot scan durable inbound queue: %s", exc)
        return []
    events = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("reason") != DURABLE_INBOUND_REASON:
                continue
            event = _deserialise_inbound_event(payload.get("data"))
            metadata = getattr(event, "metadata", None)
            if isinstance(metadata, dict):
                metadata[_DURABLE_INBOUND_ID_KEY] = str(payload["obligation_id"])
                metadata[_DURABLE_INBOUND_PATH_KEY] = path.name
            if _has_durable_outbound_transfer(
                str(payload.get("session_key") or ""),
                str(payload.get("obligation_id") or ""),
            ):
                # The matching outbound row is already authoritative.  This
                # is the recovery half of the one inbound→outbound transfer;
                # never re-run the user turn just because the previous process
                # died before it unlinked the inbound file.
                acknowledge_durable_inbound_event(event)
                continue
            try:
                arrival_sequence = int(payload.get("arrival_sequence"))
            except (TypeError, ValueError):
                # Compatibility ordering for the short-lived time-ns format
                # and older hash-named records. New writes never use wall time.
                arrival_sequence = int(payload.get("arrival_ns", payload.get("ts", 0)))
            events.append((arrival_sequence, path.name, event))
        except Exception as exc:
            logger.warning("Cannot restore durable inbound event %s: %s", path, exc)
    return [event for _arrival_sequence, _name, event in sorted(events)]


def acknowledge_durable_inbound_event(event: Any) -> None:
    """Remove one durable input only after its terminal response/transfer."""
    event = getattr(event, "terminal_event", None) or event
    if getattr(event, "input_members", None):
        from types import SimpleNamespace
        for member in event.input_members:
            acknowledge_durable_inbound_event(SimpleNamespace(metadata={
                _DURABLE_INBOUND_ID_KEY: member["inbound_id"]}))
        return
    metadata = getattr(event, "metadata", None)
    obligation_id = str(metadata.get(_DURABLE_INBOUND_ID_KEY) or "") if isinstance(metadata, dict) else ""
    if not obligation_id:
        return
    try:
        flush_dir = _get_flush_dir()
        file_name = (
            str(metadata.get(_DURABLE_INBOUND_PATH_KEY) or "")
            if isinstance(metadata, dict) else ""
        )
        if file_name and Path(file_name).name == file_name:
            (flush_dir / file_name).unlink(missing_ok=True)
            return
        # Compatibility with records written before arrival-order filenames.
        legacy_path = flush_dir / f"inbound-{obligation_id}.json"
        if legacy_path.exists():
            legacy_path.unlink()
            return
        for path in flush_dir.glob(f"inbound-*-{obligation_id}.json"):
            path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Unable to acknowledge durable inbound event %s: %s", obligation_id, exc)



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
        if value is None:
            continue
        metadata = getattr(value, "metadata", None)
        if isinstance(metadata, dict) and metadata.get(_DURABLE_INBOUND_ID_KEY):
            continue
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
        return _write_payload(_get_flush_dir(), {
            "session_key": session_id, "reason": TRANSCRIPT_CAP_DROP_REASON, "ts": int(time.time()),
            "seq": next(_TRANSCRIPT_SPOOL_SEQ),
            "data": {"session_id": session_id, "message": message},
        })
    except Exception as exc:
        logger.debug("Failed to spool cap-dropped transcript message for %s: %s", session_id, exc)
        return None


def drain_transcript_spool(session_id: str, replay) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*; return ``(replayed,
    remaining)``. ``replay(message_dict)`` runs per message in drop order; a spool file is deleted
    only after its replay succeeds. The first failure stops the drain (the DB is likely still
    unhealthy) and keeps the rest for retry.
    """
    try:
        candidates = list(_get_flush_dir().glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0
    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON
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
            logger.warning("Replay of spooled transcript message %s for %s failed; "
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


def recover_pending_to_db(session_db=None) -> int:
    """Replay flush-dir ``*.json`` files via ``SessionDB.append_message``, deleting each on success.

    ``session_db=None`` opens (and afterwards releases) the shared default ``state.db``.
    Returns the number of messages recovered.
    """
    flush_files = sorted(_get_flush_dir().glob("*.json"))
    if not flush_files:
        return 0
    own_db = session_db is None
    if own_db:
        from hermes_state_registry import acquire
        session_db = acquire()
    recovered = 0
    try:
        for path in flush_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Agent-history snapshots are for manual operator recovery, not automatic DB insertion.
            if payload.get("reason") == "shutdown-with-unpersisted-agent-history":
                continue
            if _recover_one_payload(session_db, path, payload):
                recovered += 1
                path.unlink(missing_ok=True)
    finally:
        if own_db:  # shutdown cancellation/interrupt must not strand an owned DB
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close
                release_or_close(session_db)
    if recovered:
        logger.info("Recovered %d pending message(s) from shutdown flush", recovered)
    return recovered


def _recover_one_payload(session_db, path: Path, payload: Dict[str, Any]) -> bool:
    """Append one flush payload to ``session_db``; False (file kept) when structurally invalid."""
    # Cap-dropped transcript payloads carry the full message dict keyed by session_id — replay directly
    # (#78182). This handles spool files that were never drained before a restart.
    if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
        # Cap-dropped payloads carry the full message dict keyed by session_id — replay directly.
        data = payload.get("data", {}) or {}
        spooled_sid, message = data.get("session_id", ""), data.get("message")
        if not spooled_sid or not isinstance(message, dict):
            logger.warning("Cannot recover structurally invalid transcript spool "
                           "file %s; preserved for manual inspection", path)
            return False
        session_db.append_message(session_id=spooled_sid, role=message.get("role", "unknown"),
                                  content=message.get("content") or "",
                                  timestamp=message.get("timestamp") or payload.get("ts"))
        return True
    session_key, data = payload.get("session_key", ""), payload.get("data", {})
    text = data.get("text", "")
    if not text or not session_key:
        logger.warning("Cannot recover structurally invalid pending message from %s; "
                       "the flush file has been preserved", path)
        return False
    # session_key is a gateway routing key (e.g. "agent:main:telegram:..."); appending a row
    # needs the real session_id, which only the serialised data can supply at this stage.
    session_id = data.get("session_id", "")
    if not session_id:
        logger.warning("Cannot recover pending message for %s: no session_id in flush file and "
                       "session_key-to-id resolution is not available at this recovery stage. "
                       "The message text is preserved in %s", session_key, path)
        return False
    session_db.append_message(session_id=session_id, role="user", content=text,
                              timestamp=payload.get("ts", int(time.time())))
    return True


def flush_agent_history_to_file(session_id: Optional[str], history: list) -> None:
    """Best-effort dump of an agent's in-memory transcript before teardown. Used when
    ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite corruption): the transcript is written
    outside the broken DB so an operator can salvage it after repairing state.db. Failures are
    swallowed — shutdown must never block on a best-effort backup."""
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        snapshot = []
        for _m in history:
            try:
                plain = isinstance(_m, (dict, list, str, int, float, bool, type(None)))
                snapshot.append(_m if plain else str(_m))
            except Exception:
                continue
        _write_payload(flush_dir, {
            "reason": "shutdown-with-unpersisted-agent-history", "issue": "#72680",
            "session_id": session_id, "count": len(snapshot), "messages": snapshot,
        })
        logger.warning("Preserved %d in-memory message(s) for session %s "
                       "(possible FTS corruption — recover after repairing state.db)",
                       len(snapshot), session_id)
    except Exception as _e:
        logger.warning("Agent-history shutdown preservation failed for session %s: %s", session_id,
                       _e)
