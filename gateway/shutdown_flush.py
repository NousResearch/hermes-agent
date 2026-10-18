"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 corruption blocks ``INSERT INTO messages``, ``_pending_messages`` and the live
``agent._session_messages`` are the only surviving copies; shutdown ``.clear()`` would drop them.
All hooks write atomic JSON payloads under ``<hermes_home>/pending_messages/``:
``flush_pending_to_file`` / ``flush_overflow_to_file`` (queue head / FIFO tail, before clear),
``recover_pending_to_db`` (at startup, before resume turns; replays via ``SessionDB.append_message``,
deletes each file on success), ``flush_agent_history_to_file`` (DB flush raised),
``spool_dropped_transcript_message`` / ``drain_transcript_spool``.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from hermes_dir_acl import make_user_dir

logger = logging.getLogger(__name__)

# Reason tag for transcript messages dropped by the in-memory pending cap during live
# operation. Payloads carry the full transcript message dict for verbatim replay.
# See #78182.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"
# Reason tag for agent-history snapshots: kept for manual operator recovery, never replayed.
AGENT_HISTORY_REASON = "shutdown-with-unpersisted-agent-history"
# Suffix for a spool file whose row the database rejected; no replay glob matches it.
QUARANTINE_SUFFIX = ".bad"
# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()
# Spool files whose row was written but which could not be deleted. Replaying one again would write
# its row twice, so both replay passes skip it for the rest of this process.
_REPLAYED_UNREMOVABLE: set[Path] = set()


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
            if payload.get("output_started"):
                # Visible output may already exist but no ledger row owns it:
                # replaying could execute the turn (and its tools) twice.  Move
                # the record out of the replay glob and keep it as evidence.
                quarantined = path.with_name("indeterminate-" + path.name)
                path.replace(quarantined)
                logger.warning(
                    "Durable inbound %s had started output without an outbound owner; "
                    "not replaying (kept as %s)", payload.get("obligation_id"), quarantined.name)
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


def _inbound_spool_ids(event: Any) -> list[str]:
    event = getattr(event, "terminal_event", None) or event
    members = getattr(event, "input_members", None)
    if members:
        return [str(m["inbound_id"]) for m in members if m.get("inbound_id")]
    own = durable_inbound_obligation_id(event)
    return [own] if own else []


def mark_durable_inbound_output_started(event: Any) -> int:
    """Settle the inbound spool as ``indeterminate`` before any visible stream effect.

    Streaming can reach the platform before a delivery-ledger obligation (the
    output owner) exists.  A crash in that gap must not make startup replay a
    turn whose output is already real, so each spool record is stamped with
    ``output_started`` first.  Recovery then quarantines (never replays) a
    stamped record that has no outbound transfer.  A later obligation write
    retires the record as usual.  Idempotent; returns the number of records
    stamped now.
    """
    stamped = 0
    ids = _inbound_spool_ids(event)
    if not ids:
        return 0
    flush_dir = _get_flush_dir()
    with _DURABLE_INBOUND_LOCK:
        for obligation_id in ids:
            paths = list(flush_dir.glob(f"inbound-*-{obligation_id}.json"))
            legacy_path = flush_dir / f"inbound-{obligation_id}.json"
            if legacy_path.exists():
                paths.append(legacy_path)
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("output_started"):
                    continue
                payload["output_started"] = int(time.time())
                _write_payload(flush_dir, payload, file_name=path.name)
                stamped += 1
    return stamped


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




def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home
    flush_dir = get_hermes_home() / "pending_messages"
    from hermes_constants import assert_named_profile_home_live
    assert_named_profile_home_live(flush_dir)
    make_user_dir(flush_dir, parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _write_payload(
    flush_dir: Path,
    payload: dict[str, Any],
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


def flush_pending_to_file(pending: dict[str, Any], *, reason: str = "shutdown") -> int:
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


def flush_overflow_to_file(overflow_by_session: dict[str, Any], *, reason: str = "shutdown") -> int:
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


def spool_dropped_transcript_message(session_id: str, message: dict[str, Any]) -> Optional[Path]:
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


def drain_transcript_spool(session_id: str, replay, *, db_known_failing: bool = False) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*; return ``(replayed,
    remaining)``. ``replay(message_dict)`` runs per message in drop order; a spool file is deleted
    only after its replay succeeds. The first failure stops the drain (the DB is likely still
    unhealthy) and keeps the rest for retry, unless the DB rejected that row's own values
    (:func:`is_row_rejection`): it is quarantined and the drain goes on. With ``db_known_failing``
    (the caller's last write already failed and is being logged/escalated) a replay failure is
    expected and logs at DEBUG, so a stalled session does not add one WARNING per append on top of
    its ERROR (#114266).
    """
    try:
        candidates = list(_get_flush_dir().glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0
    entries = []
    for path in candidates:
        if path in _REPLAYED_UNREMOVABLE:
            continue
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
        entries.append((_spool_sort_key(payload, path.name), path, message))
    ordered, replayed, remaining = sorted(entries, key=lambda e: e[0]), 0, 0
    for idx, (_key, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            if is_row_rejection(exc):  # no retry can write it, so it must not hold the session back
                _quarantine_spool_file(path, session_id, exc)
                continue
            (logger.debug if db_known_failing else logger.warning)(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s", path, session_id, exc)
            remaining = len(ordered) - idx
            break
        _remove_replayed(path, session_id)
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
        result: dict[str, Any] = {"text": getattr(value, "text", "")}
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


def _sort_number(value: Any) -> float:
    """Coerce a spool ordering field to a float; unusable values sort first.

    Ordering fields are read back from JSON on disk and may be missing or
    corrupt.  A sort key that mixes ``str`` and ``int`` raises ``TypeError``
    and would abort the entire recovery pass, so anything non-numeric is
    normalised to ``0.0``.  So is a JSON integer too large for a float
    (``float(10**400)`` raises ``OverflowError``) and a NaN/Infinity literal
    (NaN compares false both ways and would scramble the sort).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        number = float(value)
    except OverflowError:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _spool_sort_key(payload: dict[str, Any], name: str) -> tuple:
    """Drop-order key ``(ts, seq, filename)`` for one spool payload. The live drain and the restart
    pass both sort on it, so they replay a session's files in the same order."""
    # seq is per process (_TRANSCRIPT_SPOOL_SEQ) and ts has one-second resolution, so files two
    # processes spool in the same second can interleave; within one process the order is exact.
    # Slot heads have no seq and overflow tails start at zero, so a missing seq sorts as -1.
    return _sort_number(payload.get("ts")), _sort_number(payload.get("seq", -1)), name


def is_row_rejection(exc: BaseException) -> bool:
    """True when the database refused this row's own values, so no retry can succeed: a value sqlite
    cannot bind (an unsupported type, or an int outside 64 bits), or a malformed field that fails
    before the write. Lock, I/O, corruption and routing failures stay retryable, and so do constraint
    failures, which can be session-wide. Only a binding message counts among sqlite errors:
    ``InterfaceError("no more rows available")`` is WAL contention that ``SessionDB`` retries and
    re-raises unchanged once its patience runs out."""
    if isinstance(exc, sqlite3.Error):
        return (isinstance(exc, (sqlite3.InterfaceError, sqlite3.ProgrammingError))
                and "binding parameter" in str(exc).lower())
    return isinstance(exc, (TypeError, ValueError, AttributeError, OverflowError))


def _remove_replayed(path: Path, session_id: str) -> None:
    """Delete a spool file whose row was just written. If it cannot be deleted, this process never
    replays it again, and replay goes on with the files after it, which are still older than any
    live row. Nothing on disk marks it, so a restart writes the row a second time."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _REPLAYED_UNREMOVABLE.add(path)
        logger.error("Spooled transcript row for %s was written, but %s could not be deleted (%s); "
                     "delete it by hand, or the next start writes that row again", session_id, path, exc)


def _quarantine_spool_file(path: Path, session_id: str, exc: BaseException) -> None:
    """Move a spool file whose row the database rejected out of every replay, so it cannot hold its
    session's later rows back on every write and every restart. It stays on disk for manual recovery."""
    target = path.with_name(path.name + QUARANTINE_SUFFIX)
    try:
        path.replace(target)
    except OSError as move_exc:
        logger.error("Spooled transcript row for %s in %s was rejected by the database (%s) and could "
                     "not be quarantined: %s", session_id, path, exc, move_exc)
        return
    logger.warning("Spooled transcript row for %s was rejected by the database (%s); moved it to %s "
                   "for manual recovery and continuing with the session's later rows",
                   session_id, exc, target)


def _order_flush_files(paths) -> list[Path]:
    """Order recovery payloads by :func:`_spool_sort_key`, as :func:`drain_transcript_spool` does.
    ``SessionDB`` restores a conversation by AUTOINCREMENT id and spool files have random names, so
    any other order permanently scrambles the recovered transcript.

    Keeps only the key and path, not decoded payloads; the caller re-reads each file to replay it.
    Unparseable files sort last by name.
    """
    entries = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        # OSError: unreadable; ValueError: bad JSON or bytes; RecursionError: nesting too deep.
        except (OSError, ValueError, RecursionError):
            payload = None
        if isinstance(payload, dict):
            entries.append(((0, *_spool_sort_key(payload, path.name)), path))
        else:
            entries.append(((1, 0.0, 0.0, path.name), path))
    entries.sort(key=lambda entry: entry[0])
    return [path for _key, path in entries]


def recover_pending_to_db(session_db=None, *, session_resolver=None) -> int:
    """Replay flush-dir ``*.json`` files into state.db; return the number of messages recovered.
    See :func:`recover_pending_spool`, which also reports the sessions it held back."""
    return recover_pending_spool(session_db, session_resolver=session_resolver)[0]


def recover_pending_spool(session_db=None, *, session_resolver=None) -> tuple[int, set[str]]:
    """Replay flush-dir ``*.json`` files via ``SessionDB.append_message``, deleting each on success.

    ``session_db=None`` opens (and afterwards releases) the shared default ``state.db``.
    ``session_resolver`` (optional ``(session_key, not_after=ts) -> (session_id, db) | None``, e.g.
    ``SessionStore.resolve_session_id_for_key``) is required for real flush files: adapter
    ``MessageEvent`` objects carry no ``session_id``, so without it every recovery lands in the skip
    branch. A returned ``db`` routes the append to the profile store owning the key (multiplexed
    gateways); ``None`` falls back to ``session_db``. Returns ``(recovered, held_back)``: the number of
    messages recovered, and the session ids whose spool files this pass held back after a failed
    replay, so the live writer can drain them before that session's next row.
    """
    flush_files = _order_flush_files(_get_flush_dir().glob("*.json"))
    if not flush_files:
        return 0, set()
    own_db = session_db is None
    if own_db:
        from hermes_state_registry import acquire
        session_db = acquire()
    recovered = 0
    # Sessions whose spool replay already failed this pass -> files held back for them. Ordering is a
    # per-session property, so one unhealthy session must not hold back the others.
    blocked_sessions: dict[str, int] = {}
    try:
        for path in flush_files:
            if path in _REPLAYED_UNREMOVABLE:
                continue
            # One unparseable payload or rejected append must only skip THIS file: the file is
            # never unlinked, so aborting the pass would re-poison every later boot.
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                # Agent-history snapshots use a different schema (reason +
                # messages list) and are meant for manual operator recovery,
                # not automatic DB insertion. Skip them silently.
                if payload.get("reason") == AGENT_HISTORY_REASON:
                    continue
                if _recover_one_payload(session_db, path, payload,
                                        session_resolver=session_resolver,
                                        blocked_sessions=blocked_sessions):
                    recovered += 1
                    _remove_replayed(path, payload.get("session_key", ""))
            # health: allow BLE001 -- per-file boundary moved unchanged from recover_pending_to_db; a traceback per locked-DB file would only add noise
            except Exception as exc:
                logger.warning("Failed to recover pending message from %s: %s", path, exc)
    finally:
        if own_db:  # shutdown cancellation/interrupt must not strand an owned DB
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close
                release_or_close(session_db)
    if recovered:
        logger.info("Recovered %d pending message(s) from shutdown flush", recovered)
    if blocked_sessions:
        # Otherwise a spool dir that never empties gives an operator no reason why.
        logger.info("Held back %d spooled transcript file(s) for the next start after a failed replay: %s",
                    sum(blocked_sessions.values()),
                    ", ".join(f"{sid} ({count})" for sid, count in blocked_sessions.items()))
    return recovered, set(blocked_sessions)


def _recover_one_payload(session_db, path: Path, payload: dict[str, Any], *,
                         session_resolver=None, blocked_sessions: dict[str, int]) -> bool:
    """Append one flush payload to ``session_db``; False when it was not replayed (the file is kept,
    or quarantined when the database rejected the row itself)."""
    # Cap-dropped transcript payloads carry the full message dict keyed by session_id: replay it
    # directly (#78182). This handles spool files that were never drained before a restart.
    if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
        from gateway.session_transcript import transcript_append_kwargs
        data = payload.get("data", {}) or {}
        spooled_sid, message = data.get("session_id", ""), data.get("message")
        if not spooled_sid or not isinstance(message, dict):
            logger.warning("Cannot recover structurally invalid transcript spool "
                           "file %s; preserved for manual inspection", path)
            return False
        if spooled_sid in blocked_sessions:
            # An older message for this session could not be replayed. Writing this one now would give
            # it a lower row id than the message it follows, permanently inverting the transcript, so
            # leave it for the next start.
            blocked_sessions[spooled_sid] += 1
            return False
        try:
            session_db.append_message(**transcript_append_kwargs(spooled_sid, message, fallback_ts=payload.get("ts")))
        except Exception as exc:
            if is_row_rejection(exc):  # no retry can write it, so it must not hold the session back
                _quarantine_spool_file(path, spooled_sid, exc)
                return False
            # Same contract as drain_transcript_spool: stop this session's replay on the first failure
            # and keep the remaining spool files for the next attempt.
            blocked_sessions[spooled_sid] = 1
            raise
        return True
    session_key, data = payload.get("session_key", ""), payload.get("data", {})
    text = data.get("text", "")
    if not text or not session_key:
        logger.warning("Cannot recover structurally invalid pending message from %s; "
                       "the flush file has been preserved", path)
        return False
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
        return False
    target_db.append_message(session_id=session_id, role="user", content=text,
                             timestamp=payload.get("ts", int(time.time())))
    return True


def recover_gateway_pending(runner) -> int:
    """Replay every ``pending_messages`` spool this gateway owns into state.db; return the count.

    ``_get_flush_dir`` follows the active HERMES_HOME, so a routed turn on a multiplexed gateway spools
    its stalled transcript backlog under ``profiles/<name>/`` and the runtime drain forgets it on
    restart. After the launch home, replay each served profile inside its own home so the default
    store ``recover_pending_spool`` opens is that profile's state.db (#123584). The sessions a pass
    holds back are marked on the session store, so their next live row drains the spool first.
    """
    from gateway.run import _multiplex_profile_homes
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    store = runner.session_store
    recovered, held_back = recover_pending_spool(session_resolver=store.resolve_session_id_for_key)
    store.mark_spooled_drop_sessions(held_back)
    if not getattr(runner.config, "multiplex_profiles", False):
        return recovered
    launch_home = Path(get_hermes_home()).resolve()
    for name, home in _multiplex_profile_homes(runner.config):
        if Path(home).resolve() == launch_home or not (Path(home) / "pending_messages").is_dir():
            continue
        token = set_hermes_home_override(str(home))
        try:
            count, held_back = recover_pending_spool(session_resolver=store.resolve_session_id_for_key)
        except Exception:  # one profile's unreadable spool must not strand the others'
            logger.warning("Pending-message recovery failed for profile %s", name, exc_info=True)
            continue
        finally:
            reset_hermes_home_override(token)
        recovered += count
        store.mark_spooled_drop_sessions(held_back)
    return recovered


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
            "reason": AGENT_HISTORY_REASON, "issue": "#72680",
            "session_id": session_id, "count": len(snapshot), "messages": snapshot,
        })
        logger.warning("Preserved %d in-memory message(s) for session %s "
                       "(possible FTS corruption — recover after repairing state.db)",
                       len(snapshot), session_id)
    except Exception as _e:
        logger.warning("Agent-history shutdown preservation failed for session %s: %s", session_id,
                       _e)
