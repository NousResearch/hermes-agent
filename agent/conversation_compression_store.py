"""Compression's reads and writes of the session store outside the compaction commit, for
``agent.conversation_compression``: the durable cooldown snapshot and its rollback, the rotated-parent and
deliberate-end checks, the orphaned-parent reopen, the active watermark and transcript reads, each through the store's
interface handle when it has one. ``conversation_compression`` imports every name it calls.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Optional

# Log-record parity with the origin module.
logger = logging.getLogger("agent.conversation_compression")


def _compressor_interface(compressor: Any) -> Any:
    """The writer handle bound beside the compressor's store (read off the instance, never a class default)."""
    from agent.agent_init_session import writer_interface
    values = vars(compressor)
    return writer_interface(values.get("_persistence"), values.get("_session_db"))


def _rollback_durable_cooldown(
    compressor: Any, snapshot: dict[str, Any], authoritative: Optional[bool], durable_state: Optional[dict[str, Any]]
) -> None:
    """Recreate/clear the durable cooldown row from the attempt snapshot.
    Authoritative captures use the exact raw-row restore API (verifies read-back, propagates failure); the
    legacy path re-derives deadline/error best-effort."""
    session_db = vars(compressor).get("_session_db")
    session_id = vars(compressor).get("_session_id")
    if session_db is None or not session_id:
        return
    store = _compressor_interface(compressor)
    if authoritative is True and store is not None:
        if durable_state is None:
            raise RuntimeError("exact compression cooldown rollback API is unavailable")
        from persistence.conversation.messages import CooldownSnapshot
        store.messages.restore_compression_cooldown(session_id, CooldownSnapshot(
            durable_state["session_exists"], durable_state["cooldown_until"], durable_state["error"]))
        return
    if authoritative is True:
        restorer = getattr(type(session_db), "restore_compression_failure_cooldown_row", None)
        if not callable(restorer) or durable_state is None:
            raise RuntimeError("exact compression cooldown rollback API is unavailable")
        restorer(session_db, session_id, copy.deepcopy(durable_state))
        return
    from agent.conversation_compression import _swallow
    with _swallow('compression cooldown persistence rollback failed', exc_info=True):
        deadline = float(snapshot["_summary_failure_cooldown_until"] or 0.0)
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0 and store is not None:
            store.messages.record_compression_cooldown(
                session_id, time.time() + remaining, snapshot.get("_last_summary_error"))
        elif remaining > 0:
            recorder = getattr(type(session_db), "record_compression_failure_cooldown", None)
            if callable(recorder):
                recorder(session_db, session_id, time.time() + remaining, snapshot.get("_last_summary_error"))
        elif store is not None:
            store.messages.clear_compression_cooldown(session_id)
        else:
            clearer = getattr(type(session_db), "clear_compression_failure_cooldown", None)
            if callable(clearer):
                clearer(session_db, session_id)


def _capture_authoritative_cooldown_under_lease(
    compressor: Any, attempt_snapshot: dict[str, Any]
) -> tuple[Optional[bool], Optional[dict[str, Any]]]:
    """Refresh and snapshot built-in durable cooldown state under the lease.
    Third-party compressors are not invoked: plugin code must not run under the lease. Returns ``False`` on
    durable read failure (rollback must not mistake unknown state for an empty row) and ``None`` when the
    legacy API is absent."""
    try:
        from agent.context_compressor import ContextCompressor
        if not isinstance(compressor, ContextCompressor):
            return None, None
        values = vars(compressor)
        session_db = values.get("_session_db")
        session_id = values.get("_session_id")
        if session_db is None or not session_id:
            # Unbound compressors have no durable row to mutate or restore.
            return None, None
        # Read the raw persisted row: the active getter filters expired rows and is not a lossless rollback snapshot.
        store = _compressor_interface(compressor)
        if store is not None:
            raw = store.messages.read_compression_cooldown_snapshot(session_id)
            durable_state = {"session_exists": raw.present, "cooldown_until": raw.until, "error": raw.error}
        else:
            raw_reader = getattr(type(session_db), "get_compression_failure_cooldown_row", None)
            if not callable(raw_reader):
                return False, None
            durable_state = raw_reader(session_db, session_id)
        if not isinstance(durable_state, dict):
            raise TypeError("raw compression cooldown snapshot must be a mapping")
        ContextCompressor.get_active_compression_failure_cooldown(compressor, refresh=True)
    except Exception as exc:
        logger.debug("authoritative compression cooldown capture failed: %s", exc)
        return False, None
    authoritative = getattr(compressor, "_last_cooldown_refresh_was_authoritative", None)
    if authoritative is not True:
        return authoritative, None
    values = vars(compressor)
    from agent.conversation_compression import _COMPRESSOR_COOLDOWN_STATE_FIELDS
    for name in _COMPRESSOR_COOLDOWN_STATE_FIELDS:
        if name in values:
            attempt_snapshot[name] = copy.deepcopy(values[name])
    return True, copy.deepcopy(durable_state)


def _session_was_rotated_by_compression(session_db: Any, session_id: str, store: Any = None) -> bool:
    """Return whether another path already rotated this compression parent (read through ``store``, the writer
    handle beside ``session_db``, when given)."""
    if store is not None:
        record = store.sessions.get(session_id)
        return bool(record and record.ended_at is not None and record.end_reason == "compression")
    getter = getattr(type(session_db), "get_session", None)
    if not callable(getter):
        return False
    session = getter(session_db, session_id)
    return bool(session and session.get("ended_at") is not None and session.get("end_reason") == "compression")


def _active_watermark(store: Any, session_db: Any, session_id: str) -> int:
    """MAX(id) of the session's active rows (0 when none), through ``store`` when given, else ``session_db``."""
    if store is not None:
        return int(store.messages.active_watermark(session_id).last_active or 0)
    return session_db.get_active_message_watermark(session_id)


def _read_conversation(store: Any, session_id: str, *, include_row_ids: bool = False) -> list[dict[str, Any]]:
    """The session's active conversation as message dicts, read through the writer handle ``store``."""
    from persistence.conversation.messages import ConversationRequest
    return [message.as_openai_message() for message in store.messages.read_conversation(
        session_id, ConversationRequest(include_row_ids=include_row_ids))]


def _reopen_orphaned_parent(session_db: Any, session_id: str, store: Any = None) -> None:
    """Reopen a compression-ended parent that has no continuation and no lease holder."""
    orphan_reopener = getattr(type(session_db), "reopen_orphaned_compression_session", None)
    if store is None and not callable(orphan_reopener):
        return
    try:
        if (store.lineage.reopen_orphaned_compression(session_id) if store is not None
                else orphan_reopener(session_db, session_id)):
            logger.warning("compression recovery: reopened orphaned session=%s with no continuation", session_id)
    except Exception as exc:
        logger.warning("orphaned compression session reopen failed for %s: %s", session_id, exc)


def _parent_deliberately_ended(session_db: Any, session_id: str, store: Any = None) -> bool:
    """True when the parent row was ended by a non-automatic reason. Fails OPEN: an
    unreadable row must not turn a cheap guard into a new way to lose compression."""
    reader = getattr(session_db, "get_session", None)
    if store is None and not callable(reader):
        return False
    try:
        from persistence.conversation.sessions import is_automatic_end_reason
        if store is not None:
            record = store.sessions.get(session_id)
            ended_at, end_reason = (None, None) if record is None else (record.ended_at, record.end_reason)
        else:
            row = reader(session_id) or {}
            ended_at, end_reason = row.get("ended_at"), row.get("end_reason")
        return ended_at is not None and not is_automatic_end_reason(end_reason)
    except Exception:  # health: allow BLE001 -- the guard fails open: an unreadable row never blocks compression
        return False
