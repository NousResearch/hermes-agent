"""SessionStore transcript I/O: SQLite append with a per-session retry queue, compression-reroute
following, FTS corruption recovery, rewrite/rewind/load. Mixin split out of ``gateway/session.py``;
bound onto ``SessionStore`` via the MRO."""

from __future__ import annotations

import contextlib
import logging
import threading
from agent.turn_context import extract_api_content_sidecar
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from gateway.session import SessionEntry

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


class TranscriptReadError(RuntimeError):
    """Raised when persisted history cannot be read safely."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"transcript read failed for session {session_id}")


def _plain_text(content) -> str:
    """Text of a message content (str or text-part list); "" for anything else."""
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(t for t in parts if t)
    return content if isinstance(content, str) else ""


def _spool_dropped(session_id: str, message: Dict[str, Any]):
    """Spool one evicted/undeliverable message to disk (same machinery as the shutdown flush, so it
    is replayed after DB recovery); path or None."""
    try:
        from gateway.shutdown_flush import spool_dropped_transcript_message
        return spool_dropped_transcript_message(session_id, message)
    except Exception:
        return None


# Message keys persisted only for assistant rows (None otherwise).
_ASSISTANT_ONLY_KEYS = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items",
)


class SessionTranscriptMixin:
    """SessionStore transcript I/O: SQLite append with a per-session retry queue,
    compression-reroute following, FTS corruption recovery, rewrite/rewind/load."""

    _MAX_PENDING_PER_SESSION = 200  # in-memory pending messages per session (DB broken)

    def _compression_tip_for_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Latest compression continuation for *session_id* (heals a mapping left pointing at a
        compressed parent by a restart or failed send)."""
        db = self._db_for_session_id(session_id) if session_id else None
        if db is None:
            return session_id
        try:
            return db.get_compression_tip(session_id) or session_id
        except Exception:
            logger.debug("Compression-tip lookup failed for session %s", session_id, exc_info=True)
            return session_id

    def _heal_compression_tip_locked(
        self, entry: "SessionEntry", original_session_id: Optional[str],
        canonical_session_id: Optional[str]) -> bool:
        """Rewrite *entry* to the compression continuation if stale. Lock held."""
        if not original_session_id or not canonical_session_id:
            return False
        if entry.session_id != original_session_id or canonical_session_id == original_session_id:
            return False
        logger.info(
            "SessionStore healed compressed session mapping: %s -> %s", entry.session_id,
            canonical_session_id)
        entry.session_id = canonical_session_id
        return True

    def advance_compression_session(
        self, session_key: str, expected_session_id: str, target_session_id: str,
    ) -> Optional[SessionEntry]:
        """CAS-advance one route along an already-verified compression lineage. Unlike
        ``switch_session`` this never ends/reopens SQLite rows (the compression transaction owns
        that). ``None`` means the route moved after the caller's snapshot (e.g. /new) — caller
        must fail closed."""
        if not session_key or not expected_session_id or not target_session_id:
            return None
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            if entry.session_id == target_session_id:
                return entry
            if not self._heal_compression_tip_locked(entry, expected_session_id, target_session_id):
                return None  # route moved (session_id != expected) or nothing to heal
            self._save()  # bookkeeping, not user activity: leave ``updated_at`` alone
            return entry

    def _get_transcript_drain_lock(self):
        """Return the lock that serializes pending-queue drain boundaries."""
        return self._lazy("_transcript_drain_lock", threading.RLock)

    def append_to_transcript(self, session_id: str, message: Dict[str, Any], skip_db: bool = False) -> None:
        """Serialize transcript draining across queue migration boundaries."""
        if not self._db_for_session_id(session_id) or skip_db:
            return
        with self._get_transcript_drain_lock():
            self._append_to_transcript_serialized(self._follow_reroutes(session_id), message)

    def _follow_reroutes(self, session_id: str) -> str:
        """Follow the compression reroute chain (cycle-guarded)."""
        reroutes = self._lazy("_transcript_reroutes", dict)
        seen = set()
        while session_id in reroutes and session_id not in seen:
            seen.add(session_id)
            session_id = reroutes[session_id]
        return session_id

    def _enqueue_transcript_message(self, session_id: str, message: Dict[str, Any]) -> list:
        """Queue *message* (retry lock held); evicts + spools the oldest past the cap."""
        pending = self._dirty_transcripts.setdefault(session_id, [])
        pending.append(dict(message))
        # Cap pending messages per session to avoid unbounded memory growth when the DB is persistently
        # broken. Spool the evicted oldest message to the on-disk pending spool (same machinery
        # flush_pending_to_file uses at shutdown) so a runtime cap rotation does not silently discard it
        # (#78182); it is replayed on the next successful transcript flush.
        if len(pending) > self._MAX_PENDING_PER_SESSION:
            spool_path = _spool_dropped(session_id, pending.pop(0))
            if spool_path is not None:
                self._lazy("_spooled_drop_sessions", set).add(session_id)
                logger.warning(
                    "Session DB transcript pending queue full for %s (cap=%d); spooled oldest "
                    "message to %s for replay after DB recovery", session_id,
                    self._MAX_PENDING_PER_SESSION, spool_path)
            else:
                logger.warning(
                    "Session DB transcript pending queue full for %s (cap=%d); dropping oldest "
                    "message to make room (on-disk spool unavailable)", session_id,
                    self._MAX_PENDING_PER_SESSION)
        return pending

    def _divert_transcript_after_db_replaced(
        self, session_id: str, queue_session_id: str, exc: Exception
    ) -> None:
        """Stop SQLite writes on a replaced/quarantined handle and divert the backlog to the on-disk
        spool + JSONL fallback: retrying cannot succeed and the FTS rebuild must never run here."""
        logger.error(
            "Session DB refused further writes on this handle for %s (%s); stopping SQLite writes "
            "and diverting pending transcripts to the on-disk fallback: %s", session_id,
            type(exc).__name__, exc)
        with self._transcript_retry_lock:
            remaining = list(self._dirty_transcripts.get(queue_session_id, []))
            self._dirty_transcripts.pop(queue_session_id, None)
            self._transcript_append_failures.pop(session_id, None)
        for dropped in remaining:
            if _spool_dropped(session_id, dropped) is None:
                logger.warning(
                    "pending fallback failed for replaced state.db transcript on %s", session_id,
                    exc_info=True)
        try:
            from hermes_state import divert_session_transcript_jsonl
            divert_session_transcript_jsonl(session_id, remaining)
        except Exception:
            logger.warning(
                "JSONL divert failed for replaced state.db transcript on %s", session_id,
                exc_info=True)

    def _live_compression_child(self, session_id: str) -> str:
        """Transitive compression tip of *session_id* if it is a different, still-live row, else ""
        (a depth-1 lookup misses multi-hop lineages). Uses the PARENT's proven owner handle: the
        child's id is unpublished until its write succeeds, so a by-id lookup would hit the ambient
        store."""
        owner_db = self._db_for_session_id(session_id)
        tip = owner_db.get_compression_tip(session_id) if owner_db is not None else None
        if tip and tip != session_id:
            tip_row = owner_db.get_session(tip)
            if tip_row is not None and tip_row.get("ended_at") is None:
                return str(tip)
        return ""

    def _migrate_transcript_queue_to_child(
        self, session_id: str, queue_session_id: str, child_id: str, pending: list, msg
    ) -> list:
        """Move the retry queue + failure counter from parent to child and record the reroute (retry
        lock held); returns the child's pending list. Older parent backlog must precede messages
        already queued directly on the child; routing is published only AFTER the queue moved
        (caller), so new child writes cannot bypass older parent backlog."""
        if pending and pending[0] is msg:
            pending.pop(0)
        existing_child_pending = self._dirty_transcripts.get(child_id, [])
        if pending:
            pending.extend(existing_child_pending)
            self._dirty_transcripts[child_id] = pending
        elif existing_child_pending:
            pending = existing_child_pending
        self._dirty_transcripts.pop(queue_session_id, None)
        previous_failures = self._transcript_append_failures.pop(queue_session_id, 0)
        if previous_failures:
            self._transcript_append_failures[child_id] = max(
                previous_failures, self._transcript_append_failures.get(child_id, 0))
        self._transcript_reroutes[session_id] = child_id
        return pending

    def _publish_transcript_reroute(self, session_id: str, child_id: str) -> None:
        """Repoint every route at the compression child and save (index authoritative again)."""
        with self._lock:
            for entry in self._entries.values():
                if entry.session_id == session_id:
                    entry.session_id = child_id
            self._save()
        (getattr(self, "_session_owner_hints", None) or {}).pop(child_id, None)

    def _append_to_transcript_serialized(self, session_id: str, message: Dict[str, Any]) -> None:
        """Append a message to a session's transcript (SQLite), draining the per-session retry
        queue.

        Args: skip_db: When True, skip the SQLite write. Used when the agent already persisted messages to
        SQLite via its own _flush_messages_to_session_db(), preventing the duplicate-write bug (#860).
        """
        with self._transcript_retry_lock:
            pending = self._enqueue_transcript_message(session_id, message)
            msg = pending[0]
        queue_session_id = session_id

        def _ack_head() -> bool:
            """Pop the acknowledged head (retry lock held). True if queue drained."""
            if pending and pending[0] is msg:
                pending.pop(0)
            if not pending:
                self._dirty_transcripts.pop(queue_session_id, None)
                self._transcript_append_failures.pop(session_id, None)
                return True
            return False

        # DB write outside the retry lock so other sessions can append.
        while True:
            try:
                self._append_transcript_message(session_id, msg)
            except Exception as exc:
                from hermes_state import StateDbCorruptError, StateDbReplacedError
                from hermes_state_errors import CompressionSessionClosedError
                if isinstance(exc, (StateDbReplacedError, StateDbCorruptError)):
                    self._divert_transcript_after_db_replaced(session_id, queue_session_id, exc)
                    return
                if isinstance(exc, CompressionSessionClosedError):
                    # Adopt only a different, still-live compression tip, else fail closed.
                    _owner_key = self._owner_key_for_session_id(session_id)
                    child_id = self._live_compression_child(session_id)
                    if child_id:
                        # Record the child's owner BEFORE writing to it (the reroute is published
                        # only after the write succeeds — load-bearing for backlog order).
                        if _owner_key:
                            self._lazy("_session_owner_hints", dict)[child_id] = _owner_key
                        try:
                            self._append_transcript_message(child_id, msg)
                        except Exception as reroute_exc:
                            exc = reroute_exc
                        else:
                            with self._transcript_retry_lock:
                                pending = self._migrate_transcript_queue_to_child(
                                    session_id, queue_session_id, child_id, pending, msg)
                                queue_session_id = child_id
                            self._publish_transcript_reroute(session_id, child_id)
                            if not pending:
                                return
                            msg = pending[0]
                            session_id = child_id
                            continue
                    else:
                        # Permanent routing invariant failure, not a transient outage: drop it so it
                        # cannot poison later writes.
                        with self._transcript_retry_lock:
                            _ack_head()
                        logger.error(
                            "Session DB transcript append rejected for compression-ended %s with "
                            "no unique live child; not retrying", session_id)
                        return
                if self._is_fts_corruption_error(exc) and self._rebuild_fts_once():
                    try:
                        self._append_transcript_message(session_id, msg)
                    except Exception as retry_exc:
                        exc = retry_exc
                    else:
                        with self._transcript_retry_lock:
                            _ack_head()
                        continue
                with self._transcript_retry_lock:
                    failures = self._transcript_append_failures.get(session_id, 0) + 1
                    self._transcript_append_failures[session_id] = failures
                logger.warning(
                    "Session DB transcript append failed for %s (failure_count=%d, pending=%d); "
                    "will retry: %s", session_id, failures, len(pending), exc)
                return
            else:
                with self._transcript_retry_lock:
                    queue_empty = _ack_head()
                    if not queue_empty:
                        msg = pending[0]
                if queue_empty:
                    # Backlog clear: replay cap-dropped messages spooled to disk.
                    # See #78182.
                    self._drain_spooled_drops(session_id)
                    return
                continue

    def _drain_spooled_drops(self, session_id: str) -> None:
        """Replay cap-dropped spooled transcript messages after DB recovery. Best-effort: replay
        failures keep the spool files for the next successful flush; nothing here may raise."""
        spooled_sessions = getattr(self, "_spooled_drop_sessions", None)
        if not spooled_sessions or session_id not in spooled_sessions:
            return
        try:
            from gateway.shutdown_flush import drain_transcript_spool
            _replayed, remaining = drain_transcript_spool(
                session_id, lambda message: self._append_transcript_message(session_id, message),
            )
            if not remaining:
                spooled_sessions.discard(session_id)
        except Exception as exc:
            logger.warning("Failed to drain transcript spool for %s: %s", session_id, exc)

    def _append_transcript_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Write one transcript row. Caller handles retry queuing."""
        _db = self._db_for_session_id(session_id)
        if _db is None:
            # Named profile with no resolvable home yet: defer (caller queues) instead of writing
            # into the ambient store.
            raise RuntimeError(
                f"no owning session store for {session_id}; deferring transcript write")
        is_assistant = message.get("role") == "assistant"
        _db.append_message(
            session_id=session_id,
            role=message.get("role", "unknown"),
            content=message.get("content"),
            tool_name=message.get("tool_name"),
            tool_calls=message.get("tool_calls"),
            tool_call_id=message.get("tool_call_id"),
            **{k: message.get(k) if is_assistant else None for k in _ASSISTANT_ONLY_KEYS},
            platform_message_id=(message.get("platform_message_id") or message.get("message_id")),
            observed=bool(message.get("observed")),
            timestamp=message.get("timestamp"),
            # Exact bytes sent to the API (prompt-cache-stable replay); must survive every
            # persistence path or the next replay diverges.
            api_content=extract_api_content_sidecar(message),
            # Presentation typing (e.g. "internal_notification"); DB-only.
            # "internal_notification" for self-injected async-delegation/background notification turns,
            # #82888). DB-only; stripped from provider-bound payloads.
            display_kind=message.get("display_kind"),
            display_metadata=message.get("display_metadata"),
        )

    @staticmethod
    def _is_fts_corruption_error(exc: Exception) -> bool:
        """True only when the failure is provably scoped to the FTS index. A bare SQLITE_CORRUPT
        can mean structural B-tree damage; only errors naming ``messages_fts`` or carrying FTS
        provenance (``SessionDB._is_fts_write_corruption_error``) may authorize the one-shot
        rebuild-and-retry; everything else takes the retry path.

        A generic ``database disk image is malformed`` (bare SQLITE_CORRUPT) can mean structural damage to
        canonical B-trees, not just the FTS shadow tables — treating it as FTS-only here made the store
        rebuild the index and retry transcript writes against a structurally corrupt database (#97940).
        """
        if "messages_fts" in str(exc).lower():
            return True
        import sqlite3
        from hermes_state import SessionDB
        return isinstance(exc, sqlite3.DatabaseError) and SessionDB._is_fts_write_corruption_error(exc)

    def _rebuild_fts_once(self) -> bool:
        """Attempt FTS5 ``rebuild`` once per store lifetime; True if any index was rebuilt."""
        if self._fts_rebuild_attempted:
            return False
        self._fts_rebuild_attempted = True
        db = self._db
        if db is None or not hasattr(db, "rebuild_fts"):
            return False
        # WAL split-brain guard: skip when a foreign process holds state.db.
        foreign_holders = None
        if hasattr(db, "_foreign_state_db_holders"):
            foreign_holders = db._foreign_state_db_holders()
        if foreign_holders:
            logger.warning(
                "Skipping Session DB FTS rebuild while foreign processes hold the database or "
                "WAL sidecars (%s); canonical transcript writes remain available.", foreign_holders)
            return False
        try:
            rebuilt = db.rebuild_fts()
        except Exception as exc:
            logger.warning("Session DB FTS rebuild failed: %s", exc)
            return False
        if rebuilt:
            logger.warning("Rebuilt %d Session DB FTS index(es) after append corruption", rebuilt)
        return rebuilt > 0

    def _clear_dirty_transcript(self, session_id: str) -> None:
        """Drop queued pending messages so a rewrite/rewind doesn't re-insert them."""
        with self._transcript_retry_lock:
            self._dirty_transcripts.pop(session_id, None)
            self._transcript_append_failures.pop(session_id, None)

    def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        """Whether a message with this platform_message_id is persisted (False without a DB).

        Thin wrapper over SessionDB.has_platform_message_id(). Returns False when no DB is available
        (in-memory sessions). Used by the gateway's transient-failure dedupe guard (#47237).
        """
        db = self._db_for_session_id(session_id)
        if not db:
            return False
        try:
            return db.has_platform_message_id(session_id, platform_message_id)
        except Exception:
            logger.debug("has_platform_message_id lookup failed", exc_info=True)
            return False

    def rewrite_transcript(
        self, session_id: str, messages: List[Dict[str, Any]], active_only: bool = False,
        reject_active_turn_lease: bool = False) -> bool:
        """Replace a session's transcript (/retry, /compress). DESTRUCTIVE by default:
        ``active_only=False`` DELETEs every row incl. soft-archived compaction history (pass
        ``active_only=True`` for sessions that may carry archived rows). True when the write lands
        or there is no DB, False on failure — callers committing a destructive change on top
        (/compress repointing) must check it. ``reject_active_turn_lease`` is for user-initiated
        rewrites that do not own the cross-process turn lease."""
        db = self._db_for_session_id(session_id)
        if not db:
            return True
        with self._get_transcript_drain_lock():
            try:
                # Even when the current agent doesn't "own" persistence, the session on disk may already
                # carry compaction-archived rows — e.g. after a model switch or a /restore, both of which
                # mint a fresh agent with _session_db_created=False (so the check above is False) yet leave
                # the durable archived transcript in place. A full-history replace would DELETE those
                # archived rows just like the owned-agent case. Guard against it by replacing ONLY the live
                # (active=1) set unconditionally: on a fresh create/fork every row is active=1, so
                # active-only replace is behaviorally identical to the full replace — and when archived rows
                # DO exist they survive. An existence probe here (has_archived_messages) would fail OPEN
                # into the destructive replace on any DB error and can race a concurrent archive_and_compact
                # — the same probe failure mode #80216's /retry fix (gateway/slash_commands.py) deliberately
                # avoids.
                db.replace_messages(
                    session_id, messages, active_only=active_only,
                    reject_active_turn_lease=reject_active_turn_lease)
            except Exception as e:
                logger.debug("Failed to rewrite transcript in DB: %s", e)
                return False
            self._clear_dirty_transcript(session_id)
            return True

    def load_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages from a session's transcript (state.db is canonical). Reads follow the
        same routing writes use — the in-memory reroute map, then the durable compression tip —
        otherwise the transcript "vanishes" while every message sits under the child."""
        if not self._db_for_session_id(session_id):
            return []
        session_id = self._follow_reroutes(session_id)
        with contextlib.suppress(Exception):
            # Durable successor survives restart; the reroute map doesn't.
            db = self._db_for_session_id(session_id)
            session_id = db.get_compression_tip(session_id) or session_id
        try:
            # repair_alternation: this feeds LIVE REPLAY; heal a durable user;user wedge once here.
            return self._db_for_session_id(session_id).get_messages_as_conversation(
                session_id, repair_alternation=True)
        except Exception as e:
            # Empty history is valid data; a failed canonical read is not — live-replay callers
            # must fail closed, not start from [].
            logger.error(
                "Transcript read failed for session %s; refusing to treat the conversation as "
                "empty: %s", session_id, e, exc_info=True)
            raise TranscriptReadError(session_id) from e

    def rewind_session(
        self, session_id: str, n: int = 1, *, require_retryable_composite: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Back up ``n`` user turns via soft-delete (``active=0``), mirroring CLI ``/undo [N]``.
        Returns ``{"rewound_count", "turns_undone", "target_text"}`` or ``None`` (no DB / no user
        turn); ``n`` clamps to the oldest user turn. ``require_retryable_composite`` is the gateway
        ``/retry`` guard: the selected turn must be a composite carrier whose live payload is
        losslessly replayable as text before anything changes."""
        db = self._db_for_session_id(session_id)
        if not db:
            return None
        with self._get_transcript_drain_lock():
            n = max(n, 1)
            from agent.context_compressor import (
                retryable_user_text, split_user_originated_turn, user_originated_turn_view,
            )
            try:
                expected_active_ids = db.get_active_message_ids(session_id)
                durable = db.get_messages_as_conversation(session_id, include_row_ids=True)
                user_indices = [
                    index for index, message in enumerate(durable)
                    if user_originated_turn_view(message) is not None
                ]
                if not user_indices:
                    return None
                turns_undone = min(n, len(user_indices))
                target = durable[user_indices[-turns_undone]]
                target_id = target.get("_row_id")
                if not isinstance(target_id, int):
                    return None
                handoff, target_view = split_user_originated_turn(target)
                if target_view is None:
                    return None
                if require_retryable_composite and handoff is None:
                    return None
            except Exception as e:
                logger.debug("rewind_session: failed to resolve canonical target: %s", e)
                return None
            if require_retryable_composite:
                # Keep replay-policy failures distinct from persistence errors so /retry can explain
                # why the selected carrier is unsafe.
                target_text = retryable_user_text(target_view.get("content"))
            try:
                result = db.rewind_to_message(
                    session_id, target_id, preserve_compaction_handoff=handoff is not None,
                    expected_active_ids=expected_active_ids,
                    expected_target_content=target_view.get("content"))
            except Exception as e:
                prefix = "" if isinstance(e, ValueError) else "rewind_to_message failed: "
                logger.debug("rewind_session: %s%s", prefix, e)
                return None
            self._clear_dirty_transcript(session_id)
            # ``target_view`` is the live projection; a composite carrier's raw row holds the
            # summary wrapper and must not be echoed as prompt.
            if not require_retryable_composite:
                target_text = _plain_text(target_view.get("content") or "")
            return {
                "rewound_count": result.get("rewound_count", 0),
                "turns_undone": turns_undone,
                "target_text": target_text,
            }
