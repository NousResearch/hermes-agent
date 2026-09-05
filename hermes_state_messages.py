"""Transcript persistence for SessionDB: message append/replace/rewind, reactions, resume assembly,
replayed-user dedupe. Mixin bound via the MRO, built on SessionDB's _read_ctx/_execute_write/_read_* primitives."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY, split_user_originated_turn
from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from hermes_state_common import (
    _COMPRESSION_LOCK_ROW_SQL, _ENDED_ROW_SQL, _RESET_END_REASONS, _RESET_END_REASONS_SQL, _ended_by_compression,
    _legacy_reset_child_sql, _placeholders, _sql_json_extract)

logger = logging.getLogger("hermes_state")  # caplog tests pin the origin module's name

# One INSERT shape for every message writer (append, batch, replace, compact, import).
_INSERT_MESSAGE_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
_BUMP_GENERATION_SQL = """
            INSERT INTO conversation_generations (source, session_key, generation)
            VALUES (?, ?, 1)
            ON CONFLICT(source, session_key) DO UPDATE
                SET generation = conversation_generations.generation + 1
            """

_TURN_LEASE_ROW_SQL = "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?"
_DELETE_COMPRESSION_LOCK_SQL = "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?"
_DISPLAY_ACTIVE_CLAUSE = " AND (active = 1 OR compacted = 1)"
_DISPLAY_META_ROW_SQL = "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?"
_ACTIVE_IDS_SQL = "SELECT id FROM messages WHERE session_id = ? AND active = 1 ORDER BY id"
_SET_COUNTERS_SQL = "UPDATE sessions SET message_count = ?, tool_call_count = ?"
_RESET_COUNTERS_SQL = "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?"
_SET_DISPLAY_META_SQL = "UPDATE messages SET display_metadata = ? WHERE id = ?"
_ARCHIVE_ACTIVE_SQL = "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ? AND active = 1"
_INVALID = object()  # _json_or sentinel where the fallback must be distinguishable from JSON null


def _json_or(raw: Any, fallback: Any, warning: str) -> Any:
    """``json.loads(raw)``; on failure log *warning* and return *fallback*."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(warning)
        return fallback


def _coerce_timestamp(value: Any, default: float) -> float:
    """Explicit message timestamp (datetime or number) or *default* when invalid."""
    if value is None:
        return default
    try:
        return float(value.timestamp()) if hasattr(value, "timestamp") else float(value)
    except (TypeError, ValueError):
        logger.debug("Ignoring invalid explicit message timestamp: %r", value)
        return default


def _parse_tool_calls(tool_calls: Any) -> Any:
    """tool_calls is a list (live agent) or JSON string (import/export); parse so json.dumps never double-encodes."""
    if not isinstance(tool_calls, str):
        return tool_calls
    try:
        return json.loads(tool_calls)
    except (json.JSONDecodeError, TypeError):
        return []


def _tool_calls_count(tool_calls: Any) -> int:
    return 0 if tool_calls is None else (len(tool_calls) if isinstance(tool_calls, list) else 1)


def _tool_calls_len(raw: Any, scalar: int = 0) -> int:
    """Count of a stored ``tool_calls`` column: list length, *scalar* for a truthy non-list, else 0."""
    parsed = _parse_tool_calls(raw)
    return len(parsed) if isinstance(parsed, list) else (scalar if parsed else 0)


def _scrub_surrogates(value: Any) -> Any:
    """Lone surrogates make sqlite3 raise UnicodeEncodeError and abort the whole write."""
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def _stale_holder(row, now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process."""
    from hermes_state import _compression_lock_holder_process_is_dead
    return float(row["expires_at"]) <= now or _compression_lock_holder_process_is_dead(row["holder"])


class SessionMessagesMixin:
    """Message append/replace/rewind, reactions, resume conversations, replay dedupe."""

    def _bump_conversation_generation(self, conn, session_id: str, end_reason: str) -> None:
        """Advance the peer's conversation generation past a boundary, in the txn that writes it. Only
        ``_RESET_END_REASONS`` count (compression continues one conversation). Never derived from session
        rows (deletes/prunes could re-emit a retired affinity identity); it only ever increments."""
        if end_reason not in _RESET_END_REASONS:
            return
        row = conn.execute("SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return
        source, session_key = (str(row[k] or "").strip() for k in ("source", "session_key"))
        if source and session_key:
            conn.execute(_BUMP_GENERATION_SQL, (source, session_key))

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize list/dict content (multimodal parts) as a sentinel-prefixed JSON string (sqlite3 binds
        only scalars). Lone UTF-16 surrogates (unsanitized web-scraped tool results) are scrubbed here: left
        raw, sqlite3 raises UnicodeEncodeError and the session silently stops persisting. Pairs with
        :meth:`_decode_content`."""
        if isinstance(content, str):
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)  # ensure_ascii escapes surrogates: bindable
        except (TypeError, ValueError):
            return _sanitize_surrogates(str(content))

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            return _json_or(content[len(cls._CONTENT_JSON_PREFIX):], content,
                "Failed to decode JSON-encoded message content; returning raw string")
        return content

    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column; an already-serialized JSON string
        (import/replace paths) is not double-encoded."""
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            display_metadata = _json_or(display_metadata, _INVALID, "Ignoring non-JSON display metadata on write")
            if display_metadata is _INVALID:
                return None
            if not isinstance(display_metadata, dict):
                logger.warning("Ignoring non-object display metadata on write")
                return None
        elif not isinstance(display_metadata, dict):
            logger.warning("Ignoring unexpected display metadata type on write: %s", type(display_metadata).__name__)
            return None
        return json.dumps(display_metadata)

    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column to a dict (never raw TEXT: the desktop does ``'task_count'
        in meta``). Pre-guard rows are double-encoded, so a second string layer is unwrapped."""
        if raw is None:
            return None
        meta = raw
        for _ in range(2):  # pre-guard rows carry a second string layer
            if isinstance(meta, str):
                meta = _json_or(meta, _INVALID, "Ignoring invalid display metadata on message row")
        if meta is _INVALID:
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            return None
        return meta

    @staticmethod
    def _reasoning_json_text(value: Any) -> Optional[str]:
        """Serialize a structured reasoning field for its TEXT column. Strings are stored as-is: round-trips
        (get_messages -> replace_messages) hand back raw TEXT; re-dumping would double-encode it and
        reasoning-replay consumers (``isinstance(..., list)``) would drop it."""
        return None if not value else (value if isinstance(value, str) else json.dumps(value))

    def _check_transcript_write_guards(self, conn, session_id: str, compression_lock_holder: Optional[str],
        turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False, reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False) -> None:
        """Transcript-write admission checks, run INSIDE the write txn by every writer. Ordinary appends do
        NOT check compression_locks: the lock only stops two COMPRESSIONS colliding and archive_and_compact()
        commits against a watermark, so concurrent appends are safe (blocking them killed turns during slow
        summaries). Destructive user mutations opt in via ``reject_active_*`` so a compressor that captured
        its watermark cannot resurrect the removed turn.

        Shared by :meth:`append_message` and :meth:`append_messages_batch` so the two writers can never
        diverge on these correctness invariants (this guard has already needed targeted fixes — see the
        #74478 patience note below). User-initiated transcript mutations may opt in to rejecting an active
        unowned turn lease in that same transaction.
        """
        from hermes_state import SessionCompressionInProgressError
        from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError
        # NOTE (#75316 redesign): appends do NOT check compression_locks. The lock's job is to stop two
        # COMPRESSIONS colliding, not to fence ordinary transcript writes. Concurrent appends during a
        # compression are safe by construction: archive_and_compact() commits against a watermark captured
        # at compression start and clones every row that arrived after it back into the live transcript, in
        # the same write transaction. Blocking appends here was the root cause of a whole symptom family —
        # turns dying as session_persistence_failed while a slow provider summary held the lease (#74568,
        # #77386), including stale locks from dead PIDs blocking writes for the full TTL. Keep that narrow
        # fence opt-in so ordinary appends retain the watermark behavior.
        if reject_active_compression_lock:
            active_lock = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
            if active_lock is not None:
                if _stale_holder(active_lock, time.time()):
                    conn.execute(_DELETE_COMPRESSION_LOCK_SQL, (session_id, active_lock["holder"]))
                elif active_lock["holder"] != compression_lock_holder:
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer")
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            lease = conn.execute(_TURN_LEASE_ROW_SQL, (conversation_id,)).fetchone()
            now = time.time()
            if turn_lease_holder:
                if lease is None or lease["holder"] != turn_lease_holder:
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write for {session_id!r}")
                if float(lease["expires_at"]) <= now:
                    # Expiry makes the row reclaimable, not taken over; BEGIN IMMEDIATE serializes this
                    # renewal with acquisition, so a still-matching owner recovers from a starved refresher.
                    conn.execute("UPDATE session_turn_leases SET expires_at = ? "
                        "WHERE conversation_id = ? AND holder = ?",
                        (now + max(0.1, float(turn_lease_ttl_seconds)), conversation_id, turn_lease_holder))
            elif lease is not None:
                if not _stale_holder(lease, now):
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript mutation for {session_id!r}")
                # Same reclaim rule as acquisition; deleting also fences a stale late flush after the mutation.
                conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                    (conversation_id, lease["holder"]))
        if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()) and not allow_closed_compression_parent:
            raise CompressionSessionClosedError(session_id)

    def _message_row_params(self, session_id: str, role: str, msg: Dict[str, Any], tool_calls: Any,
        message_timestamp: float, *, keep_reasoning: bool) -> tuple:
        """Bind values for ``_INSERT_MESSAGE_SQL`` from one message dict (*tool_calls* already parsed;
        *keep_reasoning* False NULLs every reasoning column). ``platform_message_id`` falls back to
        ``message_id`` (yuanbao's message-dict convention)."""
        _str_or_none = lambda v: _scrub_surrogates(v) if isinstance(v, str) else None  # noqa: E731
        _reasoning = lambda key: msg.get(key) if keep_reasoning else None  # noqa: E731
        return (session_id, role, self._encode_content(msg.get("content")), msg.get("tool_call_id"),
            json.dumps(tool_calls) if tool_calls else None, _scrub_surrogates(msg.get("tool_name")),
            msg.get("effect_disposition"), message_timestamp, msg.get("token_count"), msg.get("finish_reason"),
            _scrub_surrogates(_reasoning("reasoning")), _scrub_surrogates(_reasoning("reasoning_content")),
            *(self._reasoning_json_text(_reasoning(k))
              for k in ("reasoning_details", "codex_reasoning_items", "codex_message_items")),
            msg.get("platform_message_id") or msg.get("message_id"),
            1 if msg.get("observed") else 0, 1 if msg.get("_compressed_summary") else 0, 1,
            _str_or_none(msg.get("api_content")), _str_or_none(msg.get("display_kind")),
            self._encode_display_metadata(msg.get("display_metadata")))

    @staticmethod
    def _bump_session_counters(conn, session_id: str, inserted: int, tool_calls: int, *, unit: bool) -> None:
        """Bump sessions.* counters after an insert; *unit* bakes the ``+ 1`` literal into the SQL."""
        inc, params = ("1", ()) if unit else ("?", (inserted,))
        if tool_calls > 0:
            conn.execute(
                f"""UPDATE sessions SET message_count = message_count + {inc},
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                (*params, tool_calls, session_id))
        elif inserted > 0:
            conn.execute(
                f"UPDATE sessions SET message_count = message_count + {inc} WHERE id = ?", (*params, session_id))

    def append_message(
        self, session_id: str, role: str, content: str = None, tool_name: str = None, tool_calls: Any = None,
        tool_call_id: str = None, token_count: int = None, finish_reason: str = None, reasoning: str = None,
        reasoning_content: str = None, reasoning_details: Any = None, codex_reasoning_items: Any = None,
        codex_message_items: Any = None, platform_message_id: str = None, observed: bool = False,
        effect_disposition: Optional[str] = None, _compressed_summary: bool = False, timestamp: Any = None,
        api_content: Optional[str] = None, display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None, compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> int:
        """Append one message; returns the row id and bumps the session counters. ``platform_message_id``:
        the platform's own id. ``api_content``: byte-fidelity sidecar, the exact string sent to the API when
        it differed from ``content``, stored as sent except lone surrogates."""
        msg = dict(locals())  # every keyword above is a message-dict field of the same name
        # Encode outside the write txn (display metadata first: log-order parity).
        msg["display_metadata"] = self._encode_display_metadata(display_metadata)
        tool_calls = _parse_tool_calls(tool_calls)
        params = self._message_row_params(
            session_id, role, msg, tool_calls, _coerce_timestamp(timestamp, time.time()), keep_reasoning=True)
        def _do(conn):
            self._check_transcript_write_guards(conn, session_id, compression_lock_holder,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            self._bump_session_counters(conn, session_id, 1, _tool_calls_count(tool_calls), unit=True)
            return msg_id
        # THE critical write (failure aborts the turn): long patience so a sibling legitimately
        # holding the lock for seconds (VACUUM, checkpoint) can't kill it.
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def append_messages_batch(
        self, session_id: str, messages: List[Dict[str, Any]], compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None,
        turn_lease_ttl_seconds: float = 300.0) -> int:
        """Append *messages* in ONE write txn (all rows land or none, guards run once); returns the inserted
        count. ``chunk_rows`` bounds txn size for LARGE copies (branch seeds; FTS triggers run per row)."""
        if not messages:
            return 0
        if chunk_rows is not None and len(messages) > chunk_rows:
            return sum(self.append_messages_batch(session_id, messages[start:start + chunk_rows],
                    compression_lock_holder=compression_lock_holder, turn_lease_holder=turn_lease_holder,
                    turn_lease_ttl_seconds=turn_lease_ttl_seconds)
                for start in range(0, len(messages), chunk_rows))
        def _do(conn):
            self._check_transcript_write_guards(conn, session_id, compression_lock_holder,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            from agent.transcript_repair import resolve_and_repair_transcript_batch
            inserted_rows = resolve_and_repair_transcript_batch(conn, session_id, messages,
                encode_content_fn=self._encode_content, decode_content_fn=self._decode_content)
            inserted, tool_calls_total = self._insert_message_rows(conn, session_id, inserted_rows)
            self._bump_session_counters(conn, session_id, inserted, tool_calls_total, unit=False)
            return inserted
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def set_latest_matching_message_display_kind(self, session_id: str, *, role: str, content: str,
                                                 display_kind: str,
                                                 display_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Stamp presentation metadata on this turn's freshly persisted row (newest active row by content,
        right after the serial turn flushed); the model still sees ``role``/``content`` unchanged, so
        producer provenance survives without classifying by content at render time."""
        if not session_id or not content or not display_kind:
            return False
        def _do(conn):
            row = conn.execute("SELECT id FROM messages WHERE session_id = ? AND role = ? "
                "AND content = ? AND active = 1 ORDER BY id DESC LIMIT 1",
                (session_id, role, self._encode_content(content))).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?",
                (_scrub_surrogates(display_kind), self._encode_display_metadata(display_metadata), row[0]))
            return True
        return self._execute_write(_do)

    def _reaction_list(self, meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Well-formed (dict) reactions stored under ``REACTIONS_METADATA_KEY``."""
        reactions = (meta or {}).get(self.REACTIONS_METADATA_KEY)
        return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []

    def set_message_reaction(self, session_id: str, message_row_id: int, emoji: Optional[str], *,
                             author: str = "user") -> Optional[List[Dict[str, Any]]]:
        """Set (``emoji=None``: clear) *author*'s reaction. Tapback semantics: one per author per message;
        the same emoji again clears, a different one replaces. Returns the list after the write, or
        ``None`` for a foreign row."""
        if not session_id or message_row_id is None:
            return None
        def _do(conn):
            row = conn.execute(_DISPLAY_META_ROW_SQL, (message_row_id, session_id)).fetchone()
            if row is None:
                return None
            meta = self._decode_display_metadata(row[0]) or {}
            existing = self._reaction_list(meta)
            reactions = [r for r in existing if r.get("author") != author]
            previous = next((r for r in existing if r.get("author") == author), None)
            if emoji and (previous is None or previous.get("emoji") != emoji):
                reactions.append({"emoji": _scrub_surrogates(emoji), "author": author, "at": time.time()})
            if reactions:
                meta[self.REACTIONS_METADATA_KEY] = reactions
            else:
                meta.pop(self.REACTIONS_METADATA_KEY, None)
            conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta) if meta else None, message_row_id))
            return reactions
        return self._execute_write(_do)

    def get_message_reactions(self, session_id: str, message_row_id: int) -> List[Dict[str, Any]]:
        """Reaction list persisted on one message row (never ``None``)."""
        if not session_id or message_row_id is None:
            return []
        row = self._read_one(_DISPLAY_META_ROW_SQL, (message_row_id, session_id))
        return self._reaction_list(self._decode_display_metadata(row[0])) if row is not None else []

    def take_unseen_reactions(self, session_id: str, *, author: str = "user") -> List[Dict[str, Any]]:
        """Return *author*'s not-yet-surfaced reactions and mark them seen. Reactions are announced on the
        NEXT user turn (never by rewriting the reacted message: cache-safe); ``seen`` makes it exactly once."""
        if not session_id:
            return []
        def _do(conn):
            pending = []
            for row in conn.execute("SELECT id, role, content, display_metadata FROM messages "
                    "WHERE session_id = ? AND active = 1 AND display_metadata IS NOT NULL ORDER BY id",
                    (session_id,)).fetchall():
                meta = self._decode_display_metadata(row["display_metadata"])
                reactions = meta.get(self.REACTIONS_METADATA_KEY) if meta else None
                if not isinstance(reactions, list):
                    continue
                changed = False
                for reaction in reactions:
                    if not isinstance(reaction, dict) or reaction.get("author") != author or reaction.get("seen"):
                        continue
                    reaction["seen"] = True
                    changed = True
                    content = self._decode_content(row["content"])
                    pending.append({
                        "row_id": row["id"], "role": row["role"], "emoji": reaction.get("emoji") or "",
                        "text": content if isinstance(content, str) else ""})
                if changed:
                    conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta), row["id"]))
            return pending
        return self._execute_write(_do)

    def latest_message_row_id(self, session_id: str, *, role: str = "user", offset: int = 0,
                              require_text: bool = True) -> Optional[int]:
        """Row id of the most recent active *role* message, or ``None``. ``offset`` steps back; ``require_text``
        skips rows without plain-text content so "the latest message" never resolves to an invisible bubble."""
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            return None
        text_filter = "AND content IS NOT NULL AND TRIM(content) != '' " if require_text else ""
        row = self._read_one("SELECT id FROM messages WHERE session_id = ? AND role = ? "
            f"AND active = 1 {text_filter}ORDER BY id DESC LIMIT 1 OFFSET ?",
            (session_id, role, int(offset)))
        return row[0] if row else None

    def get_message_role(self, session_id: str, row_id: int) -> Optional[str]:
        """Role of the active message at *row_id* in *session_id*, or ``None``."""
        if not session_id:
            return None
        row = self._read_one("SELECT role FROM messages WHERE id = ? AND session_id = ? AND active = 1", (int(row_id), session_id))
        return row[0] if row else None

    def _insert_message_rows(self, conn, session_id: str, messages: List[Dict[str, Any]]) -> tuple[int, int]:
        """Insert *messages* as fresh active rows in the caller's txn -> ``(inserted, tool_call_count)``.
        Never touches sessions.* counters (callers reconcile differently); reasoning kept for assistant rows."""
        now_ts = time.time()
        inserted = tool_calls_total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            message_timestamp = _coerce_timestamp(msg.get("timestamp"), now_ts)
            cur = conn.execute(_INSERT_MESSAGE_SQL, self._message_row_params(
                session_id, role, msg, tool_calls, message_timestamp, keep_reasoning=role == "assistant"))
            if cur.lastrowid is not None:
                msg["_row_id"] = cur.lastrowid
            inserted += 1
            tool_calls_total += _tool_calls_count(tool_calls)
            now_ts = max(now_ts, message_timestamp) + 1e-6
        return inserted, tool_calls_total

    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]], active_only: bool = False,
        archive_dropped: bool = False, reject_active_turn_lease: bool = False) -> None:
        """Atomically replace a session's messages (/retry, /undo, /compress). DESTRUCTIVE by default (rows
        DELETEd, leave FTS). ``active_only`` spares soft-archived rows (needed with in-place compaction).
        ``archive_dropped`` SOFT-archives live rows rewind-style: what rewind/edit/regenerate must use, since
        DELETE leaves nothing to recover. ``reject_active_turn_lease``: in-txn lease check for user rewrites.

        Pass ``archive_dropped=True`` to SOFT-archive the live rows instead of DELETEing them: the replaced
        turns stay on disk with ``active = 0``, ``compacted = 0`` — the same "the user took it back" marking
        :meth:`rewind_to_message` applies — and stay readable via :meth:`get_messages` with
        ``include_inactive=True``. This is the mode a rewind/edit/regenerate must use: those flows overwrite
        a transcript the user may not have meant to drop, and a plain DELETE also evicts the rows from the
        FTS index, leaving nothing to recover from (#82756). It implies active-only handling —
        already-archived rows are never touched — so ``active_only`` is redundant with it. The rewritten set
        is inserted as fresh active rows exactly as in the destructive path, so the live view is identical
        either way; only the durability of the dropped turns differs.
        """
        from hermes_state_errors import CompressionSessionClosedError
        def _do(conn):
            if reject_active_turn_lease:
                self._check_transcript_write_guards(
                    conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
            elif _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
                raise CompressionSessionClosedError(session_id)
            if archive_dropped:
                # FTS triggers don't fire on `active`: replaced turns stay searchable (include_inactive=True).
                conn.execute("UPDATE messages SET active = 0 WHERE session_id = ? AND active = 1", (session_id,))
            else:
                conn.execute(f"DELETE FROM messages WHERE session_id = ?{' AND active = 1' if active_only else ''}", (session_id,))
            conn.execute(_RESET_COUNTERS_SQL, (session_id,))
            total_messages, total_tool_calls = self._insert_message_rows(conn, session_id, messages)
            conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?", (total_messages, total_tool_calls, session_id))
        self._execute_write(_do)

    def has_archived_messages(self, session_id: str) -> bool:
        """True if the session has any soft-archived (``active = 0``) rows (tests/diagnostics).

        Cheap existence probe — does not load rows. NOTE: production rewrite paths no longer branch on this
        (they pass ``active_only=True`` unconditionally — a probe can fail open or race a concurrent
        ``archive_and_compact``, #80216); kept for tests and diagnostics.
        """
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1", (session_id,)) is not None

    def get_active_message_watermark(self, session_id: str) -> int:
        """MAX(id) of the active rows (0 if none), captured at compression START: every active row above it
        arrived concurrently and must survive compaction verbatim."""
        if not session_id:
            return 0
        return int(self._read_one(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1", (session_id,))[0])

    def _tail_rows_after_watermark(self, conn, sql: str, params) -> Tuple[List[int], int]:
        """``(ids, tool_call_count)`` of the concurrent-tail rows selected by *sql* (``SELECT id, tool_calls``)."""
        rows = conn.execute(sql, params).fetchall()
        return [int(r["id"]) for r in rows], sum(_tool_calls_len(r["tool_calls"]) for r in rows)

    def _clone_message_rows(self, conn, tail_ids: List[int], *, session_id: Optional[str] = None) -> None:
        """Pure-SQL clone of *tail_ids* as fresh live rows (new id, active=1, compacted=0, all else byte-exact;
        FTS triggers index the clones), into *session_id* when given."""
        retarget = session_id is not None
        skip = ("id", "active", "compacted") + (("session_id",) if retarget else ())
        col_list = ", ".join(c for c in self._message_column_names(conn) if c not in skip)
        conn.execute(
            f"INSERT INTO messages ({col_list}, {'session_id, ' if retarget else ''}active, compacted) "
            f"SELECT {col_list}, {'?, ' if retarget else ''}1, 0 FROM messages "
            f"WHERE id IN ({_placeholders(tail_ids)}) ORDER BY id",
            [session_id, *tail_ids] if retarget else tail_ids)

    def archive_and_compact(self, session_id: str, compacted_messages: List[Dict[str, Any]],
        model_config_patch: Optional[Dict[str, Any]] = None, watermark: Optional[int] = None,
        lock_holder: Optional[str] = None, tail_count: int = 0) -> int:
        """Non-destructive in-place compaction under ONE session id: soft-archive the active rows (``active=0,
        compacted=1``: summarized away, still searchable) and insert *compacted_messages* as fresh active
        rows, atomically; returns the new ACTIVE count (= ``message_count``). *watermark* (compression
        START): rows ``id > watermark`` arrived during the slow summary and are re-sequenced after the
        compacted set by a pure-SQL clone (fresh ids); ``None`` archives everything. *lock_holder*: verified
        in-txn so a reclaimed lease fails instead of clobbering the winner. *tail_count*: the LAST N compacted
        rows are the verbatim carried tail; their originals and the clones' originals are superseded
        duplicates and get rewind flags (``active=0, compacted=0``) so search doesn't return each carried
        message once per compaction. ``model_config_patch`` merges in the same txn (``None`` removes a key).

        Concurrent-append safety (#75316): when *watermark* is provided (the value of
        :meth:`get_active_message_watermark` captured at compression START), rows that arrived during the
        slow provider summary call (``id > watermark``) are NOT summarized away. They are re-sequenced after
        the compacted set by a pure-SQL column clone (every column except ``id`` — content, api_content,
        platform_message_id, token counts, reasoning sidecars all survive byte-exact, and the FTS triggers
        index the clones naturally), and the originals are archived. NOTE: re-sequencing assigns the tail
        rows fresh ids; consumers that reference durable row ids re-resolve by content (see 3e8ab0610).
        """
        from hermes_state import SessionCompressionInProgressError
        def _do(conn):
            if lock_holder is not None:
                lock_row = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
                if lock_row is None or lock_row["holder"] != lock_holder or float(lock_row["expires_at"]) <= time.time():
                    raise SessionCompressionInProgressError(
                        f"Compression lease for {session_id!r} lost before commit; refusing to publish a stale compaction")
            patch = model_config_patch is not None
            # on_missing="raise": never commit against a vanished session row (caller keeps the original).
            patched_model_config = self._merge_model_config_json(
                conn, session_id, model_config_patch, on_missing="raise") if patch else None
            tail_ids, tail_tool_calls = ([], 0) if watermark is None else self._tail_rows_after_watermark(
                conn, "SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 AND id > ? ORDER BY id",
                (session_id, int(watermark)))
            # Rewind targets sit AT/BELOW the watermark (all the compressor saw); unbounded, a
            # concurrent append would steal a LIMIT slot.
            rewind_ids: list[int] = []
            if tail_count > 0:
                bound = watermark is not None
                rewind_ids = [int(row["id"]) for row in conn.execute(
                    f"SELECT id FROM messages WHERE session_id = ? AND active = 1{' AND id <= ?' if bound else ''} "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, *((int(watermark),) if bound else ()), int(tail_count))).fetchall()]
            rewind_ids += tail_ids
            if rewind_ids:
                placeholders = _placeholders(rewind_ids)
                conn.execute("UPDATE messages SET active = 0, compacted = 0 "
                    f"WHERE session_id = ? AND id IN ({placeholders})", [session_id, *rewind_ids])
                conn.execute(f"{_ARCHIVE_ACTIVE_SQL} AND id NOT IN ({placeholders})", [session_id, *rewind_ids])
            else:
                conn.execute(_ARCHIVE_ACTIVE_SQL, (session_id,))
            inserted, tool_calls_total = self._insert_message_rows(conn, session_id, compacted_messages)
            if tail_ids:
                self._clone_message_rows(conn, tail_ids)
                inserted += len(tail_ids)
                tool_calls_total += tail_tool_calls
            conn.execute(f"{_SET_COUNTERS_SQL}{', model_config = ?' if patch else ''} WHERE id = ?",
                (inserted, tool_calls_total, *((patched_model_config,) if patch else ()), session_id))
            return inserted
        return self._execute_write(_do)

    def _message_column_names(self, conn) -> List[str]:
        """Column names of the messages table, cached per-connection era."""
        if not getattr(self, "_message_columns_cache", None):
            self._message_columns_cache = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        return self._message_columns_cache

    def set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row (0/1 rows). Preflight compaction
        inserts that row BEFORE the sidecar exists and the later persist identity-skips compacted dicts;
        without this a reload reopens the prompt-cache divergence. ``content`` match guards a racing rewrite."""
        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = (SELECT id FROM messages "
            "WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1"
            ") AND content IS ?",
            (_scrub_surrogates(api_content), session_id, self._encode_content(content)))

    def _dedupe_display_generations(self, rows):
        """Collapse compaction generations so each logical message appears once (the protected tail is copied
        into each generation: same role/content/timestamp, different ``active``/id); prefer the live row, then
        the newest. The ONE definition every display projection shares. *rows* must be ordered by ``id``."""
        seen: Dict[Tuple[Any, ...], Any] = {}
        for row in rows:
            dedupe_content = row["content"]
            if row["role"] == "user":
                handoff, live_view = split_user_originated_turn({
                    "role": "user", "content": self._decode_content(row["content"]),
                    "display_kind": row["display_kind"],
                    "display_metadata": self._decode_display_metadata(row["display_metadata"])})
                if handoff is not None and live_view is not None:
                    dedupe_content = self._encode_content(live_view.get("content"))
            # Tool fields key too: identical tool messages collapse, distinct calls with equal
            # role/content/timestamp never merge.
            key = (row["role"], dedupe_content, row["timestamp"],
                row["tool_call_id"], row["tool_calls"], row["tool_name"])
            cur = seen.get(key)
            if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
                seen[key] = row
        return sorted(seen.values(), key=lambda r: r["id"])

    def _row_to_message_dict(self, row, *, warn_context: str, summary_flag: bool) -> Dict[str, Any]:
        """``dict(row)`` with content/tool_calls/display_metadata decoded; *summary_flag* keeps
        ``_compressed_summary`` only as ``True``."""
        msg = dict(row)
        if summary_flag and msg.pop("_compressed_summary", 0):
            msg["_compressed_summary"] = True
        msg["content"] = self._decode_content(msg["content"])
        if msg.get("tool_calls"):
            msg["tool_calls"] = _json_or(
                msg["tool_calls"], [], f"Failed to deserialize tool_calls in {warn_context}, falling back to []")
        if msg.get("display_metadata") is not None:
            msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
        return msg

    @staticmethod
    def _active_clause(include_inactive: bool, include_compacted: bool) -> str:
        """Audit: every row; display: active plus compaction-archived (never Undo/Rewind rows); default: live."""
        return "" if include_inactive else (_DISPLAY_ACTIVE_CLAUSE if include_compacted else " AND active = 1")

    def get_messages(self, session_id: str, include_inactive: bool = False, include_compacted: bool = False,
                     limit: Optional[int] = None, offset: int = 0, latest: bool = False,
                     after_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load messages in insertion order (id, never timestamp: clocks regress). ``include_inactive``:
        rewind rows too; ``include_compacted``: compaction-archived display history (not rewind rows).
        ``latest`` pages back from the newest but returns chronological order; ``after_id``: keyset paging."""
        if after_id is not None and (latest or offset):
            raise ValueError("after_id is incompatible with latest/offset paging")
        if after_id is not None and include_compacted:
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        active_clause = self._active_clause(include_inactive, include_compacted)
        if include_compacted:
            # Full display set (the UI row cap lives in the endpoint), dedupe, then page ([:None] is a no-op).
            rows = self._dedupe_display_generations(self._read_all(
                "SELECT * FROM messages WHERE session_id = ?" + active_clause + " ORDER BY id ASC", [session_id]))
            rows = rows[::-1][offset:][:limit][::-1] if latest else rows[offset:][:limit]
        else:
            sql = (f"SELECT * FROM messages WHERE session_id = ?{active_clause}"
                f"{' AND id > ?' if after_id is not None else ''} ORDER BY id {'DESC' if latest else 'ASC'}")
            params: list = [session_id] if after_id is None else [session_id, after_id]
            if limit is not None or offset:
                # SQLite's OFFSET requires LIMIT; -1 means "no limit".
                sql += " LIMIT ? OFFSET ?"
                params.extend([-1 if limit is None else limit, offset])
            rows = self._read_all(sql, params)
            if latest:
                rows.reverse()
        return [self._row_to_message_dict(row, warn_context="get_messages", summary_flag=True) for row in rows]

    def find_pr_url_messages(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """Tool results containing ``/pull/``: a deliberately loose scan, oldest-first so the caller takes the last."""
        ids = [s for s in session_ids if s]
        chunks = (ids[start : start + 900] for start in range(0, len(ids), 900))  # SQLite's bound-variable ceiling.
        return [{"session_id": row[0], "content": row[1]} for chunk in chunks for row in self._read_all(
                f"""SELECT session_id, content FROM messages
                    WHERE session_id IN ({_placeholders(chunk)})
                      AND role = 'tool' AND content LIKE '%/pull/%'
                    ORDER BY id ASC""",
                chunk)]

    def get_messages_around(self, session_id: str, around_message_id: int, window: int = 5) -> Dict[str, Any]:
        """Up to *window* messages either side of an anchor id (ascending). ``messages_before``/``_after`` count
        strictly around the anchor (fewer than *window* = session boundary). Empty for a foreign anchor."""
        window = max(window, 0)
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                                (around_message_id, session_id)).fetchone():
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1)).fetchall()
            after_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window)).fetchall()
        window_msgs = [self._row_to_message_dict(r, warn_context="get_messages_around", summary_flag=False)
                       for r in (*reversed(before_rows), *after_rows)]
        # before_rows includes the anchor itself.
        return {"window": window_msgs, "messages_before": max(0, len(before_rows) - 1), "messages_after": len(after_rows)}

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant holding the messages: follow the compression chain to
        the live tip (lineage-aware, so delegate/branch children never hijack it), then walk
        ``parent_session_id`` forward to the DEEPEST node with messages (a continuation may hold newer
        turns), skipping branch/delegate/reset/tool children. Unchanged when nothing has messages; depth cap 32.

        Context compression ends the current session and forks a new child session (linked via
        ``parent_session_id``). The flush cursor is reset, so the child is where new messages actually land
        — the parent ends up with ``message_count = 0`` rows unless messages had already been flushed to it
        before compression. See #15000.
        """
        if not session_id:
            return session_id
        try:
            session_id = self.get_compression_tip(session_id) or session_id
        except Exception:
            pass
        with self._read_ctx() as conn:
            current = session_id
            seen = {current}
            best = None  # deepest node with messages
            for _ in range(32):
                try:
                    if conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (current,)).fetchone() is not None:
                        best = current
                    child_row = conn.execute(
                        "SELECT id FROM sessions AS child WHERE child.parent_session_id = ? "
                        f"  AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL "
                        f"  AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL "
                        f"  AND {_sql_json_extract('child.model_config', '$._reset_from')} IS NULL "
                        f"  AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
                        "  AND COALESCE(child.source, '') != 'tool' "
                        "ORDER BY child.started_at DESC, child.id DESC LIMIT 1", (current,)).fetchone()
                except Exception:
                    return session_id
                if child_row is None or not child_row["id"] or child_row["id"] in seen:
                    break
                current = child_row["id"]
                seen.add(current)
            return best if best is not None else session_id

    def _fetch_conversation_rows(self, session_ids: List[str], active_clause: str, *, with_session_id: bool):
        """``_CONVERSATION_ROW_COLUMNS`` rows for *session_ids* ORDER BY id (timestamps are not monotonic
        and would break tool-call adjacency)."""
        return self._read_all(
            f"SELECT {'session_id, ' if with_session_id else ''}{self._CONVERSATION_ROW_COLUMNS} "
            f"FROM messages WHERE session_id IN ({_placeholders(session_ids)})"
            f"{active_clause} ORDER BY id", tuple(session_ids))

    def get_messages_as_conversation(self, session_id: str, include_ancestors: bool = False,
                                     include_inactive: bool = False, repair_alternation: bool = False,
                                     include_row_ids: bool = False,
                                     include_compacted: bool = False) -> List[Dict[str, Any]]:
        """Load messages in OpenAI format. ``include_compacted`` (deduped display history) is for DISPLAY reads
        only: the model-fed restore must not regrow what compaction summarized away. ``repair_alternation``
        repairs the loaded list for LIVE REPLAY callers (a durable ``user;user`` pair would re-trigger the
        per-request repair forever); the stored transcript is never mutated."""
        rows = self._fetch_conversation_rows(
            self._resume_lineage_ids(session_id) if include_ancestors else [session_id],
            self._active_clause(include_inactive, include_compacted), with_session_id=False)
        if include_compacted:
            rows = self._dedupe_display_generations(rows)
        return self._rows_to_conversation(rows, session_id=session_id, include_ancestors=include_ancestors,
            repair_alternation=repair_alternation, include_row_ids=include_row_ids)

    def _dedupe_replayed_user(self, messages, msg, exact_user_clones) -> Tuple[bool, Any]:
        """Ancestor-lineage dedupe of one decoded user *msg* -> ``(skip, exact_clone_key)``. Rotation
        column-clones the concurrent tail into the child, so copies need not be adjacent: the exact
        ``(timestamp, canonical content)`` clone index is checked first, then the adjacent heuristic. A
        rotated child carrier wins over the ancestor copy."""
        canonical_content = self._canonical_replayed_user_content(msg)[0]
        exact_clone_key = self._exact_replayed_user_clone_key(msg.get("timestamp"), canonical_content)
        previous_exact = exact_user_clones.get(exact_clone_key) if exact_clone_key is not None else None
        duplicate = None
        if previous_exact is not None:
            previous_index = next((i for i, candidate in enumerate(messages) if candidate is previous_exact), None)
            if previous_index is not None:
                duplicate = (previous_index, True)
        if duplicate is None:
            duplicate = self._find_duplicate_replayed_user_message(messages, msg)
        if duplicate is None:
            return False, exact_clone_key
        duplicate_index, prefer_current = duplicate
        if prefer_current:
            messages.pop(duplicate_index)
        return not prefer_current, exact_clone_key

    def _rows_to_conversation(self, rows, *, session_id: str, include_ancestors: bool, repair_alternation: bool,
                              include_row_ids: bool = False,
                              include_summary_markers: bool = False) -> List[Dict[str, Any]]:
        """Decode fetched rows (ordered by id, pre-filtered) into OpenAI format, stable key order. Every dict is
        stamped ``_DB_PERSISTED_MARKER_KEY`` (born durable) so an identity-losing handoff never re-appends the
        transcript on flush. ``_row_id`` is opt-in (gateway reactions); reasoning restored on assistant rows
        only; ``api_content`` VERBATIM (no sanitize/strip) so replay keeps the provider prompt cache byte-stable."""
        from hermes_state import _strip_background_review_harness, _strip_stale_tool_call_markers
        messages = []
        exact_user_clones: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            # Underscore-prefixed like ``_row_id``: transports strip it before the wire; compression's
            # assembly copies strip it so rotated child handoffs still flush (_fresh_compaction_message_copy).
            msg = {"role": row["role"], "content": content, _DB_PERSISTED_MARKER_KEY: True}
            # Born durable (#92231): this dict is materialized FROM a durable row, so stamp the persistence
            # marker at the source instead of relying on every restore caller to thread the loaded list back
            # through a flush as ``conversation_history=`` — any identity-losing handoff (compression's
            # durable-snapshot adoption, incremental persists with no history arg) would otherwise re-append
            # the ENTIRE transcript on flush.
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
            msg.update((col, row[col]) for col in ("api_content", "display_kind") if row[col])
            if row["display_metadata"] and (decoded := self._decode_display_metadata(row["display_metadata"])) is not None:
                msg["display_metadata"] = decoded
            if include_summary_markers and row["_compressed_summary"]:
                msg["_compressed_summary"] = True
            msg.update(
                (col, row[col]) for col in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if row[col])
            if row["tool_calls"]:
                msg["tool_calls"] = _json_or(
                    row["tool_calls"], [], "Failed to deserialize tool_calls in conversation replay, falling back to []")
            if row["platform_message_id"]:  # platform-side id exposed as ``message_id`` (JSONL transcript compat)
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            if row["role"] == "assistant":
                msg.update((col, row[col]) for col in ("finish_reason", "reasoning") if row[col])
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                msg.update(
                    (col, _json_or(row[col], None, f"Failed to deserialize {col}, falling back to None"))
                    for col in ("reasoning_details", "codex_reasoning_items", "codex_message_items") if row[col])
            if include_ancestors:
                skip, exact_clone_key = self._dedupe_replayed_user(messages, msg, exact_user_clones)
                if skip:
                    continue
                if exact_clone_key is not None:
                    exact_user_clones[exact_clone_key] = msg
            messages.append(msg)
        # Defense-in-depth: strip a background-review harness turn (older builds shared the parent's
        # session_id) plus its curator reply, and bare tool-call marker content ("[memory]") persisted as an answer.
        messages = _strip_stale_tool_call_markers(_strip_background_review_harness(messages))
        if repair_alternation and messages:
            from agent.agent_runtime_helpers import repair_message_sequence
            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info("Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence", repaired, session_id)
        return messages

    def get_resume_conversations(self, session_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """``(model_history, display_history)`` for a resume from ONE SELECT; byte-identical to the separate
        reads. model: the tip's active rows, alternation-repaired, summary marker kept for pre-compress
        checkpointing. display: the full lineage (``/branch`` stands alone), compaction-archived rows deduped.

        The display projection also includes rows preserved by IN-PLACE compaction (``active=0,
        compacted=1``), deduped by :meth:`_dedupe_display_generations`. Without them a compacted
        conversation resumes showing only its summary plus the carried-forward tail — the user's own turns
        read as deleted even though every row is still on disk, and the REST transcript read (which has
        always included them) disagreed with this one about the same session (#92080).
        """
        rows = self._fetch_conversation_rows(
            self._resume_lineage_ids(session_id), _DISPLAY_ACTIVE_CLAUSE, with_session_id=True)
        # The model projection stays active-only: it is the compressed working context.
        model_history = self._rows_to_conversation(
            [r for r in rows if r["session_id"] == session_id and r["active"]], session_id=session_id,
            include_ancestors=False, repair_alternation=True, include_row_ids=True, include_summary_markers=True)
        display_history = self._rows_to_conversation(
            self._dedupe_display_generations(rows), session_id=session_id,
            include_ancestors=True, repair_alternation=False, include_row_ids=True)
        return model_history, display_history

    def _resume_lineage_ids(self, session_id: str) -> List[str]:
        """Session ids a display resume materializes: the compression lineage, or the session alone for an
        explicit ``/branch`` copy. Shared with the resume guard so it counts exactly what a resume loads."""
        return [session_id] if self._is_explicit_branch_session(session_id) else self._session_lineage_root_to_tip(session_id)

    def _resume_count_scope(self, session_id: str, tip_only: bool) -> Tuple[List[str], str]:
        """``tip_only``: the tip's ACTIVE rows (model restore); else the full-lineage DISPLAY set."""
        if tip_only:
            return [session_id], "active = 1"
        return self._resume_lineage_ids(session_id), "(active = 1 OR compacted = 1)"

    def get_resume_message_count(self, session_id: str, *, tip_only: bool = False) -> int:
        """Count the rows a resume would materialize (see ``_resume_count_scope``)."""
        session_ids, active_clause = self._resume_count_scope(session_id, tip_only)
        return int(self._read_one(
            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({_placeholders(session_ids)}) AND {active_clause}",
            tuple(session_ids))[0])

    def assert_resume_safe(self, session_id: str, max_messages: Optional[int] = None, *, tip_only: bool = False) -> int:
        """Resume row count, or raise ``SessionResumeTooLargeError``. ``max_messages=None`` reads config; 0
        disables the guard without counting. ``tip_only`` bounds only the tip's active rows for callers that
        never materialize the lineage: a heavily compressed conversation is a success, not a rejection."""
        from hermes_state import SessionResumeTooLargeError, resolved_max_resume_messages
        if max_messages is None:
            max_messages = resolved_max_resume_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        session_ids, active_clause = self._resume_count_scope(session_id, tip_only)
        message_count = int(self._read_one("SELECT COUNT(*) FROM ("
            f"SELECT 1 FROM messages WHERE session_id IN ({_placeholders(session_ids)}) "
            f"AND {active_clause} LIMIT ?)", (*session_ids, max_messages + 1))[0])
        if message_count > max_messages:
            raise SessionResumeTooLargeError(
                message_count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
        return message_count

    def get_ancestor_display_prefix(self, session_id: str) -> List[Dict[str, Any]]:
        """Ancestor-only display messages of a lineage (row ``session_id != tip``) that ``session.resume``
        prepends. Identified by row origin, not ``display[:len(display) - len(model)]``, so alternation
        repair cannot overcount."""
        session_ids = self._resume_lineage_ids(session_id)
        if len(session_ids) <= 1:
            return []
        rows = self._dedupe_display_generations(
            self._fetch_conversation_rows(session_ids, _DISPLAY_ACTIVE_CLAUSE, with_session_id=True))
        ancestor_ids = {int(row["id"]) for row in rows if row["session_id"] != session_id and row["id"] is not None}
        if not ancestor_ids:
            return []
        lineage = self._rows_to_conversation(
            rows, session_id=session_id, include_ancestors=True, repair_alternation=False, include_row_ids=True)
        return [{k: v for k, v in message.items() if k != "_row_id"}
            for message in lineage if message.get("_row_id") in ancestor_ids]

    def get_conversation_root(self, session_id: str) -> str:
        """ROOT id of the lineage: the stable conversation id across compression segments and delegate
        subagents (Nous Portal usage tagging). Unchanged when there is no recorded parent."""
        chain = self._session_lineage_root_to_tip(session_id)
        return chain[0] if chain and chain[0] else session_id

    @staticmethod
    def _canonical_replayed_user_content(msg: Dict[str, Any]) -> Tuple[Any, bool]:
        """Return canonical live content and whether *msg* is composite."""
        if msg.get("role") != "user":
            return None, False
        handoff, live_view = split_user_originated_turn(msg)
        is_composite = handoff is not None and live_view is not None
        return live_view.get("content") if is_composite else msg.get("content"), is_composite

    @staticmethod
    def _exact_replayed_user_clone_key(timestamp: Any, content: Any) -> Optional[Tuple[Any, str]]:
        """Return a hashable key for a column-exact rotation clone."""
        if timestamp is None or content in (None, "", []):
            return None
        try:
            return timestamp, json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _find_duplicate_replayed_user_message(messages: List[Dict[str, Any]],
                                              msg: Dict[str, Any]) -> Optional[Tuple[int, bool]]:
        """Adjacent replay duplicate ``(index, prefer_current)`` or None. Rotation may persist the current ask
        in the parent and again inside a composite child carrier: carriers compare by canonical live payload,
        ordinary users by exact string. The child carrier wins (it owns the durable row id and scaffold)."""
        if msg.get("role") != "user":
            return None
        canonical = SessionMessagesMixin._canonical_replayed_user_content
        content, prefer_current = canonical(msg)
        if content in (None, "", []):
            return None
        for index in range(len(messages) - 1, -1, -1):
            prev = messages[index]
            if prev.get("role") == "user":
                prev_content, prev_is_composite = canonical(prev)
                if prev_content == content and (prefer_current or prev_is_composite or isinstance(content, str)):
                    return index, prefer_current
            elif prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return None
        return None

    # ========================================================================= Rewind (soft-delete) — see
    # /rewind slash command + issue #21910
    # =========================================================================
    def get_active_message_ids(self, session_id: str) -> List[int]:
        """Ordered physical active ids for rewind CAS checks (includes legacy harness rows projections omit)."""
        return [int(row[0]) for row in self._read_all(_ACTIVE_IDS_SQL, (session_id,))]

    @staticmethod
    def _active_transcript_counts(conn, session_id: str) -> tuple[int, int]:
        """Return active message/tool-call counts inside the caller's txn."""
        rows = conn.execute("SELECT tool_calls FROM messages WHERE session_id = ? AND active = 1", (session_id,)).fetchall()
        return len(rows), sum(_tool_calls_len(row[0], scalar=1) for row in rows)

    def _split_rewind_target(self, target_row: Dict[str, Any], expected_target_content: Any, preserve_compaction_handoff: bool):
        """Validate an active rewind target; return its handoff scaffold (or None). ``ValueError``: inactive /
        non-user-originated / missing composite carrier; ``RuntimeError``: live payload changed."""
        if not target_row.get("active"):
            raise ValueError("rewind target is not active")
        handoff, live_view = split_user_originated_turn({
            **target_row, "content": self._decode_content(target_row.get("content")),
            "display_metadata": self._decode_display_metadata(target_row.get("display_metadata"))})
        if live_view is None:
            raise ValueError("rewind target is not a user-originated turn")
        live_content = live_view.get("content")
        if isinstance(live_content, str):
            live_content = sanitize_context(live_content).strip()
        if expected_target_content is not None and live_content != expected_target_content:
            raise RuntimeError("rewind target changed before it could be persisted")
        if preserve_compaction_handoff and handoff is None:
            raise ValueError("preserve_compaction_handoff requires an active composite carrier")
        return handoff if preserve_compaction_handoff else None

    def rewind_to_message(self, session_id: str, target_message_id: int, *, preserve_compaction_handoff: bool = False,
                          expected_active_ids: Optional[List[int]] = None,
                          expected_target_content: Any = None) -> Dict[str, Any]:
        """Soft-delete (``active=0``) every message with id >= *target_message_id*, target included (the caller
        pre-fills it as the next prompt). Returns ``{"rewound_count", "target_message", "new_head_id"}``, plus
        ``replacement_message_id`` with ``preserve_compaction_handoff`` (archives a composite summary carrier,
        inserts its hidden handoff scaffold as the new head). ``ValueError``: target missing or not ``user``.
        ``expected_active_ids`` / ``expected_target_content`` pin the active set and canonical live payload
        in-txn before any mutation (presentation-only metadata changes don't invalidate a rewind). A live turn
        lease refuses; expired/dead holders are reclaimed. ``rewind_count`` always increments."""
        def _do(conn):
            self._check_transcript_write_guards(
                conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
            if expected_active_ids is not None:
                active_rows = conn.execute(_ACTIVE_IDS_SQL, (session_id,)).fetchall()
                if [int(r[0]) for r in active_rows] != expected_active_ids:
                    raise RuntimeError("active transcript changed before the rewind could be persisted")
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?", (target_message_id, session_id)).fetchone()
            if row is None:
                raise ValueError(f"message {target_message_id} not found in session {session_id}")
            target_row = dict(row)
            if target_row.get("role") != "user":
                raise ValueError(
                    f"rewind target must be a 'user' message (got role={target_row.get('role')!r}, id={target_message_id})")
            replacement_message_id = replacement = None
            if preserve_compaction_handoff or expected_target_content is not None:
                replacement = self._split_rewind_target(target_row, expected_target_content, preserve_compaction_handoff)
            ids = [r[0] for r in conn.execute("SELECT id FROM messages WHERE session_id = ? AND id >= ? AND active = 1",
                                             (session_id, target_message_id)).fetchall()]
            if ids:
                conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({_placeholders(ids)})", ids)
            if replacement is not None:
                self._insert_message_rows(conn, session_id, [replacement])
                replacement_message_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = ?", (session_id,))
            message_count, tool_call_count = self._active_transcript_counts(conn, session_id)
            conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?", (message_count, tool_call_count, session_id))
            head_id = conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1", (session_id,)).fetchone()[0]
            return target_row, ids, head_id, replacement_message_id
        target_row, rewound, new_head_id, replacement_message_id = self._execute_write(_do)
        # Decode for the prompt-buffer prefill without a second fallible DB operation.
        target_row["content"] = self._decode_content(target_row.get("content"))
        return {"rewound_count": len(rewound), "target_message": target_row, "new_head_id": new_head_id,
                **({"replacement_message_id": replacement_message_id} if preserve_compaction_handoff else {})}

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        sql = "SELECT COUNT(*) FROM messages" + (" WHERE session_id = ?" if session_id else "")
        return self._read_one(sql, (session_id,) if session_id else ())[0]

    def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        """True when *platform_message_id* exists (partial-index probe; the gateway's transient-failure dedupe).

        Uses the idx_messages_platform_msg_id partial index for efficient lookup. Used by the gateway's
        transient-failure dedupe guard (#47237) to skip re-persisting a user message that was already saved
        on a prior retry of the same inbound platform message.
        """
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
            (session_id, platform_message_id)) is not None

    def _is_explicit_fork_child_row(self, session: Dict[str, Any]) -> bool:
        """True when *session* is a branch, delegate, or tool child of its parent. Markers only count when they
        point at ``parent_session_id``: compression copies ``model_config`` onto the continuation, so
        presence-only matching would misclassify it (same binding as ``_NON_CONTINUATION_CHILD_FILTER_SQL``)."""
        if session.get("source") == "tool":
            return True
        cfg = session.get("model_config")
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except json.JSONDecodeError:
                return False
        if not isinstance(cfg, dict):
            return False
        markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"))
        parent_id = session.get("parent_session_id")
        return parent_id in markers if parent_id else any(m is not None for m in markers)

    def is_explicit_fork_child(self, session_id: str) -> bool:
        """Read-only :meth:`_is_explicit_fork_child_row`; a missing row is not a fork (prompt_cache_scope keeps
        a declared conversation key from crossing the fork boundary)."""
        session = self.get_session(session_id)
        return bool(session and self._is_explicit_fork_child_row(session))

    def latest_conversation_boundary(self, session_key: str, source: str) -> Optional[int]:
        """Conversation boundaries (``_RESET_END_REASONS`` ends) this peer has crossed, or ``None`` if never
        reset. The peer is ``(session_key, source)``, never the key alone (an API caller may legally reuse a
        Telegram row's key). Read from ``conversation_generations`` (advanced in each boundary's txn), not an
        aggregate over session rows: deletes/prunes would re-emit a retired pair. Rows are never GC'd
        (dropping one re-issues generation 1: the ABA this prevents). Wall-clock-free, so a backwards NTP
        correction cannot reorder it. DBs upgraded mid-conversation take their first generation from the
        next boundary written (a pre-upgrade reset shares its predecessor's scope once: costs a warm
        prompt-cache bucket, never crosses an identity)."""
        if not session_key or not source:
            return None
        row = self._read_one(
            "SELECT generation FROM conversation_generations WHERE source = ? AND session_key = ?",
            (source, session_key))
        generation = int(row["generation"]) if row is not None and row["generation"] is not None else 0
        return generation if generation > 0 else None

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(_RESET_COUNTERS_SQL, (session_id,))
        self._execute_write(_do)

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
        """Permanently clear bare tool-call marker content ("[memory]") left by pre-fix sessions
        (``_rows_to_conversation`` repairs it in memory; this stops the re-scan). Only ``content`` is touched.
        ``backup``: ``VACUUM INTO`` snapshot first (none when nothing changes)."""
        from hermes_state import _STALE_TOOL_CALL_MARKER_RE
        def _find_affected(conn) -> List[int]:
            cursor = conn.execute("SELECT id, content FROM messages "
                "WHERE role = 'assistant' AND tool_calls IS NOT NULL AND tool_calls != ''")
            return [row["id"] for row in cursor.fetchall()
                if isinstance(row["content"], str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(row["content"].strip())]
        def _result(affected, backup_path=None):
            return {"dry_run": dry_run, "rows_affected": len(affected), "row_ids": affected, "backup_path": backup_path}
        with self._read_ctx() as conn:
            affected_ids = _find_affected(conn)
        if dry_run or not affected_ids:
            return _result(affected_ids)
        backup_path: Optional[str] = None
        if backup:
            import datetime
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.db_path.with_name(f"{self.db_path.name}.pre-clean-markers-backup-{stamp}"))
            with self._lock:
                self._conn.execute("VACUUM INTO ?", (backup_path,))
            logger.info("Backed up state.db to %s before clean-markers write", backup_path)
        def _do(conn):
            ids = _find_affected(conn)
            if ids:
                conn.execute(f"UPDATE messages SET content = '' WHERE id IN ({_placeholders(ids)})", ids)
            return ids
        affected_ids = self._execute_write(_do)
        if affected_ids:
            logger.info("Permanently cleared %d stale tool-call marker row(s) in state.db (#78148)", len(affected_ids))
        return _result(affected_ids, backup_path)
