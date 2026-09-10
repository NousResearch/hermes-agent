"""Passive conversation-history commits for SessionDB: save finalized speech without running the agent.

A trusted host caller hands over one finalized user or assistant turn (or an ordered pair) for an
existing conversation. Canonical message rows, an idempotency receipt and the external-history
watermark move in ONE ``_execute_write`` transaction, fenced by the active turn lease the same way
``append_delegation_delivery`` fences a detached delivery. Nothing here starts inference, dispatches a
tool, mints an approval or creates/reopens a session.

The receipt table is the idempotency authority: identity is ``(producer, event_id)``, never content.
An equal-identity/equal-payload retry returns the ORIGINAL row ids, so a response lost in transit is
recovered by replaying the same event id. Mixin bound via the MRO, built on SessionDB's
``_read_ctx``/``_execute_write`` primitives.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hermes_state_errors import SessionTurnLeaseLostError

# Provenance identifiers are host-assigned and bounded so an untrusted payload can never grow a
# metadata map, smuggle SQL wildcards or blow up a row.
_ID_CHARS = re.compile(r"\A[A-Za-z0-9._-]+\Z")
_PRODUCER_MAX_CHARS = 64
_EVENT_MAX_CHARS = 128
_MAX_CONTENT_BYTES = 64 * 1024
# A compression chain this deep is a corrupt lineage, not a long conversation: refuse rather than walk.
_MAX_LINEAGE_HOPS = 100
_ORDERED_PAIR_ROLES = ("user", "assistant")
_ALLOWED_ROLES = frozenset(_ORDERED_PAIR_ROLES)
_ROW_KEYS = frozenset({"role", "content"})

#: ``messages.display_kind`` stamped on every passively committed row.
PASSIVE_HISTORY_DISPLAY_KIND = "passive_conversation"

_RECEIPT_ROW_SQL = """SELECT id, origin_turn_id, payload_sha256, conversation_id, session_id,
                             message_ids_json
                      FROM passive_history_commits WHERE producer = ? AND event_id = ?"""
_INSERT_RECEIPT_SQL = """INSERT INTO passive_history_commits (
                             producer, event_id, origin_turn_id, payload_sha256, conversation_id,
                             session_id, message_ids_json, committed_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
_WATERMARK_SQL = "SELECT COALESCE(MAX(id), 0) FROM passive_history_commits WHERE conversation_id = ?"
_LINEAGE_ROW_SQL = """SELECT id, parent_session_id, source, model_config, ended_at, end_reason
                      FROM sessions WHERE id = ?"""
_COMMITTED_ROW_SQL = """SELECT session_id, role, content, display_kind, display_metadata
                        FROM messages WHERE id = ?"""


@dataclass(frozen=True)
class PassiveHistoryWatermark:
    """External-history generation for a conversation; ``revision`` 0 means nothing was ever saved."""

    conversation_id: str
    revision: int


@dataclass(frozen=True)
class PassiveHistoryReceipt:
    """Proof that one client event is committed. ``revision`` is the receipt row id — not a turn,
    approval or execution id. ``session_id`` is the physical segment that holds the rows, which stays
    the original segment across later compressions. ``replayed`` marks an acknowledged retry."""

    producer: str
    event_id: str
    origin_turn_id: str
    conversation_id: str
    session_id: str
    message_ids: Tuple[int, ...]
    revision: int
    replayed: bool


class PassiveHistoryConflictError(ValueError):
    """The same client event id was reused for a different payload, origin or conversation owner."""


class PassiveHistoryRetiredError(RuntimeError):
    """A prior receipt's canonical session or message rows are gone; the identity is spent forever.

    Deleted content is never re-inserted by a retry, and recreating a deleted session id cannot
    revive it: the content-free receipt tombstone outlives the rows it referenced.
    """


class PassiveHistoryTargetError(ValueError):
    """The requested conversation is missing, closed, ambiguous or has a malformed lineage."""


class PassiveHistoryBusyError(SessionTurnLeaseLostError):
    """A turn is running on this conversation. Retry the same event id after it finishes.

    Deliberately NOT a lease wait: a passive commit is short, and holding a lease for the length of
    a voice call would starve the conversation's real turns.
    """

    retryable = True


def _validated_identifier(value: Any, field: str, max_chars: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= max_chars or not _ID_CHARS.match(value):
        raise ValueError(
            f"{field} must be 1-{max_chars} ASCII letters, digits, '.', '_' or '-'")
    return value


def _validated_content(content: Any) -> str:
    """Finalized plain text, returned byte-for-byte (whitespace included) or refused."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("passive message content must be non-empty finalized text")
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        # Lone surrogates would be silently replaced by the canonical encoder; refuse instead of
        # storing text the caller never submitted.
        raise ValueError("passive message content is not valid Unicode") from exc
    if len(encoded) > _MAX_CONTENT_BYTES:
        raise ValueError(f"passive message content exceeds {_MAX_CONTENT_BYTES} UTF-8 bytes")
    return content


def _validated_messages(messages: Any) -> List[Dict[str, str]]:
    """One finalized user/assistant row, or an ordered ``[user, assistant]`` pair.

    Role/content are the ONLY accepted keys: system and tool roles, tool fields, reasoning sidecars
    and multimodal structures are refused before anything is written, so this entry point cannot be
    used to inject instructions or fabricate tool results.
    """
    if not isinstance(messages, list) or not 1 <= len(messages) <= 2:
        raise ValueError("messages must be a list of one or two finalized rows")
    rows: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != _ROW_KEYS:
            raise ValueError("each passive message must carry exactly 'role' and 'content'")
        if not isinstance(message["role"], str) or message["role"] not in _ALLOWED_ROLES:
            raise ValueError("passive history accepts only finalized 'user' or 'assistant' rows")
        rows.append({"role": message["role"], "content": _validated_content(message["content"])})
    if len(rows) == 2 and tuple(row["role"] for row in rows) != _ORDERED_PAIR_ROLES:
        raise ValueError("a passive message pair must be ordered [user, assistant]")
    return rows


def _payload_fingerprint(origin_turn_id: str, messages: List[Dict[str, str]]) -> str:
    """Stable payload identity. Host timestamps are excluded on purpose: they are assigned at the
    first insert, so hashing them would make every retry look like a changed payload."""
    return hashlib.sha256(
        json.dumps({"version": 1, "origin_turn_id": origin_turn_id, "messages": messages},
                   ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _committed_message_ids(raw: Any) -> Tuple[int, ...]:
    """Row ids stored on a receipt; anything unreadable retires the identity rather than guessing."""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PassiveHistoryRetiredError("passive history receipt has unreadable row ids") from exc
    if not isinstance(decoded, list) or not decoded or not all(isinstance(i, int) for i in decoded):
        raise PassiveHistoryRetiredError("passive history receipt has unreadable row ids")
    return tuple(decoded)


class SessionPassiveHistoryMixin:
    """``append_passive_messages`` / ``get_passive_history_watermark`` (see module docstring)."""

    def _passive_lineage_row(self, conn, session_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(_LINEAGE_ROW_SQL, (session_id,)).fetchone()
        return dict(row) if row is not None else None

    def _is_passive_continuation_child(self, child: Dict[str, Any]) -> bool:
        """True only for a compression continuation. Branch, delegate and tool children are caught by
        the shared filter; a reset child is an explicit conversation boundary, and an unreadable
        ``model_config`` proves nothing, so both fail closed."""
        if self._is_explicit_fork_child_row(child):
            return False
        config = child.get("model_config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                return False
        return not (isinstance(config, dict)
                    and config.get("_reset_from") == child.get("parent_session_id"))

    def _passive_continuation_children(self, conn, parent_session_id: str) -> List[Dict[str, Any]]:
        rows = conn.execute(
            f"SELECT id, parent_session_id, source, model_config, ended_at, end_reason FROM sessions "
            f"WHERE parent_session_id = ?{self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias='')}"
            "AND (ended_at IS NULL OR end_reason = 'compression') "
            "ORDER BY started_at ASC, id ASC",
            (parent_session_id, parent_session_id, parent_session_id)).fetchall()
        return [child for child in (dict(row) for row in rows)
                if self._is_passive_continuation_child(child)]

    def _passive_conversation_id(self, conn, session_id: str) -> str:
        """Ownership key for a requested session. Resolved on the WRITER connection so a failed lookup
        can never hand back an id the same transaction then commits against."""
        if self._passive_lineage_row(conn, session_id) is None:
            raise PassiveHistoryTargetError(f"Session {session_id!r} does not exist")
        return self._session_turn_lease_key_on_conn(conn, session_id)

    def _resolve_passive_history_tip(self, conn, conversation_id: str, *, requested_session_id: str) -> str:
        """The live segment a NEW event must land on, following compression edges only.

        Zero or several eligible continuations, a cycle, an over-deep chain, a missing row or a tip
        closed for any non-compression reason all refuse: this method never reopens a parent and
        never picks a "best" candidate.
        """
        current = self._passive_lineage_row(conn, conversation_id)
        if current is None:
            raise PassiveHistoryTargetError(f"Conversation {conversation_id!r} lineage row is missing")
        seen = {str(current["id"])}
        for _hop in range(_MAX_LINEAGE_HOPS):
            if current["end_reason"] != "compression":
                break
            children = self._passive_continuation_children(conn, str(current["id"]))
            if len(children) != 1:
                raise PassiveHistoryTargetError(
                    f"Conversation {conversation_id!r} names {len(children)} compression "
                    "continuations; refusing to guess its live segment")
            current = children[0]
            if str(current["id"]) in seen:
                raise PassiveHistoryTargetError(
                    f"Conversation {conversation_id!r} compression lineage is cyclic")
            seen.add(str(current["id"]))
        else:
            raise PassiveHistoryTargetError(
                f"Conversation {conversation_id!r} compression lineage is too deep to resolve")
        if requested_session_id not in seen:
            raise PassiveHistoryTargetError(
                "Requested session is not on the live compression lineage")
        if current["ended_at"] is not None:
            raise PassiveHistoryTargetError(
                f"Session {str(current['id'])!r} is closed ({current['end_reason']!r}); "
                "passive history never reopens a conversation")
        return str(current["id"])

    def get_passive_history_tip(self, session_id: str) -> str:
        """Resolve the same strict segment for admitted readers and passive writers.

        Admission holds the conversation turn lease across this read and the history load.
        Generic resume helpers may choose a preferred child or hide read failures.
        """
        with self._read_ctx() as conn:
            conversation_id = self._passive_conversation_id(conn, session_id)
            return self._resolve_passive_history_tip(
                conn, conversation_id, requested_session_id=session_id)

    def _verify_passive_receipt(self, conn, receipt, *, producer: str, event_id: str) -> None:
        """Prove a stored receipt still describes its original commit, or retire the identity.

        Row ids are historical references without cascading foreign keys, so they are re-verified
        against immutable provenance AND the stored-content fingerprint: after a delete + recreate or
        a database recovery, a normally-autoincrementing id could otherwise point at unrelated
        history, and acknowledging that as the caller's turn would be worse than refusing.
        """
        session_id = str(receipt["session_id"])
        if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
            raise PassiveHistoryRetiredError(
                f"Passive history event {event_id!r} references a session that no longer exists")
        origin_turn_id = str(receipt["origin_turn_id"])
        committed: List[Dict[str, str]] = []
        for index, message_id in enumerate(_committed_message_ids(receipt["message_ids_json"])):
            row = conn.execute(_COMMITTED_ROW_SQL, (message_id,)).fetchone()
            if (row is None or str(row["session_id"]) != session_id
                    or row["display_kind"] != PASSIVE_HISTORY_DISPLAY_KIND
                    or row["role"] not in _ALLOWED_ROLES):
                raise PassiveHistoryRetiredError(
                    f"Passive history event {event_id!r} references rows that no longer exist")
            metadata = self._decode_display_metadata(row["display_metadata"]) or {}
            expected = {"producer": producer, "event_id": event_id,
                        "origin_turn_id": origin_turn_id, "index": index}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise PassiveHistoryRetiredError(
                    f"Passive history event {event_id!r} references rows with foreign provenance")
            content = self._decode_content(row["content"])
            if not isinstance(content, str):
                raise PassiveHistoryRetiredError(
                    f"Passive history event {event_id!r} references unreadable content")
            committed.append({"role": str(row["role"]), "content": content})
        if _payload_fingerprint(origin_turn_id, committed) != str(receipt["payload_sha256"]):
            raise PassiveHistoryRetiredError(
                f"Passive history event {event_id!r} no longer matches its committed content")

    def _insert_passive_receipt(self, conn, *, producer: str, event_id: str, origin_turn_id: str,
        payload_sha256: str, conversation_id: str, session_id: str, message_ids: Tuple[int, ...],
    ) -> int:
        """Insert the receipt and return its id — the conversation's new external-history revision."""
        return conn.execute(_INSERT_RECEIPT_SQL, (
            producer, event_id, origin_turn_id, payload_sha256, conversation_id, session_id,
            json.dumps(list(message_ids)), time.time())).lastrowid

    def append_passive_messages(self, session_id: str, *, producer: str, event_id: str,
        origin_turn_id: str, messages: List[Dict[str, str]],
    ) -> PassiveHistoryReceipt:
        """Commit one finalized external turn (or ``[user, assistant]`` pair) without running the agent.

        ``producer`` is host-assigned provenance, NOT permission. ``event_id`` is the caller's stable
        client-event id and the sole dedupe key. ``origin_turn_id`` links this utterance to the
        operator interaction and to any receipt of work it causes; exactly one canonical owner records
        an utterance, so an execution-bearing turn already recorded elsewhere is adopted by its origin
        receipt rather than submitted here.

        Returns the original receipt with ``replayed=True`` for an equal-payload retry, including a
        retry that arrives through a proven compression successor. Raises
        :class:`PassiveHistoryConflictError` for a reused id with a changed payload/origin/owner,
        :class:`PassiveHistoryBusyError` (retryable) while a turn is running,
        :class:`PassiveHistoryTargetError` for an unresolvable conversation, and
        :class:`PassiveHistoryRetiredError` once the referenced content has been deleted. Rows are
        appended in committed order and never back-inserted to simulate capture order.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("append_passive_messages requires a session_id")
        producer = _validated_identifier(producer, "producer", _PRODUCER_MAX_CHARS)
        event_id = _validated_identifier(event_id, "event_id", _EVENT_MAX_CHARS)
        origin_turn_id = _validated_identifier(origin_turn_id, "origin_turn_id", _EVENT_MAX_CHARS)
        rows = _validated_messages(messages)
        fingerprint = _payload_fingerprint(origin_turn_id, rows)

        def _do(conn) -> PassiveHistoryReceipt:
            receipt = conn.execute(_RECEIPT_ROW_SQL, (producer, event_id)).fetchone()
            if receipt is not None:
                self._verify_passive_receipt(conn, receipt, producer=producer, event_id=event_id)
                if fingerprint != str(receipt["payload_sha256"]):
                    raise PassiveHistoryConflictError(
                        f"Passive history event {event_id!r} is already committed with a different "
                        "payload or origin turn")
                # Ownership only: a replay must stay recoverable after the conversation is closed.
                if self._passive_conversation_id(conn, session_id) != str(receipt["conversation_id"]):
                    raise PassiveHistoryConflictError(
                        f"Passive history event {event_id!r} belongs to another conversation; "
                        "an explicit branch needs its own event identity")
                return PassiveHistoryReceipt(
                    producer=producer, event_id=event_id, origin_turn_id=str(receipt["origin_turn_id"]),
                    conversation_id=str(receipt["conversation_id"]), session_id=str(receipt["session_id"]),
                    message_ids=_committed_message_ids(receipt["message_ids_json"]),
                    revision=int(receipt["id"]), replayed=True)

            conversation_id = self._passive_conversation_id(conn, session_id)
            tip = self._resolve_passive_history_tip(
                conn, conversation_id, requested_session_id=session_id)
            try:
                self._check_transcript_write_guards(conn, tip, None, reject_active_turn_lease=True)
            except SessionTurnLeaseLostError as exc:
                raise PassiveHistoryBusyError(
                    f"Conversation {conversation_id!r} has an active turn; "
                    "retry this passive commit once it finishes") from exc
            pending = [{"role": row["role"], "content": row["content"],
                        "display_kind": PASSIVE_HISTORY_DISPLAY_KIND,
                        "display_metadata": {"producer": producer, "event_id": event_id,
                                             "origin_turn_id": origin_turn_id, "index": index}}
                       for index, row in enumerate(rows)]
            inserted, tool_calls = self._insert_message_rows(conn, tip, pending)
            self._bump_session_counters(conn, tip, inserted, tool_calls, unit=False)
            message_ids = tuple(int(row["_row_id"]) for row in pending)
            revision = self._insert_passive_receipt(
                conn, producer=producer, event_id=event_id, origin_turn_id=origin_turn_id,
                payload_sha256=fingerprint, conversation_id=conversation_id, session_id=tip,
                message_ids=message_ids)
            return PassiveHistoryReceipt(
                producer=producer, event_id=event_id, origin_turn_id=origin_turn_id,
                conversation_id=conversation_id, session_id=tip, message_ids=message_ids,
                revision=int(revision), replayed=False)

        # Same patience as every other transcript writer: a sibling holding the lock for seconds
        # (VACUUM, checkpoint) must not turn into a lost user turn.
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def get_passive_history_watermark(self, session_id: str) -> PassiveHistoryWatermark:
        """External-history generation for *session_id*'s conversation.

        The next canonical turn compares this with the marker it last loaded, so an idle agent
        refreshes exactly when external history changed. Reads no message bodies, never falls back to
        another profile, and never reports absence for a failed read.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("get_passive_history_watermark requires a session_id")
        with self._read_ctx() as conn:
            conversation_id = self._passive_conversation_id(conn, session_id)
            revision = conn.execute(_WATERMARK_SQL, (conversation_id,)).fetchone()[0]
        return PassiveHistoryWatermark(conversation_id=conversation_id, revision=int(revision))
