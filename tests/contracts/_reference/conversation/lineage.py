"""In-memory reference implementation of ``LineageRepository`` (test fixture).

Registered onto the assembled backend as the ``lineage`` attribute. It owns no table: every walk reads the sessions reference's rows (``scope_store(scope)["sessions"]``) and the message reference's transcripts, and continuation edges are evaluated from the parent's ``end_reason`` and the child's fork markers when read, never from a stored flag. Each public method runs under the backend lock, which makes it one atomic step.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Mapping, Optional, Sequence

from persistence import CompressionBusy, CompressionPublishRefused, Conflict, ProfileScope, SessionId
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationMessage, NewMessage, RowStamp
from persistence.conversation.sessions import (
    AUTOMATIC_END_REASONS,
    END_REASON_COMPRESSION,
    END_REASON_ORPHANED_COMPRESSION,
    SessionLineageKind,
    SessionRecord,
)
from persistence.core.identity import MessageId

from ..backend import InMemoryBackend, register_repository
from .sessions import InMemorySessionRepository, blank_session_row, sync_lineage_markers
from .sessions_listing import recency_key
from .transcript_store import (
    RESET_END_REASONS,
    _hashable,
    _payload,
    _Transcript,
    clone_rows,
    conversation_dicts,
    dedupe_display,
    insert_rows,
    resume_lineage,
    row_stamp,
    rows_of,
    transcripts_of,
    _messages_namespace,
)

__all__ = ["InMemoryLineageRepository", "compression_tip"]

_FORK_KINDS = (SessionLineageKind.BRANCH, SessionLineageKind.DELEGATE)
_FORK_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")
# Raw parent walks stop after this many nodes; the forward compression walk after ``_CHAIN_CAP`` steps.
_WALK_NODES = 100
_CHAIN_CAP = 1000
_RESUME_DESCENT_DEPTH = 32
_ORPHAN_FINALIZE_AGE_SECONDS = 7 * 86400.0
# created_source values that are no creation provenance.
_UNSET_CREATED = (None, "", "unknown", "local")
# model_config keys that belong to the whole conversation: a rotation carries them onto the child.
_CONVERSATION_MODEL_CONFIG_KEYS = ("onboarding_run", "prompt_refresh_epoch")


def _sessions_table(backend: InMemoryBackend, scope: ProfileScope) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = backend.scope_store(scope).setdefault("sessions", {})
    return table


def _fork(row: Mapping[str, Any], *, include_reset: bool = False) -> bool:
    return InMemorySessionRepository._is_fork_child_row(row, include_reset=include_reset)


def _compression_child(table: Mapping[str, dict[str, Any]], row: Mapping[str, Any]) -> bool:
    """A continuation of its parent: no fork of it (reset included) and the parent ended by compression (``_is_compression_child_row``)."""
    parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
    return parent is not None and not _fork(row, include_reset=True) and parent["end_reason"] == END_REASON_COMPRESSION


def _non_continuation(row: Mapping[str, Any], parent_id: str) -> bool:
    """A child that is no compression continuation of ``parent_id``: a marker naming it, or a tool row."""
    return row["source"] == "tool" or any(row["model_config"].get(marker) == parent_id for marker in _FORK_MARKERS)


def _descent_child(table: Mapping[str, dict[str, Any]], row: Mapping[str, Any]) -> bool:
    """A child a resume descends into: no fork marker at all, no legacy reset child of a reset-ended parent on the same routing key, not a tool row."""
    if row["source"] == "tool" or any(row["model_config"].get(marker) is not None for marker in _FORK_MARKERS):
        return False
    parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
    return not (parent is not None and parent["end_reason"] in RESET_END_REASONS and row["session_key"] and row["session_key"] == parent["session_key"])


def _carry_parent_timestamps(parent: Optional[_Transcript], handoff: Sequence[NewMessage]) -> list[NewMessage]:
    """Handoff rows without a timestamp take the matching parent row's, so the child copy keeps the original's display identity (``SessionMessagesMixin._carry_parent_timestamps``): donors are the parent's active rows by role, content, tool call id, tool calls and tool name, first match wins."""
    donors: dict[tuple[Any, ...], list[float]] = {}
    for row in parent.active() if parent is not None else ():
        m = row.message
        donors.setdefault((m.role, _hashable(m.content), m.tool_call_id, _hashable(list(m.tool_calls)), m.tool_name), []).append(row.epoch)
    carried: list[NewMessage] = []
    for message in handoff:
        stored = _payload(message)
        queue = donors.get((stored.role, _hashable(stored.content), stored.tool_call_id, _hashable(list(stored.tool_calls)), stored.tool_name))
        carried.append(message if message.timestamp is not None or not queue else replace(message, timestamp=queue.pop(0)))
    return carried


def _counter(row: Optional[Mapping[str, Any]], column: str) -> int:
    """A stored counter clamped at 0; a missing row or an unreadable value reads 0."""
    try:
        return 0 if row is None else max(0, int(row.get(column) or 0))
    except (TypeError, ValueError):
        return 0


class InMemoryLineageRepository:
    """See the ``persistence.conversation.lineage`` protocol for the contract each method implements."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _chain_step(self, scope: ProfileScope, table: Mapping[str, dict[str, Any]], current: str) -> Optional[str]:
        """The preferred continuation of ``current``: a child of a compression-ended parent without a fork marker, reset or tool origin; one that compressed again first, then an open one, then the most recently active."""
        parent = table.get(current)
        if parent is None or parent["end_reason"] != END_REASON_COMPRESSION:
            return None
        candidates = [row for row in table.values() if row["parent_session_id"] == current and _descent_child(table, row)]
        if not candidates:
            return None

        def rank(row: Mapping[str, Any]) -> int:
            if row["end_reason"] == END_REASON_COMPRESSION:
                return 0
            return 1 if row["ended_at"] is None else 2

        best = min(rank(row) for row in candidates)
        pool = [row for row in candidates if rank(row) == best]
        winner = max(pool, key=lambda row: (*recency_key(self._backend.sessions._last_active(scope, row)), row["started_at"], row["session_id"]))
        return str(winner["session_id"])

    def chain(self, session_id: SessionId) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        if not session_id:
            return ()
        table = _sessions_table(self._backend, scope)
        walked = [str(session_id)]
        for _ in range(_CHAIN_CAP):
            child = self._chain_step(scope, table, walked[-1])
            if child is None or child in walked:
                break
            walked.append(child)
        return tuple(SessionId(sid) for sid in walked)

    def tip(self, session_id: SessionId) -> SessionId:
        walked = self.chain(session_id)
        return walked[-1] if walked else session_id

    def lineage(self, session_id: SessionId) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        session = table.get(str(session_id))
        if session is None:
            return ()
        if _fork(session):
            return (session_id,)
        root = session
        ancestors = {root["session_id"]}
        while _compression_child(table, root):
            parent = table.get(root["parent_session_id"])
            if parent is None or parent["session_id"] in ancestors:
                break
            root = parent
            ancestors.add(root["session_id"])
        walked = [root["session_id"]]
        current = root
        while current["end_reason"] == END_REASON_COMPRESSION:
            children = sorted((row for row in table.values() if row["parent_session_id"] == current["session_id"]), key=lambda row: row["started_at"])
            following = next((row for row in children if _compression_child(table, row)), None)
            if following is None or following["session_id"] in walked:
                break
            walked.append(following["session_id"])
            current = following
        return tuple(SessionId(sid) for sid in walked) if session_id in walked else (session_id,)

    def conversation_root(self, session_id: SessionId) -> SessionId:
        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        walked: list[str] = []
        current: Optional[str] = str(session_id)
        while current and current not in walked and len(walked) < _WALK_NODES:
            walked.append(current)
            row = table.get(current)
            if row is None:
                break
            current = row["parent_session_id"]
        return SessionId(walked[-1]) if walked else session_id

    def tool_rotation_root(self, session_id: SessionId) -> SessionId:
        table = _sessions_table(self._backend, self._backend.scope)
        row, seen = table.get(str(session_id)), {str(session_id)}
        while row is not None and row["source"] == "tool" and (parent_id := row["parent_session_id"]) not in seen:
            parent = table.get(parent_id) if parent_id else None
            if parent is None or parent["source"] != "tool" or not _compression_child(table, {**row, "source": None}):
                break
            seen.add(parent_id)
            row = parent
        return SessionId(row["session_id"]) if row is not None else session_id

    def is_fork_child(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        row = _sessions_table(self._backend, scope).get(str(session_id))
        return row is not None and _fork(row)

    def is_compression_child(self, child: SessionRecord) -> bool:
        edge = {"source": child.source, "model_config": child.model_config, "parent_session_id": child.parent_session_id}
        return _compression_child(_sessions_table(self._backend, self._backend.scope), edge)

    def resolve_resume_target(self, session_id: SessionId) -> SessionId:
        scope = self._backend.scope
        if not session_id:
            return session_id
        table = _sessions_table(self._backend, scope)
        transcripts = transcripts_of(self._backend, scope)
        current = str(self.tip(session_id))
        start, seen, best = current, {current}, None
        for _ in range(_RESUME_DESCENT_DEPTH):
            if transcripts.get(current) is not None and transcripts[current].rows:
                best = current
            children = sorted((row for row in table.values() if row["parent_session_id"] == current and _descent_child(table, row)),
                              key=lambda row: (row["started_at"], row["session_id"]), reverse=True)
            if not children or children[0]["session_id"] in seen:
                break
            current = children[0]["session_id"]
            seen.add(current)
        return SessionId(best if best is not None else start)

    def resume_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        return tuple(SessionId(sid) for sid in resume_lineage(self._backend, self._backend.scope, str(session_id)))

    def delegated_child_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        row = _sessions_table(self._backend, self._backend.scope).get(str(session_id))
        if row is None or not (row.get("model_config") or {}).get("_delegate_from"):
            return (session_id,)
        return self.resume_lineage_ids(session_id)

    def delegate_descendant_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        table = _sessions_table(self._backend, self._backend.scope)
        seeds: list[str] = []
        current: Optional[str] = str(session_id)
        while current and current not in seeds and len(seeds) < _WALK_NODES:
            row = table.get(current)
            config = row["model_config"] if row is not None else {}
            if config.get("_delegate_from"):
                break
            seeds.append(current)
            if row is None or config.get("_branched_from"):
                break
            current = row["parent_session_id"]
        found: list[str] = []
        frontier = seeds
        while frontier:
            children = [c for c in InMemorySessionRepository._delegate_children(table, frontier) if c not in found]
            segments = [seg for child in children for seg in self.chain(SessionId(child))]
            frontier = [seg for seg in dict.fromkeys(segments) if seg not in found]
            found.extend(frontier)
        return tuple(SessionId(sid) for sid in found)

    def current_segments(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionId]:
        from .._authority import authority_state
        from ..authority.bindings import _RECEIPT_PREFIX
        from ..authority.bindings_lineage import decode

        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        meta = authority_state(self._backend, scope).snapshot["meta"]
        out: dict[SessionId, SessionId] = {}
        for sid in dict.fromkeys(str(s) for s in session_ids):
            if sid not in table:
                continue
            try:
                receipt = decode(meta[_RECEIPT_PREFIX + sid])
                entry = receipt["entry"]["session_id"]
            except (KeyError, TypeError, ValueError):
                entry = None
            if isinstance(entry, str) and entry:
                out[SessionId(sid)] = SessionId(entry)
            else:
                out[SessionId(sid)] = self.tip(SessionId(sid)) if table[sid].get("end_reason") == END_REASON_COMPRESSION else SessionId(sid)
        return out

    def ancestor_display_prefix(self, session_id: SessionId) -> tuple[ConversationMessage, ...]:
        scope = self._backend.scope
        ids = resume_lineage(self._backend, scope, str(session_id))
        if len(ids) <= 1:
            return ()
        pairs = dedupe_display(rows_of(self._backend, scope, ids))
        ancestor_ids = {row.id for sid, row in pairs if sid != str(session_id)}
        if not ancestor_ids:
            return ()
        dicts = conversation_dicts([row for _sid, row in pairs], include_lineage=True, repair_alternation=False, include_row_ids=True, include_summary_markers=False)
        return tuple(ConversationMessage.from_openai_message({k: v for k, v in message.items() if k != "_row_id"})
                     for message in dicts if message.get("_row_id") in ancestor_ids)

    # ── compression write side ──────────────────

    def _leases(self, scope: ProfileScope) -> dict[str, tuple[str, float]]:
        leases: dict[str, tuple[str, float]] = self._backend.scope_store(scope).setdefault("coord_leases", {})
        return leases

    def _child_row(self, scope: ProfileScope, parent: Mapping[str, Any], child: CompressionChild) -> dict[str, Any]:
        """The continuation's session row copied from the parent: routing, user, git metadata, display name, origin and the tools pin from the parent; the conversation-wide config keys, pin and archive state too unless the child is a branch or reset copy."""
        continuation = not any(key in (child.model_config or {}) for key in ("_branched_from", "_reset_from"))
        carried = {key: value for key, value in parent["model_config"].items() if continuation and key in _CONVERSATION_MODEL_CONFIG_KEYS}
        row = blank_session_row(str(child.session_id), lineage_kind=SessionLineageKind.NONE, model_config={**carried, **(child.model_config or {})})
        created = next((label for label in (parent.get("created_source"), child.source) if label not in _UNSET_CREATED), None)
        row.update(
            source=child.source, created_source=created, model=child.model, system_prompt=child.system_prompt,
            parent_session_id=parent["session_id"], cwd=child.cwd or parent["cwd"],
            profile_name=child.profile_name or parent.get("profile_name") or self._backend.owner_profile, started_at=datetime.now(UTC), compression_fork=True,
            pinned=bool(parent["pinned"]) and continuation, archived=bool(parent["archived"]) and continuation,
            auto_archived=bool(parent.get("auto_archived")) and continuation,
        )
        for name in ("git_branch", "git_repo_root", "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin", "tool_pin"):
            row[name] = parent[name]
        sync_lineage_markers(row)
        return row

    def publish_compression_child(self, parent_session_id: SessionId, child: CompressionChild, handoff: Sequence[NewMessage], *,
                                  compression_lock_holder: Optional[str] = None, require_compression_lease: bool = True, require_lease_refresh: bool = False,
                                  lease_ttl_seconds: float = 300.0, watermark: Optional[MessageId] = None,
                                  watermark_ceiling: Optional[MessageId] = None) -> tuple[RowStamp, ...]:
        scope = self._backend.scope
        parent_id = str(parent_session_id)
        leases = self._leases(scope)
        lock = leases.get(parent_id)
        # A refresh by the holder lands first, so a lease that just expired still admits it; it commits only with the publish.
        refresh = require_lease_refresh and bool(compression_lock_holder) and lock is not None and lock[0] == compression_lock_holder
        expires = time.time() + lease_ttl_seconds if refresh else (lock[1] if lock is not None else 0.0)
        if require_compression_lease and (lock is None or not compression_lock_holder or lock[0] != compression_lock_holder or expires <= time.time()):
            raise CompressionBusy(parent_session_id)
        table = _sessions_table(self._backend, scope)
        parent = table.get(parent_id)
        if parent is None:
            raise CompressionPublishRefused(parent_session_id, f"Compression parent not found: {parent_id}")
        if parent["ended_at"] is not None and parent["end_reason"] not in AUTOMATIC_END_REASONS:
            raise CompressionPublishRefused(parent_session_id, f"Compression parent already ended: {parent_id}")
        if not handoff:
            raise CompressionPublishRefused(parent_session_id, "Compression child handoff must not be empty")
        if str(child.session_id) in table:
            raise Conflict("session", str(child.session_id))
        # Everything is built aside first, so a failure part way leaves the store as it was.
        child_row = self._child_row(scope, parent, child)
        namespace = _messages_namespace(self._backend, scope)
        parent_rows = namespace["transcripts"].get(parent_id)
        staged = _Transcript()
        published = insert_rows(namespace, staged, _carry_parent_timestamps(parent_rows, handoff))
        if watermark is not None and parent_rows is not None:
            # The parent's concurrent tail is cloned after the handoff; the originals stay in the closed parent.
            clone_rows(namespace, staged, [row for row in parent_rows.active()
                                           if row.id > int(watermark) and (watermark_ceiling is None or row.id <= int(watermark_ceiling))])
        stamps = tuple(row_stamp(row) for row in published)
        if refresh and compression_lock_holder:
            leases[parent_id] = (compression_lock_holder, expires)
        table[str(child.session_id)] = child_row
        namespace["transcripts"].setdefault(str(child.session_id), _Transcript()).rows.extend(staged.rows)
        parent["ended_at"] = datetime.now(UTC)
        parent["end_reason"] = END_REASON_COMPRESSION
        self._stamp_ancestors(scope, table, str(child.session_id))
        if parent["archived"]:
            self._backend.sessions._unarchive_auto_archived_lineage(table, str(child.session_id))
        return stamps

    def _stamp_ancestors(self, scope: ProfileScope, table: dict[str, dict[str, Any]], session_id: str) -> int:
        """Copy the session's creating surface onto compression ancestors without one, each hop the parent's preferred continuation (``stamp_compression_ancestors``); returns the rows stamped."""
        row = table.get(session_id)
        surface = row.get("created_source") if row is not None else None
        if row is None or surface in _UNSET_CREATED:
            return 0
        stamped = 0
        current, parent_id = session_id, row["parent_session_id"]
        for _ in range(_CHAIN_CAP):
            parent = table.get(parent_id) if parent_id else None
            if parent is None or parent.get("created_source") not in (*_UNSET_CREATED, surface) or self._chain_step(scope, table, parent_id) != current:
                break
            if parent.get("created_source") != surface:
                parent["created_source"] = surface
                stamped += 1
            current, parent_id = parent_id, parent["parent_session_id"]
        return stamped

    def stamp_legacy_compression_chains(self) -> int:
        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        edges = [row["session_id"] for row in list(table.values())
                 if (parent := table.get(row["parent_session_id"]) if row["parent_session_id"] else None) is not None
                 and parent["end_reason"] == END_REASON_COMPRESSION and parent.get("created_source") in _UNSET_CREATED
                 and row.get("created_source") not in _UNSET_CREATED and not _non_continuation(row, parent["session_id"])]
        return sum(self._stamp_ancestors(scope, table, sid) for sid in edges)

    def reopen_orphaned_compression(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        row = table.get(str(session_id)) if session_id else None
        if row is None or row["ended_at"] is None or row["end_reason"] != END_REASON_COMPRESSION:
            return False
        if any(child["parent_session_id"] == str(session_id) and not _non_continuation(child, str(session_id)) for child in table.values()):
            return False
        leases = self._leases(scope)
        lock = leases.get(str(session_id))
        if lock is not None:
            if lock[1] >= time.time():
                return False
            del leases[str(session_id)]
        row["ended_at"] = None
        row["end_reason"] = None
        return True

    def finalize_orphaned_compression_children(self) -> int:
        scope = self._backend.scope
        table = _sessions_table(self._backend, scope)
        transcripts = transcripts_of(self._backend, scope)
        usage_rows: dict[str, Any] = self._backend.scope_store(scope).get("usage", {}).get("sessions", {})
        now = datetime.now(UTC)
        cutoff = now.timestamp() - _ORPHAN_FINALIZE_AGE_SECONDS
        ended = 0
        for row in table.values():
            parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
            usage = usage_rows.get(str(row["session_id"]))
            transcript = transcripts.get(str(row["session_id"]))
            if (parent is not None and parent["end_reason"] == END_REASON_COMPRESSION and parent["ended_at"] is not None
                    and row["ended_at"] is None and row["end_reason"] is None and row["started_at"].timestamp() < cutoff
                    and (usage is None or usage.usage.api_call_count == 0) and transcript is not None and transcript.rows):
                row["ended_at"] = now
                row["end_reason"] = END_REASON_ORPHANED_COMPRESSION
                ended += 1
        return ended

    # ── durable compressor counters ───────────────────────────────────────────

    def _row(self, scope: ProfileScope, session_id: SessionId) -> Optional[dict[str, Any]]:
        return _sessions_table(self._backend, scope).get(str(session_id)) if session_id else None

    def _set_counter(self, scope: ProfileScope, session_id: SessionId, column: str, value: int) -> None:
        row = self._row(scope, session_id)
        if row is not None:
            row[column] = max(0, int(value))

    def compression_fallback_streak(self, session_id: SessionId) -> int:
        scope = self._backend.scope
        return _counter(self._row(scope, session_id), "compression_fallback_streak")

    def set_compression_fallback_streak(self, session_id: SessionId, value: int) -> None:
        scope = self._backend.scope
        self._set_counter(scope, session_id, "compression_fallback_streak", value)

    def compression_ineffective_count(self, session_id: SessionId) -> int:
        scope = self._backend.scope
        return _counter(self._row(scope, session_id), "compression_ineffective_count")

    def set_compression_ineffective_count(self, session_id: SessionId, value: int) -> None:
        scope = self._backend.scope
        self._set_counter(scope, session_id, "compression_ineffective_count", value)

    def compression_overload_streak(self, session_id: SessionId) -> int:
        scope = self._backend.scope
        return _counter(self._row(scope, session_id), "compression_overload_streak")

    def set_compression_overload_streak(self, session_id: SessionId, value: int) -> None:
        scope = self._backend.scope
        self._set_counter(scope, session_id, "compression_overload_streak", value)

    def increment_compression_overload_streak(self, session_id: SessionId) -> Optional[int]:
        scope = self._backend.scope
        row = self._row(scope, session_id)
        if row is None:
            return None
        row["compression_overload_streak"] = int(row.get("compression_overload_streak") or 0) + 1
        return int(row["compression_overload_streak"])

    def compression_recovery_deadline(self, session_id: SessionId) -> float:
        scope = self._backend.scope
        row = self._row(scope, session_id)
        try:
            return 0.0 if row is None else max(0.0, float(row.get("compression_recovery_deadline") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def set_compression_recovery_deadline(self, session_id: SessionId, deadline: Optional[float]) -> None:
        scope = self._backend.scope
        row = self._row(scope, session_id)
        if row is None:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        row["compression_recovery_deadline"] = normalized or None


register_repository("lineage", InMemoryLineageRepository)


def compression_tip(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> str:
    """The live continuation of ``session_id`` as a plain call, for references that resolve it inside their own step."""
    lineage: InMemoryLineageRepository = backend.lineage
    return str(lineage.tip(SessionId(session_id)))
