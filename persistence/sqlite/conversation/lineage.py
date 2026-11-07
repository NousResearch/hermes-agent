"""``LineageRepository`` on SQLite: the compression chain walks, the compression write side and the durable compressor
counters, through the compression and message store functions."""

from __future__ import annotations

import sqlite3
from typing import Optional, Sequence

from persistence import CompressionPublishRefused, Conflict, MessageId, SessionId
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationMessage, NewMessage, RowStamp
from persistence.conversation.sessions import SessionRecord

from .._backend import SQLiteStore, register_repository
from ._store.compression import (
    finalize_orphaned_compression_sessions, get_compression_chain, get_compression_fallback_streak,
    get_compression_ineffective_count, get_compression_lineage, get_compression_overload_streak,
    get_compression_recovery_deadline, get_compression_tip, increment_compression_overload_streak,
    publish_compression_child, reopen_orphaned_compression_session, set_compression_fallback_streak,
    set_compression_ineffective_count, set_compression_overload_streak, set_compression_recovery_deadline,
    stamp_legacy_compression_chains, _is_compression_child_row,
)
from ._store.messages import (
    get_ancestor_display_prefix, get_conversation_root, is_explicit_fork_child, resolve_resume_session_id,
)
from ._errors import refusals
from .messages import _message_dict, _stamp

__all__ = ["SQLiteLineageRepository"]


class SQLiteLineageRepository:
    """Lineage walks and the compression write side over the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # ── writes ────────────────────────────────────────────────────────────────

    def publish_compression_child(
        self, parent_session_id: SessionId, child: CompressionChild, handoff: Sequence[NewMessage], *,
        compression_lock_holder: Optional[str] = None, require_compression_lease: bool = True,
        require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0, watermark: Optional[MessageId] = None,
        watermark_ceiling: Optional[MessageId] = None,
    ) -> tuple[RowStamp, ...]:
        holder = self._store.holder()
        messages = [_message_dict(message) for message in handoff]
        with refusals(holder, parent_session_id):
            try:
                publish_compression_child(holder,
                    parent_session_id=parent_session_id, child_session_id=child.session_id, source=child.source,
                    messages=messages, model=child.model,
                    model_config=dict(child.model_config) if child.model_config is not None else None,
                    system_prompt=child.system_prompt, cwd=child.cwd, profile_name=child.profile_name,
                    compression_lock_holder=compression_lock_holder,
                    require_compression_lease=require_compression_lease, require_lease_refresh=require_lease_refresh,
                    lease_ttl_seconds=lease_ttl_seconds,
                    watermark=int(watermark) if watermark is not None else None,
                    watermark_ceiling=int(watermark_ceiling) if watermark_ceiling is not None else None)
            except sqlite3.IntegrityError as exc:
                if "sessions.id" not in str(exc):
                    raise
                raise Conflict("session", str(child.session_id)) from exc
            except RuntimeError as exc:
                # The store's precondition refusals are plain RuntimeErrors; its typed refusals subclass it.
                if type(exc) is not RuntimeError:
                    raise
                raise CompressionPublishRefused(parent_session_id, str(exc)) from exc
        return tuple(_stamp(message, {}) for message in messages)

    def reopen_orphaned_compression(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return reopen_orphaned_compression_session(holder, session_id)

    def finalize_orphaned_compression_children(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return finalize_orphaned_compression_sessions(holder)

    def stamp_legacy_compression_chains(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return stamp_legacy_compression_chains(holder)

    def set_compression_fallback_streak(self, session_id: SessionId, value: int) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            set_compression_fallback_streak(holder, session_id, value)

    def set_compression_ineffective_count(self, session_id: SessionId, value: int) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            set_compression_ineffective_count(holder, session_id, value)

    def set_compression_overload_streak(self, session_id: SessionId, value: int) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            set_compression_overload_streak(holder, session_id, value)

    def increment_compression_overload_streak(self, session_id: SessionId) -> Optional[int]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return increment_compression_overload_streak(holder, session_id)

    def set_compression_recovery_deadline(self, session_id: SessionId, deadline: Optional[float]) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            set_compression_recovery_deadline(holder, session_id, deadline)

    # ── reads ─────────────────────────────────────────────────────────────────

    def tip(self, session_id: SessionId) -> SessionId:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return SessionId(get_compression_tip(holder, session_id) or session_id)

    def lineage(self, session_id: SessionId) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return tuple(SessionId(sid) for sid in get_compression_lineage(holder, session_id))

    def chain(self, session_id: SessionId) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return tuple(SessionId(sid) for sid in get_compression_chain(holder, session_id))

    def conversation_root(self, session_id: SessionId) -> SessionId:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return SessionId(get_conversation_root(holder, session_id))

    def is_fork_child(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return is_explicit_fork_child(holder, session_id)

    def is_compression_child(self, child: SessionRecord) -> bool:
        holder = self._store.holder()
        row = {"source": child.source, "model_config": child.model_config, "parent_session_id": child.parent_session_id}
        with refusals(holder, child.session_id):
            return _is_compression_child_row(holder, row)

    def resolve_resume_target(self, session_id: SessionId) -> SessionId:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return SessionId(resolve_resume_session_id(holder, session_id))

    def ancestor_display_prefix(self, session_id: SessionId) -> tuple[ConversationMessage, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            messages = get_ancestor_display_prefix(holder, session_id)
        return tuple(ConversationMessage.from_openai_message(message) for message in messages)

    def compression_fallback_streak(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_compression_fallback_streak(holder, session_id)

    def compression_ineffective_count(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_compression_ineffective_count(holder, session_id)

    def compression_overload_streak(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_compression_overload_streak(holder, session_id)

    def compression_recovery_deadline(self, session_id: SessionId) -> float:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_compression_recovery_deadline(holder, session_id)


register_repository("lineage", SQLiteLineageRepository)
