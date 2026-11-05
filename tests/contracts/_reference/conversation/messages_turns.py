"""The carrier-aware user-turn rewind of the in-memory message reference: the plan (every read and check) and the one write that pins it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from persistence import ProfileScope, RewindRefused, RowVisibility
from persistence.conversation.messages import (
    CONTENT_REFUSED_DISPLAY_KIND,
    CONTENT_REFUSED_METADATA_KEY,
    DB_PERSISTED_MARKER_KEY,
    ConversationRequest,
    NewMessage,
    RewindPlan,
    RewindTargetUnavailable,
    UserTurnRewind,
)
from persistence.core.identity import MessageId, SessionId

from . import _turn_policy as policy
from .messages_base import MessageBase
from .transcript_store import _StoredRow, insert_rows


def _scaffold_message(scaffold: Mapping[str, Any]) -> NewMessage:
    """The handoff scaffold dict as the row the rewind inserts as the new head."""
    metadata = scaffold.get("display_metadata")
    return NewMessage(
        role=str(scaffold.get("role") or "unknown"), content=scaffold.get("content"), timestamp=policy.coerce_epoch(scaffold.get("timestamp")),
        display_kind=scaffold.get("display_kind"), display_metadata=dict(metadata) if isinstance(metadata, dict) else None,
        compressed_summary=bool(scaffold.get("_compressed_summary")), message_uid=scaffold.get("message_uid") or None)


def _split_target(row: _StoredRow, expected_content: Any, preserve_handoff: bool) -> Optional[NewMessage]:
    """Check an active rewind target (``SessionMessagesMixin._split_rewind_target``) and return the scaffold row to keep, if any."""
    if row.visibility is not RowVisibility.ACTIVE:
        raise RewindTargetUnavailable("rewind target is not active")
    m = row.message
    handoff, live = policy.split_user_originated_turn(
        {"role": m.role, "content": m.content, "display_kind": m.display_kind, "display_metadata": m.display_metadata, "_compressed_summary": m.compressed_summary})
    if live is None:
        raise RewindTargetUnavailable("rewind target is not a user-originated turn")
    if expected_content is not None and policy.loaded_view("user", live.get("content")) != expected_content:
        raise RewindRefused(SessionId(""), "target_changed")
    if preserve_handoff and handoff is None:
        raise RewindTargetUnavailable("preserve_compaction_handoff requires an active composite carrier")
    return _scaffold_message(handoff) if preserve_handoff and handoff is not None else None


class MessageTurnsMixin(MessageBase):
    """``plan_user_turn_rewind``, ``rewind_user_turn`` and ``retain_refused_turn_for_display``."""

    def _durable(self, scope: ProfileScope, session_id: SessionId, *, repair: bool) -> list[dict[str, Any]]:
        read = self.read_conversation(session_id, ConversationRequest(include_row_ids=True, repair_alternation=repair))  # type: ignore[attr-defined]
        return [message.as_openai_message() for message in read]

    def plan_user_turn_rewind(self, session_id: SessionId, user_ordinal: int, *, warm_history: Optional[Sequence[Mapping[str, Any]]] = None,
                              require_retryable: bool = False, require_composite: bool = False) -> RewindPlan:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        expected_active_ids = tuple(MessageId(row.id) for row in transcript.active()) if transcript else ()
        stored = self._durable(scope, session_id, repair=False)
        # Turns are addressed on the repaired projection live replay uses: a stored user;user pair is one turn there.
        durable = self._durable(scope, session_id, repair=True)
        durable_user = policy.user_indices(durable)
        if user_ordinal < 0:
            user_ordinal = max(len(durable_user) + user_ordinal, 0)
        if user_ordinal >= len(durable_user):
            raise RewindTargetUnavailable("target user message is no longer in session history")
        target = durable[durable_user[user_ordinal]]
        durable_prefix, live_view = policy.history_before_user_originated_turn(durable, durable_user[user_ordinal])
        scaffold, _live = policy.split_user_originated_turn(target)
        if require_composite and scaffold is None:
            raise RewindTargetUnavailable("target user message is not a compaction carrier")
        prefix = durable_prefix
        if warm_history is not None:
            warm = [dict(m) for m in warm_history if not policy.is_ephemeral_scaffolding(m)]
            warm_user = policy.user_indices(warm)
            if len(warm_user) != len(durable_user):
                raise RewindRefused(session_id, "history_changed")
            prefix, warm_live_view = policy.history_before_user_originated_turn(warm, warm_user[user_ordinal])
            if policy.comparison_content(live_view) != policy.comparison_content(warm_live_view):
                raise RewindRefused(session_id, "history_changed")
        live_text = policy.retryable_user_text(live_view.get("content")) if require_retryable else None
        target_row_id = target.get("_row_id")
        if not isinstance(target_row_id, int):
            raise RewindRefused(session_id, "target_unidentified")
        # The write pins the stored row, which for a merged turn holds only the first ask.
        stored_view = next((policy.user_originated_turn_view(m) for m in stored if m.get("_row_id") == target_row_id), None)
        if stored_view is None:
            raise RewindRefused(session_id, "history_changed")
        return RewindPlan(target_row_id=MessageId(target_row_id), expected_active_ids=expected_active_ids,
                          expected_target_content=stored_view.get("content"), scaffold=scaffold, prefix=tuple(prefix),
                          durable_prefix=tuple(durable_prefix), live_view=live_view, live_text=live_text, durable_user=tuple(durable_user),
                          user_ordinal=user_ordinal)

    def _rewind_to(self, scope: ProfileScope, session_id: SessionId, plan: RewindPlan) -> tuple[int, Optional[_StoredRow]]:
        """The pinned rewind write: guards, the active-id pin, the target checks, then every active row from the target on rewound, the scaffold inserted, the counters recounted. Returns the rows rewound and the scaffold row."""
        rewound, scaffold, _target = self._rewind_rows(scope, session_id, int(plan.target_row_id), plan.expected_active_ids,
                                                       expected_content=plan.expected_target_content, preserve_handoff=plan.scaffold is not None)
        return rewound, scaffold

    def _rewind_rows(self, scope: ProfileScope, session_id: SessionId, target_row_id: int, expected_active_ids: tuple[MessageId, ...], *,
                     expected_content: Any = None, preserve_handoff: bool = False) -> tuple[int, Optional[_StoredRow], _StoredRow]:
        """The rewind write of ``_rewind_to`` for a target row id; the target's content is pinned only when ``expected_content`` is given, and checked as a user-originated turn only then or with ``preserve_handoff``. Returns the rows rewound, the scaffold row and the target row."""
        stale = self._refuse_live_leases(scope, session_id, ("compaction", "turn"))
        self._refuse_compression_closed(scope, session_id)
        transcript = self._created(scope, session_id)
        if tuple(MessageId(row.id) for row in transcript.active()) != expected_active_ids:
            raise RewindRefused(session_id, "history_changed", "active transcript changed before the rewind could be persisted")
        target = transcript.find(target_row_id)
        if target is None:
            raise RewindTargetUnavailable(f"message {target_row_id} not found in session {session_id}")
        if target.message.role != "user":
            raise RewindTargetUnavailable(f"rewind target must be a 'user' message (got role={target.message.role!r}, id={target_row_id})")
        replacement = None
        if preserve_handoff or expected_content is not None:
            try:
                replacement = _split_target(target, expected_content, preserve_handoff)
            except RewindRefused:
                raise RewindRefused(session_id, "target_changed") from None
        # Stale leases are reclaimed only once every refusal has passed.
        for table, key in stale:
            del table[key]
        rewound = [row for row in transcript.active() if row.id >= target.id]
        for row in rewound:
            row.visibility = RowVisibility.REWOUND
        scaffold = insert_rows(self._ns(scope), transcript, [replacement])[0] if replacement is not None else None
        transcript.rewind_count += 1
        transcript.recount()
        return len(rewound), scaffold, target

    def retain_refused_turn_for_display(self, session_id: SessionId, *, notice: str, boundary_row_id: Optional[MessageId] = None) -> bool:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        boundary = transcript.find(int(boundary_row_id)) if transcript is not None and isinstance(boundary_row_id, int) and boundary_row_id > 0 else None
        if boundary is not None and boundary.message.role == "user" and boundary.visibility is RowVisibility.ACTIVE:
            m = boundary.message
            handoff, _live = policy.split_user_originated_turn({"role": "user", "content": m.content, "display_metadata": m.display_metadata})
            target_row_id, preserve_handoff = boundary.id, handoff is not None
            expected_active_ids = tuple(MessageId(row.id) for row in transcript.active())
        else:
            try:
                plan = self.plan_user_turn_rewind(session_id, -1)
            except RewindTargetUnavailable:
                return False
            target_row_id, preserve_handoff, expected_active_ids = int(plan.target_row_id), plan.scaffold is not None, plan.expected_active_ids
        _rewound, _scaffold, target = self._rewind_rows(scope, session_id, target_row_id, expected_active_ids, preserve_handoff=preserve_handoff)
        # The rewound target is shown again as compacted history; the model never reads it.
        metadata = dict(target.message.display_metadata or {})
        metadata[CONTENT_REFUSED_METADATA_KEY] = {"notice": notice}
        target.message = replace(target.message, display_kind=CONTENT_REFUSED_DISPLAY_KIND, display_metadata=metadata)
        target.visibility = RowVisibility.COMPACTION_ARCHIVED
        self._bump_transcript_revision(scope, session_id)
        return True

    def rewind_user_turn(self, session_id: SessionId, user_ordinal: int, *, warm_history: Optional[Sequence[Mapping[str, Any]]] = None,
                         require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False) -> UserTurnRewind:
        scope = self._backend.scope
        plan = self.plan_user_turn_rewind(session_id, user_ordinal, warm_history=warm_history,
                                          require_retryable=require_retryable, require_composite=require_composite)
        rewound, scaffold = self._rewind_to(scope, session_id, plan)
        durable_prefix = [dict(m) for m in plan.durable_prefix]
        prefix = durable_prefix if warm_history is None else [dict(m) for m in plan.prefix]
        if plan.scaffold is not None:
            if scaffold is None or not durable_prefix:
                raise RewindRefused(session_id, "handoff_lost")
            # The installed scaffold is the replacement row: it carries that row's identity.
            durable_prefix[-1].update({"_row_id": scaffold.id, DB_PERSISTED_MARKER_KEY: True, "message_uid": scaffold.message.message_uid})
            prefix[-1] = durable_prefix[-1]
        if adopt_row_ids and warm_history is not None and len(prefix) == len(durable_prefix) and all(
            warm.get("role") == durable.get("role") and bool(warm.get("display_kind")) == bool(durable.get("display_kind"))
            and policy.comparison_content(warm) == policy.comparison_content(durable)
            for warm, durable in zip(prefix, durable_prefix)
        ):
            # Clients address follow-ups by durable row: the richer warm content stays, the identities are adopted.
            for warm, durable in zip(prefix, durable_prefix):
                if isinstance(row_id := durable.get("_row_id"), int):
                    warm["_row_id"] = row_id
                if isinstance(uid := durable.get("message_uid"), str) and uid:
                    warm["message_uid"] = uid
        live_view = dict(plan.live_view)
        return UserTurnRewind(prefix=tuple(prefix), live_view=live_view,
                              live_text=plan.live_text if plan.live_text is not None else policy.flatten_message_text(live_view.get("content")),
                              rewound_count=rewound, turns_undone=len(plan.durable_user) - plan.user_ordinal)
