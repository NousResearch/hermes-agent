"""Carrier-aware user-turn rewind (``/undo``, ``/retry``) — the ONE implementation behind the CLI, the
gateway and the TUI. Rewind is a persisted-history operation: the durable transcript is the authority,
the warm (in-memory) history only has to agree with it. A composite compaction carrier (retained
summary + live human ask in one row) keeps its hidden handoff scaffold as the new head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_HISTORY_CHANGED = "session history changed before the rewind could be persisted"


class RewindTargetUnavailableError(ValueError):
    """The requested user turn is not a rewindable target of the active transcript: no user turns, an
    ordinal past the newest one, a row that is not user-originated, or a plain turn where the caller
    required a compaction carrier. Surfaces map this to their own "nothing to undo" message."""


@dataclass
class RewindOutcome:
    prefix: List[Dict[str, Any]]  # history to install: the warm prefix when ``warm_history`` was given, else durable
    live_view: Dict[str, Any]  # canonical live projection of the rewound turn (prefill / retry source)
    live_text: str  # lossless retry text when ``require_retryable``, else the display flattening (prefill)
    rewound_count: int
    turns_undone: int


def _user_indices(messages: List[Dict[str, Any]]) -> List[int]:
    from agent.context_compressor import user_originated_turn_view
    return [i for i, m in enumerate(messages) if user_originated_turn_view(m) is not None]


def _comparison_content(message: Dict[str, Any]) -> Any:
    """Project content the way the durable row stores it (flush projection, then the read-side sanitize) so a
    warm row and its durable twin compare equal."""
    from agent.memory_manager import sanitize_context
    from agent.session_persistence import _durable_content
    content = _durable_content(message.get("content"))
    if message.get("role") in {"user", "assistant"} and isinstance(content, str):
        return sanitize_context(content).strip()
    return content


class SessionRewindMixin:
    """``SessionDB`` mixin: soft-delete from one user turn onward, validated against the warm history."""

    def rewind_user_turn(
        self, session_id: str, user_ordinal: int, *, warm_history: Optional[List[Dict[str, Any]]] = None,
        require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False,
    ) -> RewindOutcome:
        """Rewind the active transcript to just before user turn ``user_ordinal`` (0 = oldest; negative counts
        back from the newest and clamps to the oldest, so ``-n`` is ``/undo n``). ``warm_history`` (CLI/TUI):
        the in-memory view must have the same user turns and the same live target text as the durable
        transcript, else ``RuntimeError`` and nothing changes; its (richer) prefix is what gets installed.
        ``require_retryable``: the live payload must be losslessly replayable as text (``ValueError`` from
        :func:`retryable_user_text` before any write). ``require_composite``: the target must be a compaction
        carrier. ``adopt_row_ids`` (TUI): copy durable ``_row_id`` identities onto the installed warm prefix so
        clients can address follow-ups by row; the CLI leaves its history shape alone. Out-of-range /
        wrong-shape targets raise :class:`RewindTargetUnavailableError`."""
        from agent.context_compressor import (
            _DB_PERSISTED_MARKER, history_before_user_originated_turn, retryable_user_text,
            split_user_originated_turn)
        from agent.message_content import flatten_message_text
        from agent.session_persistence import _is_ephemeral_scaffolding

        expected_active_ids = self.get_active_message_ids(session_id)
        durable = self.get_messages_as_conversation(session_id, include_row_ids=True)
        durable_user = _user_indices(durable)
        if user_ordinal < 0:
            user_ordinal = max(len(durable_user) + user_ordinal, 0)
        if user_ordinal >= len(durable_user):
            raise RewindTargetUnavailableError("target user message is no longer in session history")
        target_index = durable_user[user_ordinal]
        target = durable[target_index]
        durable_prefix, live_view = history_before_user_originated_turn(durable, target_index)
        scaffold, _ = split_user_originated_turn(target)
        if require_composite and scaffold is None:
            raise RewindTargetUnavailableError("target user message is not a compaction carrier")

        prefix = durable_prefix
        if warm_history is not None:
            warm = [m for m in warm_history if not _is_ephemeral_scaffolding(m)]
            warm_user = _user_indices(warm)
            if len(warm_user) != len(durable_user):
                raise RuntimeError(_HISTORY_CHANGED)
            prefix, warm_live_view = history_before_user_originated_turn(warm, warm_user[user_ordinal])
            if _comparison_content(live_view) != _comparison_content(warm_live_view):
                raise RuntimeError(_HISTORY_CHANGED)
        # Retry re-sends the stored bytes: ``"".join`` of the text parts, never the "\n"-joined display
        # flattening (wire bytes == stored bytes; ``"ab"`` must not come back as ``"a\nb"``).
        live_text = retryable_user_text(live_view.get("content")) if require_retryable else None
        target_row_id = target.get("_row_id")
        if not isinstance(target_row_id, int):
            raise RuntimeError("rewind target has no durable row identity")
        try:
            result = self.rewind_to_message(
                session_id, target_row_id, preserve_compaction_handoff=scaffold is not None,
                expected_active_ids=expected_active_ids, expected_target_content=live_view.get("content"))
        except ValueError as exc:  # target vanished / changed role under us: same class of failure as out-of-range
            raise RewindTargetUnavailableError(str(exc)) from exc
        if scaffold is not None:
            replacement_id = result.get("replacement_message_id")
            if not isinstance(replacement_id, int) or not durable_prefix:
                raise RuntimeError("rewind did not retain its compaction handoff")
            durable_prefix[-1].update({"_row_id": replacement_id, _DB_PERSISTED_MARKER: True})
            prefix[-1] = durable_prefix[-1]
        if adopt_row_ids and prefix is not durable_prefix and len(prefix) == len(durable_prefix) and all(
            warm.get("role") == durable_message.get("role")
            and bool(warm.get("display_kind")) == bool(durable_message.get("display_kind"))
            and _comparison_content(warm) == _comparison_content(durable_message)
            for warm, durable_message in zip(prefix, durable_prefix)
        ):
            # Clients address follow-ups by durable row id: keep the richer warm content, adopt the identities.
            for warm, durable_message in zip(prefix, durable_prefix):
                if isinstance(row_id := durable_message.get("_row_id"), int):
                    warm["_row_id"] = row_id
        return RewindOutcome(
            prefix=prefix, live_view=live_view,
            live_text=live_text if live_text is not None else flatten_message_text(live_view.get("content")),
            rewound_count=int(result.get("rewound_count", 0)), turns_undone=len(durable_user) - user_ordinal)
