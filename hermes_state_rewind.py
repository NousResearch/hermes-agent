"""Carrier-aware user-turn rewind (``/undo``, ``/retry``) — the ONE implementation behind the CLI, the
gateway and the TUI. Rewind is a persisted-history operation: the durable transcript is the authority,
the warm (in-memory) history only has to agree with it. A composite compaction carrier (retained
summary + live human ask in one row) keeps its hidden handoff scaffold as the new head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

_HISTORY_CHANGED = "session history changed before the rewind could be persisted"


class RewindTargetUnavailableError(ValueError):
    """The requested user turn is not a rewindable target of the active transcript: no user turns, an
    ordinal past the newest one, a row that is not user-originated, or a plain turn where the caller
    required a compaction carrier. Surfaces map this to their own "nothing to undo" message."""


class RewindIndeterminateError(RuntimeError):
    """The backend acknowledgement was lost and the caller must recover by request id."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"rewind acknowledgement is indeterminate; recover request {request_id}")


@dataclass
class RewindOutcome:
    prefix: List[Dict[str, Any]]  # history to install: the warm prefix when ``warm_history`` was given, else durable
    live_view: Dict[str, Any]  # canonical live projection of the rewound turn (prefill / retry source)
    live_text: str  # lossless retry text when ``require_retryable``, else the display flattening (prefill)
    rewound_count: int
    turns_undone: int
    request_id: str | None = None


class TranscriptRewindStore(Protocol):
    """The narrow backend capability required by the shared rewind policy."""
    def get_active_message_ids(self, session_id: str) -> list[int]: ...
    def get_messages_as_conversation(self, session_id: str, *, include_row_ids: bool = False, **kwargs: Any) -> list[dict[str, Any]]: ...
    def rewind_to_message(self, session_id: str, target_message_id: int, *, preserve_compaction_handoff: bool = False,
                          expected_active_ids: Sequence[int] | None = None, expected_target_content: Any = None,
                          request_id: str | None = None) -> dict[str, Any]: ...


def _user_indices(messages: List[Dict[str, Any]]) -> List[int]:
    from agent.context_compressor import user_originated_turn_view
    return [i for i, m in enumerate(messages) if user_originated_turn_view(m) is not None]


def _comparison_content(message: Dict[str, Any]) -> Any:
    """Project content the way the durable row stores it (flush projection, then the read-side sanitize) so a
    warm row and its durable twin compare equal."""
    from agent.session_persistence import _durable_content
    from hermes_state_messages import SessionMessagesMixin
    return SessionMessagesMixin._loaded_view_content(message.get("role"), _durable_content(message.get("content")))


def rewind_user_turn(
    store: TranscriptRewindStore, session_id: str, user_ordinal: int, *, warm_history: Optional[List[Dict[str, Any]]] = None,
    require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False, request_id: str | None = None,
) -> RewindOutcome:
    """Backend-neutral policy over one atomic backend mutation primitive."""
    return _rewind_user_turn_impl(store, session_id, user_ordinal, warm_history=warm_history,
                                  require_retryable=require_retryable, require_composite=require_composite,
                                  adopt_row_ids=adopt_row_ids, request_id=request_id)

def _rewind_user_turn_impl(
    self, session_id: str, user_ordinal: int, *, warm_history: Optional[List[Dict[str, Any]]] = None,
    require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False, request_id: str | None = None,
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
        split_user_originated_turn, user_originated_turn_view)
    from agent.message_content import flatten_message_text
    from agent.session_persistence import _is_ephemeral_scaffolding

    expected_active_ids = self.get_active_message_ids(session_id)
    stored = self.get_messages_as_conversation(session_id, include_row_ids=True)
    # Live replay (the pre-request repair, a resume) merges a stored ``user;user`` pair — an ask whose turn
    # ended with no reply, then the next ask — into ONE turn while both rows stay stored. Address turns on
    # that same repaired projection or the warm history is a turn short of the transcript forever
    # (#115493); the merged turn keeps the first row's identity, so the rewind starts at that row.
    durable = self.get_messages_as_conversation(session_id, include_row_ids=True, repair_alternation=True)
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
    # The in-txn payload pin compares against the STORED row, which for a merged turn holds only the
    # first ask, never the merged text the live views carry.
    stored_view = next(
        (user_originated_turn_view(m) for m in stored if m.get("_row_id") == target_row_id), None)
    if stored_view is None:
        raise RuntimeError(_HISTORY_CHANGED)
    from uuid import uuid4
    request_id = request_id or uuid4().hex
    try:
        receipt_capable = callable(getattr(self, "get_rewind_receipt", None))
        mutation_kwargs = {
            "preserve_compaction_handoff": scaffold is not None,
            "expected_active_ids": expected_active_ids,
            # Pin against the STORED row (upstream #115493 semantics); receipt-bearing stores
            # compare on the same projection the durable row stores.
            "expected_target_content": _comparison_content(stored_view) if receipt_capable else stored_view.get("content"),
        }
        # SQLite retains its historical primitive signature. Receipt-bearing
        # stores opt in explicitly; this is not a duck-typed SQLite extension.
        if receipt_capable:
            mutation_kwargs["request_id"] = request_id
        result = self.rewind_to_message(session_id, target_row_id, **mutation_kwargs)
    except ValueError as exc:  # target vanished / changed role under us: same class of failure as out-of-range
        raise RewindTargetUnavailableError(str(exc)) from exc
    except Exception:
        lookup = getattr(self, "get_rewind_receipt", None)
        if callable(lookup):
            try:
                receipt = lookup(request_id)
            except Exception as receipt_error:
                raise RewindIndeterminateError(request_id) from receipt_error
            if isinstance(receipt, Mapping):
                result = {
                    "request_id": request_id,
                    "rewound_count": int(receipt["retired_count"]),
                    "replacement_message_id": receipt.get("replacement_message_id"),
                }
            else:
                raise
        else:
            raise
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
        rewound_count=int(result.get("rewound_count", 0)), turns_undone=len(durable_user) - user_ordinal,
        request_id=request_id)


class SessionRewindMixin:
    """Legacy SQLite mixin retaining the public SessionDB method."""

    def rewind_user_turn(self, session_id: str, user_ordinal: int, **kwargs: Any) -> RewindOutcome:
        return rewind_user_turn(self, session_id, user_ordinal, **kwargs)
