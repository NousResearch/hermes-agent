"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record and timeline display, not
# provider-visible message content. The request builder strips them from every
# outgoing copy and the token estimator ignores them: one set, so an estimate
# never prices bytes the provider never receives (an edit's inline_diff in
# display_metadata is ~9KB and would trigger premature compaction).
# Transcript-repair bookkeeping riding on batch rows / live dicts (agent/transcript_repair.py): the
# stored-row CAS digest and the durable row adopted onto the live dict. Never transcript payload.
DB_ROW_SNAPSHOT = "_db_row_snapshot"
CANONICAL_ROW = "_canonical_row"
REPAIR_BOOKKEEPING_FIELDS = frozenset({DB_ROW_SNAPSHOT, CANONICAL_ROW})
# Unanswered text a merged user row held before the current turn was absorbed
# (agent_runtime_helpers._merge_consecutive_users); the persist override keeps it.
# It repeats the row's own content, so pricing it would double the estimate.
MERGED_TURN_PREFIX = "_merged_turn_prefix"
# The durable per-message id (``messages.message_uid``): minted once at the row's first insert and kept by
# every host copy of that logical message (in-place compaction generation, rotation child, concurrent-tail
# clone, replace re-issue, rewrite in place). Unlike ``_row_id`` (a physical id re-issued per copy, opt-in on
# restore) it is restored unconditionally, so context engines can key on it across restarts and boundaries.
MESSAGE_UID = "message_uid"
# The merge witness on a consecutive-user merge survivor: the ``message_uid`` of each absorbed row, in
# absorption order (the uid sibling of ``_absorbed_row_ids``; persisted as ``messages.absorbed_message_uids``).
# The survivor keeps the FIRST constituent's uid.
ABSORBED_MESSAGE_UIDS = "_absorbed_message_uids"
# Per-occurrence tool-call identity. Provider tool-call ids repeat (Hermes mints deterministic ``call_<12hex>``
# ids for identical calls, and models reuse ids), so an assistant row carries ``{provider id: uid}`` for its
# ``tool_calls`` (``messages.tool_call_uids``) and its tool-result rows carry the matching uid
# (``messages.tool_call_uid``). The provider-facing ``id`` is untouched; these never reach the wire.
TOOL_CALL_UIDS = "_tool_call_uids"
TOOL_CALL_UID = "_tool_call_uid"
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset(
    {"timestamp", "display_kind", "display_metadata", "_row_id", MERGED_TURN_PREFIX, MESSAGE_UID,
     ABSORBED_MESSAGE_UIDS, TOOL_CALL_UIDS, TOOL_CALL_UID}
) | REPAIR_BOOKKEEPING_FIELDS


def index_tool_call_uids(index: MutableMapping[str, str], assistant: Any) -> None:
    """Register an assistant dict's ``_tool_call_uids`` under every pairing-id variant of its tool calls, so a
    later tool-result row can be resolved by any spelling of its ``tool_call_id``."""
    from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants

    uids = assistant.get(TOOL_CALL_UIDS) if isinstance(assistant, dict) else None
    if not isinstance(uids, dict) or not uids:
        return
    for tc in assistant.get("tool_calls") or ():
        uid = uids.get(coalesce_tool_call_id(tc))
        if isinstance(uid, str) and uid:
            for variant in tool_call_id_variants(tc):
                index[variant] = uid


def resolve_tool_call_uid(index: MutableMapping[str, str], tool_call_id: Any) -> Optional[str]:
    """The uid an indexed assistant dict minted for ``tool_call_id`` (any variant), else ``None``."""
    from agent.message_sanitization import tool_result_id_variants

    if not index or not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    for variant in tool_result_id_variants(tool_call_id):
        uid = index.get(variant)
        if uid:
            return uid
    return None


def tool_call_uid_from_history(messages: Any, tool_index: int) -> Optional[str]:
    """Resolve a tool-result dict's uid from the nearest preceding assistant dict in ``messages`` that
    named its ``tool_call_id`` (the cross-flush case: the assistant row landed in an earlier batch)."""
    if not isinstance(messages, list) or not (0 <= tool_index < len(messages)):
        return None
    tool_msg = messages[tool_index]
    tool_call_id = tool_msg.get("tool_call_id") if isinstance(tool_msg, dict) else None
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    for prior in reversed(messages[:tool_index]):
        if isinstance(prior, dict) and prior.get("role") == "assistant" and prior.get(TOOL_CALL_UIDS):
            index: dict = {}
            index_tool_call_uids(index, prior)
            uid = resolve_tool_call_uid(index, tool_call_id)
            if uid:
                return uid
        elif isinstance(prior, dict) and prior.get("role") == "user":
            return None  # a tool result never pairs across a user turn
    return None

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message
