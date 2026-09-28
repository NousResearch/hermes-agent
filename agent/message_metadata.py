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
# Live-only bookkeeping on a consecutive-user merge survivor: the ``message_uid`` of each absorbed row, in
# absorption order (the uid sibling of ``_absorbed_row_ids``). The survivor keeps the FIRST constituent's uid.
ABSORBED_MESSAGE_UIDS = "_absorbed_message_uids"
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset(
    {"timestamp", "display_kind", "display_metadata", "_row_id", MERGED_TURN_PREFIX, MESSAGE_UID, ABSORBED_MESSAGE_UIDS}
) | REPAIR_BOOKKEEPING_FIELDS

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
