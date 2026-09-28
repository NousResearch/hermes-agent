"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from uuid import uuid4
from typing import Any, List, Mapping, MutableMapping, Optional, TypeVar


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


def message_uid_or_none(msg: Mapping[str, Any]) -> Optional[str]:
    """The dict's ``message_uid`` when it is a non-empty string, else ``None`` (never coerced: an int or a
    blank would be a bug upstream of the write, not an identity)."""
    uid = msg.get(MESSAGE_UID)
    return uid if isinstance(uid, str) and uid else None


def stamp_message_uid(msg: MutableMapping[str, Any]) -> str:
    """The dict's ``message_uid``, minting one (``uuid4().hex``) when it carries none.

    Minted ONCE per logical message, at its first insert, and stamped on the caller's dict so every later
    insert of that dict (compaction generation, rotation handoff, replace) writes the same uid. Never
    derived from content, timestamp or tool-call ids: two distinct messages may share all three.
    """
    uid = message_uid_or_none(msg)
    if uid is None:
        uid = msg[MESSAGE_UID] = uuid4().hex
    return uid


def _absorbed_uids(msg: Mapping[str, Any]) -> List[str]:
    return [u for u in (msg.get(ABSORBED_MESSAGE_UIDS) or ()) if isinstance(u, str) and u]


_IDENTITY_FIELD_TYPES = ((MESSAGE_UID, str), (ABSORBED_MESSAGE_UIDS, list), (TOOL_CALL_UIDS, dict), (TOOL_CALL_UID, str))


def copy_identity_fields(src: Mapping[str, Any], dst: MutableMapping[str, Any]) -> None:
    """Copy the non-empty identity fields (uid, merge witness, tool-call uids) from *src* onto *dst*: the
    live dict to its flush row, and the committed row back onto the live dict."""
    for key, kind in _IDENTITY_FIELD_TYPES:
        value = src.get(key)
        if isinstance(value, kind) and value:
            dst[key] = kind(value) if kind is not str else value


def message_identity(msg: MutableMapping[str, Any]) -> dict:
    """The identity fields a new row copied from *msg* must carry, minting *msg*'s uid first when it has none:
    a branch/seed copy writes fresh rows from the live dicts the new session keeps using, and a row without
    them would restore with a different uid than the live dict carries."""
    stamp_message_uid(msg)
    identity: dict = {}
    copy_identity_fields(msg, identity)
    return identity


def record_absorbed_message(
    survivor: MutableMapping[str, Any], dropped: Mapping[str, Any], *, dropped_leads: bool = False,
) -> None:
    """Merge-witness bookkeeping for every host fold of *dropped* into *survivor*.

    The composite keeps the uid of the constituent whose text comes first and records every other
    constituent's uid in ``_absorbed_message_uids`` (text order, no repeats). By default the survivor's
    text leads; with *dropped_leads* the dropped dict's text was put first (the real user anchor folded
    into a scaffolding turn), so its uid becomes the survivor's and the survivor's former uid is recorded.
    A dict without a uid (unflushed, scaffolding, engine-authored) contributes nothing; an empty result
    leaves the survivor untouched.
    """
    survivor_uid = message_uid_or_none(survivor)
    dropped_uid = message_uid_or_none(dropped)
    if dropped_leads and dropped_uid:
        survivor[MESSAGE_UID] = dropped_uid
        ordered = _absorbed_uids(dropped) + ([survivor_uid] if survivor_uid else []) + _absorbed_uids(survivor)
    else:
        ordered = _absorbed_uids(survivor) + ([dropped_uid] if dropped_uid else []) + _absorbed_uids(dropped)
    absorbed: List[str] = []
    for uid in ordered:
        if uid != survivor.get(MESSAGE_UID) and uid not in absorbed:
            absorbed.append(uid)
    if absorbed:
        survivor[ABSORBED_MESSAGE_UIDS] = absorbed


def _named_tool_call_variants(assistant: Any) -> List[str]:
    """Every pairing-id variant of every tool call an assistant dict names."""
    from agent.message_sanitization import tool_call_id_variants

    if not isinstance(assistant, dict):
        return []
    return [variant for tc in assistant.get("tool_calls") or () for variant in tool_call_id_variants(tc)]


def index_tool_call_uids(index: MutableMapping[str, str], assistant: Any) -> None:
    """Register an assistant dict's ``_tool_call_uids`` under every pairing-id variant of its tool calls, so a
    later tool-result row can be resolved by any spelling of its ``tool_call_id``. Provider ids repeat, so
    every id this assistant names first shadows an earlier occurrence's entry: a result pairs with the
    NEAREST preceding call, and a call without a uid (a legacy row) pairs its result with nothing."""
    from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants

    for variant in _named_tool_call_variants(assistant):
        index.pop(variant, None)
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


def tool_call_uid_from_history(messages: Any, tool_index: int, owners: Optional[dict] = None) -> Optional[str]:
    """Resolve a tool-result dict's uid from the nearest preceding assistant dict in ``messages`` that
    named its ``tool_call_id`` (the cross-flush case: the assistant row landed in an earlier batch).
    ``owners`` memoizes each assistant's (named variants, uid index) across one flush's results, so K
    parallel results of one assistant cost O(K) variant work instead of O(K^2)."""
    if not isinstance(messages, list) or not (0 <= tool_index < len(messages)):
        return None
    tool_msg = messages[tool_index]
    tool_call_id = tool_msg.get("tool_call_id") if isinstance(tool_msg, dict) else None
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    from agent.message_sanitization import tool_result_id_variants

    result_variants = set(tool_result_id_variants(tool_call_id))
    for prior_index in range(tool_index - 1, -1, -1):  # no messages[:i] copy: this runs per flushed result
        prior = messages[prior_index]
        if not isinstance(prior, dict):
            continue
        if prior.get("role") == "user":
            return None  # a tool result never pairs across a user turn
        if prior.get("role") != "assistant":
            continue
        entry = owners.get(id(prior)) if owners is not None else None
        if entry is None:
            index: dict = {}
            index_tool_call_uids(index, prior)
            entry = (frozenset(_named_tool_call_variants(prior)), index)
            if owners is not None:
                owners[id(prior)] = entry
        named, index = entry
        if result_variants.isdisjoint(named):
            continue
        # The nearest assistant naming this id owns the result: its uid, or none if it has no map (legacy).
        return resolve_tool_call_uid(index, tool_call_id)
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
