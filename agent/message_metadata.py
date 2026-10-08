"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from collections import Counter
from time import time as wall_time
from uuid import uuid4
from typing import Any, List, Mapping, MutableMapping, Optional, TypeVar

from agent.conversation_compression_archive import MERGED_DURABLE_ROWS, RETIRED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS


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
    # Membership is the real contract, NOT the leading underscore: the chat-completions transport happens
    # to sweep underscore keys, but turn_context.py pops this set from every outgoing copy and a strict
    # backend 400s on any key it does not know.
    {"timestamp", "display_kind", "display_metadata", "_row_id", "_topic_id", "_submit_row_session_id",
     MERGED_TURN_PREFIX, MESSAGE_UID, ABSORBED_MESSAGE_UIDS, TOOL_CALL_UIDS, TOOL_CALL_UID,
     # The alternation repair's row counts: an in-place compaction reads them off the live dict.
     MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS, RETIRED_DURABLE_ROWS}
) | REPAIR_BOOKKEEPING_FIELDS


def without_persistence_fields(msg: Mapping[str, Any]) -> Mapping[str, Any]:
    """*msg* itself when it carries no persistence-only field, else a shallow copy without them."""
    if PERSISTENCE_ONLY_MESSAGE_FIELDS.isdisjoint(msg):
        return msg
    return {k: v for k, v in msg.items() if k not in PERSISTENCE_ONLY_MESSAGE_FIELDS}


def mint_uid() -> str:
    """A fresh durable id (message or tool-call occurrence): random, never derived from content."""
    return uuid4().hex


def uid_list(value: Any) -> List[str]:
    """The unique non-empty string uids of a live list, in order; anything else is ``[]``."""
    return list(dict.fromkeys(u for u in (value if isinstance(value, list) else ()) if isinstance(u, str) and u))


def message_uid_or_none(msg: Mapping[str, Any]) -> Optional[str]:
    """The dict's ``message_uid`` when it is a non-empty string, else ``None`` (never coerced: an int or a
    blank would be a bug upstream of the write, not an identity)."""
    uid = msg.get(MESSAGE_UID)
    return uid if isinstance(uid, str) and uid else None


def stamp_message_uid(msg: MutableMapping[str, Any]) -> str:
    """The dict's ``message_uid``, minting one (``mint_uid``) when it carries none.

    Minted ONCE per logical message, at its first insert, and stamped on the caller's dict so every later
    insert of that dict (compaction generation, rotation handoff, replace) writes the same uid. Never
    derived from content, timestamp or tool-call ids: two distinct messages may share all three.
    """
    uid = message_uid_or_none(msg)
    if uid is None:
        uid = msg[MESSAGE_UID] = mint_uid()
    return uid


_IDENTITY_FIELD_TYPES = ((MESSAGE_UID, str), (ABSORBED_MESSAGE_UIDS, list), (TOOL_CALL_UIDS, dict), (TOOL_CALL_UID, str))


def copy_identity_fields(src: Mapping[str, Any], dst: MutableMapping[str, Any]) -> None:
    """Copy the non-empty identity fields (uid, merge witness, tool-call uids) from *src* onto *dst*: the
    live dict to its flush row, and the committed row back onto the live dict."""
    for key, kind in _IDENTITY_FIELD_TYPES:
        value = src.get(key)
        if isinstance(value, kind) and value:
            dst[key] = kind(value)


def message_identity(msg: MutableMapping[str, Any], *, with_tool_uids: bool = False) -> dict:
    """The identity fields a new row copied from *msg* must carry, minting *msg*'s uid first when it has none:
    a branch/seed copy writes fresh rows from the live dicts the new session keeps using, and a row without
    them would restore with a different uid than the live dict carries. ``with_tool_uids`` only for a copy
    that also carries ``tool_calls`` / ``tool_call_id``; a uid map on a row without its calls pairs nothing."""
    stamp_message_uid(msg)
    identity: dict = {}
    copy_identity_fields(msg, identity)
    if not with_tool_uids:
        identity.pop(TOOL_CALL_UIDS, None)
        identity.pop(TOOL_CALL_UID, None)
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
    own = uid_list(survivor.get(ABSORBED_MESSAGE_UIDS))
    theirs = uid_list(dropped.get(ABSORBED_MESSAGE_UIDS))
    if dropped_leads and dropped_uid:
        survivor[MESSAGE_UID] = dropped_uid
        ordered = theirs + ([survivor_uid] if survivor_uid else []) + own
    else:
        ordered = own + ([dropped_uid] if dropped_uid else []) + theirs
    if absorbed := [uid for uid in dict.fromkeys(ordered) if uid != survivor.get(MESSAGE_UID)]:
        survivor[ABSORBED_MESSAGE_UIDS] = absorbed


def merge_tool_call_uids(into: Mapping[str, Any], extra: Mapping[str, Any]) -> dict:
    """Union two ``_tool_call_uids`` maps without losing an occurrence. Folding two assistant turns that
    reuse a provider id (llama.cpp-style constant ids) leaves both calls in ``tool_calls``; the id then maps
    to every occurrence's uid, in call order, as a list. Provider ids themselves are never rewritten."""
    merged = dict(into)
    for call_id, uid in extra.items():
        merged[call_id] = _uid_occurrences(merged[call_id]) + _uid_occurrences(uid) if call_id in merged else uid
    return merged


def _uid_occurrences(value: Any) -> list:
    return list(value) if isinstance(value, list) else [value]


def per_occurrence_tool_call_uids(uids: Mapping[str, Any], tool_calls: List[Mapping[str, Any]]) -> dict:
    """*uids* with every provider id this row names more than once spelled out as one uid per occurrence. A
    single response that repeats an id shares ONE uid (its results carry it, so every call stays paired);
    before a fold appends another turn's occurrences, the shared uid must fill each of this row's slots or
    the appended uids would align with the wrong calls."""
    from agent.message_sanitization import coalesce_tool_call_id

    counts = Counter(call_id for tc in tool_calls if (call_id := coalesce_tool_call_id(tc)))
    expanded = dict(uids)
    for call_id, count in counts.items():
        if count > 1 and isinstance(uid := uids.get(call_id), str):
            expanded[call_id] = [uid] * count
    return expanded


def index_tool_call_uids(index: MutableMapping[str, str], assistant: Mapping[str, Any]) -> frozenset:
    """Register an assistant dict's ``_tool_call_uids`` under every pairing-id variant of its tool calls, so a
    later tool-result row can be resolved by any spelling of its ``tool_call_id``. Provider ids repeat, so
    every id this assistant names first shadows an earlier occurrence's entry: a result pairs with the
    NEAREST preceding call, and a call without a uid (a legacy row) pairs its result with nothing. A
    provider id repeated inside one row (two folded turns) maps each call to its own occurrence's uid.
    Returns every pairing-id variant the assistant names."""
    from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants

    calls = [(tc, tool_call_id_variants(tc)) for tc in assistant.get("tool_calls") or ()]
    named = frozenset(variant for _, variants in calls for variant in variants)
    for variant in named:
        index.pop(variant, None)
    uids = assistant.get(TOOL_CALL_UIDS)
    if not isinstance(uids, dict) or not uids:
        return named
    occurrence: dict = {}  # provider id -> calls seen so far: a list value holds one uid per occurrence
    for tc, variants in calls:
        call_id = coalesce_tool_call_id(tc)
        uid = uids.get(call_id)
        if isinstance(uid, list):
            nth = occurrence[call_id] = occurrence.get(call_id, -1) + 1
            uid = uid[nth] if nth < len(uid) else None
        if isinstance(uid, str) and uid:
            for variant in variants:
                index[variant] = uid
    return named


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


def tool_call_uid_from_history(messages: List[dict], tool_index: int, owners: dict) -> Optional[str]:
    """Resolve a tool-result dict's uid from the nearest preceding assistant dict in ``messages`` that
    named its ``tool_call_id`` (the cross-flush case: the assistant row landed in an earlier batch).
    ``owners`` memoizes each assistant's (named variants, uid index) across one flush's results, so K
    parallel results of one assistant cost O(K) variant work instead of O(K^2)."""
    tool_call_id = messages[tool_index].get("tool_call_id")
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
        if (entry := owners.get(id(prior))) is None:
            index: dict = {}
            entry = owners[id(prior)] = (index_tool_call_uids(index, prior), index)
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
