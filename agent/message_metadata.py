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
# Set by the agent flush on a batch row (never a live dict) whose ``message_uid`` the agent itself saw durable in
# this session (written by an earlier flush, or loaded from the store with its marker). Only such a row, with no
# row address or snapshot, may be skipped as a re-insert of an ACTIVE uid; a bare copied uid proves nothing.
RESTORED_FROM_STORE = "_restored_from_store"
REPAIR_BOOKKEEPING_FIELDS = frozenset({DB_ROW_SNAPSHOT, CANONICAL_ROW, RESTORED_FROM_STORE})
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
    {"timestamp", "display_kind", "display_metadata", "_row_id", "_submit_row_session_id",
     MERGED_TURN_PREFIX, MESSAGE_UID, ABSORBED_MESSAGE_UIDS, TOOL_CALL_UIDS, TOOL_CALL_UID,
     # The alternation repair's row counts: an in-place compaction reads them off the live dict.
     MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS, RETIRED_DURABLE_ROWS}
) | REPAIR_BOOKKEEPING_FIELDS


# Connector data-use requirement on a tool message; persisted inside ``display_metadata``, never sent to a model.
DATA_USE = "data_use"


def _set_display_metadata_key(message: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set *key* in *message*'s ``display_metadata`` as a new dict (a non-dict value starts empty)."""
    metadata = message.get("display_metadata")
    message["display_metadata"] = {**(metadata if isinstance(metadata, dict) else {}), key: value}


def _display_metadata_str(message: Any, key: str) -> Optional[str]:
    """*message*'s ``display_metadata[key]`` when it is a string, else None."""
    metadata = message.get("display_metadata") if isinstance(message, Mapping) else None
    value = metadata.get(key) if isinstance(metadata, Mapping) else None
    return value if isinstance(value, str) else None


def data_use_requirement(messages: Any) -> frozenset[str]:
    """The union of ``data_use`` over *messages*: what the model receiving them must guarantee."""
    return frozenset(r for m in messages or () if isinstance(m, Mapping) for r in m.get(DATA_USE) or ())


def stamp_data_use(message: MutableMapping[str, Any], requires: list[str]) -> None:
    """Mark a tool message with its requirement. The ``display_metadata`` copy is the durable one: that
    column already rides every writer, branch and compaction copy, and replay lifts it back (SessionDB)."""
    if not requires:
        return
    message[DATA_USE] = list(requires)
    _set_display_metadata_key(message, DATA_USE, list(requires))


# What the user typed on a turn a ``pre_llm_call`` hook vetoed, kept inside ``display_metadata``. The row's
# ``content`` is the withheld placeholder, so every reader that hands transcript text to a model (wire
# replay, compaction, session_search, memory providers, background review) only ever sees that; transcript
# views render this key instead (agent/turn_context.py::_withhold_blocked_prompt).
WITHHELD_PROMPT = "withheld_prompt"
PRE_LLM_BLOCK_WITHHELD_PROMPT = "[Message withheld: a pre_llm_call policy hook blocked it.]"


# The ask of a multimodal turn (the text of its persisted parts: image-path hints, not ``[screenshot]``), kept inside
# ``display_metadata`` when the turn's injected context (recalled memory, ``pre_llm_call`` and gateway notes)
# is appended to ``content`` as a text part: the DB stores that content as one flattened string, so nothing
# else can tell the ask from the injection.
TYPED_ASK = "typed_ask"


def stamp_typed_ask(message: MutableMapping[str, Any], text: str) -> None:
    """Record *text* as the ask ``previous_message`` reads for *message*."""
    _set_display_metadata_key(message, TYPED_ASK, text)


def drop_typed_ask(message: MutableMapping[str, Any]) -> None:
    """Forget the typed-ask stamp on a row whose content no longer is that one ask (a fold, a redaction)."""
    metadata = message.get("display_metadata")
    if isinstance(metadata, dict) and TYPED_ASK in metadata:
        message["display_metadata"] = {k: v for k, v in metadata.items() if k != TYPED_ASK}


def typed_ask_text(message: Any) -> Optional[str]:
    """The typed ask stamped on *message*, else None."""
    return _display_metadata_str(message, TYPED_ASK)


def withheld_prompt_text(message: Any) -> Optional[str]:
    """The typed text a transcript view shows for a vetoed user row, else None."""
    return _display_metadata_str(message, WITHHELD_PROMPT)


def typed_prompt_content(content: Any, message: Any) -> Any:
    """*content* (a rewind's live view of *message*) with the withheld placeholder swapped back for what
    the user typed: ``/retry`` re-submits the real ask so the hook decides again, and ``/undo`` restores it
    to the composer. Never the placeholder as a new prompt."""
    typed = withheld_prompt_text(message)
    if typed is None:
        return content
    if isinstance(content, str):
        return content.replace(PRE_LLM_BLOCK_WITHHELD_PROMPT, typed)
    if isinstance(content, list):
        return [{**part, "text": part["text"].replace(PRE_LLM_BLOCK_WITHHELD_PROMPT, typed)}
                if isinstance(part, dict) and isinstance(part.get("text"), str) else part for part in content]
    return content


def without_withheld_prompts(messages: Any) -> Any:
    """*messages* with the vetoed text dropped from each row's ``display_metadata`` (copies only where
    one is present), for hand-offs that leave the host with whole row dicts (memory providers)."""
    if not isinstance(messages, list) or not any(withheld_prompt_text(m) is not None for m in messages):
        return messages
    return [
        {**m, "display_metadata": {k: v for k, v in m["display_metadata"].items() if k != WITHHELD_PROMPT}}
        if withheld_prompt_text(m) is not None else m
        for m in messages
    ]


def lift_data_use(message: MutableMapping[str, Any]) -> None:
    """Restore a replayed row's ``data_use`` key from its durable ``display_metadata`` copy."""
    metadata = message.get("display_metadata")
    if isinstance(metadata, dict) and metadata.get(DATA_USE):
        message[DATA_USE] = list(metadata[DATA_USE])


# Reasoning tokens the provider billed for an assistant turn (``usage.*_tokens_details.reasoning_tokens``),
# kept inside ``display_metadata`` so it rides every writer, branch and compaction copy without a column.
# Encrypted reasoning (Codex/xAI ``encrypted_content``, Claude ``signature``) cannot be sized from its
# bytes, but the provider bills its replay as input at almost exactly this count (live A/B: gpt-5.6-sol
# 2,765 reported -> +2,767 billed; claude-opus-5.5 3,959 -> +3,959 while its summary text was ~1,016).
REPLAYED_REASONING_TOKENS = "reasoning_tokens"
# The same count on a per-call request copy: ``build_api_messages`` drops ``display_metadata`` from every
# outgoing copy, so the request estimate would lose it. ``assemble_api_request`` pops this key from the
# request once it is measured; it never reaches a provider.
REQUEST_REASONING_TOKENS = "_replayed_reasoning_tokens"


def stamp_reasoning_tokens(message: MutableMapping[str, Any], tokens: Any) -> None:
    """Record *tokens* as the provider-reported reasoning size of assistant *message*. An unknown or
    zero count clears any earlier stamp: it priced reasoning the row no longer carries."""
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
        _set_display_metadata_key(message, REPLAYED_REASONING_TOKENS, tokens)
        return
    metadata = message.get("display_metadata")
    if isinstance(metadata, dict) and REPLAYED_REASONING_TOKENS in metadata:
        message["display_metadata"] = {k: v for k, v in metadata.items() if k != REPLAYED_REASONING_TOKENS}


def stamped_reasoning_tokens(message: Any) -> int:
    """The provider-reported reasoning size stamped on *message* (stored row or request copy), else 0."""
    if not isinstance(message, Mapping):
        return 0
    value = message.get(REQUEST_REASONING_TOKENS)
    if value is None:
        metadata = message.get("display_metadata")
        value = metadata.get(REPLAYED_REASONING_TOKENS) if isinstance(metadata, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def drop_request_reasoning_tokens(messages: Any) -> None:
    """Pop the request-only reasoning count from every request copy in *messages* (in place)."""
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, dict):
            message.pop(REQUEST_REASONING_TOKENS, None)


# Replay carriers the preflight estimate prices at 0, exactly as the compaction tail walk does (opaque signatures,
# duplicate reasoning copies, ordered sidecars); a mismatch between the two estimates loops compaction.
_UNPRICED_CORE_FIELDS = frozenset({"_anthropic_content_blocks", "reasoning_details", "extra_content",
                                   "bedrock_content_blocks", DATA_USE, REQUEST_REASONING_TOKENS})


def unpriced_replay_fields() -> frozenset[str]:
    """The core unpriced carriers plus every message carrier key a registered profile declares
    (``ProviderProfile.message_carrier_*``). Resolve once per message, never per key."""
    from agent.reasoning_carriers import declared_carrier_keys

    opaque, text = declared_carrier_keys()
    return _UNPRICED_CORE_FIELDS.union(opaque, text)


def without_persistence_fields(msg: Mapping[str, Any]) -> Mapping[str, Any]:
    """*msg* itself when it carries no persistence-only field, else a shallow copy without them."""
    if PERSISTENCE_ONLY_MESSAGE_FIELDS.isdisjoint(msg):
        return msg
    return {k: v for k, v in msg.items() if k not in PERSISTENCE_ONLY_MESSAGE_FIELDS}


def mint_uid() -> str:
    """A fresh durable id (message or tool-call occurrence): random, never derived from content."""
    return uuid4().hex


def uid_list(value: Any) -> list[str]:
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


# Producer-owned provenance for the compressor's standalone in-flight-task restatement. It rides in
# display_metadata (persisted by SessionDB, popped from every wire copy) so gateway replay can send the
# row unstamped, as the compaction turn did; a human pasting the header text carries no flag.
INFLIGHT_TASK_RESTATEMENT = "compaction_inflight_task_restatement"


def mark_inflight_restatement(msg: MutableMapping[str, Any]) -> None:
    _set_display_metadata_key(msg, INFLIGHT_TASK_RESTATEMENT, True)


def scaffolding_replay_fields(msg: Mapping[str, Any]) -> dict:
    """The compaction-provenance fields replay keeps (summary flag, restatement flag), else ``{}``. A proactive
    prune re-persists the replayed view, so a dropped flag would get the row stamped on the next replay."""
    metadata = msg.get("display_metadata")
    fields: dict = {"_compressed_summary": True} if msg.get("_compressed_summary") else {}
    if isinstance(metadata, dict) and metadata.get(INFLIGHT_TASK_RESTATEMENT):
        fields["display_metadata"] = {INFLIGHT_TASK_RESTATEMENT: True}
    return fields


def typed_ask_fields(msg: Mapping[str, Any]) -> Optional[dict]:
    """``{TYPED_ASK: ...}`` when *msg* carries the stamp, else None: the only key a row rewrite that backfills
    the stamp writes (merge-patched, so other writers' keys stay as committed)."""
    typed = typed_ask_text(msg)
    return None if typed is None else {TYPED_ASK: typed}


def replay_metadata_fields(msg: Mapping[str, Any]) -> dict:
    """``scaffolding_replay_fields`` plus the typed-ask stamp (``previous_message`` must not read a
    multimodal turn's flattened injected context back)."""
    fields = scaffolding_replay_fields(msg)
    if (typed := typed_ask_text(msg)) is not None:
        fields["display_metadata"] = {**fields.get("display_metadata", {}), TYPED_ASK: typed}
    return fields


def is_compaction_scaffolding(msg: Mapping[str, Any]) -> bool:
    """Compaction sent its summary and restatement rows unstamped; a replay stamp would break the cached prefix."""
    return bool(scaffolding_replay_fields(msg))


def is_inflight_restatement(msg: Mapping[str, Any]) -> bool:
    """The compressor's standalone in-flight-task restatement row, which no transcript may paint.

    ``mark_inflight_restatement`` stamps it on the copied ``role=user`` row whose content leads with the
    ``[STILL IN PROGRESS — …]`` header: model-facing scaffolding that keeps the active task across the
    handoff boundary. The flag, not the header text, is the signal — a human pasting that header carries
    no ``display_metadata`` and renders as their own turn (#155495).
    """
    metadata = msg.get("display_metadata")
    return isinstance(metadata, dict) and bool(metadata.get(INFLIGHT_TASK_RESTATEMENT))


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


def _is_empty_content(content: Any) -> bool:
    """Nothing to fold: any part (an image too) is content, unlike ``is_content_blank``'s text-only test."""
    return content is None or content == [] or (isinstance(content, str) and not content.strip())


def record_absorbed_message(
    survivor: MutableMapping[str, Any], dropped: Mapping[str, Any], *, dropped_leads: bool = False,
) -> None:
    """Merge-witness bookkeeping for every host fold of *dropped* into *survivor*.

    The composite keeps the uid of the constituent whose text comes first and records every other
    constituent's uid in ``_absorbed_message_uids`` (text order, no repeats). By default the survivor's
    text leads; with *dropped_leads* the dropped dict's text was put first (the real user anchor folded
    into a scaffolding turn), so its uid becomes the survivor's and the survivor's former uid is recorded.
    A dict without a uid (unflushed, scaffolding, engine-authored) contributes nothing; an empty result
    leaves the survivor untouched. The survivor's typed-ask stamp goes when *dropped* brought content (the row
    now holds more than that one ask, so ``previous_message`` reads it as before stamps existed); an empty
    turn leaves the content, and so the stamp, as they were.
    """
    if not _is_empty_content(dropped.get("content")):
        drop_typed_ask(survivor)
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


def per_occurrence_tool_call_uids(uids: Mapping[str, Any], tool_calls: list[Mapping[str, Any]]) -> dict:
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


def tool_call_uid_from_history(messages: list[dict], tool_index: int, owners: dict) -> Optional[str]:
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
