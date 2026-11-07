"""Transcript (message) capability contract.

One repository owns a session's transcript: appends, replace intents, the user-turn rewind, in-place compaction and prune rewrites, the replay and display projections, the compression-failure cooldown, reactions, and the probes the gateway guards read. Reads return insertion order (the stored row id), never timestamp order. Times are epoch-second floats as the store keeps them. Reads of an unknown session return typed empties; ``ValueError`` is for input shape only.

Wire shapes stay the store's: ``TranscriptRow.as_wire_row()`` is the stored row dict ``read_transcript`` callers serialize, and ``ConversationMessage.as_openai_message()`` is the replay dict the agent loop feeds the model. The projection policy applied on a loaded replay list (context sanitizer, background-review strip, stale tool-call-marker blanking, role-alternation repair) and the user-turn policy the rewind runs are application code every backend applies unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Protocol, Sequence, runtime_checkable

from ..core.conventions import RowVisibility
from ..core.errors import PersistenceError
from ..core.identity import JsonValue, MessageId, SessionId

__all__ = [
    "CONTENT_JSON_PREFIX",
    "DB_PERSISTED_MARKER_KEY",
    "QUEUED_PROMPT_METADATA_KEY",
    "AnchoredView",
    "AnchoredWindow",
    "BatchAppendProgress",
    "BatchAppendResult",
    "CompactionResult",
    "CompressionCooldown",
    "ConversationMessage",
    "ConversationPeer",
    "ConversationRequest",
    "ConversationRevision",
    "CooldownSnapshot",
    "DelegationDelivery",
    "HeldCoverage",
    "HeldMessage",
    "MarkerPurgeReport",
    "MessageRecord",
    "MessageRepository",
    "MessageStorageState",
    "NewMessage",
    "PrUrlHit",
    "PromptPage",
    "PruneRewriteResult",
    "PrunedRowChange",
    "Reaction",
    "ReplaceMode",
    "ReplaceResult",
    "ResumeConversations",
    "ResumeTooLarge",
    "RewindPlan",
    "RewindTargetUnavailable",
    "RowDisposition",
    "RowOutcome",
    "RowStamp",
    "RowUids",
    "SessionTimeline",
    "TimelineEntry",
    "TranscriptRead",
    "TranscriptRow",
    "TranscriptView",
    "UnseenReaction",
    "UserTurnRewind",
    "Watermark",
    "batch_progress",
    "stamp_live",
]

#: Prefix of a stored ``content`` cell holding JSON-encoded structured content; NUL cannot collide with text.
CONTENT_JSON_PREFIX = "\x00json:"

#: The born-durable marker every replay dict carries (``agent.context_compressor._DB_PERSISTED_MARKER``).
DB_PERSISTED_MARKER_KEY = "_db_persisted"

#: The ``display_metadata`` key marking a user row accepted while a turn was running and not yet drained.
QUEUED_PROMPT_METADATA_KEY = "_queued_prompt"


class TranscriptView(Enum):
    """Which rows a read covers: ``ACTIVE`` the live rows, ``DISPLAY`` live plus compaction-archived rows deduped to one per logical message, ``ALL`` every row including rewound ones. A display read (``DISPLAY``, ``display_count``, the timeline and prompt pages, the resume display) skips a tool row when a compaction-archived tool row of the same session, uid and timestamp holds an earlier display slot, so a compacted tool result shows once, as its archived original."""

    ACTIVE = "active"
    DISPLAY = "display"
    ALL = "all"


class ReplaceMode(Enum):
    """How ``replace_transcript`` treats the rows it replaces.

    ``DESTRUCTIVE`` deletes every stored row first; ``PRESERVE_EXISTING_ARCHIVE`` deletes only the live rows; ``ARCHIVE_DROPPED`` keeps the leading live rows the replacement repeats in place and marks the rest of the live rows rewound instead of deleting them.
    """

    DESTRUCTIVE = "destructive"
    PRESERVE_EXISTING_ARCHIVE = "preserve_existing_archive"
    ARCHIVE_DROPPED = "archive_dropped"


def _live_or_column(message: Mapping[str, Any], live_key: str, column: str) -> Any:
    """The live key when the dict has it, else the column name an import payload carries."""
    return message.get(live_key) if live_key in message else message.get(column)


def _count(value: Any) -> int:
    """A positive int as itself; anything else (a bool included) is 0."""
    return value if type(value) is int and value > 0 else 0


def _json_text(value: Any, empty: Any) -> Any:
    """A value an import or export may carry as JSON text, decoded; undecodable text is ``empty``."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return empty


@dataclass(frozen=True, slots=True)
class NewMessage:
    """One transcript row to write, with the message-dict fields the store persists.

    ``timestamp`` ``None`` takes the store's clock; ``message_uid`` ``None`` is minted at insert. ``repair_target`` and ``db_row_snapshot`` name the stored row a batch row already has and its ownership version, so ``append_batch`` repairs or adopts that row instead of inserting a duplicate; ``compact`` reads them too, to tell a replacement message that carries a row newer than its watermark (see ``MessageRepository.compact``). ``append_batch`` folds ``data_use`` into ``display_metadata`` as the store keeps it; a single ``append`` drops it. String ``content`` reads back as written (lone surrogates replaced), including a string that starts with the store's reserved ``\\x00json:`` prefix, which the store escapes on write; structured content reads back as its JSON round trip.

    ``restored_from_store`` marks a row the writer already saw durable in this session (its uid came from a stored row). ``append_batch`` does not insert such a row when it has no ``repair_target`` and no ``db_row_snapshot``, its uid names an active row of the session, and it replays exactly like that row: the same owned payload, ``api_content`` and ``codex_reasoning_items``, the stored content compared as a load returns it, and the stored token count, summary flag and uid maps standing in where the row lacks them. Its outcome is then ``REPAIRED`` with the stamp of the newest active row of its role carrying that uid. An edited copy, one with a new replay sidecar, or one whose uid names no active row is inserted as usual; nothing stores the mark.

    ``merged_durable_rows``, ``unnamed_durable_rows`` and ``absorbed_row_ids`` are the alternation repair's markers, named as on ``HeldMessage``; they name no column and no read returns them: the user rows merged into the message (two or more; 0 for none), the further durable rows it stands for on a reload without row ids (rows the repair dropped behind it or folded into it), and the row ids merged into it. A ``compact`` without a holding coverage proof widens the carried tail's rewind by them for each carried row without a ``repair_target``.
    """

    role: str
    content: JsonValue
    timestamp: Optional[float] = None
    tool_calls: tuple[JsonValue, ...] = ()
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    display_kind: Optional[str] = None
    compressed_summary: bool = False
    api_content: Optional[str] = None
    display_metadata: Optional[dict[str, JsonValue]] = None
    platform_message_id: Optional[str] = None
    observed: bool = False
    effect_disposition: Optional[str] = None
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_details: Optional[JsonValue] = None
    codex_reasoning_items: Optional[JsonValue] = None
    codex_message_items: Optional[JsonValue] = None
    repair_target: Optional[MessageId] = None
    db_row_snapshot: Optional[str] = None
    message_uid: Optional[str] = None
    absorbed_message_uids: tuple[str, ...] = ()
    tool_call_uids: Optional[Mapping[str, str]] = None
    tool_call_uid: Optional[str] = None
    token_count: Optional[int] = None
    data_use: tuple[str, ...] = ()
    merged_durable_rows: int = 0
    unnamed_durable_rows: int = 0
    absorbed_row_ids: tuple[MessageId, ...] = ()
    restored_from_store: bool = False

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("NewMessage.role must be non-empty")
        if any(type(n) is not int or n < 0 for n in (self.merged_durable_rows, self.unnamed_durable_rows)):
            raise ValueError("NewMessage.merged_durable_rows and unnamed_durable_rows must be non-negative ints")
        if not isinstance(self.absorbed_row_ids, tuple) or any(type(i) is not int or i <= 0 for i in self.absorbed_row_ids):
            raise ValueError("NewMessage.absorbed_row_ids must be a tuple of positive MessageIds")

    @classmethod
    def from_live(cls, message: Mapping[str, Any]) -> NewMessage:
        """One live message dict as the row a transcript write takes: every key the store reads off a message dict. A missing role is ``unknown``; ``platform_message_id`` falls back to ``message_id``; the uid keys read the live key (``_tool_call_uids``, ``_tool_call_uid``, ``_absorbed_message_uids``) when the dict has it, else the column name; ``tool_calls`` and the uid keys given as JSON text are decoded. ``_row_id`` and ``_db_row_snapshot`` become ``repair_target`` and ``db_row_snapshot``, which ``append_batch`` and ``compact`` read. The repair's markers read ``_merged_durable_rows``, ``_unnamed_durable_rows`` and ``_absorbed_row_ids``; a marker of the wrong type is left unset. ``restored_from_store`` is set only by ``_restored_from_store`` holding ``True``."""
        get = message.get
        tool_calls = _json_text(get("tool_calls"), [])
        tool_call_uids = _json_text(_live_or_column(message, "_tool_call_uids", "tool_call_uids"), {})
        absorbed = _json_text(_live_or_column(message, "_absorbed_message_uids", "absorbed_message_uids"), [])
        tool_call_uid = _live_or_column(message, "_tool_call_uid", "tool_call_uid")
        row_id, snapshot, uid = get("_row_id"), get("_db_row_snapshot"), get("message_uid")
        absorbed_ids = get("_absorbed_row_ids")
        return cls(
            role=get("role", "unknown"), content=get("content"), timestamp=get("timestamp"),
            tool_calls=tuple(tool_calls) if isinstance(tool_calls, (list, tuple)) else (),
            tool_name=get("tool_name"), tool_call_id=get("tool_call_id"), display_kind=get("display_kind"),
            compressed_summary=bool(get("_compressed_summary")), api_content=get("api_content"),
            display_metadata=get("display_metadata"), platform_message_id=get("platform_message_id") or get("message_id") or None,
            observed=bool(get("observed")), effect_disposition=get("effect_disposition"), finish_reason=get("finish_reason"),
            reasoning=get("reasoning"), reasoning_content=get("reasoning_content"), reasoning_details=get("reasoning_details"),
            codex_reasoning_items=get("codex_reasoning_items"), codex_message_items=get("codex_message_items"),
            repair_target=MessageId(row_id) if isinstance(row_id, int) and not isinstance(row_id, bool) else None,
            db_row_snapshot=snapshot if isinstance(snapshot, str) else None,
            message_uid=uid if isinstance(uid, str) and uid else None,
            absorbed_message_uids=tuple(absorbed) if isinstance(absorbed, (list, tuple)) else (),
            tool_call_uids=tool_call_uids if isinstance(tool_call_uids, Mapping) else None,
            tool_call_uid=tool_call_uid if isinstance(tool_call_uid, str) and tool_call_uid else None,
            token_count=get("token_count"), data_use=tuple(get("data_use") or ()),
            merged_durable_rows=_count(get("_merged_durable_rows")), unnamed_durable_rows=_count(get("_unnamed_durable_rows")),
            absorbed_row_ids=tuple(MessageId(i) for i in absorbed_ids if _count(i)) if isinstance(absorbed_ids, (list, tuple)) else (),
            restored_from_store=get("_restored_from_store") is True)


@dataclass(frozen=True, slots=True)
class RowStamp:
    """The row state a write gave one message: its id, stored timestamp, ownership version and durable uids (the state the store stamps on a written message dict). ``db_row_snapshot`` is ``None`` on a row the write kept without stamping it (the kept prefix of an ``ARCHIVE_DROPPED`` replace)."""

    id: MessageId
    timestamp: float
    db_row_snapshot: Optional[str]
    message_uid: str
    tool_call_uids: Optional[Mapping[str, str]] = None
    tool_call_uid: Optional[str] = None


class RowDisposition(Enum):
    """What ``append_batch`` did with one batch row."""

    APPENDED = "appended"
    REPAIRED = "repaired"
    ADOPTED = "adopted"


@dataclass(frozen=True, slots=True)
class RowOutcome:
    """Per-row result of ``append_batch`` in batch order: the row's stamp, what happened to it, and ``canonical_row``, the durable row (or lent metadata) the caller takes into its live dict, ``None`` when there is nothing to take."""

    id: MessageId
    disposition: RowDisposition
    timestamp: float
    db_row_snapshot: str
    canonical_row: Optional[dict[str, JsonValue]] = None
    message_uid: Optional[str] = None
    tool_call_uids: Optional[Mapping[str, str]] = None
    tool_call_uid: Optional[str] = None

    def as_row_annotation(self) -> dict[str, JsonValue]:
        """The keys the store stamps on a written batch dict, ``None`` meaning the key is absent."""
        return {
            "_row_id": int(self.id),
            "timestamp": self.timestamp,
            "_db_row_snapshot": self.db_row_snapshot,
            "_canonical_row": self.canonical_row,
            "message_uid": self.message_uid,
            "_tool_call_uids": None if self.tool_call_uids is None else dict(self.tool_call_uids),
            "_tool_call_uid": self.tool_call_uid,
        }


@dataclass(frozen=True, slots=True)
class BatchAppendResult:
    """Result of ``append_batch``: one ``RowOutcome`` per batch row; ``inserted`` counts the fresh rows (the store's return value)."""

    outcomes: tuple[RowOutcome, ...]

    @property
    def inserted(self) -> int:
        return sum(1 for o in self.outcomes if o.disposition is RowDisposition.APPENDED)


@dataclass(frozen=True, slots=True)
class RowUids:
    """The durable uids one batch row carries after a write: its ``message_uid`` and tool-call uids, ``None`` where unset."""

    message_uid: Optional[str] = None
    tool_call_uids: Optional[Mapping[str, str]] = None
    tool_call_uid: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BatchAppendProgress:
    """What a failed ``append_batch`` left behind, attached to the error it raised (:func:`batch_progress`): ``outcomes`` for the rows of the chunks that committed, a prefix of the batch in batch order (empty when none did), and ``uids`` per batch row, the uids the row was given or the write had assigned to it, which a retry of the same row writes again."""

    outcomes: tuple[RowOutcome, ...]
    uids: tuple[RowUids, ...]


def batch_progress(error: BaseException) -> Optional[BatchAppendProgress]:
    """The progress a failed ``append_batch`` attached to the error it raised; ``None`` for any other error."""
    progress = getattr(error, "batch_progress", None)
    return progress if isinstance(progress, BatchAppendProgress) else None


@dataclass(frozen=True, slots=True)
class HeldMessage:
    """One message a compactor holds in memory, as the store compares it against its row.

    Identity is ``(role, content, tool_call_id, tool_calls)`` through the loaded-view lens; ``row_id`` is the id the holder believes the message has, ``timestamp`` the exact stored epoch, ``persisted`` whether it still carries the born-durable marker. ``db_row_snapshot`` (the row's ownership version) wins over the identity compare when given; ``absorbed_row_ids`` names rows the alternation repair merged into it.

    The rest names, for a ``compact`` coverage proof, the durable rows a message stands for on a reload without row ids: ``merged_durable_rows`` counts the user rows the repair merged into it (two or more; 0 for none), and ``retired_durable_rows`` records each row the repair dropped or folded behind it by role, content, ``tool_call_id`` and ``tool_call_ids`` (the ids of its tool calls), ``own_row`` marking the record of the message's own row taken before a fold rewrote its text. A message with ``retired_row`` is one such record offered on its own, matched to one active row by those fields.
    """

    role: str
    content: JsonValue
    tool_call_id: Optional[str] = None
    tool_calls: tuple[JsonValue, ...] = ()
    timestamp: Optional[float] = None
    row_id: Optional[MessageId] = None
    persisted: bool = False
    db_row_snapshot: Optional[str] = None
    message_uid: Optional[str] = None
    absorbed_row_ids: tuple[MessageId, ...] = ()
    merged_durable_rows: int = 0
    retired_durable_rows: tuple[HeldMessage, ...] = ()
    tool_call_ids: tuple[JsonValue, ...] = ()
    own_row: bool = False
    retired_row: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("HeldMessage.role must be a non-empty string")
        if self.tool_call_id is not None and not isinstance(self.tool_call_id, str):
            raise ValueError("HeldMessage.tool_call_id must be a string or None")
        if not isinstance(self.tool_calls, tuple):
            raise ValueError("HeldMessage.tool_calls must be a tuple")
        if self.timestamp is not None and (type(self.timestamp) not in (int, float)):
            raise ValueError("HeldMessage.timestamp must be an epoch number or None")
        if self.row_id is not None and (type(self.row_id) is not int or self.row_id <= 0):
            raise ValueError("HeldMessage.row_id must be a positive MessageId or None")
        if type(self.persisted) is not bool:
            raise ValueError("HeldMessage.persisted must be a bool")
        if type(self.merged_durable_rows) is not int or self.merged_durable_rows < 0:
            raise ValueError("HeldMessage.merged_durable_rows must be a non-negative int")
        if not isinstance(self.retired_durable_rows, tuple) or any(
                not isinstance(row, HeldMessage) for row in self.retired_durable_rows):
            raise ValueError("HeldMessage.retired_durable_rows must be a tuple of HeldMessage")
        if not isinstance(self.tool_call_ids, tuple):
            raise ValueError("HeldMessage.tool_call_ids must be a tuple")
        if type(self.own_row) is not bool or type(self.retired_row) is not bool:
            raise ValueError("HeldMessage.own_row and retired_row must be bools")


@dataclass(frozen=True, slots=True)
class HeldCoverage:
    """The coverage proof a compactor offers ``compact``: ``covered_ids`` are the durable ids it summarized, ``unresolved`` the held messages it could not name by id. ``HeldCoverage()`` covers nothing."""

    covered_ids: tuple[MessageId, ...] = ()
    unresolved: tuple[HeldMessage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.covered_ids, tuple) or any(type(i) is not int or i <= 0 for i in self.covered_ids):
            raise ValueError("HeldCoverage.covered_ids must be a tuple of positive MessageIds")
        if not isinstance(self.unresolved, tuple) or any(not isinstance(m, HeldMessage) for m in self.unresolved):
            raise ValueError("HeldCoverage.unresolved must be a tuple of HeldMessage")


@dataclass(frozen=True, slots=True)
class ReplaceResult:
    """Result of ``replace_transcript``: one ``RowStamp`` per replacement row in replacement order (an ``ARCHIVE_DROPPED`` kept prefix keeps its stored ids)."""

    rows: tuple[RowStamp, ...]

    @property
    def survivor_ids(self) -> tuple[MessageId, ...]:
        return tuple(row.id for row in self.rows)


def stamp_live(messages: Sequence[MutableMapping[str, Any]], result: ReplaceResult | BatchAppendResult | BatchAppendProgress | None) -> None:
    """Write the row state a transcript write gave each message back onto the live dict it was built from (``NewMessage.from_live``), as the store stamps the dicts its own writers take; ``result`` is ``None`` for any other write that raised, and the :func:`batch_progress` of an ``append_batch`` that raised.

    Every dict first loses ``_canonical_row``. An ``append_batch`` row takes each key of ``RowOutcome.as_row_annotation()`` that is set. A failed ``append_batch`` gives every row the uids of its ``RowUids`` that are set (``message_uid``, ``_tool_call_uids``, ``_tool_call_uid``) and stamps the rows that committed as a result's. A ``replace_transcript`` row takes its id and a non-empty uid; an inserted row (one with an ownership version) also its stored timestamp, ownership version and tool-call uids, while a row the replace kept keeps its other keys. Any other failed write stamps nothing more; uids the store minted on rows it reached before failing are not carried back.
    """
    for message in messages:
        message.pop("_canonical_row", None)
    if isinstance(result, BatchAppendProgress):
        for message, uids in zip(messages, result.uids):
            for key, value in (("message_uid", uids.message_uid), ("_tool_call_uid", uids.tool_call_uid),
                               ("_tool_call_uids", None if uids.tool_call_uids is None else dict(uids.tool_call_uids))):
                if value is not None:
                    message[key] = value
        result = BatchAppendResult(outcomes=result.outcomes)
    if isinstance(result, BatchAppendResult):
        for message, outcome in zip(messages, result.outcomes):
            message.update((key, value) for key, value in outcome.as_row_annotation().items() if value is not None)
        return
    for message, row in zip(messages, result.rows if result is not None else ()):
        message["_row_id"] = int(row.id)
        if row.message_uid:
            message["message_uid"] = row.message_uid
        if row.db_row_snapshot is None:
            continue
        message.update(timestamp=row.timestamp, _db_row_snapshot=row.db_row_snapshot)
        if row.tool_call_uids is not None:
            message["_tool_call_uids"] = dict(row.tool_call_uids)
        if row.tool_call_uid is not None:
            message["_tool_call_uid"] = row.tool_call_uid


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """What ``compact`` committed: the session's active row count afterwards and one ``RowStamp`` per replacement row in replacement order."""

    active_count: int
    rows: tuple[RowStamp, ...]

    @property
    def row_ids(self) -> tuple[MessageId, ...]:
        return tuple(row.id for row in self.rows)

    @property
    def row_snapshots(self) -> tuple[Optional[str], ...]:
        return tuple(row.db_row_snapshot for row in self.rows)


@dataclass(frozen=True, slots=True)
class PrunedRowChange:
    """One row a prune edited: ``original``, the held message that names the stored row and that its compare-and-set checks; ``replacement``, the row after the prune; and ``before``, the same message as written before it, whose columns that ``replacement`` changes are the ones overwritten. Where ``original`` and ``before`` both carry a field, the store names and compares the row by ``original``'s."""

    original: HeldMessage
    replacement: NewMessage
    before: NewMessage


@dataclass(frozen=True, slots=True)
class PruneRewriteResult:
    """What ``rewrite_pruned_rows`` committed: the rows rewritten and, per change, the rewritten row's new ownership version (``None`` for a row left unchanged)."""

    rewritten: int
    snapshots: tuple[Optional[str], ...]


@dataclass(frozen=True, slots=True)
class Watermark:
    """The newest active row id at compression start; ``last_active`` ``None`` means no active row existed."""

    last_active: Optional[MessageId]


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """One stored transcript row as a typed record: the row's columns with identity and visibility."""

    id: MessageId
    role: str
    content: JsonValue
    timestamp: float
    tool_calls: tuple[JsonValue, ...]
    tool_name: Optional[str]
    tool_call_id: Optional[str]
    display_kind: Optional[str]
    compressed_summary: bool
    visibility: RowVisibility
    api_content: Optional[str] = None
    display_metadata: Optional[dict[str, JsonValue]] = None
    platform_message_id: Optional[str] = None
    observed: bool = False
    effect_disposition: Optional[str] = None
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_details: Optional[JsonValue] = None
    codex_reasoning_items: Optional[JsonValue] = None
    codex_message_items: Optional[JsonValue] = None
    message_uid: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MessageStorageState:
    """The owning session and visibility of one stored row."""

    session_id: SessionId
    visibility: RowVisibility


@dataclass(frozen=True, slots=True)
class Reaction:
    """One author's reaction on a stored row: the emoji, the author (``"user"`` or ``"agent"``), when it was set, and whether it was already announced."""

    emoji: str
    author: str
    at: float
    seen: bool = False


@dataclass(frozen=True, slots=True)
class UnseenReaction:
    """One reaction ``take_unseen_reactions`` surfaced: the reacted row, its role, the emoji, and the row's content when it is a string (else ``""``)."""

    row_id: MessageId
    role: str
    emoji: str
    text: str


@dataclass(frozen=True, slots=True)
class CompressionCooldown:
    """An unexpired compression-failure cooldown: the deadline and the recorded error."""

    until: float
    error: Optional[str]


@dataclass(frozen=True, slots=True)
class CooldownSnapshot:
    """The stored cooldown columns without expiry filtering, for restore after a cancelled compression. ``present`` is whether the session row exists; an absent snapshot carries no fields."""

    present: bool
    until: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.present and (self.until is not None or self.error is not None):
            raise ValueError("an absent CooldownSnapshot cannot carry stored fields")


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    """How ``read_conversation`` shapes the replay projection: the ``view``, whether to read the compression lineage root to tip (``include_lineage``), whether to run the role-alternation repair (which also keeps the ``_compressed_summary`` marker), and whether to stamp ``_row_id``."""

    view: TranscriptView = TranscriptView.ACTIVE
    include_lineage: bool = False
    repair_alternation: bool = False
    include_row_ids: bool = False


# Replay dict keys after role, content and the born-durable marker, in emission order, with the record field each maps to.
# The dict is serialised into provider requests, so the order is part of the contract; ``_tool_call_uid`` follows the per-row columns, where a result row's uid resolved at read time goes.
_CONVERSATION_KEYS: tuple[tuple[str, str], ...] = (
    ("_db_row_snapshot", "db_row_snapshot"),
    ("_row_id", "row_id"),
    ("message_uid", "message_uid"),
    ("_tool_call_uids", "tool_call_uids"),
    ("_absorbed_message_uids", "absorbed_message_uids"),
    ("api_content", "api_content"),
    ("display_kind", "display_kind"),
    ("display_metadata", "display_metadata"),
    ("data_use", "data_use"),
    ("_compressed_summary", "compressed_summary"),
    ("timestamp", "timestamp"),
    ("tool_call_id", "tool_call_id"),
    ("tool_name", "tool_name"),
    ("effect_disposition", "effect_disposition"),
    ("tool_calls", "tool_calls"),
    ("message_id", "message_id"),
    ("observed", "observed"),
    ("finish_reason", "finish_reason"),
    ("reasoning", "reasoning"),
    ("reasoning_content", "reasoning_content"),
    ("reasoning_details", "reasoning_details"),
    ("codex_reasoning_items", "codex_reasoning_items"),
    ("codex_message_items", "codex_message_items"),
    ("_tool_call_uid", "tool_call_uid"),
    ("_absorbed_row_ids", "absorbed_row_ids"),
    ("_merged_turn_prefix", "merged_turn_prefix"),
    ("_merged_durable_rows", "merged_durable_rows"),
    ("_unnamed_durable_rows", "unnamed_durable_rows"),
    ("_retired_durable_rows", "retired_durable_rows"),
)
_CONVERSATION_FIELD_BY_KEY: dict[str, str] = dict(_CONVERSATION_KEYS)
# The keys emitted between a stored row's identity columns and a read-time ``_tool_call_uid``.
_AFTER_IDENTITY_KEYS = frozenset(key for key, _ in _CONVERSATION_KEYS[_CONVERSATION_KEYS.index(("_absorbed_message_uids", "absorbed_message_uids")):_CONVERSATION_KEYS.index(("_tool_call_uid", "tool_call_uid"))])
# Keys emitted whenever they are not None rather than when truthy, so a stored "[]" replays as [].
_PRESENT_WHEN_NOT_NONE = frozenset({"reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items"})


def _same(value: Any) -> Any:
    return value


# Keys whose wire value and record value differ in type; every other key passes through.
_TO_WIRE: dict[str, Callable[[Any], Any]] = {
    "_row_id": int,
    "tool_calls": list,
    "_absorbed_row_ids": lambda ids: [int(v) for v in ids],
    "_absorbed_message_uids": list,
    "data_use": list,
}
_FROM_WIRE: dict[str, Callable[[Any], Any]] = {
    "_row_id": lambda v: MessageId(int(v)),
    "tool_calls": tuple,
    "_absorbed_row_ids": lambda ids: tuple(MessageId(int(v)) for v in ids),
    "_absorbed_message_uids": tuple,
    "data_use": tuple,
    "_compressed_summary": bool,
    "observed": bool,
}


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One message of the replay projection: the OpenAI-format dict the store loads, as a record.

    ``as_openai_message()`` emits ``role``, ``content`` and the born-durable marker (unless ``db_persisted`` is False), then every other key in a fixed order when its value is truthy (``_tool_call_uid`` after ``_tool_call_uids`` when the row stored it, else after the per-row columns, where it is resolved at read time; ``from_openai_message`` keeps which of the two the dict carried) (``reasoning_content`` and the three JSON sidecars whenever they are not ``None``). ``from_openai_message`` is its inverse and refuses unknown keys. ``timestamp`` is the stored epoch verbatim; ``message_id`` is the platform's own id.
    """

    role: str
    content: JsonValue
    db_row_snapshot: Optional[str] = None
    row_id: Optional[MessageId] = None
    message_uid: Optional[str] = None
    tool_call_uids: Optional[JsonValue] = None
    tool_call_uid: Optional[str] = None
    absorbed_message_uids: tuple[str, ...] = ()
    api_content: Optional[str] = None
    display_kind: Optional[str] = None
    display_metadata: Optional[dict[str, JsonValue]] = None
    data_use: tuple[str, ...] = ()
    compressed_summary: bool = False
    timestamp: Optional[float] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    effect_disposition: Optional[str] = None
    tool_calls: Optional[tuple[JsonValue, ...]] = None
    message_id: Optional[str] = None
    observed: bool = False
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_details: Optional[JsonValue] = None
    codex_reasoning_items: Optional[JsonValue] = None
    codex_message_items: Optional[JsonValue] = None
    absorbed_row_ids: tuple[MessageId, ...] = ()
    merged_turn_prefix: Optional[str] = None
    merged_durable_rows: Optional[int] = None
    unnamed_durable_rows: Optional[int] = None
    retired_durable_rows: Optional[JsonValue] = None
    db_persisted: bool = True
    _tool_call_uid_stored: bool = field(default=False, repr=False, compare=False)

    def as_openai_message(self) -> dict[str, JsonValue]:
        out: dict[str, JsonValue] = {"role": self.role, "content": self.content}
        if self.db_persisted:
            out[DB_PERSISTED_MARKER_KEY] = True
        for key, field_name in _CONVERSATION_KEYS:
            if key == "_tool_call_uid" and self._tool_call_uid_stored:
                continue
            value: Any = getattr(self, field_name)
            if key == "_absorbed_message_uids" and self._tool_call_uid_stored and self.tool_call_uid:
                out["_tool_call_uid"] = self.tool_call_uid
            if key in _PRESENT_WHEN_NOT_NONE:
                if value is not None:
                    out[key] = value
                continue
            if not value:
                continue
            out[key] = _TO_WIRE.get(key, _same)(value)
        return out

    @classmethod
    def from_openai_message(cls, message: Mapping[str, Any]) -> ConversationMessage:
        """The inverse of ``as_openai_message``; an unknown key or a marker other than True is a ``ValueError``, and an absent marker reads ``db_persisted=False``."""
        if DB_PERSISTED_MARKER_KEY in message and message[DB_PERSISTED_MARKER_KEY] is not True:
            raise ValueError("a present born-durable marker must be True")
        fields: dict[str, Any] = {"db_persisted": DB_PERSISTED_MARKER_KEY in message}
        keys = list(message)
        if "_tool_call_uid" in message:
            uid_at = keys.index("_tool_call_uid")
            fields["_tool_call_uid_stored"] = not any(key in _AFTER_IDENTITY_KEYS for key in keys[:uid_at])
        for key, value in message.items():
            if key in ("role", "content", DB_PERSISTED_MARKER_KEY):
                continue
            field_name = _CONVERSATION_FIELD_BY_KEY.get(key)
            if field_name is None:
                raise ValueError(f"unknown replay projection key {key!r}")
            fields[field_name] = _FROM_WIRE.get(key, _same)(value)
        return cls(role=str(message["role"]), content=message.get("content"), **fields)


@dataclass(frozen=True, slots=True)
class ResumeConversations:
    """The two projections a resume loads from one read: ``model``, the tip's active rows alternation-repaired with row ids, and ``display``, the lineage's deduped display history with row ids."""

    model: tuple[ConversationMessage, ...]
    display: tuple[ConversationMessage, ...]


@dataclass(frozen=True, slots=True)
class ConversationRevision:
    """A fingerprint of what a live replay read would load: the lineage ids, the active row count, the highest active id and the summed per-session ``transcript_revision``."""

    lineage_ids: tuple[SessionId, ...]
    active_count: int
    max_active_id: int
    transcript_revision: int

    def as_tuple(self) -> tuple[tuple[SessionId, ...], int, int, int]:
        return (self.lineage_ids, self.active_count, self.max_active_id, self.transcript_revision)


class ResumeTooLarge(PersistenceError, ValueError):
    """The resume scope holds more rows than the caller's limit; ``message_count`` is the bounded count (at most ``limit + 1``) and ``scope`` names what was counted."""

    def __init__(self, message_count: int, limit: int, scope: str = "across its lineage") -> None:
        self.message_count, self.limit = message_count, limit
        self.scope = scope
        super().__init__(
            f"This session is too long to reload safely ({message_count} messages; limit {limit}). "
            "Start a fresh chat and keep a copy with the dashboard Sessions page's Export action or "
            "`hermes sessions export --format md --session-id <id>` (neither is capped), or raise the limit "
            "with `hermes config set sessions.max_resume_messages 0`."
        )


class RewindTargetUnavailable(PersistenceError, ValueError):
    """The requested user turn is not a rewindable target of the active transcript: no user turns, an ordinal past the newest one, a row that is not user-originated, or a plain turn where the caller required a compaction carrier."""


@dataclass(frozen=True, slots=True)
class AnchoredWindow:
    """Up to ``window`` rows either side of an anchor, ascending, anchor included; ``before`` and ``after`` count the rows strictly around it."""

    window: tuple[MessageRecord, ...]
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class AnchoredView:
    """The anchored window filtered to the kept roles (the anchor always kept) plus up to ``bookend`` kept-role rows with non-empty content from the session's start and end outside the window."""

    window: tuple[MessageRecord, ...]
    before: int
    after: int
    bookend_start: tuple[MessageRecord, ...]
    bookend_end: tuple[MessageRecord, ...]


@dataclass(frozen=True, slots=True)
class PrUrlHit:
    """One tool row whose content mentions a pull request: the owning session and the stored content."""

    session_id: SessionId
    content: str


@dataclass(frozen=True, slots=True)
class ConversationPeer:
    """The routing peer a conversation generation belongs to: ``(session_key, source)``."""

    session_key: str
    source: str


@dataclass(frozen=True, slots=True)
class DelegationDelivery:
    """A detached-delegation result delivered between turns: the row content and its display metadata, whose ``delegation_id`` (required) and ``delivery_notice`` key the delivery."""

    content: str
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.metadata.get("delegation_id"):
            raise ValueError("Delegation delivery requires a stable delegation_id")

    @property
    def identity(self) -> tuple[JsonValue, JsonValue]:
        """The idempotence key: ``(delegation_id, delivery_notice)`` with an absent notice read as ``""``."""
        return (self.metadata["delegation_id"], self.metadata.get("delivery_notice", ""))


@dataclass(frozen=True, slots=True)
class MarkerPurgeReport:
    """Outcome of ``purge_stale_tool_call_markers``: the matching rows (cleared unless ``dry_run``), each once in the store's unordered SELECT order, and the path of the snapshot taken before the write, ``None`` when none was taken."""

    dry_run: bool
    rows_affected: int
    row_ids: tuple[MessageId, ...]
    backup_path: Optional[str]


@dataclass(frozen=True, slots=True)
class RewindPlan:
    """Every read and check of a user-turn rewind without the write: the target row, the active ids and target content the write pins, the composite carrier's handoff scaffold, the prefix to install (warm and durable), the rewound turn's live view and text, the durable user-turn indices and the resolved ordinal."""

    target_row_id: MessageId
    expected_active_ids: tuple[MessageId, ...]
    expected_target_content: JsonValue
    scaffold: Optional[Mapping[str, Any]]
    prefix: tuple[Mapping[str, Any], ...]
    durable_prefix: tuple[Mapping[str, Any], ...]
    live_view: Mapping[str, Any]
    live_text: Optional[str]
    durable_user: tuple[int, ...]
    user_ordinal: int


@dataclass(frozen=True, slots=True)
class UserTurnRewind:
    """Outcome of ``rewind_user_turn``: the history to install (warm when ``warm_history`` was given, else durable), the rewound turn's live view and its text, the rows deactivated and the user turns undone."""

    prefix: tuple[dict[str, Any], ...]
    live_view: dict[str, Any]
    live_text: str
    rewound_count: int
    turns_undone: int


@dataclass(frozen=True, slots=True)
class TranscriptRead:
    """What ``read_transcript`` reads: the ``view``, a page (``limit`` ``None`` reads to the end, ``offset`` skips, ``latest`` pages back from the newest end and still returns chronological order), keyset paging after ``after_id``, and ``include_lineage`` to read the compression lineage root to tip.

    ``after_id`` combines with neither ``latest`` or ``offset``, nor the ``DISPLAY`` view, nor ``include_lineage``.
    """

    view: TranscriptView = TranscriptView.ACTIVE
    limit: Optional[int] = None
    offset: int = 0
    latest: bool = False
    after_id: Optional[MessageId] = None
    include_lineage: bool = False

    def __post_init__(self) -> None:
        if self.after_id is not None and (self.latest or self.offset):
            raise ValueError("after_id is incompatible with latest/offset paging")
        if self.after_id is not None and self.view is TranscriptView.DISPLAY:
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        if self.after_id is not None and self.include_lineage:
            raise ValueError("after_id is incompatible with include_ancestors (merged-lineage reads use offset paging)")


# The stored row dict's key order (the ``messages`` columns minus the two display-index columns); ``_compressed_summary`` follows last and only when true.
_WIRE_ROW_KEYS = (
    "id", "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition",
    "timestamp", "token_count", "finish_reason", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "platform_message_id", "observed", "active", "compacted",
    "api_content", "display_kind", "display_metadata", "message_uid", "absorbed_message_uids", "tool_call_uids",
    "tool_call_uid",
)


@dataclass(frozen=True, slots=True)
class TranscriptRow:
    """One stored row as the store serves it: ``content``, ``tool_calls`` and ``display_metadata`` decoded (reactions included), the JSON sidecars and identity columns as their stored text.

    ``as_wire_row()`` emits the stored row dict: every column in table order, the ``(active, compacted)`` pair derived from ``visibility``, and ``_compressed_summary: True`` last on a summary row.
    """

    id: MessageId
    session_id: SessionId
    role: str
    content: JsonValue
    timestamp: float
    visibility: RowVisibility
    tool_call_id: Optional[str] = None
    tool_calls: Optional[JsonValue] = None
    tool_name: Optional[str] = None
    effect_disposition: Optional[str] = None
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning_details: Optional[str] = None
    codex_reasoning_items: Optional[str] = None
    codex_message_items: Optional[str] = None
    platform_message_id: Optional[str] = None
    observed: bool = False
    compressed_summary: bool = False
    api_content: Optional[str] = None
    display_kind: Optional[str] = None
    display_metadata: Optional[dict[str, JsonValue]] = None
    message_uid: Optional[str] = None
    absorbed_message_uids: Optional[str] = None
    tool_call_uids: Optional[str] = None
    tool_call_uid: Optional[str] = None

    def as_wire_row(self) -> dict[str, JsonValue]:
        active, compacted = {
            RowVisibility.ACTIVE: (1, 0),
            RowVisibility.COMPACTION_ARCHIVED: (0, 1),
            RowVisibility.REWOUND: (0, 0),
        }[self.visibility]
        derived: dict[str, JsonValue] = {
            "id": int(self.id),
            "session_id": str(self.session_id),
            "observed": 1 if self.observed else 0,
            "active": active,
            "compacted": compacted,
        }
        out: dict[str, JsonValue] = {key: derived[key] if key in derived else getattr(self, key) for key in _WIRE_ROW_KEYS}
        if self.compressed_summary:
            out["_compressed_summary"] = True
        return out


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """One human prompt of the session timeline: its first row id, its preview and its stored timestamp."""

    row_id: MessageId
    preview: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class SessionTimeline:
    """A page of the session's prompts in chronological order with its pagination: the page bounds, how many entries it returned, the total, and ``next_cursor`` (the last entry's sort id) when more follow."""

    entries: tuple[TimelineEntry, ...]
    limit: int
    after_row_id: int
    returned: int
    total: int
    has_more: bool
    next_cursor: Optional[int]


@dataclass(frozen=True, slots=True)
class PromptPage:
    """A display page starting at one human prompt, oldest first, with its position in the display set (``offset`` rows before it, ``total`` rows in all)."""

    messages: tuple[TranscriptRow, ...]
    row_id: MessageId
    limit: int
    returned: int
    offset: int
    total: int
    has_older: bool
    has_newer: bool


@runtime_checkable
class MessageRepository(Protocol):
    """Transcript operations; scope first, each verb the store's transaction for the operation it names."""

    def append(self, session_id: SessionId, message: NewMessage, *, compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        """Append one row and bump the session counters in one transaction; returns its id. With ``turn_lease_holder`` the turn lease on the session's conversation must still name that holder (an expired row that still names it is renewed for ``turn_lease_ttl_seconds``), else ``TurnLeaseLost``; ``None`` or an empty holder appends unfenced. ``compression_lock_holder`` fences nothing: a live compression lock never blocks an append. A session ended by compression raises ``CompressionClosed`` and a missing session row ``SessionRowMissing``."""
        ...

    def append_batch(self, session_id: SessionId, messages: Sequence[NewMessage], *, compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None, turn_lease_ttl_seconds: float = 300.0) -> BatchAppendResult:
        """Append the batch in one transaction (one per ``chunk_rows`` rows when given) with the same guards as ``append``, resolving each row that names a stored row to a repair or adoption instead of a duplicate insert. Returns one ``RowOutcome`` per row; an empty batch writes nothing. A chunked batch stamps its tool-call uids over the whole batch first, so a result pairs with its call across a chunk boundary.

        On a failure the error raised carries a :class:`BatchAppendProgress` (:func:`batch_progress`): the outcomes of the chunks that committed before it and every row's uids, so a caller stamps the committed rows and a retry of the batch repairs them instead of appending them again."""
        ...

    def append_delegation_delivery(self, session_id: SessionId, delivery: DelegationDelivery) -> MessageId:
        """Record a detached delegation result once as a ``user`` row (display kind ``hidden`` when ``presentation_suppressed`` is set, else ``async_delegation_complete``): a row with the same identity in the session or its compression ancestors is returned instead. A new insert refuses under a live turn lease (``TurnLeaseLost``) or on a session ended by compression (``CompressionClosed``)."""
        ...

    def append_completion_note(self, session_id: SessionId, admission_id: str, content: str, display_kind: str, metadata: Mapping[str, JsonValue]) -> MessageId:
        """Record an unviewed completion's note once as a ``user`` row, keyed by ``admission_id`` (stored as ``metadata.completion_admission``) over the compression lineage, with the guards of ``append_delegation_delivery``."""
        ...

    def replace_transcript(self, session_id: SessionId, messages: Sequence[NewMessage], mode: ReplaceMode, *, reject_active_turn_lease: bool = False) -> ReplaceResult:
        """Replace the live transcript with ``messages`` per ``mode`` in one transaction, counters reset to the replacement. With ``reject_active_turn_lease`` a live compression lock (``CompressionInProgress``) or live turn lease (``TurnLeaseLost``) refuses and a stale one is removed; a session ended by compression raises ``CompressionClosed``."""
        ...

    def rewind_user_turn(self, session_id: SessionId, user_ordinal: int, *, warm_history: Optional[Sequence[Mapping[str, Any]]] = None, require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False) -> UserTurnRewind:
        """Deactivate the ``user_ordinal``-th user-originated turn of the active transcript and every later active row (negative ordinals count from the newest), keeping a composite carrier's handoff scaffold as the new head. Runs the ``plan_user_turn_rewind`` checks, then one write that pins the planned active ids and target content (``RewindRefused`` on a change) under the transcript write guards."""
        ...

    def compact(self, session_id: SessionId, replacement: Sequence[NewMessage], *, watermark: Optional[Watermark] = None, carried_tail_count: int = 0, lock_holder: Optional[str] = None, model_config_patch: Optional[Mapping[str, JsonValue]] = None, carried: Sequence[HeldMessage] = (), coverage: Optional[HeldCoverage] = None, held_for_cas: Optional[Sequence[HeldMessage]] = None) -> CompactionResult:
        """Archive the active rows and insert ``replacement`` as the live transcript in one transaction, cloning rows newer than ``watermark`` (or outside a ``coverage`` proof that holds) after it and merging ``model_config_patch``. With ``lock_holder`` the compression lock on the session must be live and name that holder (an expired lock is lost, never renewed), else ``CompressionInProgress``; ``None`` compacts unfenced. A rewritten ``held_for_cas`` row raises ``HeldRowRewritten``; both refusals come before any row moves.

        A row that would be cloned but that a replacement message already carries is not cloned. The replacement must name the row (``repair_target``, else ``message_uid``, with the row's role, content, tool call id and tool calls) and replay as it (its ``api_content`` and reasoning sidecars); that row is rewound, and the positional rewind of the carried tail's originals takes one row fewer for each of the last ``carried_tail_count`` replacement messages that carries a row. A replacement still unclaimed that names the row (``repair_target`` with the row's uid, else the uid alone, the uid on no other row of the tail and no active row outside it) claims it weakly when the compressor's rewrite explains the difference: image parts replaced by text parts, or, for a copy carrying the row's current ``db_row_snapshot``, the row's text kept as whole lines inside a longer text. A weakly claimed row is archived as compacted history instead of rewound. Each replacement claims at most one row; anything unproven is cloned.
        """
        ...

    def rewrite_pruned_rows(self, session_id: SessionId, changes: Sequence[PrunedRowChange], *, model_config_patch: Optional[Mapping[str, JsonValue]] = None, turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> PruneRewriteResult:
        """Rewrite each pruned row in place (an archived twin keeps its old body) and merge ``model_config_patch``, in one transaction. With ``turn_lease_holder`` the turn lease on the session's conversation must still name that holder (an expired lease that still names it is renewed for ``max(0.1, turn_lease_ttl_seconds)``), else ``TurnLeaseLost`` before any other check; ``None`` or an empty holder rewrites unfenced. Only the columns where ``replacement`` differs from ``before`` are written (a column ``replacement`` drops is written empty), and a change with none rewrites nothing. A row that cannot be named exactly raises ``PruneRowUnresolved`` and one another writer changed ``PruneRowStale``, before any write; a refused rewrite renews no lease."""
        ...

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> MarkerPurgeReport:
        """Clear the content of every assistant row in the scope that has tool calls and whose stripped content is a bare tool-call marker. ``dry_run`` only reports; ``backup`` takes a physical snapshot first when the backend has one and a row will change."""
        ...

    def set_user_message_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        """Rewrite the content of one active ``user`` row of the session; False when no such row. With ``display_metadata`` the same write also replaces the row's display metadata whole (an empty mapping clears it) and clears its ``api_content`` sidecar, so the replay sends the new content."""
        ...

    def deactivate_message(self, session_id: SessionId, message_id: MessageId) -> bool:
        """Mark one row of the session rewound (an inactive row stays as it is); False for a row outside the session."""
        ...

    def deactivate_messages_by_display_kind(self, session_id: SessionId, display_kind: str) -> int:
        """Mark every active row of the session with ``display_kind`` rewound; returns the rows changed."""
        ...

    def set_latest_matching_message_display_kind(self, session_id: SessionId, *, role: str, content: str, display_kind: str, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        """Stamp ``display_kind`` and replace ``display_metadata`` on the newest active row with this role and content; False when none matches or an argument is empty."""
        ...

    def set_latest_user_api_content(self, session_id: SessionId, content: JsonValue, api_content: str) -> bool:
        """Stamp ``api_content`` on the newest active ``user`` row when its content equals ``content``; False otherwise. Only safe when the caller knows that row is the message it sent; ``set_message_api_content`` addresses a row by id."""
        ...

    def set_message_api_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue, api_content: str) -> bool:
        """Stamp ``api_content`` on one active ``user`` row of the session whose content equals ``content``; False otherwise."""
        ...

    def record_compression_cooldown(self, session_id: SessionId, until: float, error: Optional[str] = None) -> None:
        """Record a compression-failure cooldown: the deadline keeps the later of the stored and given one, the error takes the latest. Store failures are logged, never raised."""
        ...

    def clear_compression_cooldown(self, session_id: SessionId) -> None:
        """Clear the cooldown deadline and error. Store failures are logged, never raised."""
        ...

    def restore_compression_cooldown(self, session_id: SessionId, snapshot: CooldownSnapshot) -> None:
        """Write an exact snapshot back and verify it; an absent snapshot over an existing row, or a failed verification, raises ``CooldownRestoreRefused``. A session row that vanished is not an error."""
        ...

    def set_reaction(self, session_id: SessionId, message_id: MessageId, emoji: Optional[str], *, author: str) -> Optional[tuple[Reaction, ...]]:
        """Set, replace or clear (same emoji again, or ``None``) ``author``'s one reaction on a row the session's display resume shows, and return the row's reactions after the write; ``None`` for any other row, with nothing written."""
        ...

    def take_unseen_reactions(self, session_id: SessionId, *, author: str = "user") -> tuple[UnseenReaction, ...]:
        """Return ``author``'s not yet announced reactions on the active and compaction-archived rows of the session's resume lineage, in row order, and mark them seen in the same transaction."""
        ...

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        """The stored rows ``request`` selects, chronological; ``DISPLAY`` pages logical messages (one per display identity, the live row preferred, then the newest)."""
        ...

    def active_message_ids(self, session_id: SessionId) -> tuple[MessageId, ...]:
        """The session's active row ids in order."""
        ...

    def active_watermark(self, session_id: SessionId) -> Watermark:
        """The newest active row id, ``Watermark(None)`` when the session has no active row."""
        ...

    def count_all(self) -> int:
        """Every stored transcript row in the scope, of every session and visibility."""
        ...

    def display_count(self, session_id: SessionId) -> int:
        """How many logical messages a display read of this session shows."""
        ...

    def conversation_revision(self, session_id: SessionId, *, include_lineage: bool = False) -> ConversationRevision:
        """The fingerprint of the session's live rows (with ``include_lineage``, of its resume lineage); any append, rewind, compaction or in-place rewrite changes it."""
        ...

    def read_conversation(self, session_id: SessionId, request: ConversationRequest = ConversationRequest()) -> tuple[ConversationMessage, ...]:
        """The replay projection of the session shaped by ``request``, after the shared projection policy; the stored rows are never changed by the read."""
        ...

    def read_resume_conversations(self, session_id: SessionId) -> ResumeConversations:
        """Both resume projections from one read, each equal to its separate ``read_conversation``."""
        ...

    def assert_resume_safe(self, session_id: SessionId, *, max_messages: int, tip_only: bool = False) -> int:
        """Count the rows a resume would load (the lineage display set, or with ``tip_only`` the tip's active rows) up to ``max_messages + 1`` and raise ``ResumeTooLarge`` above the limit. ``max_messages`` 0 returns 0 without counting; a negative limit is a ``ValueError``."""
        ...

    def read_around(self, session_id: SessionId, anchor: MessageId, *, window: int = 5) -> AnchoredWindow:
        """Up to ``window`` rows of every visibility either side of ``anchor``; an anchor outside the session reads an empty window."""
        ...

    def read_anchored_view(self, session_id: SessionId, anchor: MessageId, *, window: int = 5, bookend: int = 3, keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> AnchoredView:
        """``read_around`` filtered to ``keep_roles`` (``None`` keeps every role) plus the session bookends outside the window."""
        ...

    def find_pr_url_hits(self, session_ids: Sequence[SessionId]) -> tuple[PrUrlHit, ...]:
        """Tool rows of the named sessions whose content contains ``/pull/``, oldest first."""
        ...

    def latest_todo_result(self, session_id: SessionId, *, max_chars: int) -> Optional[str]:
        """The newest valid paired todo tool result of at most ``max_chars`` characters in the session's visible lineage, or ``None``. A result one of whose copies (same display identity or uid) was removed later in the lineage is skipped, as display reads hide it: an undo in a compression child keeps the parent's copy of that plan from coming back."""
        ...

    def read_session_timeline(self, session_id: SessionId, *, limit: int = 500, after_row_id: int = 0) -> SessionTimeline:
        """The session's human prompts in display order after the ``after_row_id`` cursor, at most ``limit``, from one snapshot."""
        ...

    def read_prompt_page(self, session_id: SessionId, row_id: MessageId, *, limit: int = 120) -> Optional[PromptPage]:
        """At most ``limit`` display rows starting at the human prompt ``row_id``; ``None`` when that row is not a displayed prompt."""
        ...

    def latest_conversation_role(self, session_id: SessionId) -> Optional[str]:
        """The role of the newest active row that is not ``session_meta`` or ``system``, or ``None``."""
        ...

    def latest_dialogue_message(self, session_id: SessionId) -> Optional[str]:
        """The content of the newest active ``user`` or ``assistant`` row whose content is not empty after trimming spaces, or ``None``: the stored cell unchanged."""
        ...

    def message_storage_state(self, message_id: MessageId) -> Optional[MessageStorageState]:
        """The owning session and visibility of the row ``message_id`` names in the scope, or ``None``."""
        ...

    def latest_row_id(self, session_id: SessionId, *, role: str = "user", offset: int = 0, require_text: bool = True) -> Optional[MessageId]:
        """The id of the newest active row of ``role``, ``offset`` rows back, skipping rows without text when ``require_text``; ``None`` when there is none or ``role`` is not ``user``/``assistant`` or ``offset`` is negative."""
        ...

    def message_role(self, session_id: SessionId, message_id: MessageId) -> Optional[str]:
        """The role of the active row ``message_id`` in the session, or ``None``."""
        ...

    def resolve_active_row_id(self, session_id: SessionId, message_id: MessageId) -> Optional[MessageId]:
        """The row itself while it is active, else the single active clone a compaction made of it, else ``None``."""
        ...

    def rewritten_held_row_ids(self, session_id: SessionId, held: Sequence[HeldMessage]) -> tuple[MessageId, ...]:
        """Ids of held messages whose active row no longer holds what the message holds (the compare ``compact(held_for_cas=)`` runs in its transaction); absent and inactive rows are not reported."""
        ...

    def plan_user_turn_rewind(self, session_id: SessionId, user_ordinal: int, *, warm_history: Optional[Sequence[Mapping[str, Any]]] = None, require_retryable: bool = False, require_composite: bool = False) -> RewindPlan:
        """Every read and check of ``rewind_user_turn`` without the write: ``RewindTargetUnavailable`` when no rewindable turn is at the ordinal, ``RewindRefused`` when the warm history disagrees with the durable one or the target has no row identity, ``ValueError`` when ``require_retryable`` and the turn cannot be re-sent."""
        ...

    def has_platform_message_id(self, session_id: SessionId, platform_message_id: str) -> bool:
        """Whether any row of the session, of any visibility, carries the platform message id."""
        ...

    def has_gateway_input_owner(self, session_id: SessionId, owner: str) -> bool:
        """Whether an unobserved active or compaction-archived ``user`` row of the session carries ``display_metadata.gateway_input_owner == owner``."""
        ...

    def latest_conversation_boundary(self, peer: ConversationPeer) -> Optional[int]:
        """The peer's conversation generation (reset boundaries crossed by its sessions), or ``None`` when never reset or a peer component is empty."""
        ...

    def read_compression_cooldown(self, session_id: SessionId, at: float) -> Optional[CompressionCooldown]:
        """The cooldown still running at ``at``, or ``None``."""
        ...

    def read_compression_cooldown_snapshot(self, session_id: SessionId) -> CooldownSnapshot:
        """The stored cooldown columns without expiry filtering; an absent session reads an absent snapshot."""
        ...

    def terminal_tool_call_payloads(self, *, since_ts: float) -> Iterator[str]:
        """The stored ``tool_calls`` JSON of assistant rows since ``since_ts``, streamed in batches inside one read that ends when the iterator is exhausted or closed."""
        ...

    def blocked_tool_result_payloads(self, *, since_ts: float) -> Iterator[tuple[str, Optional[str]]]:
        """``(tool_call_id, content)`` of the tool rows with a call id since ``since_ts`` whose stored text content holds ``BLOCKED`` or ``approval`` in any ASCII case, in no defined order, streamed in batches inside one read that ends when the iterator is exhausted or closed."""
        ...

    def has_any_message(self) -> bool:
        """Whether the scope holds any transcript row (False when the store has no transcript table yet)."""
        ...

    def spillover_reference_rows(self) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        """``(content, api_content)`` of at most 20001 rows that reference spilled-over tool results, in no defined order."""
        ...
