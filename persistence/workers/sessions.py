"""The worker's session writes: the ``session.*`` and ``transcript.append`` operations of the worker receipt stream.

A managed worker settles its own session's writes through the owner (``agent/runtime_session_lifecycle.py``, ``agent/runtime_session_store.py``); each verb is one operation of the receipt transaction documented in :mod:`persistence.workers.executions`, on the worker's own session. Each record carries the producer's wire payload key for key, so the receipt digest matches the producer's. Payload checks are the handler's and refuse with ``invalid_params``, after the receipt fences and the replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, NotRequired, Optional, Protocol, Required, Sequence, TypedDict, Unpack

from ..conversation.messages import ToolCallUids
from ..conversation.sessions import SessionLifecycleStatus, TitleSource
from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import JsonValue, MessageId, SessionId
from .vocabulary import ROW_ANNOTATION_KEYS

if TYPE_CHECKING:
    from .executions import WorkerExecutionId

__all__ = [
    "RowAnnotation",
    "ToolPin",
    "TranscriptAppendOptions",
    "WorkerActivity",
    "WorkerAppendResult",
    "WorkerBillingRoute",
    "WorkerMessage",
    "WorkerSessionCreate",
    "WorkerSessionVerbs",
    "WorkerTitleResult",
]

#: A ``tools[]`` freeze pin: ``{"version", "tools"[, "delegation_backend"]}`` or a legacy list of tool names.
ToolPin = Mapping[str, JsonValue] | list[str]

#: One transcript row on the worker wire: a message dict whose keys lie in :data:`persistence.workers.vocabulary.HANDOFF_MESSAGE_FIELDS` (``_compressed_summary_has_user_turn`` only on compaction handoffs).
WorkerMessage = TypedDict("WorkerMessage", {
    "role": Required[str], "content": JsonValue, "tool_name": str | None, "tool_calls": list[JsonValue] | None,
    "tool_call_id": str | None, "token_count": int | None, "finish_reason": str | None, "reasoning": str | None,
    "reasoning_content": str | None, "reasoning_details": JsonValue, "codex_reasoning_items": JsonValue,
    "codex_message_items": JsonValue, "platform_message_id": str | None, "message_id": str | None, "observed": bool,
    "effect_disposition": str | None, "_compressed_summary": bool, "timestamp": float | None, "api_content": str | None,
    "display_kind": str | None, "display_metadata": dict[str, JsonValue] | None, "_row_id": int,
    "_canonical_content": JsonValue, "_db_row_snapshot": str, "_canonical_row": dict[str, JsonValue] | None,
    "data_use": list[str], "_submit_row_session_id": str, "_merged_turn_prefix": str, "message_uid": str,
    "_absorbed_message_uids": list[str], "_tool_call_uids": ToolCallUids, "_tool_call_uid": str,
    "_merged_durable_rows": int, "_unnamed_durable_rows": int, "_retired_durable_rows": JsonValue,
    "_absorbed_row_ids": list[int], "_compressed_summary_has_user_turn": bool,
}, total=False)


class TranscriptAppendOptions(TypedDict, total=False):
    """The optional keys of a ``transcript.append`` payload, present or absent as the worker sent them."""

    turn_lease_holder: str | None


class WorkerSessionCreate(TypedDict, total=False):
    """A ``session.create`` payload: ``source`` and any subset of the other keys."""

    source: Required[str]
    model: NotRequired[str | None]
    model_config: NotRequired[Mapping[str, JsonValue] | None]
    system_prompt: NotRequired[str | None]
    user_id: NotRequired[str | None]
    session_key: NotRequired[str | None]
    chat_id: NotRequired[str | None]
    chat_type: NotRequired[str | None]
    thread_id: NotRequired[str | None]
    parent_session_id: NotRequired[str | None]
    cwd: NotRequired[str | None]
    profile_name: NotRequired[str | None]
    git_repo_root: NotRequired[str | None]
    origin_json: NotRequired[str | None]
    display_name: NotRequired[str | None]


@dataclass(frozen=True, slots=True)
class WorkerTitleResult:
    """The ``session.title`` receipt: ``value`` the title store's answer; a title the store refused (taken by another session, the canonical Bot Chat, too long) is ``value`` False with ``error`` and the ``title_candidate`` that later authorizes ``next_session_title``."""

    value: bool
    error: str | None = None
    title_candidate: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerActivity:
    """A ``session.activity`` payload: the epoch-second stamp (``None`` takes the store's clock) and the raw labels the store bounds and normalizes."""

    ts: float | None
    description: str | None
    provenance: str | None


@dataclass(frozen=True, slots=True)
class WorkerBillingRoute:
    """A ``session.billing_route`` payload: provider and base URL replace the stored ones; ``billing_mode`` ``None`` keeps the stored mode."""

    provider: str | None
    base_url: str | None
    billing_mode: str | None


@dataclass(frozen=True, slots=True)
class RowAnnotation:
    """The row state a transcript write stamped on one message (:data:`persistence.workers.vocabulary.ROW_ANNOTATION_KEYS`); ``None`` is absent."""

    row_id: int | None
    timestamp: float | None
    db_row_snapshot: str | None
    canonical_row: Mapping[str, JsonValue] | None
    message_uid: str | None
    tool_call_uids: ToolCallUids | None
    tool_call_uid: str | None

    @classmethod
    def from_wire(cls, annotation: Mapping[str, Any]) -> RowAnnotation:
        return cls(*(annotation.get(key) for key in ROW_ANNOTATION_KEYS))

    def as_wire(self) -> dict[str, Any]:
        return dict(zip(ROW_ANNOTATION_KEYS, (self.row_id, self.timestamp, self.db_row_snapshot, self.canonical_row, self.message_uid, self.tool_call_uids, self.tool_call_uid)))


@dataclass(frozen=True, slots=True)
class WorkerAppendResult:
    """A transcript append's receipt: the inserted row count and one annotation per sent message, in order."""

    count: int
    annotations: tuple[RowAnnotation, ...]


class WorkerSessionVerbs(Protocol):
    """The worker session operations; each runs the receipt transaction of :mod:`persistence.workers.executions` on the worker's session ``session_id``."""

    def set_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, title: str | None, source: TitleSource) -> WorkerTitleResult:
        """``session.title`` with ``{"title", "source"}``: the title store's write under provenance precedence; a refusal of the title store is the receipt ``value`` False with ``error`` and ``title_candidate``, not an error."""
        ...

    def set_session_title_source(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, source: TitleSource) -> bool:
        """``session.title_source`` with ``{"source"}``: overwrite the provenance of a titled session; False when untitled."""
        ...

    def next_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, base_title: str) -> str:
        """``session.next_title`` with ``{"base_title"}``: ``base_title`` must be the session's title or the ``title_candidate`` of a receipt of a running execution on the session (``permission_denied``); answers ``base`` when no stored title equals it or starts ``"base #"``, else ``"base #N"`` one past the highest number (at least 2). No row changes."""
        ...

    def touch_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, activity: WorkerActivity) -> None:
        """``session.activity``: stamp and bounded labels move together, only when the stamp moves strictly forward."""
        ...

    def clear_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        """``session.activity_clear``: description ``""`` and provenance ``unknown``; the stamp stays."""
        ...

    def set_session_billing_route(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, route: WorkerBillingRoute) -> None:
        """``session.billing_route``: provider and base URL replaced, ``billing_mode`` kept when ``None``. The stored system prompt is not touched."""
        ...

    def set_latest_user_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, content: JsonValue, api_content: str) -> int:
        """``session.api_content``: ``api_content`` (surrogates scrubbed) on the newest active user row when its stored content equals ``content`` encoded; the rows changed, 0 or 1."""
        ...

    def set_message_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, api_content: str) -> int:
        """``session.message_api_content``: as ``set_latest_user_api_content`` on the active user row ``row_id`` of this session; ``row_id`` a positive int."""
        ...

    def set_user_message_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> int:
        """``session.user_content``: rewrite the content of the active user row ``row_id`` of this session; the rows changed, 0 or 1. With ``display_metadata`` (sent as the payload key ``display_metadata``) it is merge-patched into the row's display metadata (its keys win, a ``None`` value removes the key, other keys stay) and its ``api_content`` cleared; a payload key other than ``row_id``, ``content`` and ``display_metadata`` is ``invalid_params``."""
        ...

    def create_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **fields: Unpack[WorkerSessionCreate]) -> SessionId:
        """``session.create``: the row must exist with the same ``source`` and every non-None identity facet equal to the stored one (``permission_denied``); text keys must be strings and ``model_config`` keys within the constructor set (``invalid_params``). The keep-existing upsert fills unset facets (a retired id is ``not_found``); on an owner-reserved child row (``_delegate_from``) the constructor keys merge beside the marker. Returns ``session_id``."""
        ...

    def end_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, end_reason: str) -> None:
        """``session.end``: end with a non-empty ``end_reason`` unless already ended."""
        ...

    def session_lifecycle(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> SessionLifecycleStatus:
        """``session.lifecycle``: the status the newest row of any visibility implies, ``empty`` without rows."""
        ...

    def set_session_system_prompt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, system_prompt: str | None) -> None:
        """``session.prompt``: replace the stored prompt snapshot (``None`` clears)."""
        ...

    def patch_session_sidecars(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, patch: Mapping[str, JsonValue]) -> None:
        """``session.sidecars``: merge ``patch`` (keys within :data:`persistence.workers.vocabulary.SIDECAR_KEYS`, ``None`` deletes) into ``model_config``."""
        ...

    def set_session_tool_pin(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, tool_pin: ToolPin | None) -> None:
        """``session.tools`` with ``{"tool_names"}``: ``None`` clears; a list of 1..256-character names, or a mapping with ``version`` (str) and ``tools`` (list) and an optional ``delegation_backend`` (str of at most 64), encoding to at most 4 MiB."""
        ...

    def latest_conversation_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        """``session.tail_role``: the role of the newest active row that is not ``session_meta`` or ``system``."""
        ...

    def message_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: int) -> str | None:
        """``session.row_role``: the role of active row ``row_id`` of this session, else ``None``; ``row_id`` an int."""
        ...

    def append_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], **options: Unpack[TranscriptAppendOptions]) -> WorkerAppendResult:
        """``transcript.append`` with ``{"messages"[, "turn_lease_holder"]}``: at most 1000 dicts within :data:`persistence.workers.vocabulary.WORKER_MESSAGE_FIELDS` with role user, assistant, system or tool (``invalid_params``); a ``_row_id`` outside this transcript is ``permission_denied``; then the transcript append with its write guards (``TurnLeaseLost``, ``CompressionInProgress``, ``CompressionClosed``). Backend lease reclaim: a stale compression lock or turn lease whose holder process is dead is deleted inside the append guard."""
        ...
