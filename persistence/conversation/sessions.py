"""Session rows: lifecycle, titles, model config, flags, deletion and listings.

Every verb is one synchronous call on the store of the handle's profile; no verb takes a scope. Times are epoch-second floats as the store keeps them. Expected negatives stay typed returns (a missing row reads ``None``, empty or ``False``); ``ValueError`` is for input shape only. Deletion is retirement: a deleted id is never creatable again (``create`` raises :class:`persistence.RetiredSessionError`), and the delete verbs refuse a target with live ledger work (``AuthorityRefusal('session_busy')``) or unknown work (:class:`persistence.UnknownExecution`).

The delete and archive verbs return the ids they removed or newly archived; the caller runs the post-commit work (the ``on_session_delete`` hook, retired media, transcript files) through ``session_retention``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, TypeAlias, runtime_checkable

from ..authority.vocabulary import OwnerRule
from ..core.errors import Conflict, TransientStoreError
from ..core.identity import JsonValue, SessionId

__all__ = [
    "ACTIVITY_DESCRIPTION_MAX",
    "ACTIVITY_PROVENANCES",
    "ACTIVITY_PROVENANCE_UNKNOWN",
    "AUTOMATIC_END_REASONS",
    "AUTO_LOAD_SCAFFOLD_PREFIX",
    "BOUNDARY_END_REASONS",
    "CANONICAL_BOT_CHAT_TITLE",
    "COMPACT_EXCLUDED_COLUMNS",
    "END_REASON_ADOPTED",
    "END_REASON_COMPRESSION",
    "END_REASON_ORPHANED_COMPRESSION",
    "END_REASON_SESSION_RESET",
    "ERROR_FINISH_REASONS",
    "INTERNAL_LISTING_SOURCES",
    "LOCAL_CHAT_PREFIX",
    "MAX_TITLE_LENGTH",
    "ONBOARDING_RUN_KEY",
    "RECENT_BROWSE_CANDIDATE_CAP",
    "RECENT_BROWSE_LINEAGE_CAP",
    "RECOVERABLE_END_REASONS",
    "RETAG_GATE_PREFIX",
    "RETAG_SOURCE_FROM",
    "RETAG_SOURCE_TO",
    "RUNTIME_LOCK_KEY",
    "SESSION_ROW_COLUMNS",
    "SKILL_SCAFFOLD_PREFIX",
    "YOLO_KEY",
    "ArchiveChange",
    "CanonicalTitleLocked",
    "CronRunEntry",
    "CronRunIdentity",
    "DeclaredScopeIdentity",
    "DeleteManyOutcome",
    "LiveListingFacts",
    "LiveListingRow",
    "NewSession",
    "OnboardingAdvance",
    "OnboardingEvent",
    "OnboardingStep",
    "PendingAdmissionRef",
    "RecentBrowseQuery",
    "RecentBrowseTimeout",
    "RecentSessionEntry",
    "ReplaceSessionConfig",
    "RosterSession",
    "RosterSessions",
    "RosterWorker",
    "RuntimeLock",
    "SessionCountQuery",
    "SessionCwdUsage",
    "SessionEdge",
    "SessionExists",
    "SessionFacetEntry",
    "SessionLifecycleStatus",
    "SessionLineageKind",
    "SessionListEntry",
    "SessionListRow",
    "SessionOrder",
    "SessionQuery",
    "SessionRecencyEntry",
    "SessionRecord",
    "SessionRepository",
    "SessionWindow",
    "SkillScaffoldedSession",
    "TitleConflict",
    "TitleSource",
    "bound_activity_description",
    "is_automatic_end_reason",
    "is_corrupt_model_config",
    "is_explicit_close_reason",
    "normalize_activity_provenance",
    "parse_model_config",
    "sanitize_title",
    "session_yolo_enabled",
    "workspace_key",
]

# ── vocabulary ─────────────────────────────────────────────────────────────

#: Accidental end reasons recovery treats as resumable; an intentional reset may overwrite them.
RECOVERABLE_END_REASONS: frozenset[str] = frozenset(
    {"agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap"}
)

#: End stamps written by automatic cleanup; a writer that proves liveness may clear them.
AUTOMATIC_END_REASONS: frozenset[str] = RECOVERABLE_END_REASONS | frozenset(
    {"tui_shutdown", "ws_disconnect", "idle_timeout", "lru_evict"}
)

#: The end reason that makes a later child a compression continuation.
END_REASON_COMPRESSION = "compression"

#: The reason ``promote_to_reset`` stamps by default.
END_REASON_SESSION_RESET = "session_reset"

#: The reason a donor segment of a lineage adoption is ended with (outside the recoverable set).
END_REASON_ADOPTED = "adopted_by_profile"

#: The reason the orphaned-continuation repair ends a row with.
END_REASON_ORPHANED_COMPRESSION = "orphaned_compression"

#: Reset boundaries plus CLI ``new_session``: deliberate conversation boundaries no reopen clears.
BOUNDARY_END_REASONS: frozenset[str] = frozenset(
    {END_REASON_SESSION_RESET, "session_switch", "idle", "daily", "suspended", "resume_pending_expired", "new_session"}
)

#: Sources every human session picker leaves out (workers, tool integrations, one-shot runs).
INTERNAL_LISTING_SOURCES = ("kanban", "tool", "oneshot")

#: Maximum sanitized title length; a longer title is a ``ValueError``.
MAX_TITLE_LENGTH = 100

#: Bot Mode's canonical chat title: a hidden row holding it keeps it (its title is its identity).
CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

#: The chat-id prefix of a locally created session.
LOCAL_CHAT_PREFIX = "local-"

#: State-meta key prefix of the once-per-root gate ``retag_worker_sessions`` records (prefix plus the normalized root).
RETAG_GATE_PREFIX = "kanban_worker_source_retagged:"
RETAG_SOURCE_FROM = "cli"
RETAG_SOURCE_TO = "kanban"

#: The ``model_config`` key ``set_runtime_lock`` writes and ``update_model`` drops.
RUNTIME_LOCK_KEY = "browser_model_lock"

#: The ``sessions`` columns in table order: the leading keys of every session row dict the store serves.
SESSION_ROW_COLUMNS: tuple[str, ...] = (
    "id", "runtime_revision", "runtime_generation", "source", "created_source", "user_id", "session_key", "chat_id",
    "chat_type", "thread_id", "display_name", "origin_json", "expiry_finalized", "model", "model_config",
    "system_prompt", "system_prompt_hash", "parent_session_id", "started_at", "ended_at", "end_reason",
    "message_count", "tool_call_count", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "cwd", "git_branch", "git_repo_root", "git_metadata_generation", "billing_provider",
    "billing_base_url", "billing_mode", "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source",
    "pricing_version", "title", "title_source", "last_activity_at", "last_activity_description",
    "last_activity_provenance", "api_call_count", "handoff_state", "handoff_platform", "handoff_error",
    "compression_failure_cooldown_until", "compression_failure_error", "compression_fallback_streak",
    "compression_ineffective_count", "compression_recovery_deadline", "compression_overload_streak", "profile_name",
    "transport_profile", "rewind_count", "archived", "auto_archived", "pinned", "hidden", "last_read_at",
    "tool_names", "transcript_revision",
)

#: The columns a ``compact_rows`` listing does not read.
COMPACT_EXCLUDED_COLUMNS: frozenset[str] = frozenset({"system_prompt", "system_prompt_hash", "git_metadata_generation"})

#: The ``model_config`` key of the YOLO flag.
YOLO_KEY = "yolo_mode"

#: The ``model_config`` key holding a chat's first-run onboarding run (``session_rules.onboarding_flow``'s record).
ONBOARDING_RUN_KEY = "onboarding_run"

#: Finish reasons of a last transcript row that classify the session ``ERROR``.
ERROR_FINISH_REASONS: frozenset[str] = frozenset({"error", "agent_error", "content_filter"})

#: Heads of a first user turn that was a ``/skill`` expansion or a gateway skill auto-load.
SKILL_SCAFFOLD_PREFIX = "[IMPORTANT: The user has invoked the "
AUTO_LOAD_SCAFFOLD_PREFIX = '[IMPORTANT: The "'

#: Ceilings of the bounded recent browse.
RECENT_BROWSE_CANDIDATE_CAP = 2048
RECENT_BROWSE_LINEAGE_CAP = 8192

#: Activity labels: the description cap and the provenance vocabulary (anything else stores as unknown).
ACTIVITY_DESCRIPTION_MAX = 120
ACTIVITY_PROVENANCE_UNKNOWN = "unknown"
ACTIVITY_PROVENANCES: frozenset[str] = frozenset(
    {
        ACTIVITY_PROVENANCE_UNKNOWN,
        "agent.compression",
        "agent.compression_timeout",
        "agent.compression_cooldown",
        "agent.compression_turnhold",
    }
)


def is_automatic_end_reason(reason: object) -> bool:
    """Whether ``reason`` is an automatic-cleanup end stamp."""
    return isinstance(reason, str) and reason in AUTOMATIC_END_REASONS


def is_explicit_close_reason(reason: Optional[str]) -> bool:
    """Whether the end stamp of an ended row is one ``reopen_if_explicitly_closed`` clears: not automatic, not compression, not a boundary (a ``None`` reason counts as explicit)."""
    return not is_automatic_end_reason(reason) and reason != END_REASON_COMPRESSION and reason not in BOUNDARY_END_REASONS


def bound_activity_description(description: Optional[str]) -> str:
    """The stored activity description: stripped and clamped to :data:`ACTIVITY_DESCRIPTION_MAX` with an ellipsis."""
    text = (description or "").strip()
    if len(text) <= ACTIVITY_DESCRIPTION_MAX:
        return text
    return text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


def normalize_activity_provenance(provenance: Optional[str]) -> str:
    """The stored provenance: a member of :data:`ACTIVITY_PROVENANCES`, else unknown."""
    value = (provenance or "").strip()
    return value if value in ACTIVITY_PROVENANCES else ACTIVITY_PROVENANCE_UNKNOWN


def workspace_key(row: Mapping[str, Any]) -> Optional[str]:
    """Workspace grouping key of a session row: the git repo root, else the cwd, else ``None``."""
    return (row.get("git_repo_root") or "").strip() or (row.get("cwd") or "").strip() or None


def parse_model_config(raw: Any) -> dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or a mapping becomes a dict copy; anything else ``{}``."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def is_corrupt_model_config(raw: Any) -> bool:
    """Whether a stored ``model_config`` cell is non-blank text that does not decode as JSON (a strict reader refuses it); ``None``, blank text and JSON that is not an object are not corrupt."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        json.loads(raw)
    except (ValueError, RecursionError):
        return True
    return False


def session_yolo_enabled(model_config: Any) -> bool:
    """The persisted YOLO flag of a session's ``model_config``; ``False`` on any parse failure."""
    return bool(parse_model_config(model_config).get(YOLO_KEY))


_TITLE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_TITLE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TITLE_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]")


def sanitize_title(title: Optional[str]) -> Optional[str]:
    """The stored form of a title, or ``None`` when nothing printable remains.

    Lone surrogates become U+FFFD; ASCII controls (except tab, newline, return) and zero-width, bidi, object-replacement and interlinear code points are removed; whitespace runs collapse and the ends strip. A cleaned title longer than :data:`MAX_TITLE_LENGTH` raises ``ValueError``.
    """
    if not title:
        return None
    cleaned = _TITLE_INVISIBLE_RE.sub("", _TITLE_CONTROL_RE.sub("", _TITLE_SURROGATE_RE.sub("\ufffd", title)))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_TITLE_LENGTH:
        raise ValueError(f"Title too long ({len(cleaned)} chars, max {MAX_TITLE_LENGTH})")
    return cleaned


# ── errors ─────────────────────────────────────────────────────────────────


class TitleConflict(Conflict, ValueError):
    """Another session holds the title; nothing written. ``holder`` is that session."""

    def __init__(self, title: str, holder: SessionId) -> None:
        self.title = title
        self.holder = holder
        super().__init__("session_title", title)
        self.args = (f"Title '{title}' is already in use by session {holder}",)


class CanonicalTitleLocked(Conflict, ValueError):
    """A user rename of the hidden canonical Bot Chat; nothing written."""

    def __init__(self, session_id: SessionId) -> None:
        self.session_id = session_id
        super().__init__("session", str(session_id))
        self.args = (
            "This is the bot's canonical Bot Chat — its name is its identity, and renaming it would orphan "
            "the conversation. To start fresh, create a new bot instead.",
        )


class SessionExists(Conflict):
    """``create_exclusive`` addressed an id that already has a row; nothing written. ``existing`` is that row."""

    code = "session_exists"

    def __init__(self, existing: SessionRecord) -> None:
        self.existing = existing
        super().__init__("session", str(existing.session_id))
        self.args = ("Session already exists",)


class RecentBrowseTimeout(TransientStoreError, TimeoutError):
    """``list_recent_sessions`` ran past its deadline; nothing returned. Classified by its text, like the store's ``TimeoutError``."""

    cause = "unknown"


# ── records ────────────────────────────────────────────────────────────────


class SessionLineageKind(Enum):
    """The fork marker a child carries in ``model_config`` (``_delegate_from``, ``_branched_from``, ``_reset_from``)."""

    NONE = "none"
    DELEGATE = "delegate"
    BRANCH = "branch"
    RESET = "reset"


class TitleSource(Enum):
    """Title provenance, lowest to highest authority."""

    DERIVED = "derived"
    LLM = "llm"
    USER = "user"


class SessionOrder(Enum):
    """Listing order: ``started_at`` or the lineage's last activity, newest first."""

    STARTED_AT = "started_at"
    LAST_ACTIVE = "last_active"


class SessionLifecycleStatus(Enum):
    """What a session's last transcript row says about its turn."""

    EMPTY = "empty"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NewSession:
    """Creation command. ``lineage_kind`` writes its fork marker naming ``parent_session_id``; ``origin`` is the stored ``origin_json``; ``profile_name`` stamps the owner (``None`` stamps the store's own profile). There is no ``git_branch``: a row gets one only by inheriting its parent's, or later through ``update_cwd`` / ``publish_git_metadata``."""

    session_id: SessionId
    source: str
    model: Optional[str] = None
    cwd: Optional[str] = None
    parent_session_id: Optional[SessionId] = None
    lineage_kind: SessionLineageKind = SessionLineageKind.NONE
    session_key: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    git_repo_root: Optional[str] = None
    model_config: Optional[Mapping[str, JsonValue]] = None
    system_prompt: Optional[str] = None
    display_name: Optional[str] = None
    origin: Optional[Mapping[str, JsonValue]] = None
    transport_profile: Optional[str] = None
    profile_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("NewSession.session_id must be non-empty")
        if not self.source:
            raise ValueError("NewSession.source must be non-empty")
        if self.lineage_kind is not SessionLineageKind.NONE and not self.parent_session_id:
            raise ValueError(f"lineage_kind {self.lineage_kind.value!r} requires parent_session_id")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One full ``sessions`` row. ``system_prompt`` and ``tool_pin`` are the resolved snapshots, except that ``tool_pin`` is resolved only in ``get``: listing, cron and title reads carry its unresolved content hash, so ``tool_pin`` is ``None`` there; ``origin`` is the parsed ``origin_json``; ``lineage_kind``, ``reset_from`` and ``branched_from`` are read from the ``model_config`` markers; a stored ``model_config`` cell that does not decode reads ``{}`` with ``model_config_corrupt`` set. Counter columns (``int`` as every verb writes them) carry the stored cell as read, never coerced: a damaged row's text or real cell reads as stored. ``model_config_json``, ``origin_json`` and ``tool_names`` are the stored cells as the read returned them (``tool_names`` is the resolved pin text on ``get``, the content hash elsewhere).

    ``as_session_row()`` emits the store's session row dict: every column of :data:`SESSION_ROW_COLUMNS` in order with its stored cell (flags as ``0``/``1``, ``title_source`` as its stored string), then ``_reset_from`` and ``_branched_from`` when the config names them.
    """

    session_id: SessionId
    source: str
    created_source: Optional[str]
    user_id: Optional[str]
    session_key: Optional[str]
    chat_id: Optional[str]
    chat_type: Optional[str]
    thread_id: Optional[str]
    display_name: Optional[str]
    origin: Optional[Mapping[str, JsonValue]]
    expiry_finalized: bool
    model: Optional[str]
    model_config: Mapping[str, JsonValue]
    system_prompt: Optional[str]
    system_prompt_hash: Optional[str]
    parent_session_id: Optional[SessionId]
    lineage_kind: SessionLineageKind
    reset_from: Optional[SessionId]
    branched_from: Optional[SessionId]
    started_at: float
    ended_at: Optional[float]
    end_reason: Optional[str]
    message_count: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    api_call_count: int
    estimated_cost_usd: Optional[float]
    actual_cost_usd: Optional[float]
    billing_provider: Optional[str]
    billing_base_url: Optional[str]
    billing_mode: Optional[str]
    cost_status: Optional[str]
    cost_source: Optional[str]
    pricing_version: Optional[str]
    cwd: Optional[str]
    git_branch: Optional[str]
    git_repo_root: Optional[str]
    git_metadata_generation: int
    title: Optional[str]
    title_source: Optional[TitleSource]
    last_activity_at: Optional[float]
    last_activity_description: Optional[str]
    last_activity_provenance: Optional[str]
    handoff_state: Optional[str]
    handoff_platform: Optional[str]
    handoff_error: Optional[str]
    compression_failure_cooldown_until: Optional[float]
    compression_failure_error: Optional[str]
    compression_fallback_streak: int
    compression_ineffective_count: int
    compression_recovery_deadline: Optional[float]
    compression_overload_streak: int
    profile_name: Optional[str]
    transport_profile: Optional[str]
    rewind_count: int
    archived: bool
    auto_archived: bool
    pinned: bool
    hidden: bool
    last_read_at: Optional[float]
    tool_pin: Optional[JsonValue]
    transcript_revision: int
    runtime_revision: int
    runtime_generation: int
    #: The session the fork marker names (``None`` for ``SessionLineageKind.NONE``); a continuation that inherited a marker names the original origin.
    lineage_origin_id: Optional[SessionId] = None
    model_config_corrupt: bool = False
    model_config_json: Optional[str] = None
    origin_json: Optional[str] = None
    tool_names: Optional[str] = None

    def as_session_row(self) -> dict[str, JsonValue]:
        return {**self._row_cells(), **self._marker_cells()}

    def _row_cells(self, *, compact: bool = False) -> dict[str, JsonValue]:
        """The stored cells in table order; ``compact`` leaves out :data:`COMPACT_EXCLUDED_COLUMNS`."""
        cells: dict[str, JsonValue] = {
            "id": str(self.session_id),
            "origin_json": self.origin_json,
            "expiry_finalized": int(self.expiry_finalized),
            "model_config": self.model_config_json,
            "title_source": None if self.title_source is None else self.title_source.value,
            "archived": int(self.archived),
            "auto_archived": int(self.auto_archived),
            "pinned": int(self.pinned),
            "hidden": int(self.hidden),
            "tool_names": self.tool_names,
        }
        return {
            column: cells[column] if column in cells else getattr(self, column)
            for column in SESSION_ROW_COLUMNS
            if not (compact and column in COMPACT_EXCLUDED_COLUMNS)
        }

    def _marker_cells(self) -> dict[str, JsonValue]:
        return {key: value for key, value in (("_reset_from", self.reset_from), ("_branched_from", self.branched_from)) if value}


@dataclass(frozen=True, slots=True)
class SessionQuery:
    """Filters of ``list_sessions`` with the listing's defaults: children (except branch and reset children) excluded, archived and hidden excluded, compression roots projected to their tips. ``id_query`` and ``search_text`` apply in ``LAST_ACTIVE`` order only; ``local_created_sources`` restricts ``source='local'`` rows to those creating surfaces."""

    sources: Optional[tuple[str, ...]] = None
    exclude_sources: tuple[str, ...] = ()
    cwd_prefix: Optional[str] = None
    include_children: bool = False
    min_message_count: int = 0
    project_compression_tips: bool = True
    order: SessionOrder = SessionOrder.STARTED_AT
    include_archived: bool = False
    archived_only: bool = False
    id_query: Optional[str] = None
    search_text: Optional[str] = None
    include_pinned: bool = False
    session_key: Optional[str] = None
    include_hidden: bool = False
    include_subagents: bool = False
    local_created_sources: Optional[tuple[str, ...]] = None


@dataclass(frozen=True, slots=True)
class SessionCountQuery:
    """Filters of ``count_sessions``; children are counted unless ``exclude_children``."""

    sources: Optional[tuple[str, ...]] = None
    exclude_sources: tuple[str, ...] = ()
    cwd_prefix: Optional[str] = None
    min_message_count: int = 0
    include_archived: bool = False
    archived_only: bool = False
    exclude_children: bool = False
    include_subagents: bool = False
    local_created_sources: Optional[tuple[str, ...]] = None


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """``LIMIT``/``OFFSET`` of a listing; both non-negative, ``limit=None`` for every row from ``offset`` on."""

    limit: Optional[int]
    offset: int = 0

    def __post_init__(self) -> None:
        if (self.limit is not None and self.limit < 0) or self.offset < 0:
            raise ValueError(f"SessionWindow needs limit >= 0 and offset >= 0, got {self.limit}, {self.offset}")


@dataclass(frozen=True, slots=True)
class SessionListEntry:
    """One listing row: the row (a projected compression root carries its tip's surfaced fields), its first-user-turn ``preview`` (``""`` when none), ``unread``, ``last_active`` (``None`` when no stamp is trusted, as for :class:`SessionRecencyEntry`), and for a projected root ``lineage_root_id``, the chain ``lineage_ids`` and ``continuation_kind="compression"``. ``compact`` marks a ``compact_rows`` row, whose ``system_prompt``, ``system_prompt_hash`` and ``git_metadata_generation`` are not read.

    ``as_list_row()`` emits the listing's row dict: the row's cells (a compact row without :data:`COMPACT_EXCLUDED_COLUMNS`), ``last_active``, the config markers, ``preview``, then for a projected root ``_lineage_root_id``, ``_lineage_ids`` and ``continuation_kind``, else a local lineage's ``_lineage_ids``, and ``unread`` last.
    """

    session: SessionRecord
    preview: str
    unread: bool
    last_active: Optional[float]
    lineage_root_id: Optional[SessionId] = None
    lineage_ids: tuple[SessionId, ...] = ()
    continuation_kind: Optional[str] = None
    compact: bool = False

    def as_list_row(self) -> dict[str, JsonValue]:
        row = self._head_cells()
        if self.lineage_root_id is not None:
            row["_lineage_root_id"] = str(self.lineage_root_id)
            row["_lineage_ids"] = [str(sid) for sid in self.lineage_ids]
            row["continuation_kind"] = self.continuation_kind
        elif self.lineage_ids:
            row["_lineage_ids"] = [str(sid) for sid in self.lineage_ids]
        row["unread"] = self.unread
        return row

    def _head_cells(self, *extra: tuple[str, JsonValue]) -> dict[str, JsonValue]:
        """The cells, any ``extra`` cells, ``last_active``, the markers and ``preview``: the shape every listing read shares."""
        return {
            **self.session._row_cells(compact=self.compact), **dict(extra), "last_active": self.last_active,
            **self.session._marker_cells(), "preview": self.preview,
        }


@dataclass(frozen=True, slots=True)
class SessionListRow:
    """The picker columns of one row."""

    session_id: SessionId
    source: str
    title: Optional[str]
    started_at: float
    ended_at: Optional[float]
    message_count: int
    model: Optional[str]
    hidden: bool
    archived: bool


@dataclass(frozen=True, slots=True)
class SessionRecencyEntry:
    """One row with its ``last_active``: the newest of the activity stamp and the transcript's rows, else ``started_at``, each counted only inside the trusted epoch window; ``None`` when no cell is trusted."""

    session: SessionRecord
    last_active: Optional[float]


@dataclass(frozen=True, slots=True)
class CronRunEntry:
    """One run session of a cron job with its job and execution (``None`` for a pre-unified run). ``as_cron_row()`` emits the job-runs row dict: the row's cells, ``cron_execution_id``, ``last_active``, the config markers, ``preview`` and ``cron_job_id``."""

    entry: SessionListEntry
    cron_job_id: str
    cron_execution_id: Optional[str]

    def as_cron_row(self) -> dict[str, JsonValue]:
        return {**self.entry._head_cells(("cron_execution_id", self.cron_execution_id)), "cron_job_id": self.cron_job_id}


@dataclass(frozen=True, slots=True)
class CronRunIdentity:
    """The cron job and execution a run session belongs to."""

    cron_job_id: str
    cron_execution_id: Optional[str]


@dataclass(frozen=True, slots=True)
class DeclaredScopeIdentity:
    """Whether a session is a branch, delegate or tool child of its parent, and its stripped source."""

    is_fork_child: bool
    source: str


@dataclass(frozen=True, slots=True)
class DeleteManyOutcome:
    """``deleted`` counts the requested ids that existed and went; ``skipped_guarded`` the requested ids kept by a live lease (sorted); ``removed_ids`` every row removed."""

    deleted: int
    skipped_guarded: tuple[SessionId, ...]
    removed_ids: tuple[SessionId, ...]


@dataclass(frozen=True, slots=True)
class ArchiveChange:
    """``changed`` is True when the lineage had rows; ``archived_ids`` is non-empty only for a real flip to archived with ``notify``."""

    changed: bool
    archived_ids: tuple[SessionId, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionEdge:
    """One row of a continuation walk."""

    session_id: SessionId
    parent_session_id: Optional[SessionId]
    started_at: float


@dataclass(frozen=True, slots=True)
class RosterSession:
    """A roster conversation: the row, its compression tip (``resolved_id``), the tip's title, the root's title, the tip's latest dialogue line (80 characters then ``...``), and the tip's stamps and message count."""

    id: SessionId
    resolved_id: SessionId
    title: str
    root_title: str
    preview: str
    started_at: float
    last_active: float
    message_count: int


@dataclass(frozen=True, slots=True)
class RosterWorker:
    """The freshest worker session of the profile."""

    id: SessionId
    source: str
    title: str
    last_active: float


@dataclass(frozen=True, slots=True)
class RosterSessions:
    """The Bots roster's session fields."""

    canonical_session: Optional[RosterSession]
    last_session: Optional[RosterSession]
    worker_session: Optional[RosterWorker]


@dataclass(frozen=True, slots=True)
class RecentSessionEntry:
    """One conversation of the recent browse, projected to its live tip: the tip's id and columns, ``COALESCE(tip.title, root.title)``, the root's ``started_at``, ``lineage_root_id`` when the tip is not the root."""

    session_id: SessionId
    source: str
    model: Optional[str]
    title: Optional[str]
    started_at: float
    ended_at: Optional[float]
    end_reason: Optional[str]
    message_count: int
    tool_call_count: int
    last_active: Optional[float]
    preview: str
    lineage_root_id: Optional[SessionId]
    unread: bool


@dataclass(frozen=True, slots=True)
class RecentBrowseQuery:
    """Bounds of ``list_recent_sessions``, clamped at construction as the store clamps them: ``limit`` at least 1, ``timeout_seconds`` at least 0, ``candidate_limit`` defaulting to ``max(128, limit * 8)`` within ``[limit, 2048]``, ``lineage_limit`` defaulting to ``min(8192, candidate_limit * 8)`` within ``[candidate_limit, 8192]``."""

    limit: int = 20
    exclude_sources: tuple[str, ...] = ()
    timeout_seconds: float = 3.0
    candidate_limit: Optional[int] = None
    lineage_limit: Optional[int] = None

    def __post_init__(self) -> None:
        limit = max(1, int(self.limit))
        candidate = self.candidate_limit if self.candidate_limit is not None else max(128, limit * 8)
        candidate = max(limit, min(int(candidate), RECENT_BROWSE_CANDIDATE_CAP))
        lineage = self.lineage_limit if self.lineage_limit is not None else min(RECENT_BROWSE_LINEAGE_CAP, candidate * 8)
        lineage = max(candidate, min(int(lineage), RECENT_BROWSE_LINEAGE_CAP))
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "timeout_seconds", max(0.0, float(self.timeout_seconds)))
        object.__setattr__(self, "candidate_limit", candidate)
        object.__setattr__(self, "lineage_limit", lineage)


@dataclass(frozen=True, slots=True)
class SessionCwdUsage:
    """One distinct working directory: how many rows recorded it, and the newest ``ended_at``-else-``started_at`` among them."""

    cwd: str
    sessions: int
    last_active: float


@dataclass(frozen=True, slots=True)
class SkillScaffoldedSession:
    """A titled session whose first user turn was a skill scaffold, with that turn's content."""

    session_id: SessionId
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SessionFacetEntry:
    """One facet-listing row with its ``last_active`` and first-user-turn ``preview``."""

    session: SessionRecord
    last_active: Optional[float]
    preview: str


@dataclass(frozen=True, slots=True)
class RuntimeLock:
    """A Browser or API-client runtime model lock; the stored payload adds ``updated_at``."""

    model: Optional[str] = None
    provider: Optional[str] = None
    model_options: Optional[Mapping[str, JsonValue]] = None
    route_source: Optional[str] = None
    confirmed: bool = False

    def as_mapping(self) -> dict[str, JsonValue]:
        """The stored lock payload without ``updated_at``."""
        return {
            "provider": self.provider or "",
            "model": self.model or "",
            "model_options": dict(self.model_options or {}),
            "route_source": self.route_source or "",
            "confirmed": bool(self.confirmed),
        }


@dataclass(frozen=True, slots=True)
class ReplaceSessionConfig:
    """``replace_session_config``'s command: the whole ``model_config`` and a model that, when given, overwrites the stored one."""

    model_config: Mapping[str, JsonValue]
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model is not None and not self.model:
            raise ValueError("ReplaceSessionConfig.model must be None or non-empty")


@dataclass(frozen=True, slots=True)
class LiveListingRow:
    """The live listing's columns of one row, with the ``ended_at`` of its local receipt's current target."""

    id: SessionId
    source: str
    title: Optional[str]
    started_at: float
    ended_at: Optional[float]
    message_count: int
    model: Optional[str]
    hidden: bool
    archived: bool
    runtime_revision: int
    runtime_generation: int
    target_ended_at: Optional[float]


@dataclass(frozen=True, slots=True)
class PendingAdmissionRef:
    """One queued, started or unknown admission of a session."""

    admission_id: str
    seq: int
    status: str


@dataclass(frozen=True, slots=True)
class LiveListingFacts:
    """``rows`` for the ids that exist; ``pending`` the non-terminal admissions per id, by ``seq``."""

    rows: Mapping[SessionId, LiveListingRow]
    pending: Mapping[SessionId, tuple[PendingAdmissionRef, ...]]


#: One of ``session_rules.onboarding_flow``'s event classes; ``advance_onboarding_run`` hands it to that module's ``apply`` without reading it.
OnboardingEvent: TypeAlias = Any

#: The ``Step`` that ``session_rules.onboarding_flow.apply`` returns; ``OnboardingAdvance`` carries it back without reading it.
OnboardingStep: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class OnboardingAdvance:
    """The outcome of ``advance_onboarding_run``: the compression tip it wrote (``None`` without a row), whether the stored run was in progress before the event, and the step ``session_rules.onboarding_flow.apply`` returned (``None`` when the tip holds no run record)."""

    session_id: Optional[SessionId]
    was_active: bool
    step: Optional[OnboardingStep]


# ── protocol ───────────────────────────────────────────────────────────────


@runtime_checkable
class SessionRepository(Protocol):
    """Session rows of one store."""

    # ── writes ──

    def create(self, spec: NewSession) -> SessionId:
        """Upsert the row and return its id. An existing row keeps every column an earlier writer set (a placeholder ``unknown`` source takes the spec's; ``user_id`` is never filled later) and fills the unset ones; a config holding only a ``_reset_from`` marker takes the spec's config with the marker kept. A child fills unset cwd and git columns from its parent, and its owner within the same ``agent:<ns>:`` key namespace; a compression continuation (parent ended by compression, not a delegate or branch fork of it) also fills its routing and peer columns. A retired absent id raises ``RetiredSessionError('not_found')``."""
        ...

    def create_exclusive(
        self,
        session_id: SessionId,
        *,
        source: str,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, JsonValue]] = None,
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        """Insert a bare row only if the id is new and not retired, with the sanitized title (no title source; ``created_source`` and ``profile_name`` unset), in one transaction; returns the row. An existing row raises :class:`SessionExists`, a retired id ``RetiredSessionError('session_deleted')``, a held title :class:`TitleConflict` (nothing kept), an oversized title ``ValueError``."""
        ...

    def end(self, session_id: SessionId, reason: str) -> None:
        """Stamp ``ended_at`` and ``reason`` on a live row; the first end wins. An end that landed advances the peer's conversation generation."""
        ...

    def reopen(self, session_id: SessionId) -> None:
        """Retire never-drained queued prompt rows, freeze markerless legacy reset children as ``_reset_from`` children, clear the end stamp, and un-archive a lineage only the idle sweep archived."""
        ...

    def retire_undrained_queue_rows(self, session_id: SessionId) -> int:
        """Deactivate the session's still-active user rows marked as never-drained queued prompts; returns how many. ``ended_at`` is untouched."""
        ...

    def promote_to_reset(self, session_id: SessionId, reason: str = END_REASON_SESSION_RESET) -> bool:
        """End a live row, or overwrite a recoverable end, with ``reason``; True when written. An empty id or any store failure reads False."""
        ...

    def reopen_if_explicitly_closed(
        self, session_id: SessionId, *, provenance: str, patience_s: Optional[float] = None
    ) -> Optional[str]:
        """Clear an explicit-close stamp (:func:`is_explicit_close_reason`) from an ended row with no continuation child, as a compare-and-swap on the stamp read; returns the reason cleared, else ``None``. ``provenance`` labels the store's log line; ``patience_s`` bounds the write wait (exhausted: ``StoreContention``); zero or negative still makes one attempt."""
        ...

    def discard_unadmitted(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        """Remove a row nobody was admitted against, with its transcript and without retiring the id; returns the removed ids, ``(session_id,)`` when the row existed and empty when nothing was removed. A row with an admission or worker execution takes ``delete`` instead and returns its removed ids, so the caller runs ``after_sessions_removed`` for them as after a ``delete``; ``media_root`` goes to that ``delete``."""
        ...

    def set_title(self, session_id: SessionId, title: Optional[str]) -> bool:
        """Write a user title (empty clears it and its source); True when the row changed. A compression ancestor, a retired canonical Bot Chat or an empty ended ghost yields its title; any other holder raises :class:`TitleConflict`. Renaming the hidden canonical Bot Chat raises :class:`CanonicalTitleLocked`."""
        ...

    def set_auto_title(self, session_id: SessionId, title: str, *, source: TitleSource) -> bool:
        """Write a derived or llm title over an untitled row or a strictly lower provenance; False otherwise and on the hidden canonical Bot Chat. ``TitleSource.USER`` raises ``ValueError``; a held title raises :class:`TitleConflict`."""
        ...

    def set_title_source(self, session_id: SessionId, source: TitleSource) -> bool:
        """Rewrite the provenance of a titled row; False when untitled or missing."""
        ...

    def retitle_canonical_bot_chat(self, session_id: SessionId, title: str) -> bool:
        """Move the canonical title off the row: set ``title`` with user provenance and unhide it, only while it is still titled :data:`CANONICAL_BOT_CHAT_TITLE`."""
        ...

    def replace_session_config(self, session_id: SessionId, command: ReplaceSessionConfig) -> None:
        """Replace ``model_config`` whole and the model when given."""
        ...

    def patch_model_config(self, session_id: SessionId, patch: Mapping[str, Optional[JsonValue]]) -> None:
        """Merge ``patch`` into ``model_config``; ``None`` removes a key. No-op for an empty id, an empty patch or a missing row."""
        ...

    def advance_onboarding_run(self, session_id: SessionId, event: OnboardingEvent, *, now: float) -> OnboardingAdvance:
        """Apply one onboarding event to the run stored under :data:`ONBOARDING_RUN_KEY` at the session's compression tip and store the result, in one write: the update ``_transition`` in ``agent/onboarding_record.py`` makes.

        The tip is resolved inside the write. A stored run record (a dict) takes ``session_rules.onboarding_flow.apply(record, event, Facts.from_stored(record["facts"]), now=now)`` and the step's record is stored; the outcome carries the tip, ``was_active`` (``is_active`` of the record before the event) and the step. Any other stored value is left as it is and the outcome has no step. Without a tip row nothing is written and the outcome is ``session_id=None, was_active=False, step=None``. ``event`` is one of ``session_rules.onboarding_flow``'s event classes; an error ``apply`` raises propagates and leaves the row as it was.
        """
        ...

    def update_model(
        self,
        session_id: SessionId,
        model: str,
        provider: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
    ) -> None:
        """Set the model, drop the runtime lock, and with a provider write the whole route at the top level and under ``gateway_runtime``."""
        ...

    def set_runtime_lock(self, session_id: SessionId, lock: RuntimeLock) -> None:
        """Store the lock under :data:`RUNTIME_LOCK_KEY` with ``updated_at`` now; a lock model that is not ``None`` overwrites the model."""
        ...

    def set_yolo(self, session_id: SessionId, enabled: bool) -> None:
        """Store the YOLO flag in ``model_config``."""
        ...

    def update_system_prompt(self, session_id: SessionId, system_prompt: Optional[str]) -> None:
        """Store the system prompt snapshot byte for byte; ``None`` clears it."""
        ...

    def request_prompt_refresh(self, session_id: Optional[SessionId] = None) -> int:
        """Record an operator request that stored system prompts be rebuilt: for every session of the profile, or for ``session_id`` alone when given (the row need not exist). One counter orders both kinds: in one write the new epoch is one past the newest epoch recorded so far, profile-wide or for any session, and it is stored for the profile or the session. Returns the new epoch."""
        ...

    def update_tool_pin(self, session_id: SessionId, tool_pin: Optional[JsonValue]) -> None:
        """Store the ``tools[]`` pin whole; ``None`` clears it."""
        ...

    def update_cwd(
        self,
        session_id: SessionId,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        *,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        """Set the cwd and claim the next git metadata generation, returned. A new cwd or ``replace_git_meta`` replaces both git fields; the same cwd keeps a stored value unless a new one is given. ``None`` for an empty id or cwd or a missing row."""
        ...

    def publish_git_metadata(
        self,
        session_id: SessionId,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        """Write the non-empty git fields only while ``cwd`` and ``generation`` are still current."""
        ...

    def set_archived(self, session_id: SessionId, archived: bool, *, notify: bool = True) -> ArchiveChange:
        """Set ``archived`` across the compression lineage and clear ``auto_archived``."""
        ...

    def set_pinned(self, session_id: SessionId, pinned: bool) -> bool:
        """Set ``pinned`` across the lineage; pinning also unhides it unless the row is the hidden canonical Bot Chat."""
        ...

    def set_hidden(self, session_id: SessionId, hidden: bool) -> bool:
        """Set ``hidden`` across the lineage."""
        ...

    def set_read(self, session_id: SessionId, read: bool = True) -> bool:
        """Set the ``last_read_at`` watermark across the lineage: now when read, 0 when unread."""
        ...

    def unarchive_recoverable(self, session_id: SessionId) -> bool:
        """Un-archive an archived lineage whose tip ended for a recoverable reason, then clear the tip's end stamp. Any failure reads False."""
        ...

    def delete(
        self,
        session_id: SessionId,
        *,
        expected_ids: Optional[Sequence[SessionId]] = None,
        expected_display: Optional[Mapping[SessionId, Sequence[Mapping[str, JsonValue]]]] = None,
        refuse_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> tuple[SessionId, ...]:
        """Delete the session with its local lineage and recursive delegate children, retiring every id and orphaning other children, in one transaction; returns the removed ids (empty: nothing deleted). ``expected_ids`` and ``expected_display`` fence drift (a mismatch deletes nothing); ``refuse_guarded`` raises ``SessionWriteGuarded`` when a target holds a live lease; ``include_compression_chain`` deletes the whole compression chain as ``delete_many`` does. ``media_root`` is the profile's retained media root, resolved by the caller: a retired admission's media, with the hosted documents its prompt names under that root, become retired-media candidates."""
        ...

    def delete_many(
        self,
        session_ids: Sequence[SessionId],
        *,
        skip_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> DeleteManyOutcome:
        """Delete every existing id like ``delete``, in one transaction; unknown ids are skipped. ``skip_guarded`` keeps a root whose lineage or delegate cascade holds a live lease; ``include_compression_chain`` expands each id to its compression chain; ``media_root`` as for ``delete``."""
        ...

    def delete_if_empty(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        """Delete an untitled row with no messages and no children, not under a live lease and alone in its local lineage; the retirement expires. Returns the removed id (empty when not deleted); ``media_root`` as for ``delete``."""
        ...

    def delete_empty(self, *, media_root: str) -> tuple[SessionId, ...]:
        """Delete every ended, unarchived row with no messages whose ledger is idle, orphaning children; returns the removed ids; ``media_root`` as for ``delete``."""
        ...

    def backfill_repo_roots(self, cwd_to_root: Mapping[str, str]) -> None:
        """Record a repo root on rows whose cwd is a key and whose root is unset."""
        ...

    def backfill_session_owners(self, *, stamp: Optional[str] = None) -> int:
        """Stamp an owner on rows whose owner is unset or blank; returns how many. ``stamp`` is the owner to write (stripped; blank stamps nothing); ``None`` stamps the store's own profile, so a store with no owner stamps nothing."""
        ...

    def backfill_acp_session_cwds(self) -> int:
        """Copy ``model_config.cwd`` into the cwd of ``acp`` rows that lack one; returns how many."""
        ...

    def retag_worker_sessions(self, workspaces_root: str) -> int:
        """Retag ``cli`` rows at or under the root as ``kanban`` once per root (state-meta gate); returns how many."""
        ...

    def touch_activity(
        self,
        session_id: SessionId,
        at: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> None:
        """Stamp ``last_activity_at`` (default now) with its labels when it is newer than the stored stamp. Waits at most the activity patience (``StoreContention``)."""
        ...

    def clear_activity_labels(self, session_id: SessionId) -> None:
        """Clear the activity labels and keep the stamp; skips the write when already clear."""
        ...

    # ── reads ──

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        """The row, after queued usage is flushed; on a thread running an asyncio event loop the read does not wait for the usage writer and sees the committed totals."""
        ...

    def owns_session(self, session_id: SessionId) -> bool:
        """Whether the row exists and its owner is unset or the scope's profile."""
        ...

    def store_owner(self) -> Optional[str]:
        """The profile owning this store, from where the store lives: ``default`` for the store at the default Hermes root, the profile's name for one directly under ``<root>/profiles`` with a valid profile name, else ``None``. It is the owner a row created without ``profile_name`` is stamped with. Reads no row and never raises."""
        ...

    def resolve_id(self, id_or_prefix: str) -> Optional[SessionId]:
        """The exact id, else the one id the prefix matches."""
        ...

    def is_kanban_owned(self, session_id: SessionId) -> bool:
        """Whether any row of the session's resume lineage has ``source`` or ``created_source`` ``kanban``."""
        ...

    def declared_scope_identity(self, session_id: SessionId) -> DeclaredScopeIdentity:
        """``(is_fork_child, source)`` of the row; a missing row reads ``(False, "")``."""
        ...

    def get_title(self, session_id: SessionId) -> Optional[str]:
        """The title."""
        ...

    def get_title_source(self, session_id: SessionId) -> Optional[TitleSource]:
        """The title's provenance; ``None`` when untitled."""
        ...

    def get_by_title(self, title: str) -> Optional[SessionRecord]:
        """The row with exactly this title."""
        ...

    def resolve_by_title(self, title: str) -> Optional[SessionId]:
        """The newest ``"<title> #N"`` row, else the exact title; the canonical Bot Chat title resolves exactly."""
        ...

    def next_title_in_lineage(self, base_title: str) -> str:
        """The base title when unused, else ``"<base> #N"`` past the highest number (the unnumbered title counts as #1)."""
        ...

    def title_holder(self, title: str, *, excluding: SessionId) -> Optional[SessionId]:
        """Another session holding exactly ``title``."""
        ...

    def get_model_config_value(self, session_id: SessionId, key: str, default: JsonValue = None) -> JsonValue:
        """One ``model_config`` key, else ``default``."""
        ...

    def session_ids_with_model_config_key(self, key: str) -> tuple[SessionId, ...]:
        """Ids whose ``model_config`` carries ``key``, newest ``started_at`` first."""
        ...

    def prompt_refresh_epoch(self, session_id: SessionId) -> int:
        """The newest refresh epoch that applies to the session: the larger of the profile-wide epoch and the session's own, 0 when neither was requested."""
        ...

    def count_with_stored_prompt(self) -> int:
        """How many sessions hold a stored system prompt: the ones whose next turn a refresh rebuilds."""
        ...

    def delete_targets(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """What ``delete`` would remove: the session, its local lineage (sorted), its recursive delegate children (sorted); empty when absent."""
        ...

    def is_empty_candidate(self, session_id: SessionId) -> bool:
        """Whether the row is untitled with no messages and no children."""
        ...

    def count_empty(self) -> int:
        """How many rows ``delete_empty`` selects before its ledger check."""
        ...

    def list_sessions(
        self,
        query: SessionQuery,
        window: SessionWindow = SessionWindow(limit=20),
        *,
        compact_rows: bool = False,
    ) -> tuple[SessionListEntry, ...]:
        """One window of the listing, ordered by ``started_at`` or by the chain's last activity, then the missed pins (archived pins included), then each compression root projected to its tip. Earlier segments of a local reset lineage are left out."""
        ...

    def search_sessions(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        workspace_key: Optional[str] = None,
        local_created_sources: Optional[Sequence[str]] = None,
        skip_launch_stubs: bool = False,
        window: SessionWindow = SessionWindow(limit=20),
    ) -> tuple[SessionRecencyEntry, ...]:
        """Every row, most recently active first, filtered by source and workspace; ``skip_launch_stubs`` drops rows with no messages and no admission."""
        ...

    def list_row(self, session_id: SessionId) -> Optional[SessionListRow]:
        """The picker columns of one row."""
        ...

    def summary(self, session_id: SessionId, *, compact_rows: bool = False) -> Optional[SessionListEntry]:
        """One row with the listing facets, unprojected."""
        ...

    def records_created_surface(self) -> bool:
        """Whether the store records the creating surface of its sessions (``created_source``); an older store opened read-only may not, and a listing then filters on the raw source (hermes_cli/gateway_session_store.py:404-410)."""
        ...

    def last_active(self, session_id: SessionId) -> Optional[float]:
        """The session's recency, read without the usage flush: the freshest of ``last_activity_at`` and its newest stored message timestamp, each counted only inside the trusted epoch window, else ``started_at`` inside it; None for a missing row or when no cell is trusted."""
        ...

    def search_by_id(
        self,
        text: str,
        *,
        limit: int = 20,
        include_archived: bool = True,
        sources: Optional[Sequence[str]] = None,
        exclude_sources: Sequence[str] = (),
    ) -> tuple[SessionListEntry, ...]:
        """Listed conversations whose id or chain matches ``text``, exact matches first, then prefixes, then substrings, each in recency order; empty for blank text or ``limit <= 0``."""
        ...

    def count_sessions(self, query: SessionCountQuery = SessionCountQuery()) -> int:
        """How many rows the filters admit."""
        ...

    def count_at_least(self, minimum: int = 1) -> bool:
        """Whether at least ``minimum`` rows exist, archived and hidden included."""
        ...

    def count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False
    ) -> Mapping[str, int]:
        """Row counts per source (empty counts as ``cli``), largest first."""
        ...

    def lifecycle_statuses(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionLifecycleStatus]:
        """Each non-empty id's status from its last transcript row."""
        ...

    def descendants(self, session_id: SessionId) -> tuple[SessionEdge, ...]:
        """The session and every row reachable over continuation edges (not delegate, branch, reset or tool children); empty when absent."""
        ...

    def roster_sessions(self, *, owner_user_id: Optional[str], owner: OwnerRule | None) -> RosterSessions:
        """In one read: the canonical Bot Chat (unarchived; unless ``owner_user_id`` is ``None``, its stored owner belongs to ``owner_user_id`` under ``owner``, ``session_rules.legacy_local.same_owner``, without a rule only the equal stamp), the newest visible local chat (stored owner equal to ``owner_user_id`` unless it is ``None``), and the freshest worker. A store force-closed under the handle (a profile delete's sweep, ``persistence_assembly.close_stores_under``) raises :class:`persistence.PermanentStoreError` instead of opening again."""
        ...

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True
    ) -> tuple[SessionRecencyEntry, ...]:
        """For every routing key the newest-started rows, filtered by platform (ASCII case-insensitive) and liveness, most recently active first."""
        ...

    def list_recent_sessions(self, query: RecentBrowseQuery) -> tuple[RecentSessionEntry, ...]:
        """The latency-bounded recent browse: the most recently active conversations, each projected to its freshest live tip. Past ``timeout_seconds`` raises :class:`RecentBrowseTimeout`."""
        ...

    def list_cron_job_runs(
        self, job_id: str, window: SessionWindow = SessionWindow(limit=20)
    ) -> tuple[CronRunEntry, ...]:
        """The job's run sessions (legacy ``cron_<job>_`` ids and runs its scheduler admissions name), newest first."""
        ...

    def cron_run_identity(self, session_id: SessionId) -> Optional[CronRunIdentity]:
        """The cron job and execution of a run session, else ``None``."""
        ...

    def list_skill_scaffolded_sessions(self, *, limit: int = 200) -> tuple[SkillScaffoldedSession, ...]:
        """Titled sessions whose first user row with content starts with a skill scaffold, newest first."""
        ...

    def list_unlinked_telegram_sessions(
        self, *, user_id: str, profile_name: Optional[str] = None, limit: int = 10
    ) -> tuple[SessionFacetEntry, ...]:
        """This user's Telegram sessions of one profile bound to no topic, most recently active first. ``profile_name`` names the profile, stripped, blank reading as ``default``; ``None`` takes the store's own profile (:meth:`store_owner`, ``default`` when it has none). A row's owner is compared with spaces trimmed, an unset or blank owner reading as ``default``. Before the topic tables exist (``prepare_topic_storage`` never ran) the query is unscoped: every Telegram row of the user, any owner."""
        ...

    def distinct_session_cwds(self, *, include_archived: bool = False) -> tuple[SessionCwdUsage, ...]:
        """Every non-blank cwd with its row count and newest stamp."""
        ...

    def live_listing_facts(self, session_ids: Sequence[SessionId]) -> LiveListingFacts:
        """The live listing's columns of the given rows and their non-terminal admissions, in one read."""
        ...

    def first_session_started_at(self) -> Optional[float]:
        """The earliest ``started_at``, ``None`` without rows."""
        ...

    def longest_session_id(self) -> Optional[SessionId]:
        """The session with the most transcript rows of any visibility; ``None`` only without rows. A store with no transcript names one of its sessions (zero messages); which one among equal counts is unspecified."""
        ...
