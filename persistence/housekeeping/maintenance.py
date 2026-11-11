"""Session maintenance contract: prune, orphan and ghost sweeps, idle archive, logical export and import, foreign import, backend heartbeats and the worktree retention reads.

Every verb is one transaction (the worktree and change-watcher probes are read-only). Composite chores are not verbs: ``session_retention``'s ``run_auto_prune_and_vacuum``, ``run_auto_archive`` and ``archive_sessions``, ``hermes_cli.session_export.assert_exports_safe`` and lineage adoption are helpers outside the store that call these verbs; the plugin hook, retired-media collection and transcript-file removal after a delete run in ``after_sessions_removed`` over the ids a verb returns.

Prune selection: ``older_than_days`` becomes ``last_active_before = now - days * 86400`` unless the criteria set ``last_active_before`` or ``started_before``; a negative value is a ``ValueError``. Only ended rows match unless ``include_open``. Recency (``last_active``) is the freshest of ``last_activity_at``, the newest stored message timestamp and ``started_at``. ``last_active_before`` also requires an orphan-swept row (``end_reason == ORPHAN_REAP_END_REASON``) to have ended before the cutoff. ``*_like`` filters are case-insensitive substrings, ``provider`` is compared lower-cased, ``cwd_prefix`` is the exact cwd or a path below it, the message bounds read the stored ``message_count``, the token bounds input plus output tokens, the cost bounds the actual cost else the estimate else 0, the tool-call bounds the stored ``tool_call_count``; numeric bounds are inclusive and an empty text filter is unset. ``archived`` is tri-state. Pinned rows, and the unpinned continuations of a pinned segment, are excluded unless ``include_pinned``. ``lineage_tips_only`` drops compression-ended rows and rows titled :data:`~persistence.conversation.sessions.CANONICAL_BOT_CHAT_TITLE`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..conversation.messages import TranscriptRow
from ..core.identity import JsonValue, SessionId

__all__ = [
    "AUTO_PRUNE_STALE_OPEN_SOURCES",
    "GHOST_SESSION_MIN_AGE",
    "GHOST_SESSION_SOURCE",
    "IMPORT_MAX_MESSAGES_PER_SESSION",
    "IMPORT_MAX_SESSIONS",
    "IMPORT_MAX_SESSION_BYTES",
    "IMPORT_MAX_TOTAL_BYTES",
    "IMPORT_MAX_TOTAL_MESSAGES",
    "ORPHAN_REAP_END_REASON",
    "PRUNE_FILTER_NAMES",
    "SESSION_PROJECTION_FIELDS",
    "ExportedMessage",
    "ForeignImportResult",
    "ForeignOrigin",
    "ForeignTurn",
    "ImportResult",
    "PruneCandidate",
    "PruneCriteria",
    "PruneReclaimEstimate",
    "PruneResult",
    "SessionExport",
    "SessionExportRow",
    "SessionLineageExport",
    "SessionMaintenance",
    "TreeServeFacts",
]

#: ``end_reason`` stamped on rows the orphan sweep ends.
ORPHAN_REAP_END_REASON = "startup_orphan_reap"

#: The ghost prune's source: lazy TUI creation leaves untitled empty rows.
GHOST_SESSION_SOURCE = "tui"

#: A ghost row must have started at least this long ago.
GHOST_SESSION_MIN_AGE = timedelta(hours=24)

#: Sources whose stale open rows the automatic prune reaps.
AUTO_PRUNE_STALE_OPEN_SOURCES: tuple[str, ...] = ("cli", "cron", "kanban", "acp", "api_server", "subagent", "tool", "recovered")

#: Keyword names the prune filters accept.
PRUNE_FILTER_NAMES = frozenset({
    "last_active_before", "last_active_after", "started_before", "started_after", "source", "title_like", "end_reason", "cwd_prefix",
    "min_messages", "max_messages", "model_like", "provider", "user_id", "chat_id", "chat_type", "branch_like", "min_tokens", "max_tokens",
    "min_cost", "max_cost", "min_tool_calls", "max_tool_calls", "archived", "include_pinned", "lineage_tips_only",
})

#: Import payload limits.
IMPORT_MAX_SESSIONS = 500
IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
IMPORT_MAX_TOTAL_MESSAGES = 50_000
IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

#: The ``sessions`` columns ``session_projection_digest`` hashes, in order (tui_gateway/change_watcher.py:193-198); the activity columns are left out because the heartbeat restamps them.
SESSION_PROJECTION_FIELDS: tuple[str, ...] = (
    "id", "source", "session_key", "display_name", "model", "parent_session_id", "started_at", "ended_at", "end_reason", "message_count",
    "tool_call_count", "cwd", "git_branch", "git_repo_root", "title", "title_source", "profile_name", "archived", "pinned", "hidden",
    "last_read_at", "handoff_state",
)

#: One exported transcript row: the stored row ``MessageRepository.read_transcript`` serves; ``as_wire_row()`` is its exported wire dict.
ExportedMessage = TranscriptRow


@dataclass(frozen=True, slots=True)
class PruneCriteria:
    """The prune filters, in evaluation order; ``source`` is the verbs' positional argument. Times are epoch seconds; an empty text filter is unset. ``include_open`` lifts the ended-session guard (archive only; ``prune_sessions`` refuses it)."""

    last_active_before: Optional[float] = None
    last_active_after: Optional[float] = None
    started_before: Optional[float] = None
    started_after: Optional[float] = None
    title_like: Optional[str] = None
    end_reason: Optional[str] = None
    cwd_prefix: Optional[str] = None
    min_messages: Optional[int] = None
    max_messages: Optional[int] = None
    model_like: Optional[str] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    branch_like: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    min_tool_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    archived: Optional[bool] = None
    include_pinned: bool = False
    lineage_tips_only: bool = False
    include_open: bool = False


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    """One dry-run row: ``archived`` is the stored 0/1; ``ended_at`` is None only for an open row under ``include_open``."""

    session_id: SessionId
    source: str
    title: Optional[str]
    model: Optional[str]
    started_at: float
    last_active: Optional[float]
    ended_at: Optional[float]
    message_count: int
    archived: int


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Outcome of ``prune_sessions``: the count and the deleted ids."""

    count: int
    deleted_ids: tuple[SessionId, ...] = ()


@dataclass(frozen=True, slots=True)
class PruneReclaimEstimate:
    """Outcome of ``prune_reclaim_estimate``: the rows the prune selects, every row, the stored bytes of the selected rows' messages and of every message row. ``prunable_bytes`` is 0 and ``total_bytes`` None when nothing is selected; ``total_bytes`` is None when not measured."""

    prunable: int
    total: int
    prunable_bytes: int
    total_bytes: Optional[int]


@dataclass(frozen=True, slots=True)
class SessionExportRow:
    """One exported ``sessions`` row: the 65 columns as stored, in table order (``model_config`` and ``origin_json`` are the stored text, flags the stored 0/1). ``system_prompt`` is resolved through the prompt store; ``tool_names`` is resolved by ``export_session`` and the stored hash in ``export_all``. ``last_active`` is set by ``export_all`` only; ``reset_from`` and ``branched_from`` are the ``_reset_from`` / ``_branched_from`` markers lifted out of ``model_config``. ``as_dict()`` is the row dict: an unset extra has no key."""

    id: SessionId
    runtime_revision: int
    runtime_generation: int
    source: str
    created_source: Optional[str]
    user_id: Optional[str]
    session_key: Optional[str]
    chat_id: Optional[str]
    chat_type: Optional[str]
    thread_id: Optional[str]
    display_name: Optional[str]
    origin_json: Optional[str]
    expiry_finalized: int
    model: Optional[str]
    model_config: Optional[str]
    system_prompt: Optional[str]
    system_prompt_hash: Optional[str]
    parent_session_id: Optional[SessionId]
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
    cwd: Optional[str]
    git_branch: Optional[str]
    git_repo_root: Optional[str]
    git_metadata_generation: int
    billing_provider: Optional[str]
    billing_base_url: Optional[str]
    billing_mode: Optional[str]
    estimated_cost_usd: Optional[float]
    actual_cost_usd: Optional[float]
    cost_status: Optional[str]
    cost_source: Optional[str]
    pricing_version: Optional[str]
    title: Optional[str]
    title_source: Optional[str]
    last_activity_at: Optional[float]
    last_activity_description: Optional[str]
    last_activity_provenance: Optional[str]
    api_call_count: int
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
    archived: int
    auto_archived: int
    pinned: int
    hidden: int
    last_read_at: Optional[float]
    tool_names: Optional[str]
    transcript_revision: int
    last_active: Optional[float] = None
    reset_from: Optional[SessionId] = None
    branched_from: Optional[SessionId] = None

    def as_dict(self) -> dict[str, JsonValue]:
        """The row dict: the columns, then ``last_active``, ``_reset_from`` and ``_branched_from`` when set."""
        out: dict[str, JsonValue] = {f.name: getattr(self, f.name) for f in fields(self)[:-3]}
        extras = (("last_active", self.last_active), ("_reset_from", self.reset_from), ("_branched_from", self.branched_from))
        out.update((key, value) for key, value in extras if value is not None)
        return out


@dataclass(frozen=True, slots=True)
class SessionExport:
    """One exported session: its row and its messages in storage order. Its dict adds ``messages`` (wire rows) and ``timings`` (``export_timings`` over them) to the row."""

    session: SessionExportRow
    messages: tuple[ExportedMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionLineageExport:
    """One compression lineage exported as one logical session: the segments root to tip, their ids, and the merged messages and their count. Its dict is the tip's dict with ``segments``, ``lineage_session_ids``, ``message_count``, ``messages`` and ``timings`` replaced."""

    segments: tuple[SessionExport, ...]
    lineage_session_ids: tuple[SessionId, ...]
    message_count: int
    messages: tuple[ExportedMessage, ...]


@dataclass(frozen=True, slots=True)
class ForeignOrigin:
    """Provenance of a foreign coding-agent snapshot (hermes_cli/foreign_sessions_browser.py): the tool (also the imported session's source), the snapshot path, and the tool's own session id when it has one. Identity is ``foreign_session_id`` when set, else ``path``."""

    tool: str
    path: str
    foreign_session_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("ForeignOrigin.tool must be non-empty")
        if not self.path:
            raise ValueError("ForeignOrigin.path must be non-empty")

    def as_dict(self) -> dict[str, JsonValue]:
        """The ``imported_from`` payload stored in ``origin_json``."""
        return {"tool": self.tool, "path": self.path, "foreign_session_id": self.foreign_session_id}


@dataclass(frozen=True, slots=True)
class ForeignTurn:
    """One imported turn: the foreign parsers yield role-alternating user and assistant text."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("ForeignTurn.role must be non-empty")


@dataclass(frozen=True, slots=True)
class ForeignImportResult:
    """Outcome of ``import_foreign_history``: the session id and whether it was already imported."""

    session_id: SessionId
    already_imported: bool


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Outcome of ``import_sessions``. A refused payload is ``ok=False`` with nothing written: ``errors`` holds one ``{"index", "error"[, "session_id"]}`` mapping per rejected entry, or one text per already-deleted id."""

    ok: bool
    imported: int
    skipped: int
    detached: int
    imported_ids: tuple[SessionId, ...] = ()
    skipped_ids: tuple[SessionId, ...] = ()
    errors: tuple[JsonValue, ...] = ()


@dataclass(frozen=True, slots=True)
class TreeServeFacts:
    """What the store says about a retained worktree (hermes_cli/worktree_ops.py:810-835): ``served`` when a session inside it is open or holds an active loop, heartbeat or goal; otherwise the local chat ids of its sessions, for the caller's cron check."""

    served: bool
    chat_ids: frozenset[str] = frozenset()


@runtime_checkable
class SessionMaintenance(Protocol):
    """Maintenance verbs on the handle's bound profile; see the module docstring for the prune selection."""

    def prune_sessions(self, older_than_days: Optional[float] = 90, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), exclude_active_write_guards: bool = False, media_root: str) -> PruneResult:
        """Delete the ended rows the whole-lineage selection matches, in one write: a compression ancestor is kept while any continuation after it is not selected; with ``exclude_active_write_guards`` a row under a live turn lease or compression lock is skipped (expired guards are reclaimed); a row whose ledger holds work is skipped; deleted ids are retired. Children of deleted rows are detached, never cascaded. ``criteria.include_open`` raises ``ValueError``. ``media_root`` as for ``SessionRepository.delete``."""
        ...

    def prune_empty_ghost_sessions(self, *, media_root: str) -> tuple[SessionId, ...]:
        """Delete every ended, untitled ``GHOST_SESSION_SOURCE`` row that started more than ``GHOST_SESSION_MIN_AGE`` ago and holds no message row, skipping rows whose ledger holds work, in one write; return the deleted ids. ``media_root`` as for ``SessionRepository.delete``."""
        ...

    def sweep_orphaned_sessions(self, *, max_idle_seconds: float, sources: tuple[str, ...] = ('tui', 'desktop', 'subagent'), exclude_ids: tuple[str, ...] = (), exclude_pinned: bool = False, heartbeat_staleness_seconds: Optional[float] = None, heartbeat_ownership_grace_seconds: Optional[float] = None, respect_gateway_heartbeats: bool = True) -> tuple[SessionId, ...]:
        """End the open rows of ``sources`` whose ``started_at`` and recency are both older than ``max_idle_seconds``, with ``end_reason = ORPHAN_REAP_END_REASON``, and return their ids. ``max_idle_seconds <= 0`` or no non-empty source returns () without store access. Spared: ``exclude_ids``, pinned rows (and continuations of a pinned segment) under ``exclude_pinned``, rows under a live turn lease or compression lock, and, with ``respect_gateway_heartbeats``, rows a live backend could own (a heartbeat within the staleness window, default ``2 * max_idle_seconds``, whose ``started_at`` is at most the row's ``started_at`` plus the grace, default the staleness). Selection and update are one write."""
        ...

    def auto_archive_lineage(self, session_id: SessionId) -> bool:
        """The idle sweep's archive of one match: archive the session's compression lineage in one write, stamping ``auto_archived`` only on rows this call archives. True when the lineage exists."""
        ...

    def import_foreign_history(self, origin: ForeignOrigin, messages: Sequence[ForeignTurn], *, title: str, cwd: Optional[str], profile: str) -> ForeignImportResult:
        """Adopt or mint the session for ``origin``. The id is minted and the payload validated by the ``import_sessions`` rules before the write (a refusal raises ``ValueError`` with the first error text); then one write re-checks the origin (already imported: its id with ``already_imported=True``), suffixes a title another row holds with `` (<last 12 id characters>)``, inserts the row and its messages as an import does, and stores ``{"imported_from": origin}`` as ``origin_json`` and ``profile`` as ``profile_name``."""
        ...

    def register_backend_heartbeat(self, *, backend_id: str, pid: int, started_at: float, last_heartbeat: Optional[float] = None, profile: str = '', host: str = '') -> None:
        """Upsert this backend's liveness row by ``backend_id``; ``last_heartbeat`` None stamps now. An empty ``backend_id`` writes nothing."""
        ...

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        """Delete the heartbeat row of ``backend_id``; True when a row was deleted. An empty id returns False without a write."""
        ...

    def import_sessions(self, sessions: Sequence[Mapping[str, JsonValue]]) -> ImportResult:
        """Import session dicts as ``export_session`` / ``export_all`` produce them. More than ``IMPORT_MAX_SESSIONS`` entries raises ``ValueError``. Every entry is validated first (an object with an id, no duplicate id, a message list within the per-session and total limits, typed text, JSON and number fields); any rejection returns ``ok=False`` with every error and writes nothing. In one write: an id deleted here refuses the whole payload by name; an existing id is skipped; each new row is inserted with ``source`` defaulting to ``import``, the flags kept (``auto_archived`` only on an archived row), the routing, handoff, rewind and activity columns reset, and its messages inserted (a message with a false ``active`` restored archived, compacted per its flag); the counters count live rows. A parent edge is kept when the parent exists in the store or the payload and closes no cycle, else the child is detached and counted."""
        ...

    def list_prune_candidates(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = False) -> tuple[PruneCandidate, ...]:
        """The matching rows ordered by recency then ``started_at``; ``whole_lineages`` applies the prune's ancestor rule."""
        ...

    def count_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), pinned_only: bool = False) -> int:
        """The number of rows ``list_prune_candidates`` would return without ``whole_lineages``; ``pinned_only`` counts only rows carrying the pin themselves, pinned rows included."""
        ...

    def count_open_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria()) -> int:
        """The number of open rows the same selection would match with only the ended guard inverted; 0 without a read when ``criteria.include_open`` is set."""
        ...

    def prune_reclaim_estimate(self, older_than_days: Optional[float], source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = True, measure_total_bytes: bool = True) -> PruneReclaimEstimate:
        """The prune's selection counted against every row, then, only when it selects a row, the stored size of the selected rows' messages and, with ``measure_total_bytes``, of every message row (hermes_cli/prune_advisor.py:171-191). The selection and its cutoff are built once, as ``list_prune_candidates`` builds them. A message row's size is the summed stored length of its text and binary fields; active and inactive rows both count. A negative ``older_than_days`` raises ``ValueError``."""
        ...

    def list_idle_archive_candidates(self, idle_days: float, *, exclude_pinned: bool = True) -> tuple[SessionId, ...]:
        """The unarchived lineage tips (``end_reason`` not ``compression``) whose recency is older than ``idle_days``, ordered by ``started_at``; pinned rows and continuations of a pinned segment are left out under ``exclude_pinned``, and the hidden canonical Bot Chat always. A negative ``idle_days`` returns ()."""
        ...

    def export_session(self, session_id: SessionId, include_compacted: bool = False, include_inactive: bool = False) -> Optional[SessionExport]:
        """The session's row and its messages, or None for an unknown id. Messages are the active rows; ``include_compacted`` reads the display projection (compaction-archived turns included, deduplicated); ``include_inactive`` every row in storage order with its flags. Queued usage is flushed before the row is read."""
        ...

    def export_session_lineage(self, session_id: SessionId, include_compacted: bool = False, include_inactive: bool = False) -> Optional[SessionLineageExport]:
        """The compression lineage of ``session_id`` exported root to tip: the lineage holds the later segments only when ``session_id`` itself is part of it; an explicit fork child is its own lineage. None for an unknown id."""
        ...

    def export_all(self, source: Optional[str] = None, include_compacted: bool = False, include_inactive: bool = False) -> tuple[SessionExport, ...]:
        """Every session (of ``source`` when given) with its messages, ordered by recency, ``started_at`` and id descending, at most 100000. Rows carry ``last_active`` and the stored ``tool_names``; no usage flush."""
        ...

    def assert_export_safe(self, session_id: SessionId, max_messages: int) -> int:
        """The session's stored message rows (every visibility) counted up to ``max_messages + 1``; above ``max_messages`` raises ``ExportTooLarge(session_id, message_count, limit)``. ``max_messages == 0`` returns 0 without a read; a negative value raises ``ValueError``."""
        ...

    def find_foreign_import(self, origin: ForeignOrigin) -> Optional[SessionId]:
        """The session imported from ``origin``: a row of source ``origin.tool`` whose ``origin_json.imported_from`` names the tool and the same foreign id, or, without one, the same path."""
        ...

    def session_projection_digest(self, *, busy_timeout_s: float = 0.05) -> Optional[tuple[tuple[str, ...], bytes]]:
        """The ``SESSION_PROJECTION_FIELDS`` the store has and a 16-byte digest over those columns of every row ordered by id (tui_gateway/change_watcher.py:205-262); None when the store has no sessions table. Contention past ``busy_timeout_s`` raises ``StoreContention``."""
        ...

    def retained_tree_is_live(self, worktree_path: str, *, pending_only: bool = False) -> bool:
        """Whether a session whose cwd is the tree or below it has a non-terminal admission, or (without ``pending_only``) any message (hermes_cli/worktree_ops.py:886-907)."""
        ...

    def tree_served_sessions(self, worktree_path: str) -> TreeServeFacts:
        """Whether a session inside the tree is open or has an active ``loop:``, ``heartbeat:`` or ``goal:`` state row (a row that is not JSON, or has no status, counts as active); when not served, the chat ids of the sessions inside it (hermes_cli/worktree_ops.py:804-835)."""
        ...
