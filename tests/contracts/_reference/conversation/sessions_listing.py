"""Listings, counts and multi-row reads of the in-memory session reference, mixed into ``InMemorySessionRepository`` (``sessions.py``), which supplies the row storage and the record builder."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from persistence import ProfileScope, RowVisibility, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import (
    AUTO_LOAD_SCAFFOLD_PREFIX,
    BOUNDARY_END_REASONS,
    CANONICAL_BOT_CHAT_TITLE,
    END_REASON_COMPRESSION,
    INTERNAL_LISTING_SOURCES,
    LOCAL_CHAT_PREFIX,
    SKILL_SCAFFOLD_PREFIX,
    CronRunEntry,
    CronRunIdentity,
    LiveListingFacts,
    LiveListingRow,
    PendingAdmissionRef,
    PickerFacts,
    PickerRoot,
    PickerSegment,
    RecentBrowseQuery,
    RecentBrowseTimeout,
    RecentSessionEntry,
    RosterSession,
    RosterSessions,
    RosterWorker,
    SessionCountQuery,
    SessionCwdUsage,
    SessionEdge,
    SessionFacetEntry,
    SessionLifecycleStatus,
    SessionLineageKind,
    SessionListEntry,
    SessionListRow,
    SessionOrder,
    SessionQuery,
    SessionRecencyEntry,
    SessionRecord,
    SessionWindow,
    SkillScaffoldedSession,
)
from session_rules.legacy_local import same_owner

from .._authority import authority_state
from .sessions_titles_config import ascii_lower

_PREVIEW_MAX_CHARS = 60
# The store's reset boundaries: a child riding its parent's routing key after one is a legacy reset child.
_RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")
_CRON_PRINCIPAL_ID = "cron-owner"
_CRON_REQUEST_PREFIX = "cron:"
_LEGACY_CRON_RUN_ID = re.compile(r"^cron_(.+)_\d{8}_\d{6}$")
_LEGACY_CRON_STAMP = re.compile(r"[0-9]{8}_[0-9]{6}")
_RECEIPT_PREFIX = "binding:"


def recency_key(last_active: Optional[float]) -> tuple[bool, float]:
    """A recency as a sort key: a row with none ranks below every row with one, as the store's ``NULL`` does in a descending order."""
    return (last_active is not None, last_active or 0.0)


def activity_or_start(row: Mapping[str, Any]) -> float:
    """The stored activity stamp, else ``started_at``, as an epoch float."""
    stamp: Optional[float] = row["last_activity_at"]
    return stamp if stamp is not None else row["started_at"].timestamp()


def _shape_preview(raw: Any) -> str:
    """The short first-turn preview a listing shows."""
    from ._turn_policy import skill_preview_text

    text = str(raw or "").strip()
    if not text:
        return ""
    text = skill_preview_text(text.replace("\n", " ").replace("\r", " "))
    return text[:_PREVIEW_MAX_CHARS] + "..." if len(text) > _PREVIEW_MAX_CHARS else text


def _cron_run_from_request(request_id: Any, job_id: Optional[str] = None) -> Optional[tuple[str, str]]:
    """``(job_id, execution_id)`` of a cron fire's request id."""
    if not isinstance(request_id, str) or not request_id.startswith(_CRON_REQUEST_PREFIX):
        return None
    if job_id is None:
        rest = request_id[len(_CRON_REQUEST_PREFIX):]
        if rest.count(":") != 1:
            return None
        job_id, _, execution_id = rest.partition(":")
    else:
        prefix = f"{_CRON_REQUEST_PREFIX}{job_id}:"
        execution_id = request_id[len(prefix):] if request_id.startswith(prefix) else ""
    return (job_id, execution_id) if job_id and execution_id else None


def _legacy_cron_run_job(session_id: str) -> Optional[str]:
    """The job a pre-unified ``cron_{job}_{YYYYmmdd_HHMMSS}`` run names."""
    match = _LEGACY_CRON_RUN_ID.match(str(session_id or ""))
    return match.group(1) if match else None


def _is_legacy_cron_run(session_id: str, job_id: str) -> bool:
    """Whether ``session_id`` is ``cron_{job_id}_`` followed by exactly a ``YYYYmmdd_HHMMSS`` stamp: the store's id range and ``GLOB`` of a job's pre-unified runs."""
    prefix = f"cron_{job_id}_"
    return session_id.startswith(prefix) and _LEGACY_CRON_STAMP.fullmatch(session_id[len(prefix):]) is not None


def _under(path: Optional[str], prefix: str) -> bool:
    """The store's cwd-prefix match: the prefix itself, or below it (``/`` or ``\\``, ASCII case folded as ``LIKE`` does)."""
    if not path:
        return False
    head = prefix.rstrip("/\\") or prefix
    folded = ascii_lower(path)
    return path == head or folded.startswith((ascii_lower(head + "/"), ascii_lower(head + "\\")))


def _in_workspace(row: Mapping[str, Any], key: str) -> bool:
    """``workspace_key(row) == key``: the repo root equals it, or a row without one has its cwd at or under it."""
    head = key.rstrip("/\\") or key
    if row["git_repo_root"]:
        return bool(row["git_repo_root"] == head)
    return _under(row["cwd"], head)


def _source_matches(row: Mapping[str, Any], sources: Sequence[str], local_created_sources: Optional[Sequence[str]]) -> bool:
    """The store's source predicate: with ``local_created_sources``, a ``local`` row matches exactly when its creating surface is one of them."""
    if not local_created_sources:
        return row["source"] in sources
    if row["source"] == "local":
        return row.get("created_source") in local_created_sources
    return row["source"] in sources


def _window(rows: list[Any], window: SessionWindow) -> list[Any]:
    if window.limit is None:
        return rows[window.offset :]
    return rows[window.offset : window.offset + window.limit]


class SessionListingMixin:
    """Listing and read verbs; ``InMemorySessionRepository`` supplies ``_backend``, ``_table``, ``_record``, ``_transcripts``, ``_message_count``, ``_last_active`` and ``_classify``."""

    _backend: Any
    _table: Callable[[ProfileScope], dict[str, dict[str, Any]]]
    _record: Callable[..., SessionRecord]
    _transcripts: Callable[[ProfileScope], dict[str, Any]]
    _message_count: Callable[[ProfileScope, Mapping[str, Any]], int]
    _last_active: Callable[[ProfileScope, Mapping[str, Any]], Optional[float]]
    _classify: Callable[[str, bool, Optional[str]], SessionLifecycleStatus]

    def _listed_record(self, scope: ProfileScope, row: Mapping[str, Any]) -> SessionRecord:
        """A listing row's record: the listing and cron reads join only the prompt store, so the pin stays its unresolved hash (``tool_pin`` None)."""
        return self._record(scope, row, resolve_tool_pin=False)

    # ── row predicates ────────────────────────────────────────────────────

    def _counter(self, scope: ProfileScope, session_id: str) -> int:
        """The row's message counter (the store's ``message_count`` column)."""
        transcript = self._transcripts(scope).get(session_id)
        return transcript.counters()[0] if transcript is not None else 0

    @staticmethod
    def _listable_child(table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any]) -> bool:
        """A picker-visible row: a root, a branch child (marker, or a legacy ``branched`` parent ended before it started) or a reset child (marker, or a legacy reset boundary on the same routing key)."""
        config = row["model_config"]
        parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
        if row["parent_session_id"] is None or config.get("_branched_from") is not None or config.get("_reset_from") is not None:
            return True
        if parent is None:
            return False
        if parent["end_reason"] == "branched" and parent["ended_at"] is not None and row["started_at"] >= parent["ended_at"]:
            return True
        return bool(parent["end_reason"] in _RESET_END_REASONS and row["session_key"] and row["session_key"] == parent["session_key"])

    def _admitted_child(self, table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any], include_subagents: bool) -> bool:
        """The children filter of a listing: listable rows, plus delegate runs only with ``include_subagents``."""
        delegate = row["model_config"].get("_delegate_from") is not None
        listable = self._listable_child(table, row)
        return listable or delegate if include_subagents else listable and not delegate

    @staticmethod
    def _archive_admits(row: Mapping[str, Any], query: Any) -> bool:
        if query.archived_only:
            return bool(row["archived"])
        return query.include_archived or not row["archived"]

    def _matches_filters(self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any], query: Any, exclude_children: bool) -> bool:
        """The store's shared listing and count filters over a :class:`SessionQuery` or :class:`SessionCountQuery`."""
        if exclude_children and not self._admitted_child(table, row, query.include_subagents):
            return False
        if query.sources and not _source_matches(row, query.sources, query.local_created_sources):
            return False
        session_key = getattr(query, "session_key", None)
        if session_key and row["session_key"] != session_key:
            return False
        if row["source"] in query.exclude_sources:
            return False
        if query.cwd_prefix and not _under(row["cwd"], query.cwd_prefix):
            return False
        workspace = getattr(query, "workspace_key", None)
        if workspace and not _in_workspace(row, workspace):
            return False
        if query.min_message_count > 0 and self._counter(scope, row["session_id"]) < query.min_message_count:
            return False
        return self._archive_admits(row, query)

    def _local_lineage_index(self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]]) -> tuple[set[str], dict[str, list[str]]]:
        """``(superseded, heads)`` of the local reset lineages: earlier segments a listing leaves out, and the latest segment's whole lineage."""
        superseded: set[str] = set()
        heads: dict[str, list[str]] = {}
        for key, receipt in authority_state(self._backend, scope).snapshot["meta"].items():
            if not (isinstance(key, str) and key.startswith(_RECEIPT_PREFIX) and isinstance(receipt, dict)):
                continue
            lineage = receipt.get("lineage")
            if (not isinstance(receipt.get("session_id"), str) or not isinstance(lineage, list) or len(lineage) < 2
                    or not all(isinstance(sid, str) for sid in lineage)):
                continue
            starts = [sid for i, sid in enumerate(lineage) if i == 0 or (table.get(lineage[i - 1]) or {}).get("end_reason") == "session_reset"]
            superseded.update(starts[:-1])
            heads[starts[-1]] = lineage
        return superseded, heads

    def _forward_chain(self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]], session_id: str) -> list[str]:
        """The compression chain from ``session_id`` forward to its live tip."""
        step = self._backend.lineage._chain_step
        chain = [session_id]
        while (child := step(scope, table, chain[-1])) is not None and child not in chain:
            chain.append(child)
        return chain

    @staticmethod
    def _chain_members(table: Mapping[str, Mapping[str, Any]], session_id: str) -> list[str]:
        """The row and every row reachable forward over compression-continuation edges."""
        from .sessions_delete import _continued_parent

        members = [session_id]
        for current in members:
            members += [sid for sid, row in table.items() if sid not in members and _continued_parent(table, row) == current]
        return members

    # ── listing entries ───────────────────────────────────────────────────

    def _preview(self, scope: ProfileScope, session_id: str) -> str:
        """The shaped first user turn with content that is not hidden, by (timestamp, row id)."""
        from .transcript_store import _epoch

        transcript = self._transcripts(scope).get(session_id)
        rows = [
            stored
            for stored in (transcript.rows if transcript is not None else ())
            if stored.message.role == "user" and stored.message.content is not None and stored.message.display_kind != "hidden"
        ]
        first = min(rows, key=lambda stored: (_epoch(stored.message), stored.id), default=None)
        return _shape_preview(first.message.content) if first is not None else ""

    @staticmethod
    def _projected(root: SessionRecord, tip: SessionRecord) -> SessionRecord:
        """A compression root carrying its tip's surfaced fields; the root keeps its ``started_at`` (stable ordering) and its accounting."""
        return dataclasses.replace(
            root,
            session_id=tip.session_id, ended_at=tip.ended_at, end_reason=tip.end_reason, message_count=tip.message_count,
            tool_call_count=tip.tool_call_count, title=tip.title if tip.title is not None else root.title, model=tip.model,
            system_prompt=tip.system_prompt, cwd=tip.cwd, git_branch=tip.git_branch, git_repo_root=tip.git_repo_root,
        )

    def _entry(
        self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any], *,
        project: bool = False, compact: bool = False, heads: Optional[Mapping[str, list[str]]] = None,
    ) -> SessionListEntry:
        """One listing row; ``project`` projects a compression root to its tip, ``heads`` names a local lineage's segments on its head row."""
        record = self._listed_record(scope, row)
        surfaced = row
        root_id: Optional[str] = None
        chain: list[str] = []
        if project and row["end_reason"] == END_REASON_COMPRESSION:
            chain = self._forward_chain(scope, table, row["session_id"])
            if len(chain) > 1 and chain[-1] in table:
                surfaced = table[chain[-1]]
                record = self._projected(record, self._listed_record(scope, surfaced))
                root_id = row["session_id"]
            else:
                chain = []
        found = (heads or {}).get(root_id or row["session_id"])
        lineage_ids = list(dict.fromkeys([*found, *chain])) if found is not None else chain
        if compact:
            record = dataclasses.replace(record, system_prompt=None, system_prompt_hash=None, git_metadata_generation=0)
        last_active = self._last_active(scope, surfaced)
        return SessionListEntry(
            session=record,
            preview=self._preview(scope, surfaced["session_id"]),
            # Unread: activity (else ``started_at``) postdates the lineage-stamped ``last_read_at`` watermark (unset reads as read).
            unread=record.last_read_at is not None and (last_active or record.started_at) > record.last_read_at,
            last_active=last_active,
            lineage_root_id=SessionId(root_id) if root_id else None,
            lineage_ids=tuple(SessionId(sid) for sid in lineage_ids),
            continuation_kind="compression" if root_id else None,
            compact=compact,
        )

    def _chain_matches(self, table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any], query: SessionQuery) -> bool:
        """The ``id_query`` and ``search_text`` filters: the row's own id or any id of its forward compression chain matches (search also matches titles, and a punctuation-stripped title)."""
        id_needle = (query.id_query or "").strip().lower()
        search = (query.search_text or "").strip().lower()
        if not id_needle and not search:
            return True
        members = [table[sid] for sid in self._chain_members(table, row["session_id"])]
        if id_needle and not any(id_needle in ascii_lower(member["session_id"]) for member in members):
            return False
        compact = re.sub(r"[\W_]+", "", search)
        return not search or any(_text_matches(member, search, compact) for member in members)

    def _effective_last_active(self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any]) -> Optional[float]:
        """The newest ``last_active`` over the row's forward compression chain; None when no member has one."""
        stamps = [stamp for sid in self._chain_members(table, row["session_id"]) if (stamp := self._last_active(scope, table[sid])) is not None]
        return max(stamps, default=None)

    def list_sessions(
        self,
        query: SessionQuery,
        window: SessionWindow = SessionWindow(limit=20),
        *,
        compact_rows: bool = False,
    ) -> tuple[SessionListEntry, ...]:
        scope = self._backend.scope
        table = self._table(scope)
        exclude_children = not query.include_children
        superseded, heads = self._local_lineage_index(scope, table) if exclude_children else (set(), {})
        # The archived-only view is the recovery surface, so it admits hidden rows.
        hidden_ok = query.include_hidden or query.archived_only

        def admitted(filters: SessionQuery) -> list[dict[str, Any]]:
            return [
                row
                for sid, row in table.items()
                if sid not in superseded and (hidden_ok or not row["hidden"]) and self._matches_filters(scope, table, row, filters, exclude_children)
            ]

        rows = admitted(query)
        if query.order is SessionOrder.LAST_ACTIVE:
            rows = [row for row in rows if self._chain_matches(table, row, query)]
            rows.sort(key=lambda row: (*recency_key(self._effective_last_active(scope, table, row)), row["started_at"], row["session_id"]), reverse=True)
        else:
            rows.sort(key=lambda row: (row["started_at"], row["session_id"]), reverse=True)
        rows = _window(rows, window)
        if query.include_pinned:
            # A pin is an explicit keep: the back-fill ignores the archived filter.
            seen = {row["session_id"] for row in rows}
            pins = [row for row in admitted(dataclasses.replace(query, include_archived=True, archived_only=False)) if row["pinned"] and row["session_id"] not in seen]
            rows += sorted(pins, key=lambda row: (row["started_at"], row["session_id"]), reverse=True)
        project = query.project_compression_tips and exclude_children
        return tuple(self._entry(scope, table, row, project=project, compact=compact_rows, heads=heads) for row in rows)

    def search_by_id(
        self,
        text: str,
        *,
        limit: int = 20,
        include_archived: bool = True,
        sources: Optional[Sequence[str]] = None,
        exclude_sources: Sequence[str] = (),
    ) -> tuple[SessionListEntry, ...]:
        needle = (text or "").strip().lower()
        if not needle or limit <= 0:
            return ()
        query = SessionQuery(
            sources=tuple(sources) if sources else None, exclude_sources=tuple(exclude_sources),
            include_archived=include_archived, order=SessionOrder.LAST_ACTIVE, id_query=needle,
        )
        candidates = self.list_sessions(query, SessionWindow(limit=limit * 4))

        def rank(entry: SessionListEntry) -> int:
            ids = [str(value).lower() for value in (entry.session.session_id, entry.lineage_root_id) if value]
            if needle in ids:
                return 0
            return 1 if any(value.startswith(needle) for value in ids) else 2

        ranked = sorted(enumerate(candidates), key=lambda item: (rank(item[1]), item[0]))
        return tuple(entry for _, entry in ranked[:limit])

    def summary(self, session_id: SessionId, *, compact_rows: bool = False) -> Optional[SessionListEntry]:
        scope = self._backend.scope
        table = self._table(scope)
        row = table.get(session_id)
        return None if row is None else self._entry(scope, table, row, compact=compact_rows)

    def list_row(self, session_id: SessionId) -> Optional[SessionListRow]:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return None
        record = self._listed_record(scope, row)
        return SessionListRow(
            session_id=record.session_id, source=record.source, title=record.title, started_at=record.started_at,
            ended_at=record.ended_at, message_count=record.message_count, model=record.model, hidden=record.hidden,
            archived=record.archived,
        )

    # ── counts ────────────────────────────────────────────────────────────

    def _counted(self, scope: ProfileScope, query: SessionCountQuery) -> list[dict[str, Any]]:
        table = self._table(scope)
        superseded = self._local_lineage_index(scope, table)[0] if query.exclude_children else set()
        return [row for sid, row in table.items() if sid not in superseded and self._matches_filters(scope, table, row, query, query.exclude_children)]

    def count_sessions(self, query: SessionCountQuery = SessionCountQuery()) -> int:
        scope = self._backend.scope
        return len(self._counted(scope, query))

    def count_at_least(self, minimum: int = 1) -> bool:
        scope = self._backend.scope
        return len(self._table(scope)) >= minimum

    def count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False
    ) -> Mapping[str, int]:
        scope = self._backend.scope
        query = SessionCountQuery(include_archived=include_archived, archived_only=archived_only, exclude_children=exclude_children)
        counts: dict[str, int] = {}
        for row in self._counted(scope, query):
            source = row["source"] or "cli"
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    # ── recency reads ─────────────────────────────────────────────────────

    def _recency(self, scope: ProfileScope, rows: Sequence[Mapping[str, Any]]) -> list[tuple[Optional[float], Mapping[str, Any]]]:
        """``(last_active, row)`` most recently active first, then newest started, then id."""
        pairs = [(self._last_active(scope, row), row) for row in rows]
        pairs.sort(key=lambda pair: (*recency_key(pair[0]), pair[1]["started_at"], pair[1]["session_id"]), reverse=True)
        return pairs

    def search_sessions(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        workspace_key: Optional[str] = None,
        local_created_sources: Optional[Sequence[str]] = None,
        skip_launch_stubs: bool = False,
        window: SessionWindow = SessionWindow(limit=20),
    ) -> tuple[SessionRecencyEntry, ...]:
        scope = self._backend.scope
        admissions = authority_state(self._backend, scope).snapshot["admissions"].values()
        admitted = {row.get("target_session_id") for row in admissions}
        rows = [
            row
            for sid, row in self._table(scope).items()
            # A launch stub has neither messages nor an admission.
            if not (skip_launch_stubs and self._counter(scope, sid) == 0 and sid not in admitted)
            and (not sources or _source_matches(row, sources, local_created_sources))
            and (not workspace_key or _in_workspace(row, workspace_key))
        ]
        return tuple(
            SessionRecencyEntry(session=self._listed_record(scope, row), last_active=last_active)
            for last_active, row in _window(self._recency(scope, rows), window)
        )

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True
    ) -> tuple[SessionRecencyEntry, ...]:
        scope = self._backend.scope
        keyed = [row for row in self._table(scope).values() if row["session_key"] is not None]
        newest: dict[str, datetime] = {}
        for row in keyed:
            newest[row["session_key"]] = max(newest.get(row["session_key"], row["started_at"]), row["started_at"])
        # Winner pick first (ties kept), filters second.
        rows = [
            row
            for row in keyed
            if row["started_at"] == newest[row["session_key"]]
            and (not platform or ascii_lower(row["source"] or "") == ascii_lower(platform))
            and not (active_only and row["ended_at"] is not None)
        ]
        return tuple(
            SessionRecencyEntry(session=self._listed_record(scope, row), last_active=last_active)
            for last_active, row in self._recency(scope, rows)
        )

    def lifecycle_statuses(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionLifecycleStatus]:
        scope = self._backend.scope
        transcripts = self._transcripts(scope)
        statuses: dict[SessionId, SessionLifecycleStatus] = {}
        for sid in session_ids:
            if not sid:
                continue
            transcript = transcripts.get(str(sid))
            if transcript is not None and transcript.rows:
                last = max(transcript.rows, key=lambda stored: stored.id).message
                statuses[SessionId(sid)] = self._classify(last.role, bool(last.tool_calls), last.finish_reason)
                continue
            # Imported transcripts are transcript rows too; the portable projection carries no tool calls.
            imported = self._backend.scope_store(scope).get("transcripts", {}).get(str(sid), [])
            if imported:
                _, last_imported = max(imported, key=lambda pair: pair[0])
                statuses[SessionId(sid)] = self._classify(last_imported.role, False, None)
                continue
            statuses[SessionId(sid)] = SessionLifecycleStatus.EMPTY
        return statuses

    def descendants(self, session_id: SessionId) -> tuple[SessionEdge, ...]:
        scope = self._backend.scope
        table = self._table(scope)
        if session_id not in table:
            return ()
        children: dict[Optional[str], list[str]] = {}
        for row in table.values():
            if _continues_its_parent(table, row):
                children.setdefault(row["parent_session_id"], []).append(row["session_id"])
        # A seen set: dedupe is what makes a parent cycle terminate.
        seen = {str(session_id)}
        frontier = [str(session_id)]
        while frontier:
            for child in children.get(frontier.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        rows = sorted((table[sid] for sid in seen), key=lambda row: (row["started_at"], row["session_id"]))
        return tuple(
            SessionEdge(session_id=SessionId(row["session_id"]), parent_session_id=row["parent_session_id"], started_at=row["started_at"].timestamp())
            for row in rows
        )

    # ── the bounded recent browse ─────────────────────────────────────────

    @staticmethod
    def _is_listable_root(table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any]) -> bool:
        if row["archived"] or row["hidden"] or row["model_config"].get("_delegate_from") is not None:
            return False
        parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
        if row["parent_session_id"] is None or row["lineage_kind"] in (SessionLineageKind.BRANCH, SessionLineageKind.RESET):
            return True
        # Reset continuation: the parent ended at a deliberate boundary and the child rides its non-empty routing key.
        return bool(parent is not None and parent["end_reason"] in BOUNDARY_END_REASONS and row["session_key"] and row["session_key"] == parent["session_key"])

    def list_recent_sessions(self, query: RecentBrowseQuery) -> tuple[RecentSessionEntry, ...]:
        scope = self._backend.scope
        from .sessions_delete import _continued_parent

        deadline = time.monotonic() + query.timeout_seconds

        def check_deadline() -> None:
            if time.monotonic() >= deadline:
                raise RecentBrowseTimeout(f"recent-session browse exceeded {query.timeout_seconds:g}s deadline")

        table = self._table(scope)
        check_deadline()
        candidates = [
            row for row in table.values()
            if not row["archived"] and not row["hidden"] and row["model_config"].get("_delegate_from") is None and row["source"] not in query.exclude_sources
        ]
        candidates.sort(key=lambda row: (activity_or_start(row), row["started_at"], row["session_id"]), reverse=True)
        children: dict[str, list[str]] = {}
        for sid, row in table.items():
            if (parent := _continued_parent(table, row)) is not None:
                children.setdefault(parent, []).append(sid)
        check_deadline()
        # Up to each candidate's root, then down to every tip, each walk under its own row ceiling; a cut chain yields nothing.
        ancestors = _bounded_walk([row["session_id"] for row in candidates[: query.candidate_limit]], lambda sid: [p] if (p := _continued_parent(table, table[sid])) else [], query.lineage_limit)
        roots = [sid for sid in ancestors if _continued_parent(table, table[sid]) is None]
        chain = _bounded_walk(roots, lambda sid: children.get(sid, []), query.lineage_limit)
        check_deadline()
        tips_by_root: dict[str, list[str]] = {}
        for sid, root_id in chain.items():
            if not children.get(sid):
                tips_by_root.setdefault(root_id, []).append(sid)
        ranked = []
        for root_id, tips in tips_by_root.items():
            if self._is_listable_root(table, table[root_id]):
                last_active, tip = max(((self._last_active(scope, table[sid]), sid) for sid in tips), key=lambda pair: (*recency_key(pair[0]), pair[1]))
                ranked.append((last_active, table[root_id], table[tip]))
        ranked.sort(key=lambda item: (*recency_key(item[0]), item[1]["started_at"], item[2]["session_id"]), reverse=True)
        check_deadline()
        return tuple(self._recent_entry(scope, *item) for item in ranked[: query.limit])

    def _recent_entry(self, scope: ProfileScope, last_active: Optional[float], root: Mapping[str, Any], tip: Mapping[str, Any]) -> RecentSessionEntry:
        record = self._listed_record(scope, tip)
        return RecentSessionEntry(
            session_id=record.session_id,
            source=record.source,
            model=record.model,
            title=record.title if record.title is not None else root["title"],
            started_at=root["started_at"].timestamp(),
            ended_at=record.ended_at,
            end_reason=record.end_reason,
            message_count=record.message_count,
            tool_call_count=record.tool_call_count,
            last_active=last_active,
            preview=self._preview(scope, tip["session_id"]),
            lineage_root_id=SessionId(root["session_id"]) if root["session_id"] != tip["session_id"] else None,
            # The browse selects no ``last_read_at``, so its unread rule always reads False.
            unread=False,
        )

    # ── cron runs ─────────────────────────────────────────────────────────

    def _policy_cron_job(self, meta: Mapping[Any, Any], session_id: str) -> Optional[str]:
        """The cron job a local creation receipt's request names, else ``None``."""
        receipt = meta.get(_RECEIPT_PREFIX + session_id)
        request = (receipt.get("policy") or {}).get("request_json") if isinstance(receipt, dict) else None
        try:
            parsed = json.loads(request) if isinstance(request, str) else request
        except ValueError:
            return None
        job = (parsed.get("cron_job") or {}).get("id") if isinstance(parsed, dict) else None
        return job if isinstance(job, str) else None

    def _cron_admission_run(self, scope: ProfileScope, row: Mapping[str, Any], job_id: Optional[str] = None) -> Optional[tuple[str, str]]:
        """``(job, execution)`` of the first scheduler admission of the session (``job_id``: only that job's)."""
        snapshot = authority_state(self._backend, scope).snapshot
        admissions = sorted(
            (adm for adm in snapshot["admissions"].values() if adm.get("principal_id") == _CRON_PRINCIPAL_ID and adm.get("target_session_id") == row["session_id"]),
            key=lambda adm: adm["seq"],
        )
        for admission in admissions if job_id is not None else admissions[:1]:
            run = _cron_run_from_request(admission["request_id"]) or _cron_run_from_request(admission["request_id"], self._policy_cron_job(snapshot["meta"], row["session_id"]))
            if run is not None and job_id in (None, run[0]):
                return run
        return None

    def list_cron_job_runs(self, job_id: str, window: SessionWindow = SessionWindow(limit=20)) -> tuple[CronRunEntry, ...]:
        scope = self._backend.scope
        table = self._table(scope)
        runs: list[tuple[Mapping[str, Any], Optional[str]]] = []
        for sid, row in table.items():
            if row["source"] != "cron":
                continue
            run = self._cron_admission_run(scope, row, job_id)
            if run is not None:
                runs.append((row, run[1]))
            elif _is_legacy_cron_run(sid, job_id):
                runs.append((row, None))
        runs.sort(key=lambda item: (item[0]["started_at"], item[0]["session_id"]), reverse=True)
        return tuple(
            CronRunEntry(entry=self._entry(scope, table, row), cron_job_id=job_id, cron_execution_id=execution)
            for row, execution in _window(runs, window)
        )

    def cron_run_identity(self, session_id: SessionId) -> Optional[CronRunIdentity]:
        scope = self._backend.scope
        legacy_job = _legacy_cron_run_job(session_id)
        if legacy_job is not None:
            return CronRunIdentity(cron_job_id=legacy_job, cron_execution_id=None)
        run = self._cron_admission_run(scope, {"session_id": str(session_id)})
        return None if run is None else CronRunIdentity(cron_job_id=run[0], cron_execution_id=run[1])

    # ── facets ────────────────────────────────────────────────────────────

    def list_skill_scaffolded_sessions(self, *, limit: int = 200) -> tuple[SkillScaffoldedSession, ...]:
        scope = self._backend.scope
        heads = (ascii_lower(SKILL_SCAFFOLD_PREFIX), ascii_lower(AUTO_LOAD_SCAFFOLD_PREFIX))
        transcripts = self._transcripts(scope)
        matches: list[tuple[Mapping[str, Any], str]] = []
        for sid, row in self._table(scope).items():
            transcript = transcripts.get(sid)
            if row["title"] is None or transcript is None:
                continue
            # The first user row with content over every visibility, by (timestamp, row id).
            users = [stored for stored in transcript.rows if stored.message.role == "user" and stored.message.content is not None]
            first = min(users, key=lambda stored: (stored.message.timestamp, stored.id), default=None)
            content = first.message.content if first is not None else None
            if isinstance(content, str) and ascii_lower(content).startswith(heads):
                matches.append((row, content))
        matches.sort(key=lambda item: (item[0]["started_at"], item[0]["session_id"]), reverse=True)
        return tuple(
            SkillScaffoldedSession(session_id=SessionId(row["session_id"]), title=row["title"], content=content)
            for row, content in (matches[:limit] if limit >= 0 else matches)
        )

    def list_unlinked_telegram_sessions(
        self, *, user_id: str, profile_name: Optional[str] = None, limit: int = 10,
    ) -> tuple[SessionFacetEntry, ...]:
        scope = self._backend.scope
        profile = ((self._backend.owner_profile if profile_name is None else profile_name) or "").strip() or "default"
        namespace = self._backend.scope_store(scope)
        # Without the topic tables the unscoped read runs: no owner or binding filter.
        scoped = bool(namespace.get("topic_tables"))
        bound = {str(binding.session_id) for binding in namespace.get("topic_bindings", {}).values()}
        rows = [
            row
            for sid, row in self._table(scope).items()
            if row["source"] == "telegram"
            and row["user_id"] == str(user_id)
            # A row without an owner reads as the default profile's; the store's TRIM strips spaces only.
            and (not scoped or (((row.get("profile_name") or "").strip(" ") or "default") == profile and sid not in bound))
        ]
        return tuple(
            SessionFacetEntry(session=self._listed_record(scope, row), last_active=last_active, preview=self._preview(scope, row["session_id"]))
            for last_active, row in self._recency(scope, rows)[: max(limit, 0)]
        )

    def distinct_session_cwds(self, *, include_archived: bool = False) -> tuple[SessionCwdUsage, ...]:
        scope = self._backend.scope
        groups: dict[str, list[float]] = {}
        for row in self._table(scope).values():
            cwd = row["cwd"]
            # Blank is empty or spaces only (the store's TRIM default set).
            if cwd is None or not str(cwd).strip(" ") or (row["archived"] and not include_archived):
                continue
            stamp = row["ended_at"] if row["ended_at"] is not None else row["started_at"]
            groups.setdefault(str(cwd), []).append(stamp.timestamp())
        return tuple(SessionCwdUsage(cwd=cwd, sessions=len(stamps), last_active=max(stamps)) for cwd, stamps in sorted(groups.items()))

    def live_listing_facts(self, session_ids: Sequence[SessionId]) -> LiveListingFacts:
        scope = self._backend.scope
        table = self._table(scope)
        snapshot = authority_state(self._backend, scope).snapshot
        rows: dict[SessionId, LiveListingRow] = {}
        for sid in dict.fromkeys(session_ids):
            if sid not in table:
                continue
            record = self._listed_record(scope, table[sid])
            receipt = snapshot["meta"].get(_RECEIPT_PREFIX + sid)
            target_id = (receipt.get("entry") or {}).get("session_id") if isinstance(receipt, dict) else None
            target = table.get(target_id) if isinstance(target_id, str) else None
            rows[SessionId(sid)] = LiveListingRow(
                id=record.session_id, source=record.source, title=record.title, started_at=record.started_at,
                ended_at=record.ended_at, message_count=record.message_count, model=record.model, hidden=record.hidden,
                archived=record.archived, runtime_revision=record.runtime_revision, runtime_generation=record.runtime_generation,
                target_ended_at=target["ended_at"].timestamp() if target is not None and target["ended_at"] is not None else None,
            )
        pending: dict[SessionId, list[PendingAdmissionRef]] = {}
        wanted = set(session_ids)
        for admission_id, row in sorted(snapshot["admissions"].items(), key=lambda item: item[1]["seq"]):
            if row.get("target_session_id") in wanted and row.get("status") in ("queued", "started", "unknown"):
                ref = PendingAdmissionRef(admission_id=str(admission_id), seq=int(row["seq"]), status=str(row["status"]))
                pending.setdefault(SessionId(row["target_session_id"]), []).append(ref)
        return LiveListingFacts(rows=rows, pending={sid: tuple(refs) for sid, refs in pending.items()})

    def sessions_in_workspace(self, session_ids: Sequence[SessionId], workspace: str) -> frozenset[SessionId]:
        table = self._table(self._backend.scope)
        return frozenset(SessionId(sid) for sid in session_ids if sid in table and _in_workspace(table[sid], workspace))

    def picker_session_facts(self, root_ids: Sequence[SessionId], segment_ids: Sequence[SessionId],
                             reply_ids: Sequence[SessionId]) -> PickerFacts:
        scope = self._backend.scope
        table = self._table(scope)
        roots = {SessionId(sid): PickerRoot(cwd=table[sid]["cwd"], git_repo_root=table[sid]["git_repo_root"])
                 for sid in root_ids if sid in table}
        segments = {SessionId(sid): PickerSegment(preview=self._preview(scope, sid), git_branch=table[sid]["git_branch"],
                                                  pinned=bool(table[sid]["pinned"]), last_active=self._last_active(scope, table[sid]))
                    for sid in segment_ids if sid in table}
        return PickerFacts(roots=roots, segments=segments,
                           last_replies={SessionId(sid): self._last_reply(scope, sid) for sid in dict.fromkeys(reply_ids)})

    def _last_reply(self, scope: ProfileScope, session_id: str) -> Optional[str]:
        """The newest of the segment's eight newest shown assistant replies with text whose display projection shows text, shaped as a preview."""
        from session_rules.turns import flatten_message_text, project_compaction_message_for_display

        transcript = self._transcripts(scope).get(session_id)
        replies = [row for row in (transcript.rows if transcript is not None else ())
                   if row.message.role == "assistant" and row.visibility is not RowVisibility.REWOUND
                   and row.message.content not in (None, "") and row.message.display_kind not in ("hidden", "failed_turn")
                   and not (row.message.display_metadata or {}).get("model_only")]
        for row in sorted(replies, key=lambda row: row.id, reverse=True)[:8]:
            shown = project_compaction_message_for_display(
                {"role": "assistant", "content": row.message.content, "_compressed_summary": row.message.compressed_summary})
            text = _shape_preview(flatten_message_text(shown["content"])) if shown is not None else ""
            if text:
                return text
        return None

    def subagent_session_ids(self, subagent_id: str, marker_key: str) -> tuple[SessionId, ...]:
        rows = [row for row in self._table(self._backend.scope).values()
                if row["source"] == "subagent" and (row["model_config"] or {}).get(marker_key) == subagent_id]
        rows.sort(key=lambda row: (row["started_at"], row["session_id"]), reverse=True)
        return tuple(SessionId(row["session_id"]) for row in rows)

    def first_session_started_at(self) -> Optional[float]:
        scope = self._backend.scope
        stamps = [row["started_at"] for row in self._table(scope).values()]
        return min(stamps).timestamp() if stamps else None

    def longest_session_id(self) -> Optional[SessionId]:
        scope = self._backend.scope
        # The most rows over every session: a store without transcripts still names a session; ties keep stored order.
        counts = [(self._message_count(scope, row), sid) for sid, row in self._table(scope).items()]
        if not counts:
            return None
        most = max(count for count, _sid in counts)
        return SessionId(next(sid for count, sid in counts if count == most))

    # ── the Bots roster ───────────────────────────────────────────────────

    def _roster_session(self, scope: ProfileScope, table: Mapping[str, Mapping[str, Any]], root: Mapping[str, Any]) -> RosterSession:
        """A roster conversation: the root, its compression tip's columns and the tip's latest dialogue line."""
        from .lineage import compression_tip

        tip = compression_tip(self._backend, scope, root["session_id"])
        target = table.get(tip) or root
        transcript = self._transcripts(scope).get(tip)
        lines = [
            stored.message.content
            for stored in (transcript.active() if transcript is not None else ())
            if stored.message.role in ("user", "assistant") and str(stored.message.content or "").strip()
        ]
        text = " ".join(str(lines[-1]).split()) if lines else ""
        return RosterSession(
            id=SessionId(root["session_id"]),
            resolved_id=SessionId(tip),
            title=target["title"] or "",
            root_title=root["title"] or "",
            preview=text[:80] + "..." if len(text) > 80 else text,
            started_at=target["started_at"].timestamp(),
            last_active=activity_or_start(target),
            message_count=self._counter(scope, tip),
        )

    def roster_sessions(self, *, owner_user_id: Optional[str], owner: OwnerRule | None) -> RosterSessions:
        scope = self._backend.scope
        self._backend.backend.require_held(scope)
        table = self._table(scope)
        operator = owner_user_id is None

        def local_chat(row: Mapping[str, Any]) -> bool:
            return ascii_lower(row["chat_id"] or "").startswith(LOCAL_CHAT_PREFIX)

        recency = activity_or_start

        canonical = next((row for row in table.values() if row["title"] == CANONICAL_BOT_CHAT_TITLE and local_chat(row)), None)
        canonical_session = None
        if canonical is not None and (operator or same_owner(canonical["user_id"], owner_user_id, owner)) and not canonical["archived"]:
            canonical_session = self._roster_session(scope, table, canonical)
        # The newest conversation a picker shows: a hidden row never wins.
        visible = [
            row for row in table.values()
            if (operator or row["user_id"] == owner_user_id) and local_chat(row) and not row["archived"] and not row["hidden"]
            and (row["source"] or "") not in INTERNAL_LISTING_SOURCES
        ]
        latest = max(visible, key=recency, default=None)
        last_session = None
        if latest is not None:
            last_session = canonical_session if canonical_session is not None and canonical is latest else self._roster_session(scope, table, latest)
        # The freshest worker: no owner filter, a worker's user is its dispatch identity.
        workers = [row for row in table.values() if (row["source"] or "") in INTERNAL_LISTING_SOURCES and not row["archived"] and not row["hidden"]]
        worker = max(workers, key=recency, default=None)
        return RosterSessions(
            canonical_session=canonical_session,
            last_session=last_session,
            worker_session=None if worker is None else RosterWorker(
                id=SessionId(worker["session_id"]), source=(worker["source"] or "").strip().lower(),
                title=worker["title"] or "", last_active=recency(worker),
            ),
        )


def _text_matches(row: Mapping[str, Any], needle: str, compact: str) -> bool:
    """``search_text`` on one chain member: the title or id contains it, or the title with ``-_. `` removed contains the punctuation-stripped needle."""
    title = ascii_lower(row["title"] or "")
    if needle in title or needle in ascii_lower(row["session_id"]):
        return True
    return bool(compact) and compact in re.sub(r"[-_. ]", "", title)


def _continues_its_parent(table: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any]) -> bool:
    """The ``descendants`` edge: no delegate, branch or reset intent (judged by presence, so an inherited marker stops the walk), not a tool row, not a legacy reset child."""
    if row["lineage_kind"] is not SessionLineageKind.NONE or row["source"] == "tool":
        return False
    parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
    key = row["session_key"]
    return not (parent is not None and key and parent["session_key"] == key and parent["end_reason"] in _RESET_END_REASONS)


def _bounded_walk(seeds: list[str], step: Callable[[str], list[str]], ceiling: int) -> dict[str, str]:
    """Breadth-first closure over ``step`` from ``seeds`` (each seed labelled with itself), stopping once ``ceiling`` rows (seeds included) are admitted: a recursive ``UNION`` CTE with a LIMIT on the whole CTE. Returns row id -> seed label."""
    reached: dict[str, str] = {}
    for seed in seeds:
        if len(reached) >= ceiling:
            break
        reached.setdefault(seed, seed)
    queue = list(reached)
    index = 0
    while index < len(queue) and len(reached) < ceiling:
        current = queue[index]
        index += 1
        for nxt in step(current):
            if nxt not in reached and len(reached) < ceiling:
                reached[nxt] = reached[current]
                queue.append(nxt)
    return reached
