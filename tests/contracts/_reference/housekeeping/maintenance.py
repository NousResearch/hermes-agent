"""In-memory reference implementation of ``SessionMaintenance`` (test fixture).

Registered onto the assembled backend as the ``maintenance`` attribute. Session rows are the sessions reference's row dicts and records (its ``_record`` builds the ``SessionRecord`` every read here starts from, usage and transcript counters included); transcripts are the messages reference's stored rows, read back through ``read_transcript``; deletes run through the sessions reference's ledger check and retirement on one staged authority transaction. Heartbeat rows live under the ``"gateway_heartbeats"`` key (one dict per ``backend_id``, epoch floats). Every public method runs under the backend lock, so each verb is one atomic step.

Reference limits: imported usage columns (tokens, cost, billing) are not stored (the usage reference owns them); imported rows keep ``SessionLineageKind.NONE``; shadowed-checkpoint pruning on import is not modelled; ``tool_names`` in ``export_all`` is the resolved text (the reference has no prompt-store hash); ``session_projection_digest`` hashes the reference's own values, so only digest equality within one backend is meaningful.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from persistence import Conflict, ExportTooLarge, ProfileScope, RowVisibility, SessionId
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptRow, TranscriptView
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION, SessionLineageKind, SessionRecord
from persistence.housekeeping.maintenance import (
    GHOST_SESSION_MIN_AGE,
    GHOST_SESSION_SOURCE,
    IMPORT_MAX_SESSIONS,
    ORPHAN_REAP_END_REASON,
    SESSION_PROJECTION_FIELDS,
    ForeignImportResult,
    ForeignOrigin,
    ForeignTurn,
    ImportResult,
    PruneCandidate,
    PruneCriteria,
    PruneReclaimEstimate,
    PruneResult,
    SessionExport,
    SessionExportRow,
    SessionLineageExport,
    TreeServeFacts,
)
from session_rules.imports import IMPORT_MESSAGE_JSON_FIELDS, validate_import_payload
from session_rules.session_ids import new_session_id

from .._authority import authority_state
from .._lease_fence import _lease_rows, lease_guarded
from .._lineage_intent import direct_lineage_kind
from ..backend import InMemoryBackend, register_repository
from ..conversation import _turn_policy as policy
from ..conversation.transcript_store import _messages_namespace, _Transcript, insert_rows, newest_transcript_timestamp
from .state_meta import state_meta_table

_IMPORT_PASSTHROUGH = ("user_id", "model", "end_reason", "cwd", "git_branch", "git_repo_root", "title")
_IMPORT_FLAGS = ("archived", "auto_archived", "pinned", "hidden")
_MESSAGE_PASSTHROUGH = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason", "reasoning", "reasoning_content", "platform_message_id",
    "token_count", "display_kind", "api_content", "message_uid", "tool_call_uid", "reasoning_details", "codex_reasoning_items", "codex_message_items",
)
_DAY = 86400.0


def newest_imported_timestamp(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> Optional[datetime]:
    """The newest timestamp among rows held under the scope's ``"transcripts"`` key, or None. Imports store their rows in the messages reference, so nothing is written there; the sessions and routing references still read it."""
    rows = backend.scope_store(scope).get("transcripts", {}).get(session_id, [])
    return max((message.timestamp for _, message in rows), default=None)


def _epoch(value: Optional[datetime]) -> Optional[float]:
    return None if value is None else value.timestamp()


def _ascii_lower(text: str) -> str:
    return "".join(ch.lower() if ch.isascii() else ch for ch in text)


def _like(haystack: Optional[str], needle: str) -> bool:
    """``LOWER(COALESCE(col, '')) LIKE '%' || needle.lower() || '%'``: SQLite folds ASCII only, Python lowers the needle in full."""
    return _ascii_lower(needle.lower()) in _ascii_lower(haystack or "")


def _under(path: Optional[str], prefix: str) -> bool:
    """The exact cwd (case-sensitive) or a path below it (ASCII-case-insensitive), trailing separators on the prefix ignored."""
    if not path:
        return False
    root = prefix.rstrip("/\\") or prefix
    if path == root:
        return True
    low_path, low_root = _ascii_lower(path), _ascii_lower(root)
    return low_path.startswith(low_root + "/") or low_path.startswith(low_root + "\\")


def _inside_tree(worktree_path: str) -> Callable[[Optional[str]], bool]:
    """``inside(cwd)``: the tree or below it, by native path identity, over the tree's own spelling and its resolved one."""
    def key(value: object) -> str:
        return os.path.normcase(str(value)).replace("\\", "/").rstrip("/")

    roots = {key(Path(worktree_path)), key(Path(worktree_path).resolve())}
    return lambda cwd: bool(cwd) and any(key(cwd) == r or key(cwd).startswith(r + "/") for r in roots)


def _cost(record: SessionRecord) -> float:
    if record.actual_cost_usd is not None:
        return record.actual_cost_usd
    return record.estimated_cost_usd if record.estimated_cost_usd is not None else 0.0


def _bounded(value: float, low: Optional[float], high: Optional[float]) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


# ── import row helpers ──────────────────────────────────────────────


def _coerce_or(value: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _decoded(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _imported_message(message: Mapping[str, Any]) -> NewMessage:
    """One normalized payload message as the stored row payload (the columns ``_insert_message_rows`` writes)."""
    tool_calls = _decoded(message.get("tool_calls"))
    metadata = _decoded(message.get("display_metadata"))
    absorbed = _decoded(message.get("absorbed_message_uids"))
    uids = _decoded(message.get("tool_call_uids"))
    return NewMessage(
        role=message["role"], content=message.get("content"), timestamp=policy.coerce_epoch(message["timestamp"]) if message.get("timestamp") is not None else None,
        tool_calls=tuple(tool_calls) if isinstance(tool_calls, list) else (),
        display_metadata=metadata if isinstance(metadata, dict) else None,
        observed=bool(_coerce_or(message.get("observed"), int, 0)), compressed_summary=bool(message.get("_compressed_summary")),
        absorbed_message_uids=tuple(absorbed) if isinstance(absorbed, list) else (),
        tool_call_uids=uids if isinstance(uids, dict) else None,
        **{field: _decoded(message.get(field)) if field in IMPORT_MESSAGE_JSON_FIELDS else message.get(field) for field in _MESSAGE_PASSTHROUGH},
    )


def _stored_bytes(row: TranscriptRow) -> int:
    """The summed stored length of a message row's text and binary fields: text as UTF-8, structured values as their JSON text."""
    size = 0
    for value in (getattr(row, f.name) for f in dataclasses.fields(row)):
        if isinstance(value, str):
            size += len(value.encode("utf-8", "surrogatepass"))
        elif isinstance(value, (bytes, bytearray)):
            size += len(value)
        elif isinstance(value, (dict, list, tuple)):
            size += len(json.dumps(value).encode("utf-8"))
    return size


def _mint_session_id() -> str:
    """A foreign import's session id, minted with ``new_session_id(hex_len=12)``."""
    return new_session_id(hex_len=12)


def _export_row(record: SessionRecord, *, last_active: Optional[float] = None) -> SessionExportRow:
    """The stored row shape from the reference record: parsed columns back to their stored text, flags to 0/1."""
    columns = {f.name: getattr(record, f.name) for f in dataclasses.fields(SessionExportRow) if hasattr(record, f.name)}
    columns.update(
        id=record.session_id,
        origin_json=None if record.origin is None else json.dumps(dict(record.origin), ensure_ascii=False),
        model_config=json.dumps(dict(record.model_config)) if record.model_config else None,
        title_source=getattr(record.title_source, "value", record.title_source),
        tool_names=None if record.tool_pin is None else json.dumps(record.tool_pin),
        last_active=last_active,
        **{flag: int(getattr(record, flag)) for flag in ("expiry_finalized", "archived", "auto_archived", "pinned", "hidden")},
    )
    return SessionExportRow(**columns)


class InMemoryMaintenance:
    """See module docstring; semantics follow persistence/housekeeping/maintenance.py."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    # ── storage plumbing ──────────────────────────────────────────────────

    @property
    def _scope(self) -> ProfileScope:
        return self._backend.scope

    def _rows(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self._backend.scope_store(self._scope).setdefault("sessions", {})
        return table

    def _heartbeats(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self._backend.scope_store(self._scope).setdefault("gateway_heartbeats", {})
        return table

    def _record(self, row: Mapping[str, Any]) -> SessionRecord:
        return self._backend.sessions._record(self._scope, row)

    def _recency(self, row: Mapping[str, Any]) -> float:
        """The freshest of the activity watermark and the newest stored message timestamp, else ``started_at``."""
        stamps = [stamp for stamp in (row.get("last_activity_at"), _epoch(newest_transcript_timestamp(self._backend, self._scope, row["session_id"]))) if stamp is not None]
        return max(stamps) if stamps else row["started_at"].timestamp()

    @staticmethod
    def _continues(child: Mapping[str, Any], parent: Optional[Mapping[str, Any]]) -> bool:
        """The continuation edge: the parent ended by compression and the child is not a tool session nor a branch, delegate or reset fork of it."""
        if parent is None or parent.get("end_reason") != END_REASON_COMPRESSION or child.get("source") == "tool":
            return False
        if direct_lineage_kind(child) not in (None, SessionLineageKind.NONE):
            return False
        config = child.get("model_config") or {}
        return all(config.get(marker) != parent["session_id"] for marker in ("_branched_from", "_delegate_from", "_reset_from"))

    def _pinned_tail(self) -> set[str]:
        """Unpinned continuations a pinned segment covers."""
        table = self._rows()
        tail: set[str] = set()
        frontier = [sid for sid, row in table.items() if row.get("pinned")]
        while frontier:
            parent = table[frontier.pop()]
            for sid, row in table.items():
                if sid not in tail and row.get("parent_session_id") == parent["session_id"] and self._continues(row, parent):
                    tail.add(sid)
                    frontier.append(sid)
        return tail

    # ── prune selection ───────────────────────────────────────────────────

    @staticmethod
    def _resolved(older_than_days: Optional[float], criteria: PruneCriteria) -> PruneCriteria:
        if older_than_days is None or criteria.last_active_before is not None or criteria.started_before is not None:
            return criteria
        if older_than_days < 0:
            raise ValueError(f"older_than_days must be >= 0, got {older_than_days!r}: a negative retention builds a future cutoff that matches every ended session.")
        return dataclasses.replace(criteria, last_active_before=time.time() - older_than_days * _DAY)

    @staticmethod
    def _times_match(record: SessionRecord, last_active: float, c: PruneCriteria) -> bool:
        if c.last_active_before is not None and not (
            last_active < c.last_active_before
            and (record.end_reason != ORPHAN_REAP_END_REASON or (record.ended_at is not None and record.ended_at < c.last_active_before))
        ):
            return False
        return not (
            (c.last_active_after is not None and last_active < c.last_active_after)
            or (c.started_before is not None and record.started_at >= c.started_before)
            or (c.started_after is not None and record.started_at < c.started_after)
        )

    @staticmethod
    def _text_match(record: SessionRecord, c: PruneCriteria, source: Optional[str]) -> bool:
        return not (
            (source and record.source != source)
            or (c.title_like and not _like(record.title, c.title_like))
            or (c.end_reason and record.end_reason != c.end_reason)
            or (c.cwd_prefix and not _under(record.cwd, c.cwd_prefix))
            or (c.model_like and not _like(record.model, c.model_like))
            or (c.provider and _ascii_lower(record.billing_provider or "") != c.provider.lower())
            or (c.user_id and record.user_id != c.user_id)
            or (c.chat_id and record.chat_id != c.chat_id)
            or (c.chat_type and record.chat_type != c.chat_type)
            or (c.branch_like and not _like(record.git_branch, c.branch_like))
        )

    @staticmethod
    def _number_match(record: SessionRecord, c: PruneCriteria) -> bool:
        return (
            _bounded(record.message_count, c.min_messages, c.max_messages)
            and _bounded(record.input_tokens + record.output_tokens, c.min_tokens, c.max_tokens)
            and _bounded(_cost(record), c.min_cost, c.max_cost)
            and _bounded(record.tool_call_count, c.min_tool_calls, c.max_tool_calls)
        )

    def _selected(self, c: PruneCriteria, source: Optional[str], *, open_rows: bool = False) -> list[tuple[float, SessionRecord]]:
        """``_prune_filter_where`` over every row, ordered by recency then ``started_at``; ``open_rows`` inverts the ended guard."""
        tail = set() if c.include_pinned else self._pinned_tail()
        out = []
        for row in self._rows().values():
            if not c.include_open and (row["ended_at"] is None) != open_rows:
                continue
            record = self._record(row)
            last_active = self._recency(row)
            if (
                (c.lineage_tips_only and record.end_reason == END_REASON_COMPRESSION)
                or (isinstance(c.archived, bool) and record.archived != c.archived)
                or (not c.include_pinned and (record.pinned or record.session_id in tail))
                or not self._times_match(record, last_active, c)
                or not self._text_match(record, c, source)
                or not self._number_match(record, c)
            ):
                continue
            out.append((last_active, record))
        out.sort(key=lambda pair: (pair[0], pair[1].started_at))
        return out

    def _whole_lineages(self, selected: list[tuple[float, SessionRecord]]) -> list[tuple[float, SessionRecord]]:
        """``selected`` minus every compression ancestor of a row it does not hold."""
        table = self._rows()
        chosen = {str(record.session_id) for _, record in selected}
        kept: set[str] = set()
        for row in table.values():
            if row["session_id"] in chosen:
                continue
            child = row
            while (parent := table.get(child.get("parent_session_id") or "")) is not None and self._continues(child, parent) and parent["session_id"] not in kept:
                kept.add(parent["session_id"])
                child = parent
        return [(last_active, record) for last_active, record in selected if record.session_id not in kept]

    def _prune_selection(self, older_than_days: Optional[float], source: Optional[str], criteria: PruneCriteria, whole_lineages: bool) -> list[tuple[float, SessionRecord]]:
        """The rows a prune selects, with the cutoff resolved once; ``whole_lineages`` applies the ancestor rule."""
        selected = self._selected(self._resolved(older_than_days, criteria), source)
        return self._whole_lineages(selected) if whole_lineages else selected

    def list_prune_candidates(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = False) -> tuple[PruneCandidate, ...]:
        selected = self._prune_selection(older_than_days, source, criteria, whole_lineages)
        return tuple(
            PruneCandidate(
                session_id=record.session_id, source=record.source, title=record.title, model=record.model, started_at=record.started_at,
                last_active=last_active, ended_at=record.ended_at, message_count=record.message_count, archived=int(record.archived),
            )
            for last_active, record in selected
        )

    def count_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), pinned_only: bool = False) -> int:
        if pinned_only:
            criteria = dataclasses.replace(criteria, include_pinned=True)
        selected = self._selected(self._resolved(older_than_days, criteria), source)
        return sum(1 for _, record in selected if record.pinned or not pinned_only)

    def count_open_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria()) -> int:
        if criteria.include_open:
            return 0
        return len(self._selected(self._resolved(older_than_days, criteria), source, open_rows=True))

    def prune_reclaim_estimate(self, older_than_days: Optional[float], source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = True, measure_total_bytes: bool = True) -> PruneReclaimEstimate:
        prunable = [str(record.session_id) for _, record in self._prune_selection(older_than_days, source, criteria, whole_lineages)]
        total = len(self._rows())
        if not prunable:
            return PruneReclaimEstimate(prunable=0, total=total, prunable_bytes=0, total_bytes=None)
        transcripts = _messages_namespace(self._backend, self._scope)["transcripts"]

        def measured(session_ids: Any) -> int:
            return sum(_stored_bytes(row.as_transcript_row(sid)) for sid in session_ids if sid in transcripts for row in transcripts[sid].rows)

        return PruneReclaimEstimate(
            prunable=len(prunable), total=total, prunable_bytes=measured(prunable), total_bytes=measured(list(transcripts)) if measure_total_bytes else None,
        )

    def prune_sessions(self, older_than_days: Optional[float] = 90, source: Optional[str] = None, *, criteria: PruneCriteria = PruneCriteria(), exclude_active_write_guards: bool = False, media_root: str) -> PruneResult:
        if criteria.include_open:
            raise ValueError("prune_sessions() never deletes open sessions: include_open is archive-only")
        selected = self._whole_lineages(self._selected(self._resolved(older_than_days, criteria), source))
        scope = self._scope
        sessions = self._backend.sessions
        with authority_state(self._backend, scope).transaction() as staged:
            doomed = sorted(
                str(record.session_id) for _, record in selected
                if not (exclude_active_write_guards and lease_guarded(self._backend, scope, record.session_id))
                and sessions._ledger_idle(staged.authority, str(record.session_id))
            )
            sessions._retire_staged(staged.scope, staged.authority, doomed, media_root)
            messages = staged.scope.get("messages", {})
            for sid in doomed:
                messages.get("transcripts", {}).pop(sid, None)
                messages.get("cooldowns", {}).pop(sid, None)
            for row in staged.scope.get("sessions", {}).values():
                if row["parent_session_id"] in doomed:
                    row["parent_session_id"] = None
        return PruneResult(count=len(doomed), deleted_ids=tuple(SessionId(sid) for sid in doomed))

    def prune_empty_ghost_sessions(self, *, media_root: str) -> tuple[SessionId, ...]:
        cutoff = datetime.now(UTC) - GHOST_SESSION_MIN_AGE
        sessions = self._backend.sessions
        transcripts = _messages_namespace(self._backend, self._scope)["transcripts"]
        with authority_state(self._backend, self._scope).transaction() as staged:
            ghosts = [
                sid for sid, row in staged.scope.get("sessions", {}).items()
                if row["source"] == GHOST_SESSION_SOURCE and row["title"] is None and row["ended_at"] is not None and row["started_at"] < cutoff
                and not (sid in transcripts and transcripts[sid].rows)
                and sessions._ledger_idle(staged.authority, sid)
            ]
            sessions._retire_staged(staged.scope, staged.authority, ghosts, media_root)
        return tuple(SessionId(sid) for sid in ghosts)

    # ── orphan sweep ──────────────────────────────────────────────────────

    def _guarded(self, session_id: str) -> bool:
        """A live turn lease or compression lock protects the row; expired guards are reclaimed."""
        for _kind, table, key, live in _lease_rows(self._backend, self._scope, SessionId(session_id)):
            if live:
                return True
            del table[key]
        return False

    def _owned_by_live_backend(self, started_at: float, live_after: float, grace: float) -> bool:
        return any(hb["last_heartbeat"] >= live_after and hb["started_at"] <= started_at + grace for hb in self._heartbeats().values())

    def sweep_orphaned_sessions(self, *, max_idle_seconds: float, sources: tuple[str, ...] = ('tui', 'desktop', 'subagent'), exclude_ids: tuple[str, ...] = (), exclude_pinned: bool = False, heartbeat_staleness_seconds: Optional[float] = None, heartbeat_ownership_grace_seconds: Optional[float] = None, respect_gateway_heartbeats: bool = True) -> tuple[SessionId, ...]:
        wanted = {source for source in sources if source}
        if max_idle_seconds <= 0 or not wanted:
            return ()
        staleness = heartbeat_staleness_seconds if heartbeat_staleness_seconds and heartbeat_staleness_seconds > 0 else max_idle_seconds * 2
        grace = heartbeat_ownership_grace_seconds if heartbeat_ownership_grace_seconds is not None and heartbeat_ownership_grace_seconds >= 0 else staleness
        now = time.time()
        cutoff = now - max_idle_seconds
        tail = self._pinned_tail() if exclude_pinned else set()
        excluded = {str(sid) for sid in exclude_ids if sid}
        victims = [
            row for sid, row in self._rows().items()
            if row["ended_at"] is None and row["source"] in wanted
            and not (exclude_pinned and (row.get("pinned") or sid in tail))
            and row["started_at"].timestamp() < cutoff and self._recency(row) < cutoff
            and not (respect_gateway_heartbeats and self._owned_by_live_backend(row["started_at"].timestamp(), now - staleness, grace))
            and sid not in excluded and not self._guarded(sid)
        ]
        stamp = datetime.fromtimestamp(time.time(), UTC)
        for row in victims:
            row.update(ended_at=stamp, end_reason=ORPHAN_REAP_END_REASON)
        return tuple(SessionId(row["session_id"]) for row in victims)

    # ── idle archive ──────────────────────────────────────────────────────

    def list_idle_archive_candidates(self, idle_days: float, *, exclude_pinned: bool = True) -> tuple[SessionId, ...]:
        if idle_days is None or idle_days < 0:
            return ()
        cutoff = time.time() - float(idle_days) * _DAY
        tail = self._pinned_tail() if exclude_pinned else set()
        rows = [
            row for sid, row in self._rows().items()
            if not row["archived"] and row["end_reason"] != END_REASON_COMPRESSION
            and not (exclude_pinned and (row.get("pinned") or sid in tail))
            and not (row.get("hidden") and (row.get("title") or "") == CANONICAL_BOT_CHAT_TITLE)
            and self._recency(row) < cutoff
        ]
        rows.sort(key=lambda row: row["started_at"])
        return tuple(SessionId(row["session_id"]) for row in rows)

    def auto_archive_lineage(self, session_id: SessionId) -> bool:
        table = self._rows()
        if session_id not in table:
            return False
        for sid in self._backend.sessions._lineage_ids(table, session_id):
            if sid in table and not table[sid]["archived"]:
                table[sid].update(archived=True, auto_archived=True)
        return True

    # ── logical export ────────────────────────────────────────────────────

    def _messages(self, session_id: SessionId, include_compacted: bool, include_inactive: bool) -> tuple[Any, ...]:
        view = TranscriptView.ALL if include_inactive else (TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE)
        return self._backend.messages.read_transcript(session_id, TranscriptRead(view=view))

    def export_session(self, session_id: SessionId, include_compacted: bool = False, include_inactive: bool = False) -> Optional[SessionExport]:
        record = self._backend.sessions.get(session_id)
        if record is None:
            return None
        return SessionExport(session=_export_row(record), messages=self._messages(session_id, include_compacted, include_inactive))

    def export_session_lineage(self, session_id: SessionId, include_compacted: bool = False, include_inactive: bool = False) -> Optional[SessionLineageExport]:
        segments = tuple(
            segment for segment in (self.export_session(sid, include_compacted, include_inactive) for sid in self._backend.lineage.lineage(session_id))
            if segment is not None
        )
        if not segments:
            return None
        messages = tuple(message for segment in segments for message in segment.messages)
        return SessionLineageExport(segments=segments, lineage_session_ids=tuple(segment.session.id for segment in segments), message_count=len(messages), messages=messages)

    def export_all(self, source: Optional[str] = None, include_compacted: bool = False, include_inactive: bool = False) -> tuple[SessionExport, ...]:
        rows = [row for row in self._rows().values() if not source or row["source"] == source]
        ranked = sorted(((self._recency(row), row) for row in rows), key=lambda pair: (pair[0], pair[1]["started_at"], pair[1]["session_id"]), reverse=True)[:100000]
        return tuple(
            SessionExport(session=_export_row(self._record(row), last_active=last_active), messages=self._messages(SessionId(row["session_id"]), include_compacted, include_inactive))
            for last_active, row in ranked
        )

    def assert_export_safe(self, session_id: SessionId, max_messages: int) -> int:
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        transcript = _messages_namespace(self._backend, self._scope)["transcripts"].get(session_id)
        count = min(len(transcript.rows) if transcript is not None else 0, max_messages + 1)
        if count > max_messages:
            raise ExportTooLarge(session_id, count, max_messages)
        return count

    # ── import ────────────────────────────────────────────────────────────

    def _staged_import(self) -> Any:
        """One transaction over the scope's tables with the sessions and messages tables present in the staged copy."""
        self._rows()
        _messages_namespace(self._backend, self._scope)
        return authority_state(self._backend, self._scope).transaction()

    @staticmethod
    def _insert_imported(tables: dict[str, Any], raw: Mapping[str, Any], messages: Sequence[Mapping[str, Any]], session_id: str) -> None:
        """One normalized session and its messages on the staged ``tables``; the counters count live rows."""
        from ..conversation.sessions import blank_session_row

        config = json.loads(raw["model_config"]) if raw.get("model_config") else None
        row = blank_session_row(session_id, lineage_kind=SessionLineageKind.NONE, model_config=config)
        started = policy.coerce_epoch(raw["started_at"]) if raw.get("started_at") is not None else None
        ended = _finite_or_none(raw.get("ended_at"))
        flags = {flag: bool(_coerce_or(raw.get(flag), int, 0)) for flag in _IMPORT_FLAGS}
        flags["auto_archived"] = flags["auto_archived"] and flags["archived"]
        row.update(
            source=str(raw.get("source") or "import"), system_prompt=raw.get("system_prompt"),
            started_at=datetime.fromtimestamp(time.time() if started is None else started, UTC),
            ended_at=None if ended is None else datetime.fromtimestamp(ended, UTC), parent_session_id=None,
            **{column: raw.get(column) for column in _IMPORT_PASSTHROUGH}, **flags,
        )
        tables["sessions"][session_id] = row
        namespace = tables["messages"]
        transcript = namespace["transcripts"][session_id] = _Transcript()
        stored = insert_rows(namespace, transcript, [_imported_message(message) for message in messages])
        for message, stored_row in zip(messages, stored):
            if not _coerce_or(message.get("active"), int, 1):
                stored_row.visibility = RowVisibility.COMPACTION_ARCHIVED if _coerce_or(message.get("compacted"), int, 0) else RowVisibility.REWOUND

    @staticmethod
    def _attach_parents(table: dict[str, dict[str, Any]], parent_updates: list[tuple[str, str]]) -> int:
        """Re-attach imported children whose parent exists without closing a cycle; returns the detached count."""
        parent_by_child = dict(parent_updates)

        def closes_cycle(session_id: str, parent_id: str) -> bool:
            seen, current = {session_id}, parent_id
            while current:
                if current in seen:
                    return True
                seen.add(current)
                current = parent_by_child[current] if current in parent_by_child else (table[current]["parent_session_id"] if current in table else None)
            return False

        detached = 0
        for session_id, parent_id in parent_updates:
            if parent_id in table and not closes_cycle(session_id, parent_id):
                row = table[session_id]
                row.update(parent_session_id=parent_id, compression_fork=row["source"] != "tool")
            else:
                parent_by_child.pop(session_id, None)
                detached += 1
        return detached

    def import_sessions(self, sessions: Sequence[Mapping[str, Any]]) -> ImportResult:
        if not isinstance(sessions, (list, tuple)):
            raise ValueError("sessions must be a list")
        if len(sessions) > IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {IMPORT_MAX_SESSIONS} entries")
        normalized, errors = validate_import_payload(sessions)
        if errors:
            return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(errors))
        ids = [str(item["session"].get("id") or "").strip() for item in normalized]
        imported: list[SessionId] = []
        skipped: list[SessionId] = []
        parent_updates: list[tuple[str, str]] = []
        with self._staged_import() as staged:
            if deleted := [sid for sid in ids if sid in staged.authority["retired"]]:
                return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(f"session {sid} was deleted here; a deleted session is not restored" for sid in deleted))
            for item, session_id in zip(normalized, ids):
                if session_id in staged.scope["sessions"]:
                    skipped.append(SessionId(session_id))
                    continue
                self._insert_imported(staged.scope, item["session"], item["messages"], session_id)
                if parent_id := str(item["session"].get("parent_session_id") or "").strip():
                    parent_updates.append((session_id, parent_id))
                imported.append(SessionId(session_id))
            detached = self._attach_parents(staged.scope["sessions"], parent_updates)
        return ImportResult(ok=True, imported=len(imported), skipped=len(skipped), detached=detached, imported_ids=tuple(imported), skipped_ids=tuple(skipped))

    # ── foreign import ────────────────────────────────────────────────────

    @staticmethod
    def _find_foreign(table: Mapping[str, Mapping[str, Any]], origin: ForeignOrigin) -> Optional[SessionId]:
        for row in table.values():
            if row["source"] != origin.tool or row.get("origin") is None:
                continue
            imported = row["origin"].get("imported_from") or {}
            if imported.get("tool") != origin.tool:
                continue
            if (origin.foreign_session_id and origin.foreign_session_id == imported.get("foreign_session_id")) or (
                not origin.foreign_session_id and imported.get("path") == origin.path
            ):
                return SessionId(row["session_id"])
        return None

    def find_foreign_import(self, origin: ForeignOrigin) -> Optional[SessionId]:
        return self._find_foreign(self._rows(), origin)

    def import_foreign_history(self, origin: ForeignOrigin, messages: Sequence[ForeignTurn], *, title: str, cwd: Optional[str], profile: str) -> ForeignImportResult:
        session_id = _mint_session_id()
        turns = [{"role": turn.role, "content": turn.content} for turn in messages]
        normalized, errors = validate_import_payload([{"id": session_id, "source": origin.tool, "title": title, "cwd": cwd, "messages": turns}])
        if errors:
            raise ValueError(errors[0]["error"])
        with self._staged_import() as staged:
            table = staged.scope["sessions"]
            if (existing := self._find_foreign(table, origin)) is not None:
                return ForeignImportResult(session_id=existing, already_imported=True)
            if session_id in table:
                # A minted id already taken refuses the import and the write rolls back.
                raise Conflict("session", session_id)
            item = normalized[0]
            if any(row.get("title") == title for row in table.values()):
                item["session"]["title"] = f"{title} ({session_id[-12:]})"
            self._insert_imported(staged.scope, item["session"], item["messages"], session_id)
            table[session_id].update(origin={"imported_from": origin.as_dict()}, profile_name=profile)
        return ForeignImportResult(session_id=SessionId(session_id), already_imported=False)

    # ── backend heartbeats ────────────────────────────────────────────────

    def register_backend_heartbeat(self, *, backend_id: str, pid: int, started_at: float, last_heartbeat: Optional[float] = None, profile: str = '', host: str = '') -> None:
        if not backend_id:
            return
        stamp = time.time() if last_heartbeat is None else float(last_heartbeat)
        self._heartbeats()[str(backend_id)] = {
            "backend_id": str(backend_id), "pid": int(pid), "started_at": float(started_at), "last_heartbeat": stamp, "profile": str(profile), "host": str(host),
        }

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        if not backend_id:
            return False
        return self._heartbeats().pop(backend_id, None) is not None

    # ── read-only probes ──────────────────────────────────────────────────

    def session_projection_digest(self, *, busy_timeout_s: float = 0.05) -> Optional[tuple[tuple[str, ...], bytes]]:
        rows = sorted(self._rows())
        digest = hashlib.blake2b(digest_size=16)
        for sid in rows:
            values = _export_row(self._record(self._rows()[sid])).as_dict()
            digest.update(repr(tuple(values[field] for field in SESSION_PROJECTION_FIELDS)).encode("utf-8", "backslashreplace"))
            digest.update(b"\0")
        return SESSION_PROJECTION_FIELDS, digest.digest()

    def _inside(self, worktree_path: str) -> list[Mapping[str, Any]]:
        inside = _inside_tree(worktree_path)
        return [row for row in self._rows().values() if inside(row.get("cwd"))]

    def retained_tree_is_live(self, worktree_path: str, *, pending_only: bool = False) -> bool:
        pending = {row["target_session_id"] for row in authority_state(self._backend, self._scope).snapshot["admissions"].values() if row["status"] != "terminal"}
        return any(
            row["session_id"] in pending or (not pending_only and self._record(row).message_count > 0)
            for row in self._inside(worktree_path)
        )

    @staticmethod
    def _active_state(value: Optional[str]) -> bool:
        if value is None:
            return False
        try:
            parsed = json.loads(value)
        except ValueError:
            return True
        status = parsed.get("status") if isinstance(parsed, dict) else None
        return (status if status is not None else "active") == "active"

    def tree_served_sessions(self, worktree_path: str) -> TreeServeFacts:
        rows = self._inside(worktree_path)
        meta = state_meta_table(self._backend, self._scope)
        for row in rows:
            sid = row["session_id"]
            if row["ended_at"] is None or any(self._active_state(meta.get(f"{kind}:{sid}")) for kind in ("loop", "heartbeat", "goal")):
                return TreeServeFacts(served=True)
        return TreeServeFacts(served=False, chat_ids=frozenset(row["chat_id"] for row in rows if row.get("chat_id") is not None))


register_repository("maintenance", InMemoryMaintenance)
