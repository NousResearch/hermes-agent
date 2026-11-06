"""Session listing/rich rows, export, and import (portability) for SessionDB.

Plain mixin for ``hermes_state.SessionDB`` (no ``__init__``/state of its own).
Must never import hermes_state (cycle); shared constants live in hermes_state_common.
"""

import logging
import json
import math
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.skill_commands import AUTO_LOAD_SCAFFOLD_SQL_LIKE, SKILL_SCAFFOLD_SQL_LIKE
from utils import safe_json_loads
from hermes_cli.timefmt import coerce_epoch
from hermes_state_ids import new_session_id
from hermes_state_keys import (
    CRON_PRINCIPAL_ID, CRON_REQUEST_PREFIX, cron_request_range, cron_run_from_request, legacy_cron_run_job,
    prefix_range,
)
from hermes_state_local import POLICY_PREFIX
from hermes_state_common import SCHEMA_SQL, _shape_preview, _sql_preview_raw, _sql_session_last_active
from hermes_state_messages import _parse_tool_calls
from hermes_state_messages_text import _tool_calls_count
from hermes_state_lineage import session_lineage_guard
import itertools

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

_IMPORT_SESSION_TEXT_FIELDS = (
    "source", "user_id", "model", "system_prompt", "end_reason", "cwd", "git_branch", "git_repo_root",
    "billing_provider", "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
# ``role`` is validated separately (non-empty string).
_IMPORT_MESSAGE_TEXT_FIELDS = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason",
    "reasoning", "reasoning_content", "platform_message_id", "message_id",
)
_IMPORT_MESSAGE_JSON_FIELDS = (
    "reasoning_details", "codex_reasoning_items", "codex_message_items", "absorbed_message_uids", "tool_call_uids",
)
_IMPORT_SESSION_INSERT_SQL = """INSERT INTO sessions (
                           id, source, user_id, model, model_config, system_prompt,
                           system_prompt_hash,
                           parent_session_id, started_at, ended_at, end_reason,
                           message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived, auto_archived, pinned, hidden
                       )
                       VALUES (
                           :id, :source, :user_id, :model, :model_config,
                           NULL, :system_prompt_hash, NULL, :started_at, :ended_at,
                           :end_reason, 0, 0, :input_tokens, :output_tokens,
                           :cache_read_tokens, :cache_write_tokens,
                           :reasoning_tokens, :cwd, :git_branch, :git_repo_root,
                           :billing_provider, :billing_base_url, :billing_mode,
                           :estimated_cost_usd, :actual_cost_usd, :cost_status,
                           :cost_source, :pricing_version, :title,
                           :api_call_count, :archived, :auto_archived, :pinned, :hidden
                       )"""
# Columns copied verbatim from the payload; typed columns are converted below.
_IMPORT_PASSTHROUGH_COLS = (
    "user_id", "model", "model_config", "end_reason", "cwd", "git_branch", "git_repo_root", "billing_provider",
    "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
_IMPORT_INT_COLS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_call_count",
)
_IMPORT_FLOAT_COLS = ("ended_at", "estimated_cost_usd", "actual_cost_usd")
# Flags a restore must keep: pinned/hidden/auto_archived drive the prune, listings and resume-unarchive.
_IMPORT_FLAG_COLS = ("archived", "auto_archived", "pinned", "hidden")


def _rich_select(select_cols: str, where: str, tail: str = "", prompt_select: Optional[str] = "") -> str:
    """``list_sessions_rich``-shaped SELECT: resolved prompt (``prompt_select`` fragment;
    None omits prompt columns AND the join), preview, last_active. Whitespace matches
    the historical inline queries (SQL text is pinned)."""
    prompt_join = "" if prompt_select is None else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
    return f"""
            SELECT {select_cols}{prompt_select or ""},
                {_sql_preview_raw()},
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE {where}{tail}
        """


_PROMPT_RESOLVED_SQL = "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"

# One job's scheduler admissions (``:request_lo``/``:request_hi`` = cron_request_range). The principal
# is inlined so the partial index applies (a bound parameter cannot prove its WHERE). A request id
# whose only ``:`` after ``cron:`` ends the job id is unambiguous (fresh execution ids hold none);
# any other in the range (job ``a`` vs ``a:b``, or an execution id admitted with a ``:`` before that
# was refused) belongs to the job its creation receipt's policy names, never to a ``:`` split.
# A legacy id is the prefix plus exactly ``YYYYmmdd_HHMMSS``: the GLOB keeps job ``daily`` from
# claiming job ``daily_digest``'s ``cron_daily_digest_<ts>`` runs inside the same id range.
_CRON_POLICY_JOB_SQL = (
    "(SELECT CASE WHEN json_valid(r) THEN json_extract(r, '$.cron_job.id') END FROM"
    " (SELECT CASE WHEN json_valid(value) THEN json_extract(value, '$.policy.request_json') END AS r"
    f" FROM state_meta WHERE key = '{POLICY_PREFIX}' || target_session_id))")
_CRON_ADMISSION_SQL = (
    f"principal_id = '{CRON_PRINCIPAL_ID}' AND request_id >= :request_lo AND request_id < :request_hi"
    f" AND ((instr(substr(request_id, {len(CRON_REQUEST_PREFIX) + 1}), ':') = length(:request_lo) - {len(CRON_REQUEST_PREFIX)}"
    " AND instr(substr(request_id, length(:request_lo) + 1), ':') = 0)"
    f" OR {_CRON_POLICY_JOB_SQL} = :job_id)")
_CRON_RUN_IDS_SQL = (
    "SELECT id FROM sessions WHERE source = 'cron' AND id >= :prefix AND id < :prefix_hi"
    " AND substr(id, length(:prefix) + 1) GLOB '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'"
    f" UNION ALL SELECT target_session_id FROM session_admissions WHERE {_CRON_ADMISSION_SQL}")
_CRON_EXECUTION_SQL = (
    "(SELECT substr(request_id, length(:request_lo) + 1) FROM session_admissions"
    f" WHERE target_session_id = s.id AND {_CRON_ADMISSION_SQL} LIMIT 1)")


def _export_timings(messages: list[dict[str, Any]], session_id: Optional[str] = None) -> dict[str, Any]:
    """Text-free timing evidence for a session export.

    Exports get attached to bug reports; a reader should not have to infer from raw
    timestamps whether a slow turn was one long model gap or many small tool
    intervals. Hermes persists no model/tool stopwatch samples, so message
    timestamps are the durable floor (``complete`` is therefore always False).
    Ids, roles, counts and durations only — never prompt text, arguments or results.
    Corrupt timestamp cells go through ``coerce_epoch`` like every other reader: they
    count as ``missing`` and never abort the export.
    """
    timestamped = [(msg, ts) for msg in messages
                   if (ts := coerce_epoch(msg.get("timestamp"), session_id=session_id)) is not None]
    role_counts = Counter(str(msg.get("role") or "unknown") for msg in messages)
    tool_calls_emitted = sum(
        len(tc) if isinstance(tc, list) else 1 for tc in (msg.get("tool_calls") for msg in messages) if tc)
    intervals = [{
        "from_message_id": prev.get("id"), "to_message_id": nxt.get("id"),
        "from_role": prev.get("role"), "to_role": nxt.get("role"),
        "gap_ms": max(0, round((nxt_ts - prev_ts) * 1000)),
    } for (prev, prev_ts), (nxt, nxt_ts) in itertools.pairwise(timestamped)]
    iso = lambda ts: datetime.fromtimestamp(ts, tz=UTC).isoformat()
    first_ts, last_ts = (timestamped[0][1], timestamped[-1][1]) if timestamped else (None, None)
    return {
        "source": "message_timestamps",
        "available": bool(timestamped),
        "complete": False,
        "unavailable_reason": None if timestamped else "no_timestamped_messages",
        "message_timestamps": {"available": len(timestamped), "missing": len(messages) - len(timestamped)},
        "first_message_at": iso(first_ts) if first_ts is not None else None,
        "last_message_at": iso(last_ts) if last_ts is not None else None,
        "wall_clock_ms": max(0, round((last_ts - first_ts) * 1000)) if timestamped else None,
        "largest_gap_ms": max(i["gap_ms"] for i in intervals) if intervals else None,
        "role_counts": dict(role_counts),
        "tool_result_count": role_counts.get("tool", 0),
        "tool_calls_emitted": tool_calls_emitted,
        "intervals": intervals,
    }


class SessionPortabilityMixin:
    """See module docstring — mixin for SessionDB (Port cluster)."""

    @staticmethod
    def _find_foreign_import_on_conn(conn, origin):
        rows = conn.execute("SELECT id, origin_json FROM sessions WHERE source = ? AND origin_json IS NOT NULL",
                            (origin["tool"],)).fetchall()
        for row in rows:
            imported = (safe_json_loads(row["origin_json"], default={}) or {}).get("imported_from", {})
            if imported.get("tool") != origin["tool"]:
                continue
            foreign_id = origin.get("foreign_session_id")
            if ((foreign_id and foreign_id == imported.get("foreign_session_id"))
                    or (not foreign_id and imported.get("path") == origin["path"])):
                return row["id"]
        return None

    def find_foreign_import(self, origin):
        with self._read_ctx() as conn:
            return self._find_foreign_import_on_conn(conn, origin)

    def import_foreign_history(self, origin, messages, *, title, cwd, profile):
        """Adopt or mint a foreign snapshot in one transaction, including its provenance.

        BEGIN IMMEDIATE serializes duplicate clicks across connections/processes.
        Reuse the portability validator and message writer so counters and FTS
        obey the same contract as ordinary transcript imports.
        """
        session_id = new_session_id(hex_len=12)
        normalized, errors = self._validate_import_payload([
            {"id": session_id, "source": origin["tool"], "title": title,
             "cwd": cwd, "messages": messages}])
        if errors:
            raise ValueError(errors[0]["error"])

        def _do(conn):
            existing = self._find_foreign_import_on_conn(conn, origin)
            if existing:
                return {"session_id": existing, "already_imported": True}
            # Titles are globally unique within a profile. Preserve a readable
            # title while giving unrelated conversations with the same text room.
            item = normalized[0]
            if conn.execute("SELECT 1 FROM sessions WHERE title = ?", (title,)).fetchone():
                item["session"]["title"] = f"{title} ({session_id[-12:]})"
            self._import_session_row(conn, item["session"], item["messages"], session_id)
            conn.execute("UPDATE sessions SET origin_json = ?, profile_name = ? WHERE id = ?",
                         (json.dumps({"imported_from": origin}), profile, session_id))
            return {"session_id": session_id, "already_imported": False}

        return self._execute_write(_do)

    @classmethod
    def _compact_session_cols(cls) -> str:
        """``s.``-prefixed SELECT list of every SCHEMA_SQL ``sessions`` column except
        prompt storage internals (the compact_rows projection)."""
        if cls._session_compact_cols_sql is None:
            declared = cls._parse_schema_columns(SCHEMA_SQL)["sessions"]
            cls._session_compact_cols_sql = ", ".join(
                f"s.{name}" for name in declared if name not in cls._SESSION_COMPACT_EXCLUDED
            )
        return cls._session_compact_cols_sql

    @classmethod
    def _rich_row(cls, row) -> dict[str, Any]:
        """Session row dict with ``_preview_raw`` shaped into ``preview``."""
        s = cls._session_row_dict(row)
        s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
        return s

    def _read_rows(self, sql: str, params=()) -> list:
        """Pure-read query via ``_read_ctx()`` (never the writer lock: turn persistence must not convoy)."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchall()

    def distinct_session_cwds(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Distinct non-empty session cwds with usage stats, for repo discovery. Aggregates
        across ALL history; children/branches count (a worktree session is a real
        workspace signal)."""
        where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
        if not include_archived:
            where += " AND archived = 0"
        rows = self._read_rows(
            "SELECT cwd AS cwd, COUNT(*) AS sessions, MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
            f"FROM sessions WHERE {where} GROUP BY cwd"
        )
        return [{"cwd": r["cwd"], "sessions": int(r["sessions"] or 0), "last_active": float(r["last_active"] or 0)}
                for r in rows]

    def list_cron_job_runs(self, job_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Run sessions of one cron job, newest first, in the ``list_sessions_rich`` row shape,
        stamped with ``cron_job_id`` and ``cron_execution_id`` (None for a pre-unified run).

        Pre-unified runs are flat ``cron_{job_id}_{timestamp}`` sessions: a ``[prefix,
        prefix_hi)`` id range. Unified-runtime runs keep their runtime-minted ``local-*`` id, which
        must not change (exact-retry recovery re-derives it), but the scheduler principal admitted
        each under ``cron:<job>:<execution>``: a range of ``idx_session_admissions_cron_request``.
        Both scans scale with this job's runs, never with total cron history or with every local
        session's policy receipt, and skip ``list_sessions_rich``'s compression-chain CTE.

        Limitation (legacy runs had it too): only a run's FIRST session is listed. A run that
        compressed continues in a child session this does not return, and the row's liveness
        (``scheduler_owned``) and detail identity follow that first session, not the tip."""
        prefix, prefix_hi = prefix_range(f"cron_{job_id}_")
        request_lo, request_hi = cron_request_range(job_id)
        query = _rich_select(
            f"s.*, {_CRON_EXECUTION_SQL} AS cron_execution_id", f"s.source = 'cron' AND s.id IN ({_CRON_RUN_IDS_SQL})",
            "\n            ORDER BY s.started_at DESC, s.id DESC\n            LIMIT :limit OFFSET :offset",
            prompt_select=f",\n                {_PROMPT_RESOLVED_SQL}",
        )
        rows = self._read_rows(query, {"prefix": prefix, "prefix_hi": prefix_hi, "request_lo": request_lo,
                                       "request_hi": request_hi, "job_id": job_id, "limit": limit, "offset": offset})
        return [{**self._rich_row(row), "cron_job_id": job_id} for row in rows]

    def cron_run_identity(self, session_id: str) -> Optional[dict[str, Any]]:
        """``{"cron_job_id", "cron_execution_id"}`` of a cron run session, else None: a legacy id
        names its job (no execution), a unified-runtime run is one probe of its scheduler admission
        (and, when its request id is ambiguous, of the job its creation receipt's policy names)."""
        legacy_job = legacy_cron_run_job(session_id)
        if legacy_job is not None:
            return {"cron_job_id": legacy_job, "cron_execution_id": None}
        rows = self._read_rows(
            "SELECT request_id, " + _CRON_POLICY_JOB_SQL + " AS policy_job FROM session_admissions"
            " WHERE principal_id = ? AND target_session_id = ? ORDER BY seq LIMIT 1", (CRON_PRINCIPAL_ID, session_id))
        run = None
        if rows:
            run = cron_run_from_request(rows[0]["request_id"])
            if run is None and isinstance(rows[0]["policy_job"], str):
                run = cron_run_from_request(rows[0]["request_id"], rows[0]["policy_job"])
        return {"cron_job_id": run[0], "cron_execution_id": run[1]} if run else None

    def _get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> Optional[dict[str, Any]]:
        """One session with the ``list_sessions_rich`` enriched columns, or None.
        ``compact_rows=True`` omits the ``system_prompt`` blob. Public alias:
        :meth:`get_session_rich_row` (web server hydration)."""
        return self._get_session_rich_rows_batch([session_id], compact_rows=compact_rows).get(session_id)

    get_session_rich_row = _get_session_rich_row

    def _get_session_rich_rows_batch(self, session_ids, compact_rows: bool = False) -> dict[str, dict[str, Any]]:
        """Enriched rows for many sessions in one query, keyed by id; missing ids are absent
        (a page of compression tips resolves in one round trip)."""
        ids = [sid for sid in session_ids if sid]
        if not ids:
            return {}
        # Old SQLite caps bound variables at 999 (SQLITE_MAX_VARIABLE_NUMBER); limit=10000
        # callers exist. Chunk here — the single choke point.
        _CHUNK = 900
        if len(ids) > _CHUNK:
            result: dict[str, dict[str, Any]] = {}
            for start in range(0, len(ids), _CHUNK):
                result.update(self._get_session_rich_rows_batch(ids[start:start + _CHUNK], compact_rows=compact_rows))
            return result
        # Same read-your-writes guarantee as list_sessions_rich.
        self.flush_token_counts()
        query = _rich_select(
            self._compact_session_cols() if compact_rows else "s.*", f"s.id IN ({','.join('?' for _ in ids)})",
            prompt_select=None if compact_rows else f", {_PROMPT_RESOLVED_SQL}",
        )
        return {s["id"]: s for s in map(self._rich_row, self._read_rows(query, ids))}

    def list_skill_scaffolded_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        """Titled sessions whose first user turn was a ``/skill`` invocation or a gateway
        auto-load scaffold (their titles describe the expanded skill body, not the
        request). Returns ``id``, ``title`` and the first-turn ``content`` so callers can
        re-derive what was typed. Newest first."""
        rows = self._read_rows("""
                SELECT s.id, s.title, m.content
                FROM sessions s
                JOIN messages m ON m.id = (
                    SELECT m2.id FROM messages m2
                    WHERE m2.session_id = s.id AND m2.role = 'user'
                      AND m2.content IS NOT NULL
                    ORDER BY m2.timestamp, m2.id LIMIT 1
                )
                WHERE s.title IS NOT NULL AND (m.content LIKE ? OR m.content LIKE ?)
                ORDER BY s.started_at DESC
                LIMIT ?
                """, (SKILL_SCAFFOLD_SQL_LIKE, AUTO_LOAD_SCAFFOLD_SQL_LIKE, int(limit)))
        return [dict(row) for row in rows]

    # ── Export ─────────────────────────────────────────────────────────────

    def _with_messages(self, session: dict[str, Any], include_compacted: bool = False,
                       include_inactive: bool = False) -> dict[str, Any]:
        messages = self.get_messages(session["id"], include_inactive=include_inactive, include_compacted=include_compacted)
        return {**session, "messages": messages, "timings": _export_timings(messages, session["id"])}

    def export_session(self, session_id: str, include_compacted: bool = False,
                       include_inactive: bool = False) -> Optional[dict[str, Any]]:
        """Export a single session with all its messages as a dict. ``include_compacted`` adds the turns
        in-place compaction archived (the history the user still sees); it stays off for payloads that go
        back through :meth:`import_sessions`, because that projection is deduped and ordered for display, and
        importing it would make those turns live context. ``include_inactive`` exports every row in storage
        order with its ``active``/``compacted`` flags, which :meth:`import_sessions` restores as archived."""
        session = self.get_session(session_id)
        return self._with_messages(session, include_compacted, include_inactive) if session else None

    def export_session_lineage(self, session_id: str, include_compacted: bool = False,
                               include_inactive: bool = False) -> Optional[dict[str, Any]]:
        """Export a compression lineage as one logical session dict (flags as in :meth:`export_session`)."""
        lineage_ids = self.get_compression_lineage(session_id)
        if not lineage_ids:
            return None
        segments = [seg for seg in (self.export_session(sid, include_compacted, include_inactive)
                                    for sid in lineage_ids) if seg]
        if not segments:
            return None
        messages = [msg for seg in segments for msg in (seg.get("messages") or [])]
        return {
            **segments[-1], "segments": segments,
            "lineage_session_ids": [seg["id"] for seg in segments], "message_count": len(messages),
            "messages": messages, "timings": _export_timings(messages, session_id),
        }

    def export_all(self, source: str | None = None, include_compacted: bool = False,
                   include_inactive: bool = False) -> list[dict[str, Any]]:
        """Export all sessions (with messages) as dicts, e.g. for JSONL backup (flags as in
        :meth:`export_session`; that display read dedupes per session, so it skips the batched read).
        Backups that go back through :meth:`import_sessions` pass ``include_inactive`` so
        compaction-archived turns survive the round trip as archived rows."""
        sessions = self.search_sessions(source=source, limit=100000)
        if include_compacted:
            return [self._with_messages(session, True, include_inactive) for session in sessions]
        messages_by_session = {session["id"]: [] for session in sessions}
        session_ids = list(messages_by_session)
        active_clause = self._active_clause(include_inactive, False)
        # Stay below SQLite's legacy 999-variable limit while replacing the per-session N+1 reads.
        for start in range(0, len(session_ids), 900):
            chunk = session_ids[start:start + 900]
            rows = self._read_all(
                f"SELECT * FROM messages WHERE session_id IN ({','.join('?' for _ in chunk)}) "
                f"{active_clause} ORDER BY session_id, id",
                chunk,
            )
            for row in rows:
                messages_by_session[row["session_id"]].append(
                    self._row_to_message_dict(row, warn_context="get_messages", summary_flag=True)
                )
        return [{**session, "messages": messages_by_session[session["id"]],
                 "timings": _export_timings(messages_by_session[session["id"]], session["id"])} for session in sessions]

    def adopt_session_lineage_from(self, donor_db: Any, session_id: str, *, retire_donor: bool = True,
                                   _pending_only: bool = False) -> dict[str, Any]:
        """Copy a complete lineage, durably retaining membership for retirement retries.

        Copies and immutable membership commit together. Same-home member locks remain
        held through donor retirement; standalone pairs retain their existing contract.
        Existing rows are skipped and donor-ahead copies are never retired.
        """
        from hermes_state_adoption_records import AdoptionConflict
        pair_error = self._validate_adoption_store_pair(donor_db.db_path)
        if pair_error:
            return {"ok": False, "adopted": False, "donor_retired": False, "error": pair_error}
        try:
            for _attempt in range(3):
                outcome = self._attempt_lineage_adoption(donor_db, session_id, retire_donor=retire_donor, pending_only=_pending_only)
                if outcome is not None:
                    return outcome
        except AdoptionConflict as exc:
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "conflict": True, "error": str(exc)}
        return {"ok": False, "adopted": False, "donor_retired": False,
                "error": "donor membership kept changing; retry after its lineage stabilizes"}

    def _adoption_segments(self, donor_db: Any, session_id: str, record) -> list[dict[str, Any]]:
        from hermes_state_adoption_records import AdoptionConflict
        if record is None:
            payload = donor_db.export_session_lineage(session_id, include_inactive=True)
            return (payload.get("segments") or [payload]) if payload else []
        segments = [donor_db.export_session(mid, include_inactive=True) for mid in record["member_ids"]]
        if any(segment is None for segment in segments):
            raise AdoptionConflict("Saved adoption donor member is missing; recover the complete transfer")
        return segments

    def _attempt_lineage_adoption(self, donor_db: Any, session_id: str, *, retire_donor: bool, pending_only: bool):
        from hermes_state_adoption_records import find, ownership, validate_fresh, validate_members, validate_handles
        validate_handles(self, donor_db)
        record = find(self, session_id, donor_db)
        if pending_only and (record is None or record["status"] != "pending"):
            return {"ok": False, "adopted": False, "donor_retired": False, "retry_skipped": True}
        segments = self._adoption_segments(donor_db, session_id, record)
        if not segments:
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "error": f"session {session_id!r} not found in donor store"}
        ids = tuple(segment["id"] for segment in segments)
        authority = None
        if record is not None:
            validate_members(self, donor_db, record)
            authority = ownership(record, donor_db.db_path, self.db_path)
        with session_lineage_guard(self.db_path, ids, transfer_from=donor_db.db_path,
                                   adoption=authority, timeout_s=self._WRITE_PATIENCE_S) as snapshot:
            # Reload authority under the complete lock set. Returning releases it before
            # the bounded outer loop discovers/reacquires any expanded membership.
            validate_handles(self, donor_db)
            current = find(self, session_id, donor_db)
            if pending_only and (current is None or current["status"] != "pending"):
                return {"ok": False, "adopted": False, "donor_retired": False, "retry_skipped": True}
            if (current is None) != (record is None):
                return None
            if current is not None and current["manifest"] != record["manifest"]:
                return None
            refreshed = self._adoption_segments(donor_db, session_id, current)
            if tuple(segment["id"] for segment in refreshed) != ids:
                return None
            pair_error = self._validate_adoption_store_pair(donor_db.db_path)
            if pair_error:
                return {"ok": False, "adopted": False, "donor_retired": False, "error": pair_error}
            self._validate_adoption_inventory(snapshot, donor_db.db_path, ids)
            if current is None:
                validate_fresh(self, refreshed)
            else:
                validate_members(self, donor_db, current)
            return self._adopt_lineage_segments_locked(
                donor_db, session_id, refreshed, retire_donor=retire_donor, record=current, pending_only=pending_only)

    def _validate_adoption_inventory(self, snapshot, donor_path, ids) -> None:
        if snapshot is None:
            return  # both resolutions positively classified the standalone compatibility path
        from hermes_state_lineage import SessionLineageAmbiguousError, SessionLineageInventoryError, _same_file
        for session_id in ids:
            rows = snapshot.rows_by_id.get(session_id, ())
            if not any(_same_file(row.store_path, Path(donor_path)) for row in rows):
                raise SessionLineageInventoryError("Cannot inventory the adoption donor; use a managed profile store")
            if any(not _same_file(row.store_path, Path(donor_path))
                   and not _same_file(row.store_path, self.db_path) for row in rows):
                raise SessionLineageAmbiguousError("Adoption has a third store copy; adjudicate its ownership first")

    def _adoption_import_and_record(self, donor_db, segments, record, *, retire_donor, pending_only):
        from hermes_state_adoption_records import AdoptionConflict, build_manifest, find_in, insert, set_retirement_intent, validate_handles
        normalized, errors = self._validate_import_payload(segments)
        if errors:
            return {"ok": False, "imported": 0, "skipped": 0, "detached": 0, "errors": errors}, record
        first_id = segments[0]["id"]

        def write(conn):
            validate_handles(self, donor_db)
            current = find_in(conn, first_id, donor_db, self)
            if pending_only and (current is None or current["status"] != "pending"):
                return {"ok": False, "imported": 0, "skipped": 0, "retry_skipped": True}, current
            if (current is None) != (record is None):
                raise AdoptionConflict("Adoption record appeared or disappeared; retry with its complete membership")
            if current is not None and current["manifest"] != record["manifest"]:
                raise AdoptionConflict("Adoption membership changed; inspect the saved transfer")
            result = self._import_sessions_locked(conn, normalized)
            if result.get("ok"):
                if current is None:
                    insert(conn, build_manifest(self, donor_db, segments, conn), retire_donor=retire_donor)
                else:
                    set_retirement_intent(conn, current, retire_donor)
                current = find_in(conn, first_id, donor_db, self)
            return result, current

        return self._execute_write(write)

    def _adopt_lineage_segments_locked(self, donor_db: Any, session_id: str, segments: list[dict[str, Any]],
                                       *, retire_donor: bool, record=None, pending_only: bool = False) -> dict[str, Any]:
        from hermes_state_adoption_records import mark_complete
        members = [segment["id"] for segment in segments]
        donor_ahead = any(
            self.get_session(segment["id"]) is not None
            and len(segment.get("messages") or []) > len(self.get_messages(segment["id"], include_inactive=True))
            for segment in segments)
        if donor_ahead:
            logger.warning("adoption divergence: donor %s has newer messages; leaving it unretired", session_id)
        result, saved = self._adoption_import_and_record(donor_db, segments, record, retire_donor=retire_donor, pending_only=pending_only)
        adopted = result.get("ok", False) and (result.get("imported", 0) + result.get("skipped", 0)) == len(members)
        donor_retired = False
        pending = list(members)
        if adopted and retire_donor and not donor_ahead:
            pending = []
            for order, member in enumerate(members):
                if not self._retire_donor_segment(donor_db, member):
                    pending = members[order:]
                    break
            for member in members:
                row = donor_db.get_session(member)
                if (row is None or row.get("end_reason") != "adopted_by_profile" or not row.get("archived")) and member not in pending:
                    pending.append(member)
            donor_retired = not pending
            if donor_retired:
                mark_complete(self, saved, donor_db)
        return {**result, "adopted": adopted, "donor_retired": donor_retired,
                "member_ids": members, "pending_ids": pending}

    def _retire_donor_segment(self, donor_db: Any, seg_id: str) -> bool:
        """Archive one adopted donor segment; False when skipped or failed. TOCTOU close-out:
        the divergence guard used EXPORT-TIME counts; re-read both stores right before
        stamping so donor growth never lands behind a non-recoverable archive
        (equal-count CONTENT divergence is accepted — bytes stay in the donor either way).
        A retirement failure must not fail the adoption (a later resume retries
        idempotently), but never claims success it didn't have."""
        try:
            donor_now = len(donor_db.get_messages(seg_id, include_inactive=True))
            local_now = len(self.get_messages(seg_id, include_inactive=True))
            if donor_now > local_now:
                logger.warning(
                    "adoption divergence at retire time: donor segment %s grew to %d messages (local %d) — "
                    "leaving donor unretired", seg_id, donor_now, local_now,
                )
                return False
            # Stamp the adoption boundary without reopening the donor: a concurrent target
            # owner must never see the donor become live again between import and retirement.
            if not donor_db.end_session_as_adopted(seg_id):
                return False
            # The session lives on in the adopting profile: not a delete for plugins.
            # _attempt_lineage_adoption still holds the complete, transfer-validated
            # member set. The public archive API must refuse this donor once the
            # target owns it. Retire this one recorded member under that guard;
            # the caller visits every member, including compression continuations.
            donor_db._execute_write(lambda conn: conn.execute(
                'UPDATE sessions SET archived=1, auto_archived=0 WHERE id=?', (seg_id,)))
            return True
        except Exception:
            logger.warning("failed to retire donor segment %s after adoption", seg_id, exc_info=True)
            return False

    def _validate_adoption_store_pair(self, donor_path: Path) -> str | None:
        from hermes_state_lineage import _managed_root_and_current, _same_file
        # Resolution errors are unverifiable stores, never standalone permission.
        donor = _managed_root_and_current(donor_path)
        target = _managed_root_and_current(self.db_path)
        if _same_file(Path(donor_path), self.db_path):
            return "Donor and target are the same database; choose distinct stores"
        if (donor is None) != (target is None):
            standalone = "donor" if donor is None else "target"
            return f"Adoption refuses a mixed managed/standalone pair (standalone {standalone}); use one managed home"
        if donor is not None and donor[0] != target[0]:
            return "Adoption requires both stores in the same managed home"
        return None

    # ── Import ─────────────────────────────────────────────────────────────

    @staticmethod
    def _import_text_or_none(value: Any, field: str) -> Optional[str]:
        if value is None or isinstance(value, str):
            return value
        raise ValueError(f"{field} must be a string")

    @staticmethod
    def _import_int_or_none(value: Any, field: str) -> Optional[int]:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc

    @staticmethod
    def _import_non_finite_path(value: Any, path: str) -> Optional[str]:
        """First ``path`` of a NaN/inf number anywhere in decoded JSON data, else None.

        Iterative walk (a recursive visitor would overflow on crafted nesting); cycles
        can't arrive through ``json.loads`` but guard anyway.
        """
        pending = [(value, path)]
        while pending:
            current, current_path = pending.pop()
            if isinstance(current, float) and not math.isfinite(current):
                return current_path
            if isinstance(current, dict):
                pending.extend((child, f"{current_path}.{key}") for key, child in current.items())
            elif isinstance(current, list):
                pending.extend((child, f"{current_path}[{key}]") for key, child in enumerate(current))
        return None

    @classmethod
    def _import_load_json(cls, value: str, path: str) -> Any:
        """``json.loads`` that refuses NaN/Infinity anywhere in the decoded value.

        A JSON exponent like ``1e309`` decodes to ``float("inf")``, which strict
        serializers reject and cost analytics mis-handle. Callers that intentionally
        decode a JSON-backed column (model_config, reasoning items) re-check through
        here so non-finite values can't reappear after the outer payload was inspected.
        """
        parsed = json.loads(value)  # NaN/Infinity are accepted syntax; catch them after decode
        non_finite = cls._import_non_finite_path(parsed, path)
        if non_finite is not None:
            raise ValueError(f"non-finite number at {non_finite}")
        return parsed

    @classmethod
    def _import_json_object_or_none(cls, value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        path = f"session.{field}"
        if isinstance(value, str):
            try:
                parsed = cls._import_load_json(value, path)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field} must be a JSON object")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object")
        if cls._import_non_finite_path(value, path) is not None:
            raise ValueError(f"non-finite number at {path}")
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @staticmethod
    def _coerce_or(value: Any, cast, default):
        """``cast(value)``; *default* for None or an unparsable value."""
        try:
            return default if value is None else cast(value)
        except (TypeError, ValueError):
            return default

    # SQLite binds Python ints only inside the signed 64-bit range; anything wider
    # raises OverflowError at INSERT time (an unhandled 500, not a 400).
    _SQLITE_INT_MIN = -(1 << 63)
    _SQLITE_INT_MAX = (1 << 63) - 1

    @classmethod
    def _import_int_or_bounded(cls, value: Any, field: str, default: int) -> int:
        """Import an int column: SQL-int-range or *default* (never fails the import)."""
        number = cls._coerce_or(value, int, default)
        if number is None:
            return default
        if isinstance(number, bool) or not (cls._SQLITE_INT_MIN <= number <= cls._SQLITE_INT_MAX):
            return default
        return number

    def _normalize_import_message(self, message: dict[str, Any], message_index: int) -> dict[str, Any]:
        """Type-check one payload message; raises ValueError naming the offending field."""
        clean_message = dict(message)
        role = clean_message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{message_index}].role must be a non-empty string")
        for field in _IMPORT_MESSAGE_TEXT_FIELDS:
            clean_message[field] = self._import_text_or_none(clean_message.get(field), field)
        token_count = self._import_int_or_none(clean_message.get("token_count"), "token_count")
        if token_count is not None and not (self._SQLITE_INT_MIN <= token_count <= self._SQLITE_INT_MAX):
            raise ValueError("token_count is outside SQLite's integer range")
        clean_message["token_count"] = token_count
        message_path = f"session.messages[{message_index}]"
        timestamp = clean_message.get("timestamp")
        if timestamp is not None and coerce_epoch(timestamp, field="messages[].timestamp") is None:
            raise ValueError(f"{message_path}.timestamp must be a supported Unix timestamp")
        self._validate_import_message_content(clean_message.get("content"), message_path)
        for field in _IMPORT_MESSAGE_JSON_FIELDS:
            json_value = clean_message.get(field)
            if isinstance(json_value, str):
                # Malformed JSON text persists verbatim (readers fall back via _json_or);
                # only a PARSED non-finite number is rejected — it would break strict
                # JSON serializers once decoded back out of the row.
                try:
                    self._import_load_json(json_value, f"{message_path}.{field}")
                except json.JSONDecodeError:
                    pass
                except ValueError:
                    raise
            elif json_value is not None:
                non_finite = self._import_non_finite_path(json_value, f"{message_path}.{field}")
                if non_finite is not None:
                    raise ValueError(f"non-finite number at {non_finite}")
        return clean_message

    @classmethod
    def _validate_import_message_content(cls, content: Any, message_path: str) -> None:
        """Scalar message content must be bindable and later serializable."""
        if isinstance(content, int) and not isinstance(content, bool) and not (
                cls._SQLITE_INT_MIN <= content <= cls._SQLITE_INT_MAX):
            # SQLite can't bind a wider int: this reached INSERT and raised OverflowError.
            raise ValueError(f"{message_path}.content is outside SQLite's integer range")
        if isinstance(content, float) and not math.isfinite(content):
            raise ValueError(f"non-finite number at {message_path}.content")
        if isinstance(content, (bytes, bytearray)):
            raise ValueError(f"{message_path}.content must be a string, list, object, finite number, or null")

    @classmethod
    def _import_float_or_finite(cls, value: Any, field: str) -> Optional[float]:
        """Import a float column: finite float or None (never fails the import).

        A JSON exponent like ``1e309`` decodes to ``float("inf")``; persisted, it later
        breaks strict JSON serialization in the session APIs. Missing stays missing.
        """
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def _normalize_import_session(self, raw: dict[str, Any], session_id: str, messages: list) -> dict[str, Any]:
        """Type-check one payload session + its messages; raises ValueError."""
        clean_session = dict(raw)
        clean_session["id"] = session_id
        clean_session["model_config"] = self._import_json_object_or_none(clean_session.get("model_config"), "model_config")
        for field in ("parent_session_id", *_IMPORT_SESSION_TEXT_FIELDS):
            clean_session[field] = self._import_text_or_none(clean_session.get(field), field)
        started_at = clean_session.get("started_at")
        if started_at is not None and coerce_epoch(started_at, field="started_at") is None:
            raise ValueError("session.started_at must be a supported Unix timestamp")
        for field in _IMPORT_FLOAT_COLS:
            value = clean_session.get(field)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite number at session.{field}")
        for field in _IMPORT_INT_COLS:
            value = clean_session.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and not (
                    self._SQLITE_INT_MIN <= value <= self._SQLITE_INT_MAX):
                raise ValueError(f"{field} is outside SQLite's integer range")
        clean_messages: list[dict[str, Any]] = []
        for message_index, message in enumerate(messages):
            clean_messages.append(self._normalize_import_message(message, message_index))
        return {"session": clean_session, "messages": clean_messages}

    def _validate_import_payload(self, sessions: list[dict[str, Any]]) -> tuple:
        """Size/shape/type validation of the whole payload; returns ``(normalized_items,
        errors)``. Every rejected entry is reported."""
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries")
        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        totals = {"messages": 0, "bytes": 0}
        for index, raw in enumerate(sessions):
            session_id = str(raw.get("id") or "").strip() if isinstance(raw, dict) else ""
            try:
                item = self._validate_import_session(raw, session_id, seen_ids, totals)
            except ValueError as exc:
                item = {"index": index, "error": str(exc)}
                if session_id:
                    item["session_id"] = session_id
                errors.append(item)
                continue
            seen_ids.add(session_id)
            normalized.append({"index": index, **item})
        return normalized, errors

    def _validate_import_session(self, raw: Any, session_id: str, seen_ids: set, totals: dict[str, int]) -> dict[str, Any]:
        """One payload session -> normalized item; ValueError(message) on rejection. *totals*
        accumulate before their limit check (a rejected oversize entry still counts)."""
        if not isinstance(raw, dict):
            raise ValueError("session must be an object")
        if not session_id:
            raise ValueError("session id is required")
        if session_id in seen_ids:
            raise ValueError("duplicate session id")
        messages = raw.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if len(messages) > self._IMPORT_MAX_MESSAGES_PER_SESSION:
            raise ValueError("messages exceeds the per-session import limit")
        if any(not isinstance(msg, dict) for msg in messages):
            raise ValueError("messages must contain only objects")
        try:
            # `timings` is derived from the messages at export time and rebuilt on the next export;
            # it must not eat into the size budget of the content it merely describes.
            measured = {k: v for k, v in raw.items() if k != "timings"}
            session_bytes = len(json.dumps(measured, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError):
            raise ValueError("session must be JSON serializable") from None
        if session_bytes > self._IMPORT_MAX_SESSION_BYTES:
            raise ValueError("session exceeds the import size limit")
        totals["bytes"] += session_bytes
        if totals["bytes"] > self._IMPORT_MAX_TOTAL_BYTES:
            raise ValueError("import exceeds the total size limit")
        item = self._normalize_import_session(raw, session_id, messages)
        totals["messages"] += len(item["messages"])
        if totals["messages"] > self._IMPORT_MAX_TOTAL_MESSAGES:
            raise ValueError("messages exceeds the total import limit")
        return item

    def _import_session_row(self, conn, raw: dict[str, Any], messages: list[dict[str, Any]], session_id: str) -> None:
        """INSERT one normalized session + its messages; counts fixed up after."""
        started_at = coerce_epoch(raw.get("started_at"), session_id=session_id, field="started_at")
        params = {
            "id": session_id, "source": str(raw.get("source") or "import"),
            "system_prompt_hash": self._store_system_prompt(conn, raw.get("system_prompt")),
            "started_at": time.time() if started_at is None else started_at,
            **{col: 1 if self._coerce_or(raw.get(col), int, 0) else 0 for col in _IMPORT_FLAG_COLS},
            **{col: raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS},
            **{col: self._import_float_or_finite(raw.get(col), col) for col in _IMPORT_FLOAT_COLS},
            **{col: self._import_int_or_bounded(raw.get(col), col, 0) for col in _IMPORT_INT_COLS},
        }
        params["auto_archived"] &= params["archived"]  # only the sweep sets it, and only on rows it archives
        conn.execute(_IMPORT_SESSION_INSERT_SQL, params)
        def _json_value(value: Any) -> Any:
            return safe_json_loads(value, default=value) if isinstance(value, str) else value
        sanitized_messages = [
            {**msg, **{key: _json_value(msg.get(key)) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages
        ]
        # A row exported archived (``include_inactive``) must stay archived: inserted live, compacted or
        # rewound turns would re-enter model context. Flags are coerced like the int session columns, so a
        # hand-edited "0" archives and a missing/null/unparsable flag imports live (older exports have none).
        live: list[dict[str, Any]] = []
        archived: list[dict[str, Any]] = []
        for msg in sanitized_messages:
            (live if self._coerce_or(msg.get("active"), int, 1) else archived).append(msg)
        self._insert_message_rows(conn, session_id, sanitized_messages, prune_checkpoints=False)
        if archived:
            conn.executemany("UPDATE messages SET active = 0, compacted = ? WHERE id = ?",
                             [(1 if self._coerce_or(msg.get("compacted"), int, 0) else 0, msg["_row_id"])
                              for msg in archived])
        # Pruning keys on live rows, so it runs only now: while every row was still live, an archived row's
        # newer checkpoint would strip the newest live one, and archived rows keep theirs as in the donor.
        self._prune_shadowed_checkpoints(conn, session_id, live)
        # Session counters count live rows only.
        conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                     (len(live), sum(_tool_calls_count(_parse_tool_calls(msg.get("tool_calls"))) for msg in live),
                      session_id))

    @staticmethod
    def _attach_import_parents(conn, parent_updates: list[tuple]) -> int:
        """Re-attach imported children whose parent exists (in the store or the same payload)
        without creating a cycle; returns the detached count. Only the closing edge of a
        cycle is dropped, so later entries can still attach to the now-root session."""
        parent_by_child = dict(parent_updates)

        def _would_create_cycle(session_id: str, parent_id: str) -> bool:
            seen = {session_id}
            current = parent_id
            while current:
                if current in seen:
                    return True
                seen.add(current)
                if current in parent_by_child:
                    current = parent_by_child[current]
                    continue
                row = conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1", (current,),
                ).fetchone()
                if row is None:
                    return False
                current = row["parent_session_id"]
            return False

        detached = 0
        for session_id, parent_id in parent_updates:
            parent_exists = conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (parent_id,)).fetchone()
            if parent_exists and not _would_create_cycle(session_id, parent_id):
                conn.execute("UPDATE sessions SET parent_session_id = ? WHERE id = ?", (parent_id, session_id))
            else:
                parent_by_child.pop(session_id, None)
                detached += 1
        return detached

    def import_sessions(
        self: Any, sessions: list[dict[str, Any]], *, transfer_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Import sessions exported by :meth:`export_session` or ``export_all``. Existing ids
        are skipped. A child keeps its parent only when the parent exists or is in the
        same payload; otherwise it is detached so partial imports pass FK validation.
        Gateway routing, handoff, rewind and other live runtime state are reset: this
        restores history, not ownership of a live channel or process. Export INCLUDES
        ``last_activity_*`` but import RESETS them to NULL — resurrecting a stale
        "working ..." label would fabricate activity the watchdog acts on (pinned).

        Activity contract (#76354 review S4): export INCLUDES the live activity fields (``last_activity_at``
        / ``last_activity_description`` / ``last_activity_provenance``) because they are part of the durable
        row, but import deliberately RESETS them to NULL. This asymmetry is intentional and covered by
        regression
        (tests/gateway/test_watchdog_review.py::test_s4_export_includes_activity_import_resets_it).
        """
        normalized, errors = self._validate_import_payload(sessions)
        if errors:
            return {"ok": False, "imported": 0, "skipped": 0, "detached": 0, "errors": errors}

        session_ids = tuple(dict.fromkeys(
            str(item["session"].get("id") or "").strip() for item in normalized
            if str(item["session"].get("id") or "").strip()
        ))

        def _do(conn):
            return self._import_sessions_locked(conn, normalized)

        with session_lineage_guard(self.db_path, session_ids, transfer_from=transfer_from, timeout_s=self._WRITE_PATIENCE_S):
            # Every payload id is guarded, not only the absent ones: an id that was present
            # when ``present_ids`` was computed can be moved to a sibling store before this
            # transaction opens, and re-importing it here would recreate it behind the fence
            # (review finding 2). The guard itself rejects ids owned elsewhere; inside the
            # write, the per-id presence recheck skips them instead of inserting.
            return self._execute_write(_do)

    def _import_sessions_locked(self, conn: sqlite3.Connection, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert normalized payloads on the CALLER's open transaction (per-id presence
        recheck inside). Shared by ``import_sessions`` and adoption, which commits the
        durable transfer record in the SAME transaction as these inserts."""
        from hermes_state_mutation_retirement import retired_in
        # A user-deleted id stays fenced for good: its row restored beside the fence would list
        # fine and refuse every turn, so the import is refused by name and writes nothing.
        ids = [str(item["session"].get("id") or "").strip() for item in normalized]
        if deleted := [sid for sid in ids if retired_in(conn, sid)]:
            return {"ok": False, "imported": 0, "skipped": 0, "detached": 0, "errors": [
                f"session {sid} was deleted here; a deleted session is not restored" for sid in deleted]}
        imported_ids: list[str] = []
        skipped_ids: list[str] = []
        parent_updates: list[tuple[str, str]] = []
        for item in normalized:
            raw = item["session"]
            session_id = str(raw.get("id") or "").strip()
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
                skipped_ids.append(session_id)
                continue
            self._import_session_row(conn, raw, item["messages"], session_id)
            parent_id = str(raw.get("parent_session_id") or "").strip()
            if parent_id:
                parent_updates.append((session_id, parent_id))
            imported_ids.append(session_id)
        return {
            "ok": True, "imported": len(imported_ids), "skipped": len(skipped_ids),
            "detached": self._attach_import_parents(conn, parent_updates),
            "imported_ids": imported_ids, "skipped_ids": skipped_ids, "errors": [],
        }
