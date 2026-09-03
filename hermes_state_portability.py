"""Session listing/rich rows, export, and import (portability) for SessionDB.

Plain mixin for ``hermes_state.SessionDB`` (no ``__init__``/state of its own).
Must never import hermes_state (cycle); shared constants live in hermes_state_common.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional

from agent.skill_commands import SKILL_SCAFFOLD_SQL_LIKE
from utils import safe_json_loads
from hermes_state_common import SCHEMA_SQL, _PREVIEW_RAW_SUBQUERY_SQL, _shape_preview, _sql_session_last_active

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

_MESSAGE_CLONE_LINEAGE_EXPORT_VERSION = 1

_IMPORT_SESSION_TEXT_FIELDS = (
    "source", "user_id", "model", "system_prompt", "end_reason", "cwd", "git_branch", "git_repo_root",
    "billing_provider", "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
# ``role`` is validated separately (non-empty string).
_IMPORT_MESSAGE_TEXT_FIELDS = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason",
    "reasoning", "reasoning_content", "platform_message_id", "message_id",
)
_IMPORT_MESSAGE_JSON_FIELDS = ("reasoning_details", "codex_reasoning_items", "codex_message_items")
_IMPORT_SESSION_INSERT_SQL = """INSERT INTO sessions (
                           id, source, user_id, model, model_config, system_prompt,
                           system_prompt_hash,
                           parent_session_id, started_at, ended_at, end_reason,
                           message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived
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
                           :api_call_count, :archived
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


def _rich_select(select_cols: str, where: str, tail: str = "", prompt_select: Optional[str] = "") -> str:
    """``list_sessions_rich``-shaped SELECT: resolved prompt (``prompt_select`` fragment;
    None omits prompt columns AND the join), preview, last_active. Whitespace matches
    the historical inline queries (SQL text is pinned)."""
    prompt_join = "" if prompt_select is None else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
    return f"""
            SELECT {select_cols}{prompt_select or ""},
                {_PREVIEW_RAW_SUBQUERY_SQL},
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE {where}{tail}
        """


_PROMPT_RESOLVED_SQL = "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"


class SessionPortabilityMixin:
    """See module docstring — mixin for SessionDB (Port cluster)."""

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
    def _rich_row(cls, row) -> Dict[str, Any]:
        """Session row dict with ``_preview_raw`` shaped into ``preview``."""
        s = cls._session_row_dict(row)
        s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
        return s

    def _read_rows(self, sql: str, params=()) -> list:
        """Pure-read query via ``_read_ctx()`` (never the writer lock: turn persistence must not convoy)."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchall()

    def distinct_session_cwds(self, include_archived: bool = False) -> List[Dict[str, Any]]:
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

    def list_cron_job_runs(self, job_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Run sessions of one cron job, newest first, in the ``list_sessions_rich`` row shape.
        Cron runs are flat ``cron_{job_id}_{timestamp}`` sessions that never compress or
        branch, so this skips ``list_sessions_rich``'s compression-chain CTE /
        leading-wildcard ``id_query`` path (which seeds from EVERY ``source='cron'`` row)
        for a ``[prefix, prefix_hi)`` id range scan that scales with the window."""
        prefix = f"cron_{job_id}_"
        # Half-open upper bound: bump the final byte so the range covers exactly the prefix.
        prefix_hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        query = _rich_select(
            "s.*", "s.source = 'cron' AND s.id >= ? AND s.id < ?",
            "\n            ORDER BY s.started_at DESC, s.id DESC\n            LIMIT ? OFFSET ?",
            prompt_select=f",\n                {_PROMPT_RESOLVED_SQL}",
        )
        return [self._rich_row(row) for row in self._read_rows(query, (prefix, prefix_hi, limit, offset))]

    def _get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> Optional[Dict[str, Any]]:
        """One session with the ``list_sessions_rich`` enriched columns, or None.
        ``compact_rows=True`` omits the ``system_prompt`` blob. Public alias:
        :meth:`get_session_rich_row` (web server hydration)."""
        return self._get_session_rich_rows_batch([session_id], compact_rows=compact_rows).get(session_id)

    get_session_rich_row = _get_session_rich_row

    def _get_session_rich_rows_batch(self, session_ids, compact_rows: bool = False) -> Dict[str, Dict[str, Any]]:
        """Enriched rows for many sessions in one query, keyed by id; missing ids are absent
        (a page of compression tips resolves in one round trip)."""
        ids = [sid for sid in session_ids if sid]
        if not ids:
            return {}
        # Old SQLite caps bound variables at 999 (SQLITE_MAX_VARIABLE_NUMBER); limit=10000
        # callers exist. Chunk here — the single choke point.
        _CHUNK = 900
        if len(ids) > _CHUNK:
            result: Dict[str, Dict[str, Any]] = {}
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

    def list_skill_scaffolded_sessions(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Titled sessions whose first user turn was a ``/skill`` invocation (their titles
        describe the expanded skill body, not the request). Returns ``id``, ``title`` and
        the first-turn ``content`` so callers can re-derive what was typed. Newest first."""
        rows = self._read_rows("""
                SELECT s.id, s.title, m.content
                FROM sessions s
                JOIN messages m ON m.id = (
                    SELECT m2.id FROM messages m2
                    WHERE m2.session_id = s.id AND m2.role = 'user'
                      AND m2.content IS NOT NULL
                    ORDER BY m2.timestamp, m2.id LIMIT 1
                )
                WHERE s.title IS NOT NULL AND m.content LIKE ?
                ORDER BY s.started_at DESC
                LIMIT ?
                """, (SKILL_SCAFFOLD_SQL_LIKE, int(limit)))
        return [dict(row) for row in rows]

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_message_clone_edges(self, message_ids) -> List[Dict[str, int]]:
        ids = {int(message_id) for message_id in message_ids if message_id is not None}
        if not ids:
            return []
        edges: List[Dict[str, int]] = []
        ordered_ids = list(ids)
        with self._read_ctx() as conn:
            for start in range(0, len(ordered_ids), 900):
                chunk = ordered_ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT source_message_id, clone_message_id FROM message_clone_lineage "
                    f"WHERE source_message_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                edges.extend(
                    {
                        "source_message_id": int(row["source_message_id"]),
                        "clone_message_id": int(row["clone_message_id"]),
                    }
                    for row in rows
                    if int(row["clone_message_id"]) in ids
                )
        edges.sort(key=lambda edge: (edge["source_message_id"], edge["clone_message_id"]))
        return edges

    def _with_messages(self, session: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.get_messages(session["id"])
        result = {**session, "messages": messages}
        edges = self._export_message_clone_edges(message.get("id") for message in messages)
        if edges:
            result["message_clone_lineage_version"] = _MESSAGE_CLONE_LINEAGE_EXPORT_VERSION
            result["message_clone_lineage"] = edges
        return result

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        return self._with_messages(session) if session else None

    def export_session_lineage(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a compression lineage as one logical session dict."""
        lineage_ids = self.get_compression_lineage(session_id)
        if not lineage_ids:
            return None
        segments = [seg for seg in map(self.export_session, lineage_ids) if seg]
        if not segments:
            return None
        lineage_message_ids = {
            message.get("id")
            for segment in segments
            for message in (segment.get("messages") or [])
            if message.get("id") is not None
        }
        lineage_edges = self._export_message_clone_edges(lineage_message_ids)
        for segment in segments:
            segment.pop("message_clone_lineage_version", None)
            segment.pop("message_clone_lineage", None)
        if lineage_edges:
            segments[-1]["message_clone_lineage_version"] = _MESSAGE_CLONE_LINEAGE_EXPORT_VERSION
            segments[-1]["message_clone_lineage"] = lineage_edges
        messages = [msg for seg in segments for msg in (seg.get("messages") or [])]
        return {
            **segments[-1], "segments": segments,
            "lineage_session_ids": [seg["id"] for seg in segments], "message_count": len(messages),
            "messages": messages,
        }

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """Export all sessions (with messages) as dicts, e.g. for JSONL backup."""
        results = [self._with_messages(s) for s in self.search_sessions(source=source, limit=100000)]
        message_owner_by_id = {
            int(message["id"]): result
            for result in results
            for message in (result.get("messages") or [])
            if isinstance(message.get("id"), int) and not isinstance(message.get("id"), bool)
        }
        edges = self._export_message_clone_edges(message_owner_by_id)
        edges_by_carrier: Dict[str, List[Dict[str, int]]] = {}
        carrier_by_id: Dict[str, Dict[str, Any]] = {}
        for result in results:
            result.pop("message_clone_lineage_version", None)
            result.pop("message_clone_lineage", None)
        for edge in edges:
            carrier = message_owner_by_id[edge["clone_message_id"]]
            carrier_id = str(carrier["id"])
            carrier_by_id[carrier_id] = carrier
            edges_by_carrier.setdefault(carrier_id, []).append(edge)
        for carrier_id, carrier_edges in edges_by_carrier.items():
            carrier = carrier_by_id[carrier_id]
            carrier["message_clone_lineage_version"] = _MESSAGE_CLONE_LINEAGE_EXPORT_VERSION
            carrier["message_clone_lineage"] = carrier_edges
        return results

    def adopt_session_lineage_from(self, donor_db: Any, session_id: str, *, retire_donor: bool = True) -> Dict[str, Any]:
        """Adopt *session_id*'s full compression lineage from *donor_db* (stranded-bot-session
        heal: a profile bot's rows accumulated in the DEFAULT profile's state.db before the
        desktop routed session RPCs by target session). Pure composition
        ``donor_db.export_session_lineage()`` -> ``self.import_sessions()``: runtime
        fields reset, already-present ids skipped (idempotent). With ``retire_donor`` and
        a complete adoption, donor rows are ARCHIVED (never deleted) with
        ``end_reason='adopted_by_profile'`` — deliberately NOT in the recoverable set, so
        resurrection cannot undo an adoption. Returns the ``import_sessions`` dict plus
        ``adopted`` and ``donor_retired`` (True only when EVERY segment retired).

        Once routing was fixed, the profile backend correctly received the RPCs but had no such session, so
        the same chat 4001'd for the opposite reason. This method moves the conversation to where routing
        now looks for it. See #93091, #93296.
        """
        payload = donor_db.export_session_lineage(session_id)
        if not payload:
            return {"ok": False, "adopted": False, "donor_retired": False,
                    "error": f"session {session_id!r} not found in donor store"}
        segments = payload.get("segments") or [payload]

        # Divergence guard: a segment we will SKIP (already here) may have kept growing in
        # the donor after a partial adoption; retiring it would strand those messages
        # behind a non-recoverable archive. Still import, but refuse to retire.
        donor_ahead = False
        for seg in segments:
            seg_id = seg.get("id")
            if not seg_id or self.get_session(seg_id) is None:
                continue
            donor_count = len(seg.get("messages") or [])
            local_count = len(self.get_messages(seg_id))
            if donor_count > local_count:
                donor_ahead = True
                logger.warning("adoption divergence: donor segment %s has %d messages, "
                               "local copy has %d — donor will NOT be retired", seg_id, donor_count, local_count)

        result = self.import_sessions([dict(seg) for seg in segments])
        imported = int(result.get("imported") or 0)
        skipped = int(result.get("skipped") or 0)
        adopted = result.get("ok", False) and (imported + skipped) == len(segments)
        if not adopted:
            logger.warning("adoption of %s did not complete: imported=%s skipped=%s of %s segment(s); errors=%s",
                           session_id, imported, skipped, len(segments), result.get("errors"))

        donor_retired = False
        if adopted and retire_donor and not donor_ahead:
            donor_retired = all(self._retire_donor_segment(donor_db, seg["id"]) for seg in segments if seg.get("id"))
        return {**result, "adopted": adopted, "donor_retired": donor_retired}

    def _retire_donor_segment(self, donor_db: Any, seg_id: str) -> bool:
        """Archive one adopted donor segment; False when skipped or failed. TOCTOU close-out:
        the divergence guard used EXPORT-TIME counts; re-read both stores right before
        stamping so donor growth never lands behind a non-recoverable archive
        (equal-count CONTENT divergence is accepted — bytes stay in the donor either way).
        A retirement failure must not fail the adoption (a later resume retries
        idempotently), but never claims success it didn't have."""
        try:
            donor_now = len(donor_db.get_messages(seg_id))
            local_now = len(self.get_messages(seg_id))
            if donor_now > local_now:
                logger.warning(
                    "adoption divergence at retire time: donor segment %s grew to %d messages (local %d) — "
                    "leaving donor unretired", seg_id, donor_now, local_now,
                )
                return False
            # First end_reason wins in end_session(); reopen so the adoption boundary is
            # stamped even on ended segments.
            donor_db.reopen_session(seg_id)
            donor_db.end_session(seg_id, "adopted_by_profile")
            donor_db.set_session_archived(seg_id, True)
            return True
        except Exception:
            logger.warning("failed to retire donor segment %s after adoption", seg_id, exc_info=True)
            return False

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
    def _import_json_object_or_none(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field} must be a JSON object")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object")
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

    def _normalize_import_session(self, raw: Dict[str, Any], session_id: str, messages: list) -> Dict[str, Any]:
        """Type-check one payload session + its messages; raises ValueError."""
        clean_session = dict(raw)
        clean_session["id"] = session_id
        clean_session["model_config"] = self._import_json_object_or_none(clean_session.get("model_config"), "model_config")
        for field in ("parent_session_id", *_IMPORT_SESSION_TEXT_FIELDS):
            clean_session[field] = self._import_text_or_none(clean_session.get(field), field)
        clean_messages: List[Dict[str, Any]] = []
        for message_index, message in enumerate(messages):
            clean_message = dict(message)
            role = clean_message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"messages[{message_index}].role must be a non-empty string")
            for field in _IMPORT_MESSAGE_TEXT_FIELDS:
                clean_message[field] = self._import_text_or_none(clean_message.get(field), field)
            clean_message["token_count"] = self._import_int_or_none(clean_message.get("token_count"), "token_count")
            clean_messages.append(clean_message)
        raw_clone_edges = raw.get("message_clone_lineage") or []
        if not isinstance(raw_clone_edges, list):
            raise ValueError("message_clone_lineage must be a list")
        if raw_clone_edges and raw.get("message_clone_lineage_version") != _MESSAGE_CLONE_LINEAGE_EXPORT_VERSION:
            raise ValueError("unsupported message_clone_lineage_version")
        clean_clone_edges: List[tuple[int, int]] = []
        for edge_index, edge in enumerate(raw_clone_edges):
            if not isinstance(edge, dict):
                raise ValueError(f"message_clone_lineage[{edge_index}] must be an object")
            source_id = edge.get("source_message_id")
            clone_id = edge.get("clone_message_id")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (source_id, clone_id)
            ):
                raise ValueError(f"message_clone_lineage[{edge_index}] ids must be positive integers")
            if source_id == clone_id:
                raise ValueError(f"message_clone_lineage[{edge_index}] cannot self-reference")
            clean_clone_edges.append((source_id, clone_id))
        return {
            "session": clean_session,
            "messages": clean_messages,
            "message_clone_lineage": clean_clone_edges,
        }

    def _validate_import_payload(self, sessions: List[Dict[str, Any]]) -> tuple:
        """Size/shape/type validation of the whole payload; returns ``(normalized_items,
        errors)``. Every rejected entry is reported."""
        normalized: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
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
        exported_id_counts: Dict[int, int] = {}
        for item in normalized:
            for message in item["messages"]:
                exported_id = message.get("id")
                if isinstance(exported_id, int) and not isinstance(exported_id, bool) and exported_id > 0:
                    exported_id_counts[exported_id] = exported_id_counts.get(exported_id, 0) + 1
        for item in normalized:
            for source_id, clone_id in item["message_clone_lineage"]:
                for field, message_id in (("source_message_id", source_id), ("clone_message_id", clone_id)):
                    if exported_id_counts.get(message_id, 0) != 1:
                        error = {
                            "index": item["index"],
                            "session_id": item["session"]["id"],
                            "error": (
                                f"message_clone_lineage {field} must reference exactly "
                                "one imported message"
                            ),
                        }
                        if error not in errors:
                            errors.append(error)
        return normalized, errors

    def _validate_import_session(self, raw: Any, session_id: str, seen_ids: set, totals: Dict[str, int]) -> Dict[str, Any]:
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
            session_bytes = len(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
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

    def _import_session_row(
        self, conn, raw: Dict[str, Any], messages: List[Dict[str, Any]], session_id: str
    ) -> Dict[int, int]:
        """INSERT one normalized session + its messages; counts fixed up after."""
        started_at = self._coerce_or(raw.get("started_at"), float, None)
        params = {
            "id": session_id, "source": str(raw.get("source") or "import"),
            "system_prompt_hash": self._store_system_prompt(conn, raw.get("system_prompt")),
            "started_at": time.time() if started_at is None else started_at,
            "archived": 1 if raw.get("archived") else 0,
            **{col: raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS},
            **{col: self._coerce_or(raw.get(col), float, None) for col in _IMPORT_FLOAT_COLS},
            **{col: self._coerce_or(raw.get(col), int, 0) for col in _IMPORT_INT_COLS},
        }
        conn.execute(_IMPORT_SESSION_INSERT_SQL, params)
        def _json_value(value: Any) -> Any:
            return safe_json_loads(value, default=value) if isinstance(value, str) else value
        sanitized_messages = [
            {**msg, **{key: _json_value(msg.get(key)) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages
        ]
        total_messages, total_tool_calls = self._insert_message_rows(conn, session_id, sanitized_messages)
        conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                     (total_messages, total_tool_calls, session_id))
        return {
            int(message["id"]): int(message["_row_id"])
            for message in sanitized_messages
            if isinstance(message.get("id"), int)
            and not isinstance(message.get("id"), bool)
            and message.get("id") > 0
            and message.get("_row_id") is not None
        }

    def _attach_import_parents(self, conn, parent_updates: List[tuple]) -> int:
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
        topology_ids: set[str] = set()
        topology_roots_before: set[str] = set()
        for session_id, parent_id in parent_updates:
            parent_exists = conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (parent_id,)).fetchone()
            if parent_exists and not _would_create_cycle(session_id, parent_id):
                pair_ids = {session_id, parent_id}
                pair_roots_before = self._display_roots_from_conn(conn, pair_ids)
                attached = conn.execute(
                    "UPDATE sessions SET parent_session_id = ? WHERE id = ?", (parent_id, session_id)
                )
                if attached.rowcount == 1:
                    topology_ids.update(pair_ids)
                    topology_roots_before.update(pair_roots_before)
            else:
                parent_by_child.pop(session_id, None)
                detached += 1
        if topology_ids:
            self._invalidate_display_topology(conn, topology_roots_before, topology_ids)
        return detached

    def import_sessions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        (tests/gateway/test_watchdog_review_76354.py::test_s4_export_includes_activity_import_resets_it).
        """
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries")
        normalized, errors = self._validate_import_payload(sessions)
        if errors:
            return {"ok": False, "imported": 0, "skipped": 0, "detached": 0, "errors": errors}

        exported_message_session_by_id = {
            int(message["id"]): item["session"]["id"]
            for item in normalized
            for message in item["messages"]
            if isinstance(message.get("id"), int)
            and not isinstance(message.get("id"), bool)
            and message.get("id") > 0
        }
        clone_edge_records = [
            (item["index"], item["session"]["id"], source_id, clone_id)
            for item in normalized
            for source_id, clone_id in item["message_clone_lineage"]
        ]

        def _do(conn):
            imported_ids: List[str] = []
            skipped_ids: List[str] = []
            parent_updates: List[tuple[str, str]] = []
            imported_message_id_map: Dict[int, int] = {}
            payload_session_ids = [item["session"]["id"] for item in normalized]
            existing_payload_session_ids: set[str] = set()
            for start in range(0, len(payload_session_ids), 900):
                chunk = payload_session_ids[start : start + 900]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                existing_payload_session_ids.update(
                    row["id"]
                    for row in conn.execute(
                        f"SELECT id FROM sessions WHERE id IN ({placeholders})", chunk
                    ).fetchall()
                )
            for index, session_id, source_id, clone_id in clone_edge_records:
                source_skipped = exported_message_session_by_id[source_id] in existing_payload_session_ids
                clone_skipped = exported_message_session_by_id[clone_id] in existing_payload_session_ids
                if source_skipped != clone_skipped:
                    return {
                        "ok": False, "imported": 0, "skipped": 0, "detached": 0,
                        "errors": [{
                            "index": index,
                            "session_id": session_id,
                            "error": (
                                "message clone provenance crosses skipped and imported sessions; "
                                "refusing a partial component import"
                            ),
                        }],
                    }
            for item in normalized:
                raw = item["session"]
                session_id = str(raw.get("id") or "").strip()
                if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
                    skipped_ids.append(session_id)
                    continue
                imported_message_id_map.update(
                    self._import_session_row(conn, raw, item["messages"], session_id)
                )
                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(session_id)
            detached = self._attach_import_parents(conn, parent_updates)
            remapped_edges = {
                (imported_message_id_map[source_id], imported_message_id_map[clone_id])
                for _index, _session_id, source_id, clone_id in clone_edge_records
                if source_id in imported_message_id_map and clone_id in imported_message_id_map
            }
            if remapped_edges:
                conn.executemany(
                    "INSERT INTO message_clone_lineage "
                    "(source_message_id, clone_message_id) VALUES (?, ?)",
                    sorted(remapped_edges),
                )
            return {
                "ok": True, "imported": len(imported_ids), "skipped": len(skipped_ids),
                "detached": detached,
                "imported_ids": imported_ids, "skipped_ids": skipped_ids, "errors": [],
            }

        return self._execute_write(_do)
