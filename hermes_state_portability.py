"""Session listing/rich rows, export, and import (portability) for SessionDB.

Mixin contract: this is a plain mixin class consumed by
``hermes_state.SessionDB``. It defines no ``__init__`` and no state of its
own; methods access the host's attributes (``self._conn``, ``self.db_path``,
``self._execute_write`` and other SessionDB methods) established by
``SessionDB.__init__``. It must never import hermes_state (cycle) — shared
module-level constants live in hermes_state_common.
"""

import logging
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.skill_commands import SKILL_SCAFFOLD_SQL_LIKE
from hermes_state_common import (
    SCHEMA_SQL,
    _PREVIEW_ELIGIBLE_SQL,
    _PREVIEW_RAW_SELECT,
    _shape_preview,
    _sql_session_last_active,
)

# Moved methods logged under the "hermes_state" logger before the split;
# keep that logger identity so log filtering/capture behavior is unchanged.
logger = logging.getLogger("hermes_state")


class SessionPortabilityMixin:
    """See module docstring — mixin for SessionDB (Port cluster)."""

    @classmethod
    def _compact_session_cols(cls) -> str:
        """SELECT list for compact_rows: every ``sessions`` column declared in
        SCHEMA_SQL except prompt storage internals, aliased with the ``s``
        prefix used by list_sessions_rich/_get_session_rich_row queries."""
        if cls._session_compact_cols_sql is None:
            declared = cls._parse_schema_columns(SCHEMA_SQL)["sessions"]
            cls._session_compact_cols_sql = ", ".join(
                f"s.{name}" for name in declared
                if name not in cls._SESSION_COMPACT_EXCLUDED
            )
        return cls._session_compact_cols_sql

    def distinct_session_cwds(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Distinct non-empty session cwds with usage stats, for repo discovery.

        Aggregates across ALL session history (not a single page), so the desktop
        can surface every git repo the user has worked in — not just the repos
        that happen to be in the currently-loaded recents. Children/branches
        count: a worktree session is still a real workspace signal.
        """
        where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
        if not include_archived:
            where += " AND archived = 0"
        with self._lock:
            rows = self._conn.execute(
                "SELECT cwd AS cwd, COUNT(*) AS sessions, "
                "MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
                f"FROM sessions WHERE {where} GROUP BY cwd"
            ).fetchall()
        return [
            {
                "cwd": r["cwd"],
                "sessions": int(r["sessions"] or 0),
                "last_active": float(r["last_active"] or 0),
            }
            for r in rows
        ]

    def list_cron_job_runs(
        self,
        job_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List the run sessions produced by a single cron job, newest first.

        Cron runs are flat, independent sessions whose id is
        ``cron_{job_id}_{timestamp}`` (see ``cron/scheduler.run_job``). They are
        never compression roots and never branch, so this deliberately skips the
        ``list_sessions_rich`` recursive compression-chain CTE / leading-wildcard
        ``id_query`` path — that path seeds from *every* ``source='cron'`` row in
        the DB and only filters to one job's runs after the scan, so it scales
        with the whole cron pile (a heavy history makes the desktop run-history
        endpoint time out before it eventually populates).

        Instead this binds to one job with a ``[prefix, prefix_hi)`` range over
        the id (an index range scan, not a ``%...%`` substring), filters
        ``source='cron'``, and orders by ``started_at DESC``. Work scales with
        the requested window, not the total cron history.

        Returns the same enriched row shape as ``list_sessions_rich`` (adds
        ``preview`` + ``last_active``) so callers can reuse it.
        """
        prefix = f"cron_{job_id}_"
        # Half-open upper bound for an index range scan: increment the final
        # byte of the prefix so the range covers exactly the ids that start
        # with ``prefix`` and nothing else. ``prefix`` always ends in '_', but
        # compute it generically rather than hardcoding the successor char.
        prefix_hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)

        query = f"""
            SELECT s.*,
                COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved,
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                       AND {_PREVIEW_ELIGIBLE_SQL}
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
            WHERE s.source = 'cron' AND s.id >= ? AND s.id < ?
            ORDER BY s.started_at DESC, s.id DESC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            cursor = self._conn.execute(query, (prefix, prefix_hi, limit, offset))
            rows = cursor.fetchall()

        runs: List[Dict[str, Any]] = []
        for row in rows:
            s = self._session_row_dict(row)
            s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
            runs.append(s)
        return runs

    def _get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch a single session with the same enriched columns as
        ``list_sessions_rich`` (preview + last_active). Returns None if the
        session doesn't exist.

        Pass ``compact_rows=True`` to omit the ``system_prompt`` blob (see
        ``list_sessions_rich`` for details).

        Thin wrapper over ``_get_session_rich_rows_batch`` so the enriched
        SELECT lives in exactly one place.
        """
        return self._get_session_rich_rows_batch(
            [session_id], compact_rows=compact_rows
        ).get(session_id)

    def _get_session_rich_rows_batch(
        self, session_ids, compact_rows: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple sessions with the same enriched columns as
        ``_get_session_rich_row``, in a single query.

        Used by ``list_sessions_rich``'s compression-tip projection to resolve
        every tip row for a page in one round trip instead of one query per
        compression-root row. Returns a dict keyed by session id; ids that
        don't exist are simply absent from the result (same as
        ``_get_session_rich_row`` returning ``None`` for them).
        """
        ids = [sid for sid in session_ids if sid]
        if not ids:
            return {}
        # Old SQLite builds cap bound variables at 999
        # (SQLITE_MAX_VARIABLE_NUMBER); large pages (limit=10000 callers
        # exist) could exceed it. Chunk the IN list so the helper is safe at
        # any page size — this is the single choke point for the enriched
        # multi-row fetch, so the bound lives here, not at call sites.
        _CHUNK = 900
        if len(ids) > _CHUNK:
            result: Dict[str, Dict[str, Any]] = {}
            for start in range(0, len(ids), _CHUNK):
                result.update(
                    self._get_session_rich_rows_batch(
                        ids[start:start + _CHUNK], compact_rows=compact_rows
                    )
                )
            return result
        # Same read-your-writes guarantee as list_sessions_rich.
        self.flush_token_counts()
        _sel = self._compact_session_cols() if compact_rows else "s.*"
        placeholders = ",".join("?" for _ in ids)
        prompt_select = (
            "" if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            "" if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )
        query = f"""
            SELECT {_sel}{prompt_select},
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                       AND {_PREVIEW_ELIGIBLE_SQL}
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE s.id IN ({placeholders})
        """
        with self._lock:
            cursor = self._conn.execute(query, ids)
            rows = cursor.fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            s = self._session_row_dict(row)
            s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
            result[s["id"]] = s
        return result

    def get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> Optional[Dict[str, Any]]:
        """Public wrapper for :meth:`_get_session_rich_row`.

        Exposes the single-session enriched row (same columns as
        ``list_sessions_rich``: preview + last_active) for callers outside
        this module, e.g. the web server's session-search hydration.
        """
        return self._get_session_rich_row(session_id, compact_rows=compact_rows)

    def list_skill_scaffolded_sessions(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Titled sessions whose first user turn was a ``/skill`` invocation.

        Those titles were generated from the expanded message, which embeds the
        whole skill body — so they describe the skill rather than the request.
        Returns ``id``, ``title``, and the full first-turn ``content`` so a
        caller can re-derive what the user typed. Newest first.
        """
        with self._lock:
            rows = self._conn.execute(
                """
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
                """,
                (SKILL_SCAFFOLD_SQL_LIKE, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_first_assistant_text(self, session_id: str) -> str:
        """The session's first assistant reply as plain text ('' when none).

        Pairs with :meth:`list_skill_scaffolded_sessions` so a re-title can feed
        the titler the same (request, reply) shape the live path uses.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM messages "
                "WHERE session_id = ? AND role = 'assistant' AND content IS NOT NULL "
                "ORDER BY timestamp, id LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return ""
        decoded = self._decode_content(row["content"])
        return decoded if isinstance(decoded, str) else ""

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_session_lineage(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a compression lineage as one logical session dict."""
        lineage_ids = self.get_compression_lineage(session_id)
        if not lineage_ids:
            return None
        segments = []
        for sid in lineage_ids:
            segment = self.export_session(sid)
            if segment:
                segments.append(segment)
        if not segments:
            return None
        base = dict(segments[-1])
        total_messages = sum(len(seg.get("messages") or []) for seg in segments)
        base["segments"] = segments
        base["lineage_session_ids"] = [seg["id"] for seg in segments]
        base["message_count"] = total_messages
        base["messages"] = [msg for seg in segments for msg in (seg.get("messages") or [])]
        return base

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    @staticmethod
    def _import_text_or_none(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise ValueError(f"{field} must be a string")

    @classmethod
    def _import_non_finite_path(cls, value: Any, path: str = "session") -> Optional[str]:
        """Return the first path containing NaN/Infinity in decoded JSON."""
        pending = [(value, path)]
        seen: set[int] = set()
        while pending:
            current, current_path = pending.pop()
            if isinstance(current, float) and not math.isfinite(current):
                return current_path
            if not isinstance(current, (dict, list)) or id(current) in seen:
                continue
            seen.add(id(current))
            children = current.items() if isinstance(current, dict) else enumerate(current)
            for key, child in reversed(list(children)):
                child_path = (
                    f"{current_path}.{key}"
                    if isinstance(current, dict)
                    else f"{current_path}[{key}]"
                )
                pending.append((child, child_path))
        return None

    @classmethod
    def _validate_import_finite_numbers(cls, value: Any, path: str) -> None:
        non_finite_path = cls._import_non_finite_path(value, path)
        if non_finite_path:
            raise ValueError(f"non-finite number at {non_finite_path}")

    @classmethod
    def _import_load_json(cls, value: str, path: str) -> Any:
        """Decode JSON text and reject values unsafe for strict JSON output."""
        try:
            parsed = json.loads(value)
        except RecursionError as exc:
            raise ValueError(
                f"{path} exceeds the maximum JSON nesting depth"
            ) from exc
        cls._validate_import_finite_numbers(parsed, path)
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
        cls._validate_import_finite_numbers(value, path)
        try:
            return json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @staticmethod
    def _import_float_or_none(value: Any, field: str) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must be a finite number")
        return number

    @classmethod
    def _import_timestamp_or_none(cls, value: Any, field: str) -> Optional[float]:
        number = cls._import_float_or_none(value, field)
        if number is None:
            return None
        try:
            datetime.fromtimestamp(number)
            datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{field} must be a supported Unix timestamp") from exc
        return number

    @staticmethod
    def _is_sqlite_integer(value: int) -> bool:
        return -(1 << 63) <= value <= (1 << 63) - 1

    @classmethod
    def _import_int_or_none(cls, value: Any, field: str) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if not cls._is_sqlite_integer(number):
            raise ValueError(f"{field} is outside SQLite's integer range")
        return number

    @classmethod
    def _import_reasoning_json_value(cls, value: Any, path: str) -> Any:
        if not isinstance(value, str):
            cls._validate_import_finite_numbers(value, path)
            return value
        try:
            return cls._import_load_json(value, path)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _import_message_content(cls, value: Any, path: str) -> Any:
        if value is None or isinstance(value, (str, list, dict)):
            return value
        if isinstance(value, int):
            if not cls._is_sqlite_integer(value):
                raise ValueError(f"{path} is outside SQLite's integer range")
            return value
        if isinstance(value, float):
            cls._validate_import_finite_numbers(value, path)
            return value
        raise ValueError(
            f"{path} must be a string, list, object, finite number, or null"
        )

    @staticmethod
    def _import_error(index: int, session_id: str, error: str) -> Dict[str, Any]:
        item: Dict[str, Any] = {"index": index, "error": error}
        if session_id:
            item["session_id"] = session_id
        return item

    def import_sessions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import sessions exported by :meth:`export_session` or ``export_all``.

        Existing session IDs are skipped. Imported child sessions keep their
        parent only when that parent already exists or is included in the same
        import payload; otherwise the child is detached so partial imports don't
        fail foreign-key validation. Gateway routing, handoff, rewind, and other
        live runtime state are intentionally reset: this restores conversation
        history, not ownership of a live channel or process.

        Activity contract (#76354 review S4): export INCLUDES the live
        activity fields (``last_activity_at`` / ``last_activity_description``
        / ``last_activity_provenance``) because they are part of the durable
        row, but import deliberately RESETS them to NULL. Resurrecting a
        stale "working ..." label on a machine where no agent is running
        would fabricate activity the watchdog and session listings act on.
        This asymmetry is intentional and covered by regression
        (tests/gateway/test_watchdog_review_76354.py::test_s4_export_includes_activity_import_resets_it).
        """
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:
            raise ValueError(
                f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries"
            )

        normalized: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        total_messages = 0
        total_bytes = 0
        session_text_fields = (
            "source",
            "user_id",
            "model",
            "system_prompt",
            "end_reason",
            "cwd",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
        )
        message_text_fields = (
            "role",
            "tool_call_id",
            "tool_name",
            "effect_disposition",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "platform_message_id",
            "message_id",
        )
        session_timestamp_fields = (
            "started_at",
            "ended_at",
        )
        session_float_fields = (
            "estimated_cost_usd",
            "actual_cost_usd",
        )
        session_int_fields = (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "api_call_count",
        )
        message_json_fields = (
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
        )

        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                errors.append(self._import_error(index, "", "session must be an object"))
                continue
            session_id = str(raw.get("id") or "").strip()
            if not session_id:
                errors.append(self._import_error(index, "", "session id is required"))
                continue
            if session_id in seen_ids:
                errors.append(self._import_error(index, session_id, "duplicate session id"))
                continue
            try:
                self._validate_import_finite_numbers(raw, "session")
            except ValueError as exc:
                errors.append(
                    self._import_error(index, session_id, str(exc))
                )
                continue
            messages = raw.get("messages") or []
            if not isinstance(messages, list):
                errors.append(self._import_error(index, session_id, "messages must be a list"))
                continue
            if len(messages) > self._IMPORT_MAX_MESSAGES_PER_SESSION:
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the per-session import limit",
                    )
                )
                continue
            if any(not isinstance(msg, dict) for msg in messages):
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages must contain only objects",
                    )
                )
                continue

            try:
                session_bytes = len(
                    json.dumps(
                        raw,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
            except (TypeError, ValueError, RecursionError):
                errors.append(
                    self._import_error(index, session_id, "session must be JSON serializable")
                )
                continue
            if session_bytes > self._IMPORT_MAX_SESSION_BYTES:
                errors.append(
                    self._import_error(index, session_id, "session exceeds the import size limit")
                )
                continue
            total_bytes += session_bytes
            if total_bytes > self._IMPORT_MAX_TOTAL_BYTES:
                errors.append(
                    self._import_error(index, session_id, "import exceeds the total size limit")
                )
                continue

            try:
                clean_session = dict(raw)
                clean_session["id"] = session_id
                clean_session["model_config"] = self._import_json_object_or_none(
                    clean_session.get("model_config"), "model_config"
                )
                clean_session["parent_session_id"] = self._import_text_or_none(
                    clean_session.get("parent_session_id"), "parent_session_id"
                )
                for field in session_text_fields:
                    clean_session[field] = self._import_text_or_none(
                        clean_session.get(field), field
                    )
                for field in session_timestamp_fields:
                    clean_session[field] = self._import_timestamp_or_none(
                        clean_session.get(field), field
                    )
                for field in session_float_fields:
                    clean_session[field] = self._import_float_or_none(
                        clean_session.get(field), field
                    )
                for field in session_int_fields:
                    clean_session[field] = self._import_int_or_none(
                        clean_session.get(field), field
                    )

                clean_messages: List[Dict[str, Any]] = []
                for message_index, message in enumerate(messages):
                    clean_message = dict(message)
                    role = clean_message.get("role")
                    if not isinstance(role, str) or not role:
                        raise ValueError(f"messages[{message_index}].role must be a non-empty string")
                    for field in message_text_fields:
                        if field == "role":
                            continue
                        clean_message[field] = self._import_text_or_none(
                            clean_message.get(field), field
                        )
                    clean_message["token_count"] = self._import_int_or_none(
                        clean_message.get("token_count"),
                        "token_count",
                    )
                    clean_message["timestamp"] = self._import_timestamp_or_none(
                        clean_message.get("timestamp"),
                        f"messages[{message_index}].timestamp",
                    )
                    message_path = f"session.messages[{message_index}]"
                    clean_message["content"] = self._import_message_content(
                        clean_message.get("content"), f"{message_path}.content"
                    )
                    for field in message_json_fields:
                        clean_message[field] = self._import_reasoning_json_value(
                            clean_message.get(field), f"{message_path}.{field}"
                        )
                    clean_messages.append(clean_message)
            except ValueError as exc:
                errors.append(self._import_error(index, session_id, str(exc)))
                continue

            total_messages += len(clean_messages)
            if total_messages > self._IMPORT_MAX_TOTAL_MESSAGES:
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the total import limit",
                    )
                )
                continue
            seen_ids.add(session_id)
            normalized.append(
                {"index": index, "session": clean_session, "messages": clean_messages}
            )

        if errors:
            return {
                "ok": False,
                "imported": 0,
                "skipped": 0,
                "detached": 0,
                "errors": errors,
            }

        def _do(conn):
            imported_ids: List[str] = []
            skipped_ids: List[str] = []
            parent_updates: List[tuple[str, str]] = []
            detached = 0

            for item in normalized:
                raw = item["session"]
                messages = item["messages"]
                session_id = str(raw.get("id") or "").strip()
                exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
                if exists:
                    skipped_ids.append(session_id)
                    continue

                started_at = raw.get("started_at")
                if started_at is None:
                    started_at = time.time()
                archived = 1 if raw.get("archived") else 0
                system_prompt_hash = self._store_system_prompt(
                    conn, raw.get("system_prompt")
                )

                conn.execute(
                    """INSERT INTO sessions (
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
                       )""",
                    {
                        "id": session_id,
                        "source": str(raw.get("source") or "import"),
                        "user_id": raw.get("user_id"),
                        "model": raw.get("model"),
                        "model_config": raw.get("model_config"),
                        "system_prompt_hash": system_prompt_hash,
                        "started_at": started_at,
                        "ended_at": raw.get("ended_at"),
                        "end_reason": raw.get("end_reason"),
                        "input_tokens": raw.get("input_tokens") or 0,
                        "output_tokens": raw.get("output_tokens") or 0,
                        "cache_read_tokens": raw.get("cache_read_tokens") or 0,
                        "cache_write_tokens": raw.get("cache_write_tokens") or 0,
                        "reasoning_tokens": raw.get("reasoning_tokens") or 0,
                        "cwd": raw.get("cwd"),
                        "git_branch": raw.get("git_branch"),
                        "git_repo_root": raw.get("git_repo_root"),
                        "billing_provider": raw.get("billing_provider"),
                        "billing_base_url": raw.get("billing_base_url"),
                        "billing_mode": raw.get("billing_mode"),
                        "estimated_cost_usd": raw.get("estimated_cost_usd"),
                        "actual_cost_usd": raw.get("actual_cost_usd"),
                        "cost_status": raw.get("cost_status"),
                        "cost_source": raw.get("cost_source"),
                        "pricing_version": raw.get("pricing_version"),
                        "title": raw.get("title"),
                        "api_call_count": raw.get("api_call_count") or 0,
                        "archived": archived,
                    },
                )

                total_messages, total_tool_calls = self._insert_message_rows(
                    conn,
                    session_id,
                    messages,
                )
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                    (total_messages, total_tool_calls, session_id),
                )

                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(session_id)

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
                        "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                    if row is None:
                        return False
                    current = row["parent_session_id"]
                return False

            for session_id, parent_id in parent_updates:
                parent_exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (parent_id,),
                ).fetchone()
                if parent_exists and not _would_create_cycle(session_id, parent_id):
                    conn.execute(
                        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                        (parent_id, session_id),
                    )
                else:
                    # Drop only the closing edge. Later entries can still attach
                    # to this now-root session, preserving the acyclic portion
                    # of a malformed imported lineage.
                    parent_by_child.pop(session_id, None)
                    detached += 1

            return {
                "ok": True,
                "imported": len(imported_ids),
                "skipped": len(skipped_ids),
                "detached": detached,
                "imported_ids": imported_ids,
                "skipped_ids": skipped_ids,
                "errors": [],
            }

        return self._execute_write(_do)
