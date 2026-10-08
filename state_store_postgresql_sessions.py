"""Session lifecycle, accounting, insights, browsing, and title operations for PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from hermes_state_common import _RECOVERABLE_END_REASONS, _RESET_END_REASONS

logger = logging.getLogger(__name__)

_USAGE_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_USAGE_SUM_FIELDS = (*_USAGE_COUNTERS, "api_call_count")
_USAGE_ROUTE_FIELDS = ("model", "cost_status", "cost_source", "pricing_version", "billing_provider", "billing_base_url", "billing_mode")
_SESSION_METADATA_COLUMNS = (
    "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json",
    "model", "model_config", "parent_session_id", "cwd", "profile_name", "git_repo_root",
)
_TITLE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TITLE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_TITLE_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]")
_NUMBERED_TITLE_RE = re.compile(r"^(.*?) #(\d+)$")
_TITLE_SOURCE_RANK = {"derived": 0, "llm": 1, "user": 2}
_CANONICAL_BOT_CHAT_TITLE = "Bot Chat"
_MAX_TITLE_LENGTH = 100

def _sanitize_title(title: str | None) -> str | None:
    if not title:
        return None
    cleaned = _TITLE_INVISIBLE_RE.sub("", _TITLE_CONTROL_RE.sub("", _TITLE_SURROGATE_RE.sub("\ufffd", str(title))))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > _MAX_TITLE_LENGTH:
        raise ValueError(f"Title too long ({len(cleaned)} chars, max {_MAX_TITLE_LENGTH})")
    return cleaned


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class PostgreSQLSessionsMixin:
    def queue_token_counts(self, session_id: str, **kwargs: Any) -> None:
        self._token_usage_transport.queue_delta(session_id, kwargs)

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        return self._token_usage_transport.flush(timeout)

    def read_insights_snapshot(self, *, cutoff: float, source: str | None):
        """Return canonical analytics rows from this tenant; callers never receive a PG cursor."""
        from agent.insights import InsightsSnapshot
        predicate, params = "s.started_at >= %s", [cutoff]
        if source is not None:
            predicate += " AND s.source = %s"
            params.append(source)
        session_columns = ("s.id, s.source, s.model, s.started_at, s.ended_at, "
                           "(SELECT COUNT(*) FROM {schema}.messages sm WHERE sm.session_id=s.id) AS message_count, "
                           "(SELECT COUNT(*) FROM {schema}.messages tm WHERE tm.session_id=s.id AND tm.role='tool') AS tool_call_count, "
                           "s.input_tokens, s.output_tokens, s.cache_read_tokens, s.cache_write_tokens, s.billing_provider, "
                           "s.billing_base_url, s.billing_mode, s.estimated_cost_usd, s.actual_cost_usd, s.cost_status, "
                           "s.cost_source, s.api_call_count").format(schema=self._schema)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            def rows(sql: str):
                cursor.execute(sql, params)
                return tuple(dict(row) for row in cursor.fetchall())
            sessions = rows(f"SELECT {session_columns} FROM {self._schema}.sessions s WHERE {predicate} ORDER BY s.started_at DESC")
            tool_rows = rows(f"SELECT m.session_id, m.tool_name, COUNT(*) AS count FROM {self._schema}.messages m "
                             f"JOIN {self._schema}.sessions s ON s.id=m.session_id WHERE {predicate} "
                             "AND m.role='tool' AND m.tool_name IS NOT NULL GROUP BY m.session_id, m.tool_name")
            assistant_rows = rows(f"SELECT m.session_id, m.tool_calls FROM {self._schema}.messages m "
                                  f"JOIN {self._schema}.sessions s ON s.id=m.session_id WHERE {predicate} "
                                  "AND m.role='assistant' AND m.tool_calls IS NOT NULL")
            skill_rows = rows(f"SELECT m.tool_calls, m.created_at AS timestamp FROM {self._schema}.messages m "
                              f"JOIN {self._schema}.sessions s ON s.id=m.session_id WHERE {predicate} "
                              "AND m.role='assistant' AND m.tool_calls IS NOT NULL AND "
                              "(position('skill_view' in m.tool_calls::text) > 0 OR position('skill_manage' in m.tool_calls::text) > 0)")
            stats_rows = rows(f"SELECT COUNT(*) AS total_messages, SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS user_messages, "
                              "SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) AS assistant_messages, "
                              "SUM(CASE WHEN m.role='tool' THEN 1 ELSE 0 END) AS tool_messages "
                              f"FROM {self._schema}.messages m JOIN {self._schema}.sessions s ON s.id=m.session_id WHERE {predicate}")
            usage_rows = rows(f"SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url, u.api_call_count, "
                              "u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens, "
                              "u.estimated_cost_usd, u.actual_cost_usd, u.cost_status, u.cost_source, u.billing_mode "
                              f"FROM {self._schema}.session_model_usage u JOIN {self._schema}.sessions s ON s.id=u.session_id WHERE {predicate}")
        stats = stats_rows[0] if stats_rows else {}
        return InsightsSnapshot(sessions, tool_rows, assistant_rows, skill_rows, stats, usage_rows)

    def _persist_token_usage_delta(self, session_id: str, **kwargs: Any) -> None:
        self.update_token_counts(session_id, **kwargs)

    def update_token_counts(self, session_id: str, input_tokens: int = 0, output_tokens: int = 0, model: str | None = None, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, estimated_cost_usd: float | None = None, actual_cost_usd: float | None = None, cost_status: str | None = None, cost_source: str | None = None, pricing_version: str | None = None, billing_provider: str | None = None, billing_base_url: str | None = None, billing_mode: str | None = None, api_call_count: int = 0, absolute: bool = False, source: str | None = None) -> None:
        counters = (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)
        has_usage = bool(any(counters) or api_call_count or estimated_cost_usd)
        accounted = bool(has_usage or actual_cost_usd is not None)
        # The first token delta may arrive before the normal session creator succeeds.
        # Preserve its real surface instead of leaving a durable unknown placeholder.
        self.ensure_session(session_id, source=source or "unknown")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT model, billing_provider, api_call_count FROM {self._schema}.sessions WHERE id=%s FOR UPDATE", (session_id,))
            row = cursor.fetchone() or {}
            if int(row.get("api_call_count") or 0) == 0 and accounted and model and billing_provider and (row.get("model") != model or row.get("billing_provider") != billing_provider):
                cursor.execute(f"UPDATE {self._schema}.sessions SET model=%s, billing_provider=%s, billing_base_url=%s, billing_mode=%s WHERE id=%s", (model, billing_provider, billing_base_url, billing_mode, session_id))
            additions = not absolute
            set_counters = ", ".join(f"{field} = {'%s' if not additions else field + ' + %s'}" for field in _USAGE_COUNTERS)
            estimated = "COALESCE(%s, 0)" if absolute else "COALESCE(estimated_cost_usd, 0) + COALESCE(%s, 0)"
            actual = "CASE WHEN %s::double precision IS NULL THEN actual_cost_usd ELSE %s::double precision END" if absolute else "CASE WHEN %s::double precision IS NULL THEN actual_cost_usd ELSE COALESCE(actual_cost_usd, 0) + %s::double precision END"
            calls = "%s" if absolute else "api_call_count + %s"
            route = (billing_provider if accounted else None, billing_base_url if accounted else None, billing_mode if accounted else None, model if accounted else None)
            cursor.execute(f"UPDATE {self._schema}.sessions SET {set_counters}, estimated_cost_usd={estimated}, actual_cost_usd={actual}, cost_status=COALESCE(%s,cost_status), cost_source=COALESCE(%s,cost_source), pricing_version=COALESCE(%s,pricing_version), billing_provider=COALESCE(billing_provider,%s), billing_base_url=COALESCE(billing_base_url,%s), billing_mode=COALESCE(billing_mode,%s), model=COALESCE(model,%s), api_call_count={calls} WHERE id=%s", (*counters, estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version, *route, api_call_count, session_id))
            if not absolute and has_usage:
                self._record_model_usage(cursor, session_id, model=model, billing_provider=billing_provider, billing_base_url=billing_base_url, billing_mode=billing_mode, input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens, reasoning_tokens=reasoning_tokens, estimated_cost_usd=estimated_cost_usd, actual_cost_usd=actual_cost_usd, cost_status=cost_status, cost_source=cost_source, api_call_count=api_call_count)

    def _record_model_usage(self, cursor: Any, session_id: str, *, model: str | None = None, billing_provider: str | None = None, billing_base_url: str | None = None, billing_mode: str | None = None, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, estimated_cost_usd: float | None = None, actual_cost_usd: float | None = None, cost_status: str | None = None, cost_source: str | None = None, api_call_count: int = 0, task: str = "") -> None:
        session = {}
        if not task:
            cursor.execute(f"SELECT model, billing_provider, billing_base_url, billing_mode FROM {self._schema}.sessions WHERE id=%s", (session_id,))
            session = cursor.fetchone() or {}
        now = time.time()
        cursor.execute(f"""INSERT INTO {self._schema}.session_model_usage (session_id,model,billing_provider,billing_base_url,billing_mode,task,api_call_count,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,reasoning_tokens,estimated_cost_usd,actual_cost_usd,cost_status,cost_source,first_seen,last_seen) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id,model,billing_provider,billing_base_url,billing_mode,task) DO UPDATE SET api_call_count=session_model_usage.api_call_count+EXCLUDED.api_call_count,input_tokens=session_model_usage.input_tokens+EXCLUDED.input_tokens,output_tokens=session_model_usage.output_tokens+EXCLUDED.output_tokens,cache_read_tokens=session_model_usage.cache_read_tokens+EXCLUDED.cache_read_tokens,cache_write_tokens=session_model_usage.cache_write_tokens+EXCLUDED.cache_write_tokens,reasoning_tokens=session_model_usage.reasoning_tokens+EXCLUDED.reasoning_tokens,estimated_cost_usd=session_model_usage.estimated_cost_usd+EXCLUDED.estimated_cost_usd,actual_cost_usd=session_model_usage.actual_cost_usd+EXCLUDED.actual_cost_usd,cost_status=COALESCE(EXCLUDED.cost_status,session_model_usage.cost_status),cost_source=COALESCE(EXCLUDED.cost_source,session_model_usage.cost_source),last_seen=EXCLUDED.last_seen""", (session_id,model or session.get("model") or "unknown",billing_provider or session.get("billing_provider") or "",billing_base_url or session.get("billing_base_url") or "",billing_mode or session.get("billing_mode") or "",task or "",api_call_count or 0,input_tokens or 0,output_tokens or 0,cache_read_tokens or 0,cache_write_tokens or 0,reasoning_tokens or 0,float(estimated_cost_usd or 0),float(actual_cost_usd or 0),cost_status,cost_source,now,now))

    def record_auxiliary_usage(self, session_id: str, task: str, **kwargs: Any) -> None:
        if not session_id or not task:
            return
        self.ensure_session(session_id)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            self._record_model_usage(cursor, session_id, task=task, api_call_count=int(kwargs.pop("api_call_count", 1) if kwargs.get("api_call_count") is not None else 1), **kwargs)

    def end_session(self, session_id: str, end_reason: str) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET ended_at = %s, end_reason = %s "
                "WHERE id = %s AND ended_at IS NULL RETURNING source, session_key",
                (time.time(), end_reason, session_id),
            )
            self._bump_conversation_generation(cursor, cursor.fetchone(), end_reason)

    def _bump_conversation_generation(self, cursor: Any, row: Any, end_reason: str) -> None:
        """Advance only for the row this transaction newly marked as a reset.

        The row-returning update makes first-end-reason and generation increment
        one commit. The durable table has no session FK and is never pruned.
        """
        if row is None or end_reason not in _RESET_END_REASONS:
            return
        source, session_key = (str(value or "").strip() for value in row)
        if source and session_key:
            cursor.execute(
                f"INSERT INTO {self._schema}.conversation_generations (source, session_key, generation) VALUES (%s, %s, 1) "
                "ON CONFLICT (source, session_key) DO UPDATE "
                "SET generation = conversation_generations.generation + 1",
                (source, session_key),
            )

    def promote_to_session_reset(self, session_id: str, reason: str = "session_reset") -> bool:
        """Promote a live/recoverably closed row atomically, matching SQLite.

        Explicitly ended rows are immutable; only a successful promotion may
        advance the peer generation.
        """
        if not session_id:
            return False
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._schema}.sessions SET ended_at = %s, end_reason = %s "
                    "WHERE id = %s AND (ended_at IS NULL OR end_reason = ANY(%s)) "
                    "RETURNING source, session_key",
                    (time.time(), reason, session_id, list(_RECOVERABLE_END_REASONS)),
                )
                row = cursor.fetchone()
                self._bump_conversation_generation(cursor, row, reason)
                return row is not None
        except Exception:
            logger.debug("Failed to promote PostgreSQL session reset", exc_info=True)
            return False

    def latest_conversation_boundary(self, session_key: str, source: str) -> int | None:
        """Return the durable source-qualified generation, never an aggregate."""
        if not session_key or not source:
            return None
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT generation FROM {self._schema}.conversation_generations WHERE source = %s AND session_key = %s",
                (source, session_key),
            )
            row = cursor.fetchone()
        generation = int(row[0]) if row is not None and row[0] is not None else 0
        return generation if generation > 0 else None

    @staticmethod
    def _model_config_object(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return {}
        return dict(value) if isinstance(value, Mapping) else {}

    def update_session_meta(self, session_id: str, model_config_json: str, model: str | None = None) -> None:
        """Replace model config and optionally fill a missing model after queued usage is durable."""
        self.flush_token_counts()
        config = self._model_config_object(model_config_json)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET model_config = %s, model = COALESCE(%s, model) WHERE id = %s",
                (self._psycopg.types.json.Jsonb(config) if config else None, model, session_id),
            )

    def patch_session_model_config(self, session_id: str, patch: Mapping[str, Any]) -> None:
        """Atomically shallow-merge config; a ``None`` value removes its key."""
        if not session_id or not patch:
            return
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT model_config FROM {self._schema}.sessions WHERE id = %s FOR UPDATE", (session_id,))
            row = cursor.fetchone()
            if row is None:
                return
            config = self._model_config_object(row.get("model_config"))
            for key, value in patch.items():
                if value is None:
                    config.pop(key, None)
                else:
                    config[key] = value
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET model_config = %s WHERE id = %s",
                (self._psycopg.types.json.Jsonb(config) if config else None, session_id),
            )

    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any:
        session = self.get_session(session_id) or {}
        return self._model_config_object(session.get("model_config")).get(key, default)

    def update_session_model(
        self, session_id: str, model: str, provider: str | None = None, *,
        base_url: str | None = None, api_mode: str | None = None,
    ) -> None:
        """Switch the persisted route after queued pre-switch usage has drained."""
        self.flush_token_counts()
        patch: dict[str, Any] = {"browser_model_lock": None}
        if model:
            patch["model"] = model
        if provider:
            route = {"provider": provider, "base_url": base_url or None, "api_mode": api_mode or None}
            patch.update(route, gateway_runtime=route)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT model_config FROM {self._schema}.sessions WHERE id = %s FOR UPDATE", (session_id,))
            row = cursor.fetchone()
            if row is None:
                return
            config = self._model_config_object(row.get("model_config"))
            for key, value in patch.items():
                if value is None:
                    config.pop(key, None)
                else:
                    config[key] = value
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET model = %s, model_config = %s WHERE id = %s",
                (model, self._psycopg.types.json.Jsonb(config) if config else None, session_id),
            )

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: str | None = None,
    ) -> None:
        """Persist the latest billable route after queued pre-switch usage has drained."""
        self.flush_token_counts()
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET billing_provider = %s, billing_base_url = %s, "
                "billing_mode = COALESCE(%s, billing_mode) WHERE id = %s",
                (provider, base_url, billing_mode, session_id),
            )

    def get_recent_session_model_route(self, session_id: str) -> dict[str, Any] | None:
        """Most recent main-loop billing route, with a portable deterministic tie-break."""
        if not self.flush_token_counts():
            raise RuntimeError("token accounting flush did not complete")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT model, billing_provider, billing_base_url, billing_mode, api_call_count "
                f"FROM {self._schema}.session_model_usage WHERE session_id=%s AND task='' "
                "AND model <> 'unknown' AND billing_provider <> '' "
                "ORDER BY last_seen DESC, first_seen DESC, model DESC, billing_provider DESC, billing_base_url DESC, billing_mode DESC LIMIT 1",
                (session_id,),
            )
            return cursor.fetchone()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self.flush_token_counts()
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT s.id, s.source, s.started_at, s.ended_at, s.end_reason, s.title, s.title_source, s.hidden, s.archived, s.pinned, "
                f"s.system_prompt_hash, s.git_branch, s.git_metadata_generation, s.last_activity_at, s.last_activity_description, s.last_activity_provenance, "
                f"s.compression_failure_cooldown_until, s.compression_failure_error, s.compression_fallback_streak, s.compression_ineffective_count, s.compression_recovery_deadline, p.prompt AS system_prompt, {', '.join('s.' + column for column in _SESSION_METADATA_COLUMNS)} "
                f"FROM {self._schema}.sessions s LEFT JOIN {self._schema}.system_prompts p ON p.hash = s.system_prompt_hash WHERE s.id = %s", (session_id,),
            )
            return cursor.fetchone()

    def get_message_storage_state(self, message_id: int) -> dict[str, Any] | None:
        """Return only the tenant-local visibility state required by contextual recall."""
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT session_id, active, compacted FROM {self._schema}.messages WHERE id = %s",
                (message_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {"session_id": row["session_id"], "active": int(row["active"]), "compacted": int(row["compacted"])}

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: str | None = None,
        git_repo_root: str | None = None, replace_git_meta: bool = False,
    ) -> int | None:
        """Claim Git enrichment authority atomically, matching SessionDB's A→B→A fence."""
        if not session_id or not cwd:
            return None
        branch, repo_root = (git_branch or "").strip(), (git_repo_root or "").strip()
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET cwd = %s, "
                "git_metadata_generation = git_metadata_generation + 1, "
                "git_branch = CASE WHEN cwd IS DISTINCT FROM %s OR %s THEN %s "
                "WHEN %s <> '' THEN %s ELSE git_branch END, "
                "git_repo_root = CASE WHEN cwd IS DISTINCT FROM %s OR %s THEN %s "
                "WHEN %s <> '' THEN %s ELSE git_repo_root END "
                "WHERE id = %s RETURNING git_metadata_generation",
                (cwd, cwd, replace_git_meta, branch or None, branch, branch or None,
                 cwd, replace_git_meta, repo_root or None, repo_root, repo_root or None, session_id),
            )
            row = cursor.fetchone()
            return None if row is None else int(row[0])

    def publish_session_git_metadata(
        self, session_id: str, cwd: str, generation: int, git_branch: str | None = None,
        git_repo_root: str | None = None,
    ) -> bool:
        """Publish an async Git probe only if its claim has not been superseded."""
        if not session_id or not cwd or not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            return False
        fields = [("git_branch", (git_branch or "").strip()), ("git_repo_root", (git_repo_root or "").strip())]
        fields = [(column, value) for column, value in fields if value]
        if not fields:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET {', '.join(f'{column} = %s' for column, _ in fields)} "
                "WHERE id = %s AND cwd = %s AND git_metadata_generation = %s RETURNING id",
                [*(value for _, value in fields), session_id, cwd, generation],
            )
            return cursor.fetchone() is not None

    def set_system_prompt(self, session_id: str, system_prompt: str | None) -> None:
        prompt_hash = None if system_prompt is None else hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        with self._connection() as connection, connection.cursor() as cursor:
            if prompt_hash is not None:
                cursor.execute(
                    f"INSERT INTO {self._schema}.system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
                    (prompt_hash, system_prompt),
                )
            cursor.execute(f"UPDATE {self._schema}.sessions SET system_prompt_hash = %s WHERE id = %s", (prompt_hash, session_id))
            cursor.execute(
                f"DELETE FROM {self._schema}.system_prompts p WHERE NOT EXISTS "
                f"(SELECT 1 FROM {self._schema}.sessions s WHERE s.system_prompt_hash = p.hash)"
            )

    def get_system_prompt(self, session_id: str) -> str | None:
        row = self.get_session(session_id)
        return None if row is None else row.get("system_prompt")

    def clear_stored_system_prompts(self) -> dict[str, Any]:
        """Invalidate every stored system-prompt snapshot (StateStoreInterface).

        The PG schema is always out-of-line (``sessions.system_prompt_hash`` FK →
        ``system_prompts.hash``), so in one transaction the references are NULLed
        first — the same never-dangle-FK ordering as the SQLite store — and the
        now-unreferenced snapshot rows are deleted. ``system_prompts`` is
        referenced only by that FK (creation is the sole REFERENCES in the DDL),
        so truncating it after the UPDATE cannot orphan anything. Session rows
        are never removed. Idempotent: a schema with nothing stored (or already
        cleared) reports ``cleared == 0``. ``cleared`` counts affected sessions,
        matching the SQLite implementation's UPDATE rowcount semantics.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET system_prompt_hash = NULL "
                "WHERE system_prompt_hash IS NOT NULL"
            )
            cleared = cursor.rowcount
            cursor.execute(f"DELETE FROM {self._schema}.system_prompts")
        return {"cleared": cleared, "storage_mode": "out-of-line"}

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        return self._set_lineage_column("hidden", session_id, hidden)

    def _set_lineage_column(self, column: str, session_id: str, value: bool) -> bool:
        """Apply a visibility flag to the whole compression lineage in one transaction."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""WITH RECURSIVE
                    ancestors(id) AS (
                        SELECT %s
                        UNION
                        SELECT parent.id FROM ancestors a
                        JOIN {self._schema}.sessions child ON child.id = a.id
                        JOIN {self._schema}.sessions parent ON parent.id = child.parent_session_id
                        WHERE parent.end_reason = 'compression'
                    ),
                    descendants(id) AS (
                        SELECT %s
                        UNION
                        SELECT child.id FROM descendants d
                        JOIN {self._schema}.sessions parent ON parent.id = d.id
                        JOIN {self._schema}.sessions child ON child.parent_session_id = parent.id
                        WHERE parent.end_reason = 'compression'
                    ), lineage(id) AS (
                        SELECT id FROM ancestors UNION SELECT id FROM descendants
                    )
                    UPDATE {self._schema}.sessions SET {column} = %s WHERE id IN (SELECT id FROM lineage)""",
                (session_id, session_id, value),
            )
            return cursor.rowcount > 0

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        return self._set_lineage_column("archived", session_id, archived)

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT title, hidden FROM {self._schema}.sessions WHERE id = %s FOR UPDATE", (session_id,))
            row = cursor.fetchone()
            if row is None:
                return False
            cursor.execute(
                f"""WITH RECURSIVE
                    ancestors(id) AS (SELECT %s UNION SELECT parent.id FROM ancestors a JOIN {self._schema}.sessions child ON child.id = a.id JOIN {self._schema}.sessions parent ON parent.id = child.parent_session_id WHERE parent.end_reason = 'compression'),
                    descendants(id) AS (SELECT %s UNION SELECT child.id FROM descendants d JOIN {self._schema}.sessions parent ON parent.id = d.id JOIN {self._schema}.sessions child ON child.parent_session_id = parent.id WHERE parent.end_reason = 'compression'),
                    lineage(id) AS (SELECT id FROM ancestors UNION SELECT id FROM descendants)
                    UPDATE {self._schema}.sessions SET pinned = %s WHERE id IN (SELECT id FROM lineage)""",
                (session_id, session_id, pinned),
            )
            changed = cursor.rowcount > 0
            if pinned and not (row["hidden"] and row["title"] == _CANONICAL_BOT_CHAT_TITLE):
                cursor.execute(
                    f"""WITH RECURSIVE
                        ancestors(id) AS (SELECT %s UNION SELECT parent.id FROM ancestors a JOIN {self._schema}.sessions child ON child.id = a.id JOIN {self._schema}.sessions parent ON parent.id = child.parent_session_id WHERE parent.end_reason = 'compression'),
                        descendants(id) AS (SELECT %s UNION SELECT child.id FROM descendants d JOIN {self._schema}.sessions parent ON parent.id = d.id JOIN {self._schema}.sessions child ON child.parent_session_id = parent.id WHERE parent.end_reason = 'compression'),
                        lineage(id) AS (SELECT id FROM ancestors UNION SELECT id FROM descendants)
                        UPDATE {self._schema}.sessions SET hidden = false WHERE id IN (SELECT id FROM lineage)""",
                    (session_id, session_id),
                )
            return changed

    def list_recent_sessions_bounded(
        self, *, limit: int = 20, exclude_sources: list[str] | None = None,
        timeout_seconds: float = 3.0, candidate_limit: int | None = None,
        lineage_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the bounded, compression-aware contextual browse projection.

        This is intentionally a direct StateStore operation, not an opt-in to
        ``ContextualSessionSearchStore``: PostgreSQL still lacks the other
        contextual shapes.  The CTE follows only compression-continuation
        edges, rejects incomplete/cyclic/cap-exhausted lineages, and projects a
        visible root to its freshest terminal tip in one tenant-scoped snapshot.
        """
        limit = max(1, int(limit))
        timeout_seconds = max(0.0, float(timeout_seconds))
        if candidate_limit is None:
            candidate_limit = max(128, limit * 8)
        candidate_limit = max(limit, min(int(candidate_limit), 2048))
        if lineage_limit is None:
            lineage_limit = min(8192, candidate_limit * 8)
        lineage_limit = max(candidate_limit, min(int(lineage_limit), 8192))
        excluded = list(exclude_sources or [])
        candidate_filter = "NOT s.archived AND NOT s.hidden AND NOT (COALESCE(s.model_config, '{}'::jsonb) ? '_delegate_from')"
        params: list[Any] = []
        if excluded:
            candidate_filter += " AND NOT (s.source = ANY(%s))"
            params.append(excluded)
        edge = (
            "parent.end_reason = 'compression' AND child.parent_session_id = parent.id "
            "AND NOT (COALESCE(child.model_config, '{}'::jsonb) ? '_branched_from') "
            "AND NOT (COALESCE(child.model_config, '{}'::jsonb) ? '_delegate_from') "
            "AND COALESCE(child.source, '') <> 'tool'"
        )
        query = f"""
            WITH RECURSIVE
            recent_candidates(id) AS (
                SELECT s.id FROM {self._schema}.sessions AS s
                WHERE {candidate_filter}
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC, s.started_at DESC, s.id DESC
                LIMIT %s
            ),
            ancestors(candidate_id, cur_id, depth, path) AS (
                SELECT id, id, 1, ARRAY[id]::text[] FROM recent_candidates
                UNION ALL
                SELECT a.candidate_id, parent.id, a.depth + 1, a.path || parent.id
                FROM ancestors AS a
                JOIN {self._schema}.sessions AS child ON child.id = a.cur_id
                JOIN {self._schema}.sessions AS parent ON {edge}
                WHERE a.depth < %s AND NOT parent.id = ANY(a.path)
            ),
            candidate_roots(root_id) AS (
                SELECT DISTINCT a.cur_id
                FROM ancestors AS a
                JOIN {self._schema}.sessions AS child ON child.id = a.cur_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self._schema}.sessions AS parent WHERE {edge}
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM ancestors AS clipped
                    WHERE clipped.candidate_id = a.candidate_id AND clipped.depth >= %s
                )
            ),
            chain(root_id, cur_id, depth, path) AS (
                SELECT root_id, root_id, 1, ARRAY[root_id]::text[] FROM candidate_roots
                UNION ALL
                SELECT c.root_id, child.id, c.depth + 1, c.path || child.id
                FROM chain AS c
                JOIN {self._schema}.sessions AS parent ON parent.id = c.cur_id
                JOIN {self._schema}.sessions AS child ON {edge}
                WHERE c.depth < %s AND NOT child.id = ANY(c.path)
            ),
            valid_roots(root_id) AS (
                SELECT root_id FROM chain GROUP BY root_id
                HAVING MAX(depth) < %s AND COUNT(DISTINCT cur_id) < %s
            ),
            ranked_tips AS (
                SELECT c.root_id, c.cur_id,
                       COALESCE((SELECT MAX(m.created_at) FROM {self._schema}.messages AS m WHERE m.session_id = tip.id), tip.last_activity_at, tip.started_at) AS activity,
                       ROW_NUMBER() OVER (PARTITION BY c.root_id ORDER BY COALESCE((SELECT MAX(m.created_at) FROM {self._schema}.messages AS m WHERE m.session_id = tip.id), tip.last_activity_at, tip.started_at) DESC, tip.id DESC) AS rank_in_root
                FROM chain AS c
                JOIN valid_roots AS valid ON valid.root_id = c.root_id
                JOIN {self._schema}.sessions AS tip ON tip.id = c.cur_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self._schema}.sessions AS parent
                    JOIN {self._schema}.sessions AS child ON {edge}
                    WHERE parent.id = c.cur_id
                )
            )
            SELECT tip.id, tip.source, tip.model, COALESCE(tip.title, root.title) AS title,
                   root.started_at, tip.ended_at, tip.end_reason, rt.activity AS last_active,
                   COALESCE((
                       SELECT m.content FROM {self._schema}.messages AS m
                       WHERE m.session_id = tip.id AND m.role = 'user' AND m.content IS NOT NULL
                         AND (m.active OR m.compacted) AND COALESCE(m.display_kind, '') <> 'hidden'
                       ORDER BY m.created_at, m.id LIMIT 1
                   ), '') AS preview,
                   CASE WHEN root.id <> tip.id THEN root.id ELSE NULL END AS _lineage_root_id
            FROM ranked_tips AS rt
            JOIN {self._schema}.sessions AS root ON root.id = rt.root_id
            JOIN {self._schema}.sessions AS tip ON tip.id = rt.cur_id
            WHERE rt.rank_in_root = 1 AND NOT root.archived AND NOT root.hidden
              AND NOT (COALESCE(root.model_config, '{{}}'::jsonb) ? '_delegate_from')
            ORDER BY rt.activity DESC, root.started_at DESC, tip.id DESC
            LIMIT %s
        """
        params.extend([candidate_limit, lineage_limit, lineage_limit, lineage_limit, lineage_limit, lineage_limit, limit])
        try:
            with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
                cursor.execute("SELECT set_config('statement_timeout', %s, true)", (f"{max(1, int(timeout_seconds * 1000))}ms",))
                cursor.execute(query, params)
                rows = list(cursor.fetchall())
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "57014":
                raise TimeoutError(f"recent-session browse exceeded {timeout_seconds:g}s deadline") from exc
            raise
        for row in rows:
            row["preview"] = self._decode_content(row["preview"])
        return rows

    def list_session_summaries(
        self, *, source: str | None = None, exclude_sources: tuple[str, ...] = (),
        limit: int = 20, offset: int = 0, include_archived: bool = False,
        archived_only: bool = False, include_hidden: bool = False,
        include_pinned: bool = False,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if source is not None:
            clauses.append("s.source = %s"); params.append(source)
        if exclude_sources:
            clauses.append("NOT (s.source = ANY(%s))"); params.append(list(exclude_sources))
        if archived_only:
            clauses.append("s.archived")
        elif not include_archived:
            clauses.append("NOT s.archived")
        if not include_hidden:
            clauses.append("NOT s.hidden")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        projection = (
            "s.id, s.source, s.started_at, s.ended_at, s.end_reason, s.parent_session_id, s.title, "
            "s.title_source, s.hidden, s.archived, s.pinned, "
            "COALESCE(MAX(m.created_at), s.started_at) AS last_active, COUNT(m.id)::integer AS message_count"
        )
        grouped = f" FROM {self._schema}.sessions s LEFT JOIN {self._schema}.messages m ON m.session_id = s.id{where} GROUP BY s.id"
        order = " ORDER BY last_active DESC, s.started_at DESC, s.id DESC"
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT {projection}{grouped}{order} LIMIT %s OFFSET %s", [*params, limit, offset])
            rows = list(cursor.fetchall())
            if include_pinned:
                seen = {row["id"] for row in rows}
                pinned_where = where + (" AND s.pinned" if where else " WHERE s.pinned")
                cursor.execute(f"SELECT {projection} FROM {self._schema}.sessions s LEFT JOIN {self._schema}.messages m ON m.session_id = s.id{pinned_where} GROUP BY s.id{order}", params)
                rows.extend(row for row in cursor.fetchall() if row["id"] not in seen)
            return rows

    def session_lifecycle_statuses(self, session_ids: list[str]) -> dict[str, str]:
        """Classify every requested session from its final message, like SessionDB.

        The picker contract is deliberately a single indexed batched lookup: an
        empty requested session is ``empty`` and an unknown ID stays ``empty``.
        """
        from hermes_state_sessions import classify_session_status

        ids = [session_id for session_id in (session_ids or []) if session_id]
        if not ids:
            return {}
        statuses = {session_id: "empty" for session_id in ids}
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"""SELECT message.session_id, message.role,
                           message.tool_calls IS NOT NULL AS has_tool_calls,
                           message.finish_reason
                    FROM {self._schema}.messages AS message
                    JOIN (
                        SELECT session_id, MAX(id) AS max_id
                        FROM {self._schema}.messages
                        WHERE session_id = ANY(%s)
                        GROUP BY session_id
                    ) AS latest ON message.id = latest.max_id""",
                (ids,),
            )
            for row in cursor.fetchall():
                statuses[str(row["session_id"])] = classify_session_status(
                    role=row["role"], has_tool_calls=bool(row["has_tool_calls"]),
                    finish_reason=row["finish_reason"],
                )
        return statuses

    def list_skill_scaffolded_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the same first-user-turn /skill candidates as SQLite's repair command."""
        from agent.skill_commands import SKILL_SCAFFOLD_SQL_LIKE

        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"""SELECT session.id, session.title, message.content
                    FROM {self._schema}.sessions AS session
                    JOIN {self._schema}.messages AS message ON message.id = (
                        SELECT first_message.id FROM {self._schema}.messages AS first_message
                        WHERE first_message.session_id = session.id
                          AND first_message.role = 'user'
                          AND first_message.content IS NOT NULL
                        ORDER BY first_message.created_at, first_message.id LIMIT 1
                    )
                    WHERE session.title IS NOT NULL AND message.content LIKE %s
                    ORDER BY session.started_at DESC LIMIT %s""",
                (SKILL_SCAFFOLD_SQL_LIKE, int(limit)),
            )
            return list(cursor.fetchall())

    def _set_session_title(self, session_id: str, title: str, *, source: str) -> bool:
        cleaned_title = _sanitize_title(title)
        is_user = source == "user"
        if not is_user and source not in {"derived", "llm"}:
            raise ValueError(f"invalid automatic title source: {source!r}")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT title, title_source, hidden FROM {self._schema}.sessions WHERE id = %s FOR UPDATE", (session_id,))
            current = cursor.fetchone()
            if current is None:
                return False
            if current["title"] == _CANONICAL_BOT_CHAT_TITLE and current["hidden"] and cleaned_title != _CANONICAL_BOT_CHAT_TITLE:
                if is_user:
                    raise ValueError("This is the bot's canonical Bot Chat — its name is its identity, and renaming it would orphan the conversation. To start fresh, create a new bot instead.")
                return False
            rank = _TITLE_SOURCE_RANK.get(current["title_source"], 2 if current["title_source"] is None else 0)
            if not is_user and current["title"] is not None and rank >= _TITLE_SOURCE_RANK[source]:
                return False
            if cleaned_title:
                cursor.execute(f"SELECT id FROM {self._schema}.sessions WHERE title = %s AND id != %s FOR UPDATE", (cleaned_title, session_id))
                conflict = cursor.fetchone()
                if conflict:
                    conflict_id = conflict["id"]
                    cursor.execute(
                        f"WITH RECURSIVE ancestors(id) AS (SELECT %s UNION SELECT parent.id FROM ancestors a JOIN {self._schema}.sessions child ON child.id = a.id JOIN {self._schema}.sessions parent ON parent.id = child.parent_session_id WHERE parent.end_reason = 'compression') SELECT 1 FROM ancestors WHERE id = %s AND id != %s LIMIT 1",
                        (session_id, conflict_id, session_id),
                    )
                    if cursor.fetchone() is None:
                        raise ValueError(f"Title '{cleaned_title}' is already in use by session {conflict_id}")
                    cursor.execute(f"UPDATE {self._schema}.sessions SET title = NULL WHERE id = %s", (conflict_id,))
            cursor.execute(f"UPDATE {self._schema}.sessions SET title = %s, title_source = %s WHERE id = %s", (cleaned_title, source if cleaned_title else None, session_id))
            return cursor.rowcount > 0

    def set_session_title(self, session_id: str, title: str) -> bool:
        return self._set_session_title(session_id, title, source="user")

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        return self._set_session_title(session_id, title, source=source)

    def get_session_title(self, session_id: str) -> str | None:
        row = self.get_session(session_id)
        return None if row is None else row["title"]

    def get_session_title_source(self, session_id: str) -> str | None:
        row = self.get_session(session_id)
        return None if row is None or row["title"] is None else row["title_source"]

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        if source not in _TITLE_SOURCE_RANK:
            raise ValueError(f"invalid title source: {source!r}")
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"UPDATE {self._schema}.sessions SET title_source = %s WHERE id = %s AND title IS NOT NULL", (source, session_id))
            return cursor.rowcount > 0

    def get_session_by_title(self, title: str) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT id, source, started_at, ended_at, end_reason, title, title_source, hidden, archived, pinned, git_branch, git_metadata_generation, {', '.join(_SESSION_METADATA_COLUMNS)} FROM {self._schema}.sessions WHERE title = %s", (title,))
            return cursor.fetchone()

    def resolve_session_by_title(self, title: str) -> str | None:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT id FROM {self._schema}.sessions WHERE title LIKE %s ESCAPE '\\' ORDER BY started_at DESC", (_escape_like(title) + " #%",))
            row = cursor.fetchone()
            if row:
                return str(row["id"])
            cursor.execute(f"SELECT id FROM {self._schema}.sessions WHERE title = %s", (title,))
            row = cursor.fetchone()
            return None if row is None else str(row["id"])

    def get_next_title_in_lineage(self, base_title: str) -> str:
        match = _NUMBERED_TITLE_RE.match(base_title)
        base = match.group(1) if match else base_title
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT title FROM {self._schema}.sessions WHERE title = %s OR title LIKE %s ESCAPE '\\'", (base, _escape_like(base) + " #%"))
            rows = cursor.fetchall()
        if not rows:
            return base
        numbers = [int(match.group(2)) for row in rows if (match := _NUMBERED_TITLE_RE.match(row["title"]))]
        return f"{base} #{max([1, *numbers]) + 1}"
