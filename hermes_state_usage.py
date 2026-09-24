"""Token/usage accounting mixin for SessionDB: the coalescing background token writer,
per-model usage rows, and billing-route columns. Writer thread state lives on the instance."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from token_usage_transport import TokenUsageTransport

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

_TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _token_update_sql(delta: bool) -> str:
    """``UPDATE sessions`` for one usage report: *delta* adds to the stored counters (CLI
    per-call path), otherwise sets them (gateway cumulative path). Cost/route columns
    COALESCE-fill either way (statement text is pinned by the SQL trace harness)."""
    def add(col: str) -> str:  # "col + ?" / "COALESCE(col, 0) + ?" in delta mode, bare "?" otherwise
        return f"{col} + ?" if delta else "?"
    def add0(col: str) -> str:
        return f"COALESCE({col}, 0) + ?" if delta else "?"
    counters = "".join(f"                   {c} = {add(c)},\n" for c in _TOKEN_COUNTERS)
    estimated = "COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0)" if delta else "COALESCE(?, 0)"
    return (
        "UPDATE sessions SET\n" + counters
        + f"""                   estimated_cost_usd = {estimated},
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE {add0("actual_cost_usd")}
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = {add0("api_call_count")}
                   WHERE id = ?"""
    )


_TOKEN_UPDATE_ABSOLUTE_SQL = _token_update_sql(delta=False)
_TOKEN_UPDATE_DELTA_SQL = _token_update_sql(delta=True)

_MODEL_USAGE_UPSERT_SQL = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = api_call_count + excluded.api_call_count,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens,
                   cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, cost_status),
                   cost_source = COALESCE(excluded.cost_source, cost_source),
                   last_seen = excluded.last_seen"""


# Kwargs forwarded verbatim from update_token_counts / record_auxiliary_usage into
# _record_model_usage (the per-route attribution row).
_MODEL_USAGE_FIELDS = frozenset((
    "model", "billing_provider", "billing_base_url", "billing_mode", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count"))


class SessionUsageMixin:
    """Coalesced token writer, per-model usage rows, billing route."""

    # Transitional observability aliases: callers remain on SessionDB's established
    # surface while the mutable state is owned by TokenUsageTransport.
    @property
    def _token_queue(self):
        return self._token_usage_transport.queue

    @property
    def _token_queue_cond(self):
        return self._token_usage_transport.condition

    @property
    def _token_writer_thread(self):
        return self._token_usage_transport.writer_thread

    @_token_writer_thread.setter
    def _token_writer_thread(self, value):
        self._token_usage_transport.writer_thread = value

    @property
    def _token_writer_stop(self):
        return self._token_usage_transport.stop_requested

    @_token_writer_stop.setter
    def _token_writer_stop(self, value):
        self._token_usage_transport.stop_requested = value

    @property
    def _token_writer_busy(self):
        return self._token_usage_transport.busy

    @_token_writer_busy.setter
    def _token_writer_busy(self, value):
        self._token_usage_transport.busy = value

    @property
    def _token_atexit_hook(self):
        return self._token_usage_transport._atexit_hook

    @_token_atexit_hook.setter
    def _token_atexit_hook(self, value):
        self._token_usage_transport._atexit_hook = value

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str] = None,
    ) -> None:
        """Unconditionally set the billing route (``update_token_counts`` only COALESCE-fills
        NULLs) so the dashboard reflects the latest /model switch; also nulls
        ``system_prompt`` so the cached snapshot header is rebuilt.

        See #48173, #48248.
        """
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        def _do(conn):
            conn.execute("""UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""", (provider, base_url, billing_mode, session_id))
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    # SQLite's SessionDB is the first concrete consumer.  These forwarding methods
    # preserve its established caller/test surface while the transport itself remains
    # persistence-neutral and directly contract-testable.
    def _token_transport(self) -> TokenUsageTransport:
        return self._token_usage_transport

    def queue_token_counts(self, session_id: str, **kwargs) -> None:
        self._token_transport().queue_delta(session_id, kwargs)

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        return self._token_transport().flush(timeout)

    def _apply_token_batch(self, batch) -> None:
        self._token_transport().apply_batch(batch)

    def _coalesce_token_deltas(self, batch):
        return self._token_transport().coalesce(batch)

    def _stop_token_writer(self, join_timeout: float = 10.0) -> None:
        self._token_transport().stop(join_timeout)

    def update_token_counts(
        self, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str=None, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None,
        pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None,
        billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False,
        source: Optional[str]=None,
    ) -> None:
        """Update token counters and backfill model if unset. *absolute*=False increments
        (per-API-call deltas, CLI path); *absolute*=True sets directly (gateway path,
        where the cached agent holds cumulative totals). ``source`` is the session's real surface
        for the row-existence guard; callers that don't know it leave the placeholder."""
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        # Ensure the row exists: under concurrent load create_session() may have failed on
        # locking, and the UPDATE would silently affect 0 rows. When this guard is the first
        # writer it must carry the agent's real source: the turn lease treats an existing row as
        # proof the create already happened, so the creator never returns to repair an anonymous
        # ``unknown`` placeholder and the session stays a phantom for life (#111999).
        self._insert_session_row(session_id, source or "unknown", model=model)
        sql = _TOKEN_UPDATE_ABSOLUTE_SQL if absolute else _TOKEN_UPDATE_DELTA_SQL
        has_usage = bool(input_tokens or output_tokens or cache_read_tokens or cache_write_tokens or reasoning_tokens
                         or api_call_count or estimated_cost_usd)
        has_accounted_usage = bool(has_usage or actual_cost_usd)
        params = (
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
            estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None, model if has_accounted_usage else None,
            api_call_count, session_id)
        # Per-model attribution: the sessions row keeps one (model, provider) pair, so a
        # mid-session /model switch would attribute every token to the initial model. Only
        # the incremental path records here — absolute cumulative updates cannot be split
        # back into routes; Insights reconciles the residual instead.
        # ``update_token_counts`` is the single chokepoint every per-API-call delta flows through (CLI,
        # gateway, cron, delegated runs — see conversation_loop / codex_runtime), and each call carries the
        # model/provider *active at the time of that call*. Recording the per-call delta into
        # session_model_usage keyed by the live model preserves an accurate per-model breakdown regardless
        # of how many times the user switches. See #51607.
        record_model_usage = (not absolute) and has_usage

        def _do(conn):
            row = conn.execute(
                "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = ?", (session_id,),
            ).fetchone()
            existing = dict(row) if row is not None else {}
            # create_session records the requested route before any API call. If that fails
            # and fallback succeeds, the first accounted usage is the authoritative route;
            # after that keep the row as is (one row cannot represent mixed usage).
            first_accounted_route = (
                int(existing.get("api_call_count") or 0) == 0 and has_accounted_usage and bool(model)
                and bool(billing_provider)
                and (existing.get("model") != model or existing.get("billing_provider") != billing_provider)
            )
            if first_accounted_route:
                conn.execute("""UPDATE sessions
                       SET model = ?, billing_provider = ?,
                       billing_base_url = ?, billing_mode = ?
                       WHERE id = ?""", (model, billing_provider, billing_base_url, billing_mode, session_id))
            conn.execute(sql, params)
            if record_model_usage:
                self._record_model_usage(conn, session_id, **usage)
        self._execute_write(_do)

    def _record_model_usage(
        self, conn, session_id: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, billing_mode: Optional[str]=None, input_tokens: int=0,
        output_tokens: int=0, cache_read_tokens: int=0, cache_write_tokens: int=0, reasoning_tokens: int=0,
        estimated_cost_usd: Optional[float]=None, actual_cost_usd: Optional[float]=None,
        cost_status: Optional[str]=None, cost_source: Optional[str]=None, api_call_count: int=0, task: str="",
    ) -> None:
        """Accumulate a per-API-call usage delta into session_model_usage, inside the caller's
        write txn after the ``sessions`` UPDATE. A missing model/provider falls back to
        the session row — except for aux rows (``task`` set), which must NOT inherit the
        main-loop route (vision on gemini while the main loop runs anthropic): missing
        info stays 'unknown'/empty.

        ``task`` distinguishes what kind of work consumed the tokens: ``''`` (empty) is the main agent loop;
        auxiliary calls record their task name (``vision``, ``compression``, ``title_generation``, ...) via
        :meth:`record_auxiliary_usage` (issue #23270).
        """
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        sess = dict(row) if (row is not None and not task) else {}
        counts = [v or 0 for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)]
        now = time.time()
        conn.execute(_MODEL_USAGE_UPSERT_SQL, (
            session_id, model or sess.get("model") or "unknown",
            billing_provider or sess.get("billing_provider") or "",
            billing_base_url or sess.get("billing_base_url") or "",
            billing_mode or sess.get("billing_mode") or "", task or "", api_call_count or 0, *counts,
            float(estimated_cost_usd or 0.0), float(actual_cost_usd or 0.0), cost_status, cost_source, now, now))

    def record_auxiliary_usage(
        self, session_id: str, task: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        api_call_count: int=1,
    ) -> None:
        """Record an auxiliary LLM call's usage (vision, compression, title generation, ...)
        as a per-(model, provider, task) delta in ``session_model_usage`` WITHOUT touching
        the ``sessions`` summary row (the gateway overwrites those counters with absolute
        main-loop totals). ``api_call_count`` may aggregate N calls. Best-effort.

        See #23270.
        Background-review forks record an aggregate of N fork API calls in one write with
        ``task='background_review'`` (issue #87250).
        """
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        if not session_id or not task:
            return
        usage["api_call_count"] = 1 if api_call_count is None else int(api_call_count)
        # FK to sessions.id: same guard as update_token_counts; the aux path carries no surface, so
        # the placeholder stays repairable by the creator's upsert (_insert_session_row).
        self._insert_session_row(session_id, "unknown")
        self._execute_write(lambda conn: self._record_model_usage(conn, session_id, task=task, **usage))

    def auxiliary_usage_by_task(self, session_id: str) -> Dict[str, Dict[str, float]]:
        """Per-task auxiliary usage (``task != ''``: vision, compression, title_generation, ...) summed
        over the session's compression lineage. Aux calls bill to the id the turn STARTED with while
        compression mints child ids mid-turn, so a single-id read misses rows (#112848)."""
        if not session_id:
            return {}
        chain = self._session_lineage_root_to_tip(session_id)
        # An explicit ``/branch`` copy owns its spend: cut the walk at the nearest branch node so a
        # resumed branch never absorbs aux rows billed to its source (hermes_state_messages does the same).
        for i in range(len(chain) - 1, -1, -1):
            if self._is_explicit_branch_session(chain[i]):
                chain = chain[i:]
                break
        rows = self._read_all(
            f"""SELECT task,
                       COALESCE(SUM(api_call_count), 0) AS api_calls,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                       COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                       COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                  FROM session_model_usage
                 WHERE session_id IN ({','.join('?' * len(chain))}) AND task != ''
                 GROUP BY task""",
            chain,
        )
        return {row["task"]: {k: row[k] for k in row.keys() if k != "task"} for row in rows}

    def usage_totals(self, *, min_message_count: int = 1, include_archived: bool = False) -> Dict[str, float]:
        """Tokens and spend across the whole store (one scan), so the sidebar total does not
        shrink with paging. Spend prefers the billed figure over the estimate."""
        where = ["parent_session_id IS NULL", "message_count >= ?"]
        params: List[Any] = [min_message_count]
        if not include_archived:
            where.append("COALESCE(archived, 0) = 0")
        row = self._read_one(f"""
            SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0),
                   COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0)
              FROM sessions
             WHERE {' AND '.join(where)}
            """, params)
        return {"tokens": int(row[0] or 0), "cost_usd": float(row[1] or 0.0)}
