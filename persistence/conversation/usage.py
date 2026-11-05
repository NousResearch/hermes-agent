"""Backend-neutral usage accounting contract.

``UsageRepository`` is the token and cost accounting of one profile's store: per-call deltas queued onto the session row and its per-route attribution rows, auxiliary usage kept off the session counters, the stored billing route, and the reads the status, dashboard and one-shot callers use. Every verb acts on the handle's bound profile.

Main-loop accounting: callers queue one :class:`UsageDelta` per API call (``queue_usage``; callers ``agent/turn_usage.py``, ``agent/codex_runtime.py``) and drain the queue with ``flush_usage`` (callers the CLI shutdown, ``agent/session_persistence.py`` at finalize, ``agent/managed_worker.py``, ``agent/insights.py``). The queue and its writer belong to the backend's shared store holder, not to the caller: deltas apply in queue order, a backend may merge adjacent deltas whose merge equals applying them in sequence (same session, route, cost fields, source and task), and an apply error is logged and the delta dropped, never raised into a turn. The exact readers (``SessionRepository.get``, ``get_model_config_value``, ``update_model``, ``replace_session_config``, ``set_billing_route``, ``recent_route``) drain the queue first, except that ``SessionRepository.get``, ``recent_route`` and the session listings, called on a thread running an asyncio event loop, read the committed totals without waiting for the writer; ``usage_totals`` and ``auxiliary_usage_by_task`` do not drain.

Applying one delta:

1. Row guard, its own write: a missing session row is inserted with the delta's ``source`` (``unknown`` without one) and the store owner's profile; an ``unknown`` placeholder takes a real source; a deleted session is not recreated (the apply fails and the delta is dropped).
2. One write: the counters and ``api_call_count`` add; ``estimated_cost_usd`` adds with None as 0; ``actual_cost_usd`` adds only when the delta carries one; ``cost_status``, ``cost_source`` and ``pricing_version`` take the delta's value when it carries one. When the delta accounts usage (a counter, a call or a truthy estimated or actual cost), the row's billing provider, base URL, mode and model fill only while unset; but the first accounted main-loop delta (stored call count 0, no ``task``, model and provider set, either differing from the row) replaces the row's model and route. When the delta has usage (a counter, a call or a truthy estimate; an actual cost alone is not usage), the attribution row keyed by (model, provider, base URL, mode, task) adds the same counters, calls and costs and stamps ``last_seen``; missing main-loop route fields come from the session row, ``model`` ``unknown`` and the others empty when the row has none.

``task`` on a delta names a per-turn route (``voice_chat``): its calls count in the session counters and land in their own attribution row, which never becomes the recent route and is summed by ``auxiliary_usage_by_task``.

``UsageAnalytics`` is the read-only analytics surface over the same store: the dashboard's usage and model reads and the insights reads, one verb per read. Each returns fresh mutable dicts keyed by the names its docstring lists. Aggregates follow SQL: a sum over no rows or only unset cells is ``None`` (a sum documented as 0 when unset is 0 then), a count is 0, an average over no rows ``None``, and a distinct count counts sessions once. The dashboard reads take sessions started strictly after ``started_after``; the insights reads take sessions started at or after ``started_since`` and, with a non-empty ``source``, only sessions of that source plus ``local`` sessions whose creating surface is that source. Per-call usage rows are the attribution rows, with an unset provider, base URL or mode as ``''``. No analytics read drains the usage queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, runtime_checkable

from ..core.identity import JsonValue, SessionId

__all__ = [
    "AuxiliaryTaskUsage",
    "BillingRoute",
    "RouteUsage",
    "TokenCounts",
    "UsageAnalytics",
    "UsageDelta",
    "UsageRepository",
    "UsageTotals",
    "UsageTotalsQuery",
]


@dataclass(frozen=True, slots=True)
class BillingRoute:
    """The billing identity of one serving route; with the task, the key of an attribution row. Unset and empty base URL or mode name the same row. A delta's route may name a provider with no model (``None``) or an empty model: its billing fields still fill the session row, an empty model fills an unset session model with ``''``, and its attribution row takes the session's model (``unknown`` without one); a route read back always has a model."""

    model: Optional[str]
    billing_provider: Optional[str] = None
    billing_base_url: Optional[str] = None
    billing_mode: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """One bundle of token counters; non-negative, all defaulting to zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"TokenCounts.{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class UsageDelta:
    """One API call's accounting: counters and costs to add, the route that served the call, the session's real ``source`` for the row guard, and the per-turn route ``task`` (empty for the main loop)."""

    tokens: TokenCounts = field(default_factory=TokenCounts)
    api_call_count: int = 0
    estimated_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    cost_status: Optional[str] = None
    cost_source: Optional[str] = None
    pricing_version: Optional[str] = None
    route: Optional[BillingRoute] = None
    source: Optional[str] = None
    task: str = ""

    def __post_init__(self) -> None:
        if self.api_call_count < 0:
            raise ValueError("UsageDelta.api_call_count must be >= 0")
        if self.source is not None and not self.source:
            raise ValueError("UsageDelta.source must be None or non-empty")


@dataclass(frozen=True, slots=True)
class RouteUsage:
    """One attribution row as read back: the billing tuple and the calls it accumulated."""

    route: BillingRoute
    api_call_count: int


@dataclass(frozen=True, slots=True)
class UsageTotalsQuery:
    """The totals filter: a message-count floor (default 1) and the archived toggle; child sessions are always excluded."""

    min_message_count: int = 1
    include_archived: bool = False

    def __post_init__(self) -> None:
        if self.min_message_count < 0:
            raise ValueError("UsageTotalsQuery.min_message_count must be >= 0")


@dataclass(frozen=True, slots=True)
class UsageTotals:
    """Store-wide sums: input plus output tokens, and per session the billed cost when present, else the estimate."""

    tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskUsage:
    """Auxiliary usage summed for one task over a session's lineage chain: calls, counters and the estimated spend."""

    task: str
    api_call_count: int
    tokens: TokenCounts
    estimated_cost_usd: float


@runtime_checkable
class UsageRepository(Protocol):
    """Token and cost accounting of one profile's store; the runtime check is an assembly diagnostic only."""

    # ── writes ────────────────────────────────────────────────────────────────

    def record_auxiliary_usage(
        self,
        session_id: SessionId,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        tokens: TokenCounts = TokenCounts(),
        estimated_cost_usd: Optional[float] = None,
        api_call_count: int = 1,
    ) -> None:
        """Record an auxiliary call's usage (vision, compression, title generation, background review) under ``task``.

        No-op for an empty id or task. One write: a missing session row is inserted as the ``unknown`` placeholder with the store owner's profile (a deleted session is refused), an existing row with no profile takes the owner's; then the attribution row keyed by (model or ``unknown``, provider or empty, base URL or empty, empty mode, task) adds the counters, ``api_call_count`` (None counts 1) and the estimate. The session counters, cost fields and recent route are untouched.
        """
        ...

    def set_billing_route(self, session_id: SessionId, *, provider: str, base_url: str, billing_mode: Optional[str] = None) -> None:
        """Drain the usage queue, then replace the session row's billing provider and base URL unconditionally; ``billing_mode=None`` keeps the stored mode.

        The model, ``model_config`` and the stored prompt are untouched; an unknown session is left unknown. Later deltas fill the route only while unset, so a set route stays once usage is accounted; before any accounted usage the first accounted delta's route replaces it.
        """
        ...

    # ── reads ─────────────────────────────────────────────────────────────────

    def recent_route(self, session_id: SessionId) -> Optional[RouteUsage]:
        """Drain the usage queue (not waiting on a thread running an event loop), then the most recently recorded main-loop attribution row: no task, model not ``unknown``, provider not empty, newest ``last_seen`` with ties to the row created later. ``None`` when there is none."""
        ...

    def usage_totals(self, query: UsageTotalsQuery = UsageTotalsQuery()) -> UsageTotals:
        """One read over root sessions with at least ``min_message_count`` messages, archived ones only with ``include_archived``. The usage queue is not drained first."""
        ...

    def auxiliary_usage_by_task(self, session_id: SessionId) -> Mapping[str, AuxiliaryTaskUsage]:
        """Attribution rows with a task summed per task over the session's lineage chain.

        The chain is the session and its ancestors over ``parent_session_id`` (at most 100, cycle-safe), cut at the explicit branch nearest the session, inclusive. Main-loop rows never count; an empty id reads as empty. The usage queue is not drained first.
        """
        ...

    # ── queue and barrier (no transaction of their own) ───────────────────────

    def queue_usage(self, session_id: SessionId, delta: UsageDelta) -> None:
        """Enqueue ``delta`` on the store holder's usage writer, which applies queued deltas in order (module docstring). After the writer was stopped by close, the delta applies synchronously and may raise."""
        ...

    def flush_usage(self, timeout: float = 5.0) -> bool:
        """Block until every delta queued before the call is applied; ``False`` when ``timeout`` seconds pass first. Never raises; with no live writer the caller applies the leftovers itself."""
        ...


@runtime_checkable
class UsageAnalytics(Protocol):
    """Usage analytics of one profile's store (module docstring); every verb is one read. The runtime check is an assembly diagnostic only."""

    # ── dashboard reads: sessions started after ``started_after`` ─────────────

    def daily_usage(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Sessions grouped by the local calendar day of ``started_at`` in the process timezone (``day``, ``YYYY-MM-DD``), ordered by day: ``input_tokens``, ``output_tokens``, ``cache_read_tokens``, ``reasoning_tokens``, ``estimated_cost`` and ``actual_cost`` (0 when unset), ``sessions`` and ``api_calls``."""
        ...

    def session_usage_by_model(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Sessions with a model (an empty model is its own group) grouped by ``model``, ordered by input plus output tokens descending, ties unordered: ``input_tokens``, ``output_tokens``, ``estimated_cost`` (0 when unset), ``sessions`` and ``api_calls``."""
        ...

    def session_usage_totals(self, *, started_after: float, model_rows_only: bool = False) -> dict[str, JsonValue]:
        """One row over the sessions, with ``model_rows_only`` only those with a non-empty model: ``total_input``, ``total_output``, ``total_cache_read``, ``total_reasoning``, ``total_estimated_cost`` and ``total_actual_cost`` (0 when unset), ``total_sessions`` and ``total_api_calls``."""
        ...

    def session_usage_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Sessions with a non-empty model grouped by (``model``, ``billing_provider``), ordered by input plus output tokens descending, ties unordered: the four token sums, ``estimated_cost`` and ``actual_cost`` (0 when unset), ``sessions``, ``api_calls``, ``tool_calls`` (the stored tool-call counts summed, ``None`` when every cell is unset), ``last_used_at`` (the latest ``started_at``) and ``avg_tokens_per_session``."""
        ...

    def session_tool_calls_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Sessions with a non-empty model grouped by (``model``, ``billing_provider``), unordered: ``tool_calls``, the stored tool-call counts summed with an unset cell as 0."""
        ...

    def call_usage_by_route(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Main-loop per-call usage rows (no task) of the sessions grouped by (``model``, ``billing_provider``), ordered by input plus output tokens descending, ties unordered: the four token sums, ``estimated_cost``, ``actual_cost``, ``sessions`` (distinct), ``api_calls``, ``last_used_at`` (the latest row write) and ``avg_tokens_per_session`` (mean input plus output tokens per usage row). ``[]`` on a store that predates per-call usage rows."""
        ...

    def auxiliary_usage_by_model_task(self, *, started_after: float) -> list[dict[str, JsonValue]]:
        """Per-call usage rows with a task of the sessions grouped by (``model``, ``task``, ``billing_provider``), ordered by input plus output tokens descending, ties unordered: the four token sums, ``estimated_cost``, ``sessions`` (distinct), ``api_calls`` and ``last_used_at``. ``[]`` on a store that predates the task dimension."""
        ...

    # ── insights reads: sessions started at or after ``started_since`` ────────

    def insight_sessions(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        """The window's sessions, newest ``started_at`` first, ties unordered: ``id``, ``source``, ``model``, ``started_at``, ``ended_at``, ``message_count``, ``tool_call_count``, ``input_tokens``, ``output_tokens``, ``cache_read_tokens``, ``cache_write_tokens``, ``billing_provider``, ``billing_base_url``, ``billing_mode``, ``estimated_cost_usd``, ``actual_cost_usd``, ``cost_status``, ``cost_source`` and ``api_call_count``, as stored."""
        ...

    def insight_tool_result_counts(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        """Tool-result rows (role ``tool`` with a ``tool_name``) of every visibility in the window's sessions, counted per (``session_id``, ``tool_name``) as ``count``, unordered."""
        ...

    def insight_tool_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        """``session_id`` and ``tool_calls``, the stored JSON text of the calls, of every assistant row with tool calls (any visibility) in the window's sessions, unordered."""
        ...

    def insight_skill_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        """``tool_calls`` and ``timestamp`` of the assistant rows ``insight_tool_call_payloads`` reads whose ``tool_calls`` text contains ``skill_view`` or ``skill_manage`` anywhere, unordered."""
        ...

    def insight_message_stats(self, *, started_since: float, source: Optional[str] = None) -> dict[str, JsonValue]:
        """One row over the messages of every visibility in the window's sessions: ``total_messages``, and ``user_messages``, ``assistant_messages`` and ``tool_messages`` (``None`` over no rows)."""
        ...

    def insight_model_usage(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, JsonValue]]:
        """Every per-call usage row (any task) of the window's sessions, unordered: ``session_id``, ``model``, ``billing_provider``, ``billing_base_url``, ``billing_mode``, ``api_call_count``, the five token counters, ``estimated_cost_usd``, ``actual_cost_usd``, ``cost_status`` and ``cost_source``. ``[]`` on a store that predates per-call usage rows."""
        ...
