"""``UsageRepository`` on SQLite: token and cost accounting through the usage store functions, with deltas queued on the store holder's accounting queue, which applies each through ``update_token_counts``."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from persistence import SessionId
from persistence.conversation.usage import (
    AuxiliaryTaskUsage, BillingRoute, RouteUsage, TokenCounts, UsageDelta, UsageTotals, UsageTotalsQuery,
)

from .._backend import SQLiteStore, register_repository
from .._holder import StoreHolder, set_token_delta_applier
from ._store.sessions import get_recent_session_model_route
from ._store.usage import (
    auxiliary_usage_by_task, record_auxiliary_usage, update_session_billing_route, update_token_counts, usage_totals,
)
from ._errors import refusals

__all__ = ["SQLiteUsageRepository"]


def _apply_usage_delta(holder: StoreHolder, session_id: str, kwargs: dict[str, Any]) -> None:
    """Write one queued delta: ``update_token_counts``."""
    update_token_counts(holder, session_id, **kwargs)


def _delta_kwargs(delta: UsageDelta) -> dict[str, Any]:
    """``update_token_counts`` keywords for one delta."""
    route = delta.route
    tokens = delta.tokens
    return {
        "source": delta.source,
        "input_tokens": tokens.input_tokens,
        "output_tokens": tokens.output_tokens,
        "cache_read_tokens": tokens.cache_read_tokens,
        "cache_write_tokens": tokens.cache_write_tokens,
        "reasoning_tokens": tokens.reasoning_tokens,
        "estimated_cost_usd": delta.estimated_cost_usd,
        "actual_cost_usd": delta.actual_cost_usd,
        "cost_status": delta.cost_status,
        "cost_source": delta.cost_source,
        "pricing_version": delta.pricing_version,
        "billing_provider": route.billing_provider if route is not None else None,
        "billing_base_url": route.billing_base_url if route is not None else None,
        "billing_mode": route.billing_mode if route is not None else None,
        "model": route.model if route is not None else None,
        "api_call_count": delta.api_call_count,
        "task": delta.task,
    }


class SQLiteUsageRepository:
    """Token and cost accounting of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # ── writes ────────────────────────────────────────────────────────────────

    def record_auxiliary_usage(
        self, session_id: SessionId, task: str, *, model: Optional[str] = None, billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None, tokens: TokenCounts = TokenCounts(),
        estimated_cost_usd: Optional[float] = None, api_call_count: int = 1,
    ) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            record_auxiliary_usage(holder,
                session_id, task, model=model, billing_provider=billing_provider, billing_base_url=billing_base_url,
                input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens,
                cache_read_tokens=tokens.cache_read_tokens, cache_write_tokens=tokens.cache_write_tokens,
                reasoning_tokens=tokens.reasoning_tokens, estimated_cost_usd=estimated_cost_usd,
                api_call_count=api_call_count)

    def set_billing_route(self, session_id: SessionId, *, provider: str, base_url: str,
                          billing_mode: Optional[str] = None) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_session_billing_route(holder, session_id, provider=provider, base_url=base_url, billing_mode=billing_mode)

    # ── reads ─────────────────────────────────────────────────────────────────

    def recent_route(self, session_id: SessionId) -> Optional[RouteUsage]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            row = get_recent_session_model_route(holder, session_id)
        if row is None:
            return None
        route = BillingRoute(
            model=row["model"], billing_provider=row["billing_provider"] or None,
            billing_base_url=row["billing_base_url"] or None, billing_mode=row["billing_mode"] or None)
        return RouteUsage(route=route, api_call_count=int(row["api_call_count"] or 0))

    def usage_totals(self, query: UsageTotalsQuery = UsageTotalsQuery()) -> UsageTotals:
        holder = self._store.holder()
        with refusals(holder):
            totals = usage_totals(holder, min_message_count=query.min_message_count, include_archived=query.include_archived)
        return UsageTotals(tokens=totals["tokens"], cost_usd=totals["cost_usd"])

    def auxiliary_usage_by_task(self, session_id: SessionId) -> Mapping[str, AuxiliaryTaskUsage]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            rows = auxiliary_usage_by_task(holder, session_id)
        return {
            task: AuxiliaryTaskUsage(
                task=task, api_call_count=int(row["api_calls"]),
                tokens=TokenCounts(
                    input_tokens=int(row["input_tokens"]), output_tokens=int(row["output_tokens"]),
                    cache_read_tokens=int(row["cache_read_tokens"]), cache_write_tokens=int(row["cache_write_tokens"]),
                    reasoning_tokens=int(row["reasoning_tokens"])),
                estimated_cost_usd=float(row["estimated_cost_usd"]))
            for task, row in rows.items()
        }

    # ── queue and barrier ─────────────────────────────────────────────────────

    def queue_usage(self, session_id: SessionId, delta: UsageDelta) -> None:
        with refusals(session_id=session_id):
            self._store.holder().queue_token_counts(session_id, **_delta_kwargs(delta))

    def flush_usage(self, timeout: float = 5.0) -> bool:
        with refusals():
            return self._store.holder().flush_token_counts(timeout)


set_token_delta_applier(_apply_usage_delta)
register_repository("usage", SQLiteUsageRepository)
