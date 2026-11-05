"""In-memory reference ``UsageRepository`` test fixture.

Session rows (message counts, parent links, archived flags) are created by the sessions capability, so ``seed_session`` and ``set_session_archived`` provide this fixture's totals-filter state. Usage lives under the scope-store key ``usage``: ``sessions`` maps a session id to its :class:`_SessionRow` (counters, cost fields and stored billing route, which the sessions reference reads into ``SessionRecord``), and ``routes``, ``route_usage``, ``route_costs`` (summed actual cost, cost status and cost source) and ``route_last_write`` hold the attribution rows keyed by ``(session_id, (task or None, model, provider, base_url, mode))``. The workers reference writes the same tables.

The usage queue has no writer thread here: ``queue_usage`` applies the delta at once under the backend lock, as a writer that ran before any later read, and logs and drops an apply error. Every read barrier is therefore already met and ``flush_usage`` returns ``True``; merging adjacent deltas is not modelled, since it never changes a stored value.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from persistence import SessionId, ProfileScope
from persistence.conversation.sessions import SessionLineageKind
from persistence.conversation.usage import (
    AuxiliaryTaskUsage,
    BillingRoute,
    RouteUsage,
    TokenCounts,
    UsageDelta,
    UsageTotals,
    UsageTotalsQuery,
)

from ..backend import InMemoryBackend, register_repository

logger = logging.getLogger(__name__)

# One attribution row per (task, model, provider, base_url, mode); task is None
# for the main loop.
_RouteKey = tuple[Optional[str], str, Optional[str], Optional[str], Optional[str]]

# Lineage walk cap.
_LINEAGE_HOP_CAP = 100


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """The accounting columns of one session row."""

    tokens: TokenCounts
    api_call_count: int
    estimated_cost_usd: Optional[float]
    actual_cost_usd: Optional[float]
    cost_status: Optional[str]
    cost_source: Optional[str]
    pricing_version: Optional[str]


@dataclass(frozen=True, slots=True)
class SessionBillingRoute:
    """The billing route columns of one session row; any may be unset."""

    billing_provider: Optional[str]
    billing_base_url: Optional[str]
    billing_mode: Optional[str]


@dataclass
class _SessionRow:
    message_count: int
    parent_session_id: Optional[str]
    archived: bool
    usage: SessionUsage
    # Replaced by set_billing_route; recorded usage replaces it while the prior API-call count is zero, otherwise fills unset columns.
    route: SessionBillingRoute = field(
        default_factory=lambda: SessionBillingRoute(None, None, None)
    )
    route_model: Optional[str] = None


def _empty_usage() -> SessionUsage:
    return SessionUsage(
        tokens=TokenCounts(),
        api_call_count=0,
        estimated_cost_usd=None,
        actual_cost_usd=None,
        cost_status=None,
        cost_source=None,
        pricing_version=None,
    )


def _add_tokens(a: TokenCounts, b: TokenCounts) -> TokenCounts:
    return TokenCounts(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
    )


def _add_cost(stored: Optional[float], delta: Optional[float]) -> Optional[float]:
    if delta is None:
        return stored
    return (stored or 0.0) + delta


def _has_usage(delta: UsageDelta) -> bool:
    """The attribution gate: a counter, a call or a truthy estimate (an actual cost alone is not usage)."""
    return delta.api_call_count > 0 or delta.tokens != TokenCounts() or bool(delta.estimated_cost_usd)


def _route_key(task: Optional[str], route: BillingRoute) -> _RouteKey:
    return (task, route.model, route.billing_provider, route.billing_base_url, route.billing_mode)


class InMemoryUsageRepository:
    """Reference implementation of the ``UsageRepository`` protocol plus the fixture seeding seam."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _tables(self, scope: ProfileScope) -> dict:
        store = self._backend.scope_store(scope)
        return store.setdefault(
            "usage",
            {"sessions": {}, "routes": {}, "route_clock": 0, "route_last_write": {}, "route_usage": {}, "route_costs": {}},
        )

    # ── fixture seeding seam (stand-in for the sessions capability) ──────────

    def seed_session(
        self,
        session_id: str,
        *,
        message_count: int = 0,
        parent_session_id: Optional[str] = None,
        archived: bool = False,
    ) -> SessionId:
        scope = self._backend.scope
        session = self._backfill_session_row(scope, session_id, "cli", None)
        session["parent_session_id"] = parent_session_id
        session["archived"] = archived
        self._tables(scope)["sessions"][session_id] = _SessionRow(
            message_count=message_count,
            parent_session_id=parent_session_id,
            archived=archived,
            usage=_empty_usage(),
        )
        return SessionId(session_id)

    def set_session_archived(self, session_id: str, archived: bool) -> None:
        scope = self._backend.scope
        self._tables(scope)["sessions"][session_id].archived = archived
        self._backend.scope_store(scope)["sessions"][session_id]["archived"] = archived

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
        scope = self._backend.scope
        if not session_id or not task:
            return
        with self._backend._lock:
            self._backfill_session_row(scope, session_id, None, None)
            # Auxiliary rows never inherit the main-loop route: missing fields stay unknown or unset.
            route = BillingRoute(model=model or "unknown", billing_provider=billing_provider or None, billing_base_url=billing_base_url or None)
            calls = 1 if api_call_count is None else int(api_call_count)
            delta = UsageDelta(tokens=tokens, api_call_count=calls, estimated_cost_usd=estimated_cost_usd, route=route)
            self._attribute(self._tables(scope), session_id, task, delta, gated=False)

    def set_billing_route(self, session_id: SessionId, *, provider: str, base_url: str, billing_mode: Optional[str] = None) -> None:
        scope = self._backend.scope
        row = self._known_row(scope, session_id)
        if row is None:
            return  # no row to update: an unknown session stays unknown
        row.route = SessionBillingRoute(
            billing_provider=provider,
            billing_base_url=base_url,
            billing_mode=billing_mode if billing_mode is not None else row.route.billing_mode,
        )

    # ── reads ─────────────────────────────────────────────────────────────────

    def recent_route(self, session_id: SessionId) -> Optional[RouteUsage]:
        scope = self._backend.scope
        tables = self._tables(scope)
        routes: dict[tuple[str, _RouteKey], RouteUsage] = tables["routes"]
        last_write: dict[tuple[str, _RouteKey], float] = tables["route_last_write"]
        main_loop = [
            key for key, value in routes.items()
            if key[0] == session_id and key[1][0] is None
            and value.route.model != "unknown" and value.route.billing_provider
        ]
        if not main_loop:
            return None
        # Equal-time writes prefer the route inserted later.
        return routes[max(enumerate(main_loop), key=lambda item: (last_write[item[1]], item[0]))[1]]

    def usage_totals(self, query: UsageTotalsQuery = UsageTotalsQuery()) -> UsageTotals:
        scope = self._backend.scope
        tokens, cost = 0, 0.0
        for session_id in list(self._tables(scope)["sessions"]):
            row = self._known_row(scope, session_id)
            if row is None or row.parent_session_id is not None:
                continue
            if row.message_count < query.min_message_count:
                continue
            if row.archived and not query.include_archived:
                continue
            tokens += row.usage.tokens.input_tokens + row.usage.tokens.output_tokens
            session_cost = (
                row.usage.actual_cost_usd if row.usage.actual_cost_usd is not None
                else row.usage.estimated_cost_usd
            )
            cost += session_cost or 0.0
        return UsageTotals(tokens=tokens, cost_usd=cost)

    def auxiliary_usage_by_task(self, session_id: SessionId) -> Mapping[str, AuxiliaryTaskUsage]:
        scope = self._backend.scope
        if not session_id:
            return {}
        chain = set(self._lineage_chain(scope, session_id))
        sums: dict[str, AuxiliaryTaskUsage] = {}
        for (sid, key), row in self._tables(scope)["routes"].items():
            task = key[0]
            if sid not in chain or task is None:
                continue
            tokens, cost = self._tables(scope)["route_usage"].get((sid, key), (TokenCounts(), 0.0))
            prior = sums.get(task)
            sums[task] = AuxiliaryTaskUsage(
                task=task,
                api_call_count=row.api_call_count + (prior.api_call_count if prior else 0),
                tokens=_add_tokens(tokens, prior.tokens if prior else TokenCounts()),
                estimated_cost_usd=cost + (prior.estimated_cost_usd if prior else 0.0),
            )
        return {task: sums[task] for task in sorted(sums)}

    # ── queue and barrier ─────────────────────────────────────────────────────

    def queue_usage(self, session_id: SessionId, delta: UsageDelta) -> None:
        try:
            self._apply_delta(session_id, delta)
        except Exception as exc:
            # The writer logs an apply failure and drops the delta; accounting loss never reaches the turn.
            logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def flush_usage(self, timeout: float = 5.0) -> bool:
        return True

    # ── internals ─────────────────────────────────────────────────────────────

    def _apply_delta(self, session_id: SessionId, delta: UsageDelta) -> None:
        """The row guard, then the counters, the route fill and the attribution row as one staged write."""
        scope = self._backend.scope
        tables = self._tables(scope)
        with self._backend._lock:
            session = self._backfill_session_row(scope, session_id, delta.source, delta.route.model if delta.route else None)
            row = replace(tables["sessions"].get(
                session_id,
                _SessionRow(message_count=0, parent_session_id=None, archived=False, usage=_empty_usage()),
            ))
            row.route_model = session["model"]
            prior_calls = row.usage.api_call_count
            row.usage = self._staged_delta(row, delta)
            self._fill_stored_route(row, delta, prior_calls)
            staged_tables = {
                **tables,
                "routes": dict(tables["routes"]),
                "route_usage": dict(tables["route_usage"]),
                "route_last_write": dict(tables["route_last_write"]),
                "route_costs": dict(tables.setdefault("route_costs", {})),
            }
            self._attribute(staged_tables, session_id, delta.task or None, replace(delta, route=self._attribution_route(row, delta)))
            tables["sessions"][session_id] = row
            session["model"] = row.route_model
            for name in ("routes", "route_usage", "route_last_write", "route_costs", "route_clock"):
                tables[name] = staged_tables[name]

    @staticmethod
    def _attribution_route(row: _SessionRow, delta: UsageDelta) -> BillingRoute:
        """Missing main-loop fields come from the session row; a task row keeps them unknown or unset."""
        incoming = delta.route
        stored = SessionBillingRoute(None, None, None) if delta.task else row.route
        model = None if delta.task else row.route_model
        return BillingRoute(
            model=(incoming.model if incoming else None) or model or "unknown",
            billing_provider=(incoming.billing_provider if incoming else None) or stored.billing_provider or None,
            billing_base_url=(incoming.billing_base_url if incoming else None) or stored.billing_base_url or None,
            billing_mode=(incoming.billing_mode if incoming else None) or stored.billing_mode or None,
        )

    def _backfill_session_row(self, scope: ProfileScope, session_id: SessionId, source: Optional[str], model: Optional[str]) -> dict[str, Any]:
        """The row guard on the sessions capability's table: a missing row is created with ``source`` (``unknown`` without one) and the store owner's profile; an ``unknown`` placeholder takes a real source; a real source is kept; an unset profile or model is filled."""
        from .._authority import authority_state, require_not_retired
        from .sessions import blank_session_row  # the sessions reference imports nothing from here; kept late for symmetry with the other cross-family reads

        table: dict[str, dict[str, object]] = self._backend.scope_store(scope).setdefault("sessions", {})
        row = table.get(session_id)
        if row is None:
            # The row insert refuses to recreate a deleted session beside its tombstone.
            require_not_retired(authority_state(self._backend, scope).snapshot, session_id)
            row = blank_session_row(session_id, lineage_kind=SessionLineageKind.NONE, model_config=None)
            row["source"] = row["created_source"] = source or "unknown"
            table[session_id] = row
        elif row.get("source") == "unknown" and source:
            row["source"] = source
        if row["profile_name"] is None:
            row["profile_name"] = self._backend.owner_profile
        if row["model"] is None:
            row["model"] = model
        return row

    def _known_row(self, scope: ProfileScope, session_id: str) -> Optional[_SessionRow]:
        """The usage row of a session known to this fixture OR to the sessions capability (whose rows are minted through ``sessions.create``); ``None`` when neither knows it."""
        rows: dict[str, _SessionRow] = self._tables(scope)["sessions"]
        row = rows.get(session_id)
        if row is None and session_id in self._backend.scope_store(scope).get("sessions", {}):
            row = rows[session_id] = _SessionRow(
                message_count=0, parent_session_id=None, archived=False, usage=_empty_usage()
            )
        if row is not None:
            store = self._backend.scope_store(scope)
            session = store.get("sessions", {}).get(session_id)
            if session is not None:
                row.parent_session_id = session["parent_session_id"]
                row.archived = session["archived"]
            transcript = store.get("messages", {}).get("transcripts", {}).get(session_id)
            if transcript is not None:
                row.message_count = transcript.counters()[0]
            elif session_id in store.get("transcripts", {}):
                row.message_count = len(store["transcripts"][session_id])
        return row

    def _lineage_chain(self, scope: ProfileScope, session_id: str) -> tuple[str, ...]:
        """The session and its ancestors over parent edges (any lineage kind, deduplicated, capped), cut at the nearest explicit branch node inclusive."""
        sessions: dict[str, dict[str, object]] = self._backend.scope_store(scope).get("sessions", {})
        seeded: dict[str, _SessionRow] = self._tables(scope)["sessions"]
        chain: list[str] = []
        current: Optional[str] = session_id
        while current and current not in chain and len(chain) < _LINEAGE_HOP_CAP:
            chain.append(current)
            if current in sessions:
                if sessions[current]["lineage_kind"] is SessionLineageKind.BRANCH:
                    break
                parent = sessions[current]["parent_session_id"]
                current = str(parent) if parent is not None else None
            elif current in seeded:
                current = seeded[current].parent_session_id
            else:
                break
        return tuple(chain)

    @staticmethod
    def _fill_stored_route(row: _SessionRow, delta: UsageDelta, prior_calls: int) -> None:
        """The session-row route fill: a delta that accounts usage with a model and provider replaces the model and route when it is a main-loop delta, the prior API-call count is zero and either identity differs; otherwise each column fills only while unset (COALESCE)."""
        route = delta.route
        accounted = _has_usage(delta) or bool(delta.actual_cost_usd)  # truthiness: a 0.0 actual cost is not accounted usage
        if not accounted or route is None:
            return
        stored = row.route
        first_accounted = (
            not delta.task and prior_calls == 0 and bool(route.model) and bool(route.billing_provider)
            and (row.route_model != route.model or stored.billing_provider != route.billing_provider)
        )
        if first_accounted:
            row.route_model = route.model
            row.route = SessionBillingRoute(route.billing_provider, route.billing_base_url, route.billing_mode)
            return
        row.route = SessionBillingRoute(
            billing_provider=stored.billing_provider if stored.billing_provider is not None else route.billing_provider,
            billing_base_url=stored.billing_base_url if stored.billing_base_url is not None else route.billing_base_url,
            billing_mode=stored.billing_mode if stored.billing_mode is not None else route.billing_mode,
        )

    @staticmethod
    def _staged_delta(row: _SessionRow, delta: UsageDelta) -> SessionUsage:
        usage = row.usage
        return replace(
            usage,
            tokens=_add_tokens(usage.tokens, delta.tokens),
            api_call_count=usage.api_call_count + delta.api_call_count,
            estimated_cost_usd=(usage.estimated_cost_usd or 0.0) + (delta.estimated_cost_usd or 0.0),
            actual_cost_usd=_add_cost(usage.actual_cost_usd, delta.actual_cost_usd),
            cost_status=delta.cost_status if delta.cost_status is not None else usage.cost_status,
            cost_source=delta.cost_source if delta.cost_source is not None else usage.cost_source,
            pricing_version=(
                delta.pricing_version if delta.pricing_version is not None else usage.pricing_version
            ),
        )

    @staticmethod
    def _attribute(tables: dict, session_id: str, task: Optional[str], delta: UsageDelta, *, gated: bool = True) -> None:
        if delta.route is None or (gated and not _has_usage(delta)):
            return
        key = (session_id, _route_key(task, delta.route))
        existing = tables["routes"].get(key)
        calls = delta.api_call_count + (existing.api_call_count if existing else 0)
        tables["routes"][key] = RouteUsage(route=delta.route, api_call_count=calls)
        # The row's token and estimate columns, read back only by the per-task auxiliary sum.
        tokens, cost = tables["route_usage"].get(key, (TokenCounts(), 0.0))
        tables["route_usage"][key] = (_add_tokens(tokens, delta.tokens), cost + float(delta.estimated_cost_usd or 0.0))
        actual, status, source = tables.setdefault("route_costs", {}).get(key, (0.0, None, None))
        tables["route_costs"][key] = (
            actual + float(delta.actual_cost_usd or 0.0),
            delta.cost_status if delta.cost_status is not None else status,
            delta.cost_source if delta.cost_source is not None else source,
        )
        tables["route_clock"] += 1
        tables["route_last_write"][key] = time.time()


register_repository("usage", InMemoryUsageRepository)
