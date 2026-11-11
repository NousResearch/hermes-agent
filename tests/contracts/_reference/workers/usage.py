"""The worker's usage operations on the usage reference's tables: the session counter update and the model-usage attribution.

Session counters, costs and the stored route live in the usage reference's ``_SessionRow``; attribution rows in its ``routes`` tables (``InMemoryUsageRepository._attribute``). Both run inside the receipt transaction on the staged tables.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from session_rules.surfaces import created_surface

from ..conversation.usage import InMemoryUsageRepository, SessionBillingRoute, _add_cost, _add_tokens

__all__ = ["TOKEN_COUNTERS", "apply_auxiliary_usage", "apply_main_usage", "stamp_created_source"]

TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_UNSET_CREATED = (None, "", "unknown", "local")


def _tokens(payload: Mapping[str, Any]) -> TokenCounts:
    return TokenCounts(**{name: payload.get(name, 0) for name in TOKEN_COUNTERS})


def _session_usage(stored: Any, payload: Mapping[str, Any], absolute: bool) -> Any:
    """The session's counters after one report: added, or set when ``absolute``; a text column the report leaves unset keeps its stored value."""
    tokens, calls, estimated, actual = _tokens(payload), payload.get("api_call_count", 0), payload.get("estimated_cost_usd"), payload.get("actual_cost_usd")
    if absolute:
        counted = {"tokens": tokens, "api_call_count": calls, "estimated_cost_usd": float(estimated or 0),
                   "actual_cost_usd": stored.actual_cost_usd if actual is None else actual}
    else:
        counted = {"tokens": _add_tokens(stored.tokens, tokens), "api_call_count": stored.api_call_count + calls,
                   "estimated_cost_usd": (stored.estimated_cost_usd or 0.0) + (estimated or 0.0),
                   "actual_cost_usd": _add_cost(stored.actual_cost_usd, actual)}
    texts = {name: payload.get(name) if payload.get(name) is not None else getattr(stored, name) for name in ("cost_status", "cost_source", "pricing_version")}
    return replace(stored, **counted, **texts)


def apply_main_usage(store: Any, session_id: str, payload: Mapping[str, Any]) -> None:
    """``_update_token_counts_in_transaction(**payload)`` on the worker's session."""
    usage: InMemoryUsageRepository = store.usage
    scope = store.scope
    row = usage._known_row(scope, session_id)
    session = store.scope_store(scope).get("sessions", {}).get(session_id)
    if row is None or session is None:
        return
    model, provider, task = payload.get("model"), payload.get("billing_provider"), payload.get("task") or ""
    tokens = _tokens(payload)
    has_usage = bool(tokens != TokenCounts() or payload.get("api_call_count") or payload.get("estimated_cost_usd"))
    accounted = has_usage or bool(payload.get("actual_cost_usd"))
    row.route_model = session.get("model")
    if (not task and row.usage.api_call_count == 0 and accounted and model and provider
            and (row.route_model != model or row.route.billing_provider != provider)):
        row.route_model = model
        row.route = SessionBillingRoute(provider, payload.get("billing_base_url"), payload.get("billing_mode"))
    row.usage = _session_usage(row.usage, payload, bool(payload.get("absolute")))
    if accounted:
        stored = row.route
        row.route = SessionBillingRoute(*(getattr(stored, name) if getattr(stored, name) is not None else payload.get(name)
                                          for name in ("billing_provider", "billing_base_url", "billing_mode")))
        row.route_model = row.route_model if row.route_model is not None else model
    session["model"] = row.route_model
    if not payload.get("absolute") and has_usage:
        route = BillingRoute(
            model=model or (None if task else row.route_model) or "unknown",
            billing_provider=provider or (None if task else row.route.billing_provider) or None,
            billing_base_url=payload.get("billing_base_url") or (None if task else row.route.billing_base_url) or None,
            billing_mode=payload.get("billing_mode") or (None if task else row.route.billing_mode) or None,
        )
        delta = UsageDelta(tokens=tokens, api_call_count=payload.get("api_call_count", 0), estimated_cost_usd=payload.get("estimated_cost_usd"), route=route,
                           actual_cost_usd=payload.get("actual_cost_usd"), cost_status=payload.get("cost_status"), cost_source=payload.get("cost_source"))
        usage._attribute(usage._tables(scope), session_id, task or None, delta)


def apply_auxiliary_usage(store: Any, session_id: str, payload: Mapping[str, Any]) -> None:
    """``_record_model_usage(**({"api_call_count": 1} | payload))``: the task's attribution row only; missing route fields stay unknown or unset."""
    usage: InMemoryUsageRepository = store.usage
    route = BillingRoute(model=payload.get("model") or "unknown", billing_provider=payload.get("billing_provider") or None,
                         billing_base_url=payload.get("billing_base_url") or None)
    delta = UsageDelta(tokens=_tokens(payload), api_call_count=payload.get("api_call_count", 1), estimated_cost_usd=payload.get("estimated_cost_usd"), route=route,
                       actual_cost_usd=payload.get("actual_cost_usd"), cost_status=payload.get("cost_status"), cost_source=payload.get("cost_source"))
    usage._attribute(usage._tables(store.scope), session_id, payload["task"], delta, gated=False)


def stamp_created_source(store: Any, session_id: str, source: Any) -> None:
    """The worker platform as ``created_source`` of an unset row, then on its compression ancestors."""
    surfaced = created_surface(source)
    table = store.scope_store(store.scope).get("sessions", {})
    row = table.get(session_id)
    if not surfaced or row is None:
        return
    if row.get("created_source") in _UNSET_CREATED:
        row["created_source"] = surfaced
    store.lineage._stamp_ancestors(store.scope, table, session_id)
