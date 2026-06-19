"""Deterministic data tools for fundamental evidence collection.

These tools intentionally do not author an investment conclusion. They inspect
artifact freshness, collect real SEC/companyfacts context, and persist typed
artifacts for a later FundamentalAgent to interpret.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from .adapters import normalize_market_symbol
from .schemas import Candidate, DiscoveryData, FutuData
from .sec_provider import SecFilingsProvider
from .storage import InvestmentAssistantStore, utc_now

FUNDAMENTAL_CONTEXT_ARTIFACT = "fundamental_context"
FUNDAMENTAL_REFRESH_REQUEST_ARTIFACT = "fundamental_refresh_request"
DEFAULT_CACHE_STALE_AFTER_HOURS = 24
_CRITICAL_FUNDAMENTAL_FIELDS = (
    "ttm_revenue",
    "ttm_net_income",
    "total_assets",
    "total_liabilities",
)


def read_data_layer_catalog() -> dict[str, Any]:
    """Return the data-layer catalog available to fundamental research agents."""

    return {
        "artifact_type": "data_layer_catalog",
        "generated_at": utc_now(),
        "layers": [
            {
                "layer": "theme_discovery",
                "status": "available",
                "purpose": "主题候选池、产业链分层、候选进入池子的初始理由。",
                "artifact_type": "theme_discovery",
                "freshness_policy": {"stale_after_days": 7},
                "llm_may_interpret": True,
                "numeric_source": False,
            },
            {
                "layer": "sec_companyfacts",
                "status": "available",
                "purpose": "SEC companyfacts 硬数字，例如收入、利润、资产负债、ROE、margin。",
                "artifact_type": FUNDAMENTAL_CONTEXT_ARTIFACT,
                "source": "sec.gov via edgartools",
                "freshness_policy": {
                    "cache_stale_after_hours": _cache_stale_after_hours(),
                    "periodic_filing_stale_after_days": _sec_periodic_stale_days(),
                },
                "llm_may_interpret": True,
                "numeric_source": True,
                "numeric_llm_generated": False,
            },
            {
                "layer": "filing_metadata",
                "status": "available",
                "purpose": "最新 10-K / 10-Q / 8-K 的披露时间、source stale 和近期事件风险。",
                "artifact_type": FUNDAMENTAL_CONTEXT_ARTIFACT,
                "source": "sec.gov via edgartools",
                "freshness_policy": {
                    "cache_stale_after_hours": _cache_stale_after_hours(),
                    "recent_8k_days": _sec_recent_8k_days(),
                },
                "llm_may_interpret": True,
                "numeric_source": False,
            },
            {
                "layer": "filing_narrative_summary",
                "status": "planned",
                "purpose": "10-K/10-Q/8-K 长文、管理层讨论、风险因素、业务分部和主题叙事摘要。",
                "planned_pipeline": "edgartools_or_mineru_plus_sub_llm",
                "numeric_source": False,
                "numeric_extraction_allowed": False,
            },
            {
                "layer": "technical_market_data",
                "status": "out_of_scope_for_fundamental_tools",
                "purpose": "行情、K 线、技术指标、流动性、期权面。由后续 market/technical tools 提供。",
                "numeric_source": True,
            },
        ],
        "rules": [
            "Tools collect and label facts; they do not recommend holdings or weights.",
            "Missing or stale cache data should trigger one real collection attempt before the FundamentalAgent runs.",
            "If the provider has no fresher source data, mark stale_source and do not loop refresh.",
            "LLMs may interpret only persisted artifact fields and must not invent missing numbers.",
        ],
    }


def inspect_fundamental_freshness(
    session_id: str,
    *,
    symbols: list[str] | None = None,
    store: InvestmentAssistantStore | None = None,
    cache_stale_after_hours: int | None = None,
) -> dict[str, Any]:
    """Inspect whether persisted fundamental context is present and fresh."""

    store = store or InvestmentAssistantStore()
    resolved_symbols = _resolve_symbols(store, session_id, symbols)
    max_age = cache_stale_after_hours or _cache_stale_after_hours()
    artifact = store.latest_artifact(session_id, FUNDAMENTAL_CONTEXT_ARTIFACT)
    generated_at = _artifact_generated_at(artifact)
    cache_age_hours = _age_hours(generated_at) if generated_at else None
    cache_stale = cache_age_hours is None or cache_age_hours > max_age
    payload = artifact["payload"] if artifact else {}
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    top_status = str(payload.get("source_status") or "missing") if isinstance(payload, dict) else "missing"

    per_symbol = {}
    for symbol in resolved_symbols:
        item = items.get(symbol) if isinstance(items, dict) else None
        per_symbol[symbol] = _symbol_freshness(
            symbol,
            item if isinstance(item, dict) else None,
            top_status=top_status,
            artifact_missing=artifact is None,
            cache_stale=cache_stale,
        )

    should_refresh = any(value["should_refresh"] for value in per_symbol.values())
    return {
        "artifact_type": "fundamental_freshness_report",
        "generated_at": utc_now(),
        "session_id": session_id,
        "symbols": resolved_symbols,
        "context_artifact_id": artifact["artifact_id"] if artifact else None,
        "context_artifact_created_at": artifact.get("created_at") if artifact else None,
        "context_generated_at": generated_at,
        "cache_stale_after_hours": max_age,
        "cache_age_hours": cache_age_hours,
        "cache_stale": cache_stale,
        "source_status": top_status,
        "should_refresh": should_refresh,
        "refresh_reasons": sorted(
            {
                reason
                for value in per_symbol.values()
                for reason in value.get("refresh_reasons", [])
            }
        ),
        "per_symbol": per_symbol,
    }


def build_fundamental_context(
    session_id: str,
    *,
    symbols: list[str] | None = None,
    store: InvestmentAssistantStore | None = None,
    sec_provider: Any | None = None,
    trigger: str = "manual",
    reason: str = "",
) -> dict[str, Any]:
    """Collect real SEC/companyfacts context and persist it as an artifact."""

    store = store or InvestmentAssistantStore()
    resolved_symbols = _resolve_symbols(store, session_id, symbols)
    if not resolved_symbols:
        raise ValueError("No symbols available for fundamental context collection.")

    provider = sec_provider or SecFilingsProvider()
    candidates = [_candidate_for_sec(symbol) for symbol in resolved_symbols]
    sec_context = provider.get_sec_context(candidates)
    payload = {
        "artifact_type": FUNDAMENTAL_CONTEXT_ARTIFACT,
        "generated_at": sec_context.get("generated_at") or utc_now(),
        "session_id": session_id,
        "trigger": trigger,
        "reason": reason,
        "source": sec_context.get("source", "edgartools"),
        "source_status": sec_context.get("source_status", "unknown"),
        "requested_symbols": sec_context.get("requested_symbols", resolved_symbols),
        "fetched_symbols": sec_context.get("fetched_symbols", []),
        "items": sec_context.get("items", {}),
        "warnings": sec_context.get("warnings", []),
        "collection_policy": {
            "cache_stale_after_hours": _cache_stale_after_hours(),
            "periodic_filing_stale_after_days": _sec_periodic_stale_days(),
            "recent_8k_days": _sec_recent_8k_days(),
        },
    }
    artifact = store.add_artifact(session_id, FUNDAMENTAL_CONTEXT_ARTIFACT, payload)
    freshness = inspect_fundamental_freshness(
        session_id,
        symbols=resolved_symbols,
        store=store,
    )
    return {
        "artifact_type": "fundamental_context_build_result",
        "generated_at": utc_now(),
        "session_id": session_id,
        "fundamental_context_artifact_id": artifact["artifact_id"],
        "fundamental_context_version": artifact["version"],
        "source_status": payload["source_status"],
        "requested_symbols": payload["requested_symbols"],
        "fetched_symbols": payload["fetched_symbols"],
        "warnings": payload["warnings"],
        "freshness": freshness,
    }


def read_fundamental_context(
    session_id: str,
    *,
    symbols: list[str] | None = None,
    store: InvestmentAssistantStore | None = None,
) -> dict[str, Any]:
    """Read the latest fundamental context artifact, optionally filtered."""

    store = store or InvestmentAssistantStore()
    artifact = store.latest_artifact(session_id, FUNDAMENTAL_CONTEXT_ARTIFACT)
    if not artifact:
        return {
            "artifact_type": FUNDAMENTAL_CONTEXT_ARTIFACT,
            "session_id": session_id,
            "source_status": "missing",
            "context_artifact_id": None,
            "items": {},
            "warnings": ["No fundamental_context artifact exists for this session."],
        }

    payload = dict(artifact["payload"])
    selected = _normalize_symbols(symbols or [], _session_market(store, session_id))
    if selected:
        items = payload.get("items", {})
        payload["items"] = {
            symbol: items[symbol]
            for symbol in selected
            if isinstance(items, dict) and symbol in items
        }
        payload["requested_symbols"] = selected
    payload["context_artifact_id"] = artifact["artifact_id"]
    payload["context_artifact_version"] = artifact["version"]
    payload["context_artifact_created_at"] = artifact["created_at"]
    return payload


def request_fundamental_refresh(
    session_id: str,
    *,
    symbols: list[str] | None = None,
    mode: str = "sync",
    reason: str = "",
    store: InvestmentAssistantStore | None = None,
    sec_provider: Any | None = None,
) -> dict[str, Any]:
    """Record a refresh request and optionally execute it synchronously."""

    store = store or InvestmentAssistantStore()
    resolved_symbols = _resolve_symbols(store, session_id, symbols)
    request_payload = {
        "artifact_type": FUNDAMENTAL_REFRESH_REQUEST_ARTIFACT,
        "generated_at": utc_now(),
        "session_id": session_id,
        "symbols": resolved_symbols,
        "mode": mode,
        "reason": reason,
    }
    request_artifact = store.add_artifact(session_id, FUNDAMENTAL_REFRESH_REQUEST_ARTIFACT, request_payload)
    if mode != "sync":
        return {
            "artifact_type": "fundamental_refresh_result",
            "generated_at": utc_now(),
            "session_id": session_id,
            "refresh_request_artifact_id": request_artifact["artifact_id"],
            "status": "queued",
            "message": "Async refresh worker is not implemented yet; request artifact was recorded.",
        }
    build_result = build_fundamental_context(
        session_id,
        symbols=resolved_symbols,
        store=store,
        sec_provider=sec_provider,
        trigger="refresh_request",
        reason=reason,
    )
    return {
        "artifact_type": "fundamental_refresh_result",
        "generated_at": utc_now(),
        "session_id": session_id,
        "refresh_request_artifact_id": request_artifact["artifact_id"],
        "status": "completed",
        "build_result": build_result,
    }


def _symbol_freshness(
    symbol: str,
    item: dict[str, Any] | None,
    *,
    top_status: str,
    artifact_missing: bool,
    cache_stale: bool,
) -> dict[str, Any]:
    if artifact_missing:
        return {
            "symbol": symbol,
            "freshness": "missing",
            "source_status": "missing",
            "should_refresh": True,
            "refresh_reasons": ["missing_context"],
            "missing_fields": list(_CRITICAL_FUNDAMENTAL_FIELDS),
            "data_gaps": ["No fundamental_context artifact exists."],
        }
    if item is None:
        return {
            "symbol": symbol,
            "freshness": "missing",
            "source_status": top_status,
            "should_refresh": True,
            "refresh_reasons": ["missing_symbol_context"],
            "missing_fields": list(_CRITICAL_FUNDAMENTAL_FIELDS),
            "data_gaps": ["No symbol-level SEC/companyfacts item exists."],
        }

    source_status = str(item.get("source_status") or top_status or "unknown")
    fundamentals = item.get("fundamentals", {}) if isinstance(item.get("fundamentals"), dict) else {}
    missing_fields = [key for key in _CRITICAL_FUNDAMENTAL_FIELDS if fundamentals.get(key) is None]
    event_context = item.get("event_context", {}) if isinstance(item.get("event_context"), dict) else {}
    source_stale = bool(event_context.get("periodic_filing_stale"))
    data_gaps: list[str] = []
    refresh_reasons: list[str] = []

    if source_status not in {"available", "partial"}:
        data_gaps.append(str(item.get("error") or f"SEC source status is {source_status}."))
        refresh_reasons.append("source_unavailable")
    if cache_stale:
        refresh_reasons.append("cache_stale")
    if source_stale:
        data_gaps.append("Latest SEC periodic filing is stale or unavailable at source.")
    if missing_fields:
        data_gaps.append("Critical SEC/companyfacts fields are missing: " + ", ".join(missing_fields))

    if source_status not in {"available", "partial"}:
        freshness = "unavailable"
    elif cache_stale:
        freshness = "cache_stale"
    elif source_stale:
        freshness = "stale_source"
    elif missing_fields:
        freshness = "partial"
    else:
        freshness = "fresh"

    return {
        "symbol": symbol,
        "freshness": freshness,
        "source_status": source_status,
        "should_refresh": bool(refresh_reasons),
        "refresh_reasons": refresh_reasons,
        "source_stale": source_stale,
        "missing_fields": missing_fields,
        "data_gaps": data_gaps,
        "latest_periodic_filing_date": event_context.get("latest_periodic_filing_date"),
        "periodic_filing_age_days": event_context.get("periodic_filing_age_days"),
        "event_risk_level": event_context.get("event_risk_level", "unknown"),
    }


def _resolve_symbols(
    store: InvestmentAssistantStore,
    session_id: str,
    symbols: list[str] | None,
) -> list[str]:
    market = _session_market(store, session_id)
    provided = _normalize_symbols(symbols or [], market)
    if provided:
        return provided
    return _symbols_from_session(store, session_id, market)


def _symbols_from_session(store: InvestmentAssistantStore, session_id: str, market: str) -> list[str]:
    symbols: list[str] = []
    policy = store.latest_artifact(session_id, "policy")
    if policy:
        symbols.extend((policy["payload"] or {}).get("required_symbols") or [])
    discovery = store.latest_artifact(session_id, "theme_discovery")
    if discovery:
        payload = discovery["payload"] or {}
        for seed in payload.get("seed_symbols") or []:
            symbols.append(seed.get("symbol", ""))
        for requirement in payload.get("coverage_requirements") or []:
            symbols.extend(requirement.get("candidate_symbols") or [])
            symbols.extend(requirement.get("must_consider_symbols") or [])
        for domain in payload.get("domain_tree") or []:
            for subdomain in domain.get("subdomains") or []:
                for candidate in subdomain.get("candidates") or []:
                    symbols.append(candidate.get("symbol", ""))
    return _normalize_symbols(symbols, market)


def _session_market(store: InvestmentAssistantStore, session_id: str) -> str:
    initial = store.latest_artifact(session_id, "initial_request")
    if initial:
        payload = initial["payload"] or {}
        return str(payload.get("market") or "US").strip().upper() or "US"
    return "US"


def _normalize_symbols(symbols: list[Any], market: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in symbols:
        symbol = normalize_market_symbol(str(value or ""), market)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _candidate_for_sec(symbol: str) -> Candidate:
    return Candidate(
        symbol=symbol,
        name=symbol,
        theme_role="fundamental context collection target",
        source="fundamental_tools",
        score=0,
        discovery_data=DiscoveryData(
            source="fundamental_tools",
            role="fundamental context collection target",
            rationale="Synthetic candidate wrapper used only to call SEC provider.",
        ),
        futu_data=FutuData(),
    )


def _artifact_generated_at(artifact: dict[str, Any] | None) -> str | None:
    if not artifact:
        return None
    payload = artifact.get("payload") if isinstance(artifact, dict) else None
    if isinstance(payload, dict) and payload.get("generated_at"):
        return str(payload["generated_at"])
    return str(artifact.get("created_at") or "") or None


def _age_hours(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - parsed).total_seconds() / 3600, 4)


def _cache_stale_after_hours() -> int:
    return max(1, int(os.getenv("IA_FUNDAMENTAL_CONTEXT_STALE_HOURS", str(DEFAULT_CACHE_STALE_AFTER_HOURS))))


def _sec_periodic_stale_days() -> int:
    return max(30, int(os.getenv("IA_SEC_PERIODIC_STALE_DAYS", "140")))


def _sec_recent_8k_days() -> int:
    return max(1, int(os.getenv("IA_SEC_RECENT_8K_DAYS", "30")))
