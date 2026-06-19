"""AI-planned lightweight Futu enrichment for discovered candidates."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .adapters import (
    FutuAdapterError,
    FutuOpenDConfig,
    MarketDataAdapter,
    _candidate_score_breakdown,
    _daily_returns,
    _import_futu,
    _market_factors,
    _owner_plate_supported_codes,
    _return_over,
    _row_get,
    _safe_float,
    _safe_str,
    _spread_bps,
    _trend_label,
)
from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .schemas import ThemeDiscoveryPlan
from .storage import utc_now


_LIGHTWEIGHT_ENRICHMENT_INSTRUCTIONS = """
You are the investment assistant's lightweight Futu enrichment planner.

Your job is to decide which cheap Futu data checks should run after theme
discovery and before SEC / long-form fundamental research. You are not a
portfolio architect and must not recommend allocations.

Principles:
- Preserve discovery layers and user-required symbols.
- Quote snapshots are mandatory for every allowed symbol; quote validity,
  market cap, valuation snapshot, turnover, and bid/ask are the cheapest first
  gate.
- Daily K-line checks are mandatory for every allowed symbol; trend, relative
  strength, and volatility should be available before triage.
- Use owner-plate checks when layer mapping or theme classification needs
  validation.
- Use option-surface checks sparingly; options are useful for later strategy
  design but are not required for every candidate.
- Do not ask for SEC filings, earnings transcripts, news summaries, holdings,
  current portfolio, orders, or trade plans.
- Do not invent symbols. Use only allowed_symbols.
- Explain why a check is useful, especially when you do not check every symbol.

Return LightweightEnrichmentPlan only.
"""


class LightweightCheckRequest(BaseModel):
    check_type: Literal[
        "quote_snapshot",
        "daily_kline",
        "owner_plate",
        "option_surface",
    ]
    symbols: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    fields_needed: list[str] = Field(default_factory=list)
    rationale: str = ""


class LightweightEnrichmentPlan(BaseModel):
    theme: str
    market: str = "US"
    planning_summary: str = ""
    check_requests: list[LightweightCheckRequest] = Field(default_factory=list)
    layer_budget_notes: list[str] = Field(default_factory=list)
    deferred_to_later_enrichment: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class LightweightCandidateEvidence(BaseModel):
    symbol: str
    name: str = ""
    layers: list[str] = Field(default_factory=list)
    role: str = ""
    security_type: str = ""
    quote_status: Literal["ok", "missing", "invalid", "not_requested", "error"] = "not_requested"
    kline_status: Literal["ok", "missing", "not_requested", "error"] = "not_requested"
    owner_plate_status: Literal["ok", "empty", "not_requested", "unsupported", "error"] = "not_requested"
    option_status: Literal["ok", "missing", "not_requested", "error", "disabled"] = "not_requested"
    quote_asof: str = ""
    last_price: float | None = None
    change_rate: float | None = None
    total_market_val: float | None = None
    circular_market_val: float | None = None
    turnover: float | None = None
    volume: float | None = None
    pe_ttm_ratio: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_ratio_ttm: float | None = None
    highest52weeks_price: float | None = None
    lowest52weeks_price: float | None = None
    spread_bps: float | None = None
    trend: str = ""
    relative_strength_60d: float | None = None
    realized_volatility: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None
    daily_kline_rows: int | None = None
    liquidity_score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    plate_memberships: list[dict[str, str]] = Field(default_factory=list)
    has_option_data: bool | None = None
    option_contracts_sampled: int | None = None
    option_avg_implied_volatility: float | None = None
    option_avg_spread_bps: float | None = None
    data_quality: Literal["fresh", "partial", "unavailable"] = "partial"
    warnings: list[str] = Field(default_factory=list)


class LightweightEnrichmentArtifact(BaseModel):
    artifact_type: str = "futu_lightweight_enrichment"
    theme: str
    market: str = "US"
    generated_at: str = ""
    source: str = "futu_opend"
    plan: LightweightEnrichmentPlan
    candidates: list[LightweightCandidateEvidence] = Field(default_factory=list)
    check_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


def build_lightweight_enrichment_artifact(
    discovery: ThemeDiscoveryPlan,
    *,
    config: FutuOpenDConfig | None = None,
) -> LightweightEnrichmentArtifact:
    """Plan and execute lightweight Futu enrichment for a discovery artifact."""

    allowed_symbols = _allowed_symbols(discovery)
    if not allowed_symbols:
        raise ValueError("Discovery artifact has no seed symbols to enrich.")

    plan, runtime = _run_lightweight_enrichment_planner(discovery, allowed_symbols)
    plan = _normalize_plan(plan, discovery, allowed_symbols)
    artifact = FutuLightweightExecutor(config or FutuOpenDConfig.from_env()).execute(discovery, plan)
    artifact.pydantic_ai = {**runtime, "plan": plan.model_dump(mode="json")}
    return artifact


class FutuLightweightExecutor:
    """Execute an AI-authored lightweight data plan through Futu OpenD."""

    def __init__(self, config: FutuOpenDConfig):
        self.config = config
        self.adapter = MarketDataAdapter(config)

    def execute(
        self,
        discovery: ThemeDiscoveryPlan,
        plan: LightweightEnrichmentPlan,
    ) -> LightweightEnrichmentArtifact:
        self.adapter._check_opend()
        futu = _import_futu()
        quote_ctx = futu.OpenQuoteContext(host=self.config.host, port=self.config.port)
        generated_at = utc_now()
        warnings: list[str] = []
        evidence_by_symbol = {
            symbol: LightweightCandidateEvidence(
                symbol=symbol,
                layers=_layers_for_symbol(discovery, symbol),
                role=_role_for_symbol(discovery, symbol),
            )
            for symbol in _allowed_symbols(discovery)
        }
        requested = _requests_by_type(plan)

        try:
            quote_symbols = requested.get("quote_snapshot", [])
            snapshot_by_code: dict[str, Any] = {}
            snapshot_errors: dict[str, str] = {}
            if quote_symbols:
                snapshot_by_code, snapshot_errors = self.adapter._get_market_snapshot_result(
                    quote_ctx,
                    futu,
                    quote_symbols,
                    warnings,
                )
            for symbol in quote_symbols:
                item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                row = snapshot_by_code.get(symbol)
                if row is None:
                    item.quote_status = "missing"
                    item.data_quality = "unavailable"
                    item.warnings.append(snapshot_errors.get(symbol) or "Futu snapshot missing.")
                    continue
                _fill_quote_fields(item, row)

            kline_symbols = requested.get("daily_kline", [])
            for symbol in kline_symbols:
                item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                try:
                    rows = self.adapter._get_daily_kline(quote_ctx, futu, symbol)
                except FutuAdapterError as exc:
                    item.kline_status = "error"
                    item.warnings.append(str(exc))
                    warnings.append(str(exc))
                    continue
                _fill_kline_fields(item, rows)

            plate_symbols = requested.get("owner_plate", [])
            if plate_symbols:
                basic_info = self.adapter._get_stock_basicinfo_by_code(quote_ctx, futu, plate_symbols)
                supported_plate_symbols = _owner_plate_supported_codes(plate_symbols, basic_info)
                unsupported_plate_symbols = [symbol for symbol in plate_symbols if symbol not in supported_plate_symbols]
                for symbol in plate_symbols:
                    item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                    info = basic_info.get(symbol) or {}
                    item.security_type = _safe_str(info.get("stock_type")) or item.security_type
                for symbol in unsupported_plate_symbols:
                    item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                    item.owner_plate_status = "unsupported"
                    item.warnings.append(
                        f"owner_plate skipped because Futu stock_type={item.security_type or 'unknown'}."
                    )
                try:
                    memberships = self.adapter._get_owner_plates(quote_ctx, futu, supported_plate_symbols)
                    for symbol in supported_plate_symbols:
                        item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                        item.plate_memberships = memberships.get(symbol, [])
                        item.owner_plate_status = "ok" if item.plate_memberships else "empty"
                except FutuAdapterError as exc:
                    warnings.append(str(exc))
                    for symbol in supported_plate_symbols:
                        item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                        item.owner_plate_status = "error"
                        item.warnings.append(str(exc))

            option_symbols = requested.get("option_surface", [])
            if not self.config.fetch_options and option_symbols:
                for symbol in option_symbols:
                    item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                    item.option_status = "disabled"
                    item.warnings.append("Futu options fetch disabled by config.")
            for symbol in option_symbols if self.config.fetch_options else []:
                item = evidence_by_symbol.setdefault(symbol, LightweightCandidateEvidence(symbol=symbol))
                last_price = item.last_price or _safe_float(_row_get(snapshot_by_code.get(symbol, {}), "last_price"))
                if last_price <= 0:
                    item.option_status = "missing"
                    item.warnings.append("Option surface skipped because last_price is unavailable.")
                    continue
                options, warning = self.adapter._get_options_surface(quote_ctx, futu, symbol, last_price)
                if warning:
                    item.warnings.append(warning)
                    warnings.append(warning)
                _fill_option_fields(item, options)
        finally:
            quote_ctx.close()

        candidates = [evidence_by_symbol[symbol] for symbol in _allowed_symbols(discovery)]
        _mark_unrequested_statuses(candidates, requested)
        return LightweightEnrichmentArtifact(
            theme=discovery.theme,
            market=discovery.market,
            generated_at=generated_at,
            plan=plan,
            candidates=candidates,
            check_summary=_check_summary(plan, candidates),
            warnings=warnings,
        )


def _run_lightweight_enrichment_planner(
    discovery: ThemeDiscoveryPlan,
    allowed_symbols: list[str],
) -> tuple[LightweightEnrichmentPlan, dict[str, Any]]:
    from pydantic_ai import ModelRetry

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=LightweightEnrichmentPlan,
        instructions=_LIGHTWEIGHT_ENRICHMENT_INSTRUCTIONS,
        agent_kind="lightweight_futu_enrichment_planner",
        output_retries=2,
        agent_skill_names=["candidate-triage"],
    )

    @agent.output_validator
    def validate_plan(data: LightweightEnrichmentPlan) -> LightweightEnrichmentPlan:
        try:
            _validate_plan(data, allowed_symbols)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return data

    payload = _planner_payload(discovery, allowed_symbols)
    result = agent.run_sync(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("lightweight_futu_enrichment_planner"),
    )
    plan = result.output
    plan.theme = discovery.theme
    plan.market = discovery.market
    return plan, {**runtime, "usage": usage_metadata(result), "input": payload}


def _planner_payload(discovery: ThemeDiscoveryPlan, allowed_symbols: list[str]) -> dict[str, Any]:
    full_symbol_count = len(allowed_symbols)
    return {
        "task": "Plan first-pass lightweight Futu data checks before SEC/fundamental deep enrichment.",
        "theme": discovery.theme,
        "market": discovery.market,
        "theme_description": discovery.theme_description,
        "initial_thesis": discovery.initial_thesis,
        "allowed_symbols": allowed_symbols,
        "budgets": {
            "max_quote_snapshot_symbols": max(
                full_symbol_count,
                _env_int("IA_LIGHTWEIGHT_MAX_SNAPSHOT_SYMBOLS", 140),
            ),
            "max_daily_kline_symbols": max(
                full_symbol_count,
                _env_int("IA_LIGHTWEIGHT_MAX_KLINE_SYMBOLS", 140),
            ),
            "max_owner_plate_symbols": _env_int("IA_LIGHTWEIGHT_MAX_OWNER_PLATE_SYMBOLS", 80),
            "max_option_surface_symbols": _env_int("IA_LIGHTWEIGHT_MAX_OPTION_SYMBOLS", 8),
            "goal": (
                "Quote snapshot and daily K-line should cover all allowed symbols. "
                "Spend selectivity on owner_plate and option_surface only. Do not run SEC or long-form research."
            ),
        },
        "domain_tree": [
            {
                "key": domain.key,
                "name": domain.name,
                "importance": domain.importance,
                "thesis": _short(domain.thesis, 500),
                "subdomains": [
                    {
                        "key": sub.key,
                        "name": sub.name,
                        "importance": sub.importance,
                        "candidate_symbols": [candidate.symbol for candidate in sub.candidates],
                        "omission_risk": _short(sub.omission_risk, 240),
                    }
                    for sub in domain.subdomains
                ],
            }
            for domain in discovery.domain_tree
        ],
        "coverage_requirements": [
            requirement.model_dump(mode="json") for requirement in discovery.coverage_requirements
        ],
        "candidate_summaries": [
            {
                "symbol": seed.symbol,
                "role": seed.role,
                "subthemes": seed.subthemes,
                "value_chain_stage": seed.value_chain_stage,
                "exposure_type": seed.exposure_type,
                "exposure_purity": seed.exposure_purity,
                "rationale": _short(seed.rationale, 360),
            }
            for seed in discovery.seed_symbols
        ],
        "executed_probe_summary": [
            {
                "layer_key": probe.layer_key,
                "probe_type": probe.probe_type,
                "result_count": probe.result_count,
                "candidate_symbols": probe.candidate_symbols[:30],
                "rationale": _short(probe.rationale, 240),
            }
            for probe in discovery.executed_filter_probes
        ],
        "output_contract": (
            "Return check_requests only for quote_snapshot, daily_kline, owner_plate, "
            "and option_surface. quote_snapshot and daily_kline should include every allowed symbol. "
            "Do not rank final candidates and do not produce portfolio weights."
        ),
    }


def _validate_plan(plan: LightweightEnrichmentPlan, allowed_symbols: list[str]) -> None:
    allowed = set(allowed_symbols)
    if not plan.check_requests:
        raise ValueError("Plan must include at least one lightweight check_request.")
    seen_check_types = {request.check_type for request in plan.check_requests}
    if "quote_snapshot" not in seen_check_types:
        raise ValueError("Plan must include quote_snapshot as the cheapest first-pass check.")
    if "daily_kline" not in seen_check_types:
        raise ValueError("Plan must include daily_kline so every candidate has technical evidence.")
    budgets = {
        "quote_snapshot": max(len(allowed_symbols), _env_int("IA_LIGHTWEIGHT_MAX_SNAPSHOT_SYMBOLS", 140)),
        "daily_kline": max(len(allowed_symbols), _env_int("IA_LIGHTWEIGHT_MAX_KLINE_SYMBOLS", 140)),
        "owner_plate": _env_int("IA_LIGHTWEIGHT_MAX_OWNER_PLATE_SYMBOLS", 80),
        "option_surface": _env_int("IA_LIGHTWEIGHT_MAX_OPTION_SYMBOLS", 8),
    }
    for request in plan.check_requests:
        unknown = sorted({symbol for symbol in request.symbols if symbol not in allowed})
        if unknown:
            raise ValueError(f"{request.check_type} includes symbols outside discovery: {unknown[:10]}")
        if len(request.symbols) > budgets[request.check_type]:
            raise ValueError(
                f"{request.check_type} requested {len(request.symbols)} symbols, "
                f"budget is {budgets[request.check_type]}."
            )


def _normalize_plan(
    plan: LightweightEnrichmentPlan,
    discovery: ThemeDiscoveryPlan,
    allowed_symbols: list[str],
) -> LightweightEnrichmentPlan:
    plan.theme = discovery.theme
    plan.market = discovery.market
    allowed = set(allowed_symbols)
    normalized_requests: list[LightweightCheckRequest] = [
        _all_symbol_request(
            plan,
            "quote_snapshot",
            allowed_symbols,
            "Mandatory full-universe first-pass quote, liquidity, valuation, market-cap, and spread check.",
        ),
        _all_symbol_request(
            plan,
            "daily_kline",
            allowed_symbols,
            "Mandatory full-universe trend, relative-strength, return, and volatility check.",
        ),
    ]
    for request in plan.check_requests:
        if request.check_type in {"quote_snapshot", "daily_kline"}:
            continue
        symbols = []
        for symbol in request.symbols:
            if symbol in allowed and symbol not in symbols:
                symbols.append(symbol)
        normalized_requests.append(request.model_copy(update={"symbols": symbols}))
    plan.check_requests = normalized_requests
    return plan


def _all_symbol_request(
    plan: LightweightEnrichmentPlan,
    check_type: Literal["quote_snapshot", "daily_kline"],
    allowed_symbols: list[str],
    default_rationale: str,
) -> LightweightCheckRequest:
    existing = next((request for request in plan.check_requests if request.check_type == check_type), None)
    if existing is None:
        return LightweightCheckRequest(
            check_type=check_type,
            symbols=allowed_symbols,
            priority="high",
            rationale=default_rationale,
        )
    rationale = existing.rationale or default_rationale
    if "full-universe" not in rationale.lower():
        rationale = f"{rationale} Forced to full-universe coverage by lightweight enrichment policy."
    return existing.model_copy(
        update={
            "symbols": allowed_symbols,
            "priority": "high",
            "rationale": rationale,
        }
    )


def _fill_quote_fields(item: LightweightCandidateEvidence, row: Any) -> None:
    item.name = _safe_str(_row_get(row, "name")) or item.name
    item.quote_status = "ok"
    item.quote_asof = _safe_str(_row_get(row, "update_time"))
    item.last_price = _safe_float(_row_get(row, "last_price"))
    item.change_rate = _safe_float(_row_get(row, "change_rate"))
    item.total_market_val = _safe_float(_row_get(row, "total_market_val"))
    item.circular_market_val = _safe_float(_row_get(row, "circular_market_val"))
    item.turnover = _safe_float(_row_get(row, "turnover"))
    item.volume = _safe_float(_row_get(row, "volume"))
    item.pe_ttm_ratio = _safe_float(_row_get(row, "pe_ttm_ratio"))
    item.pe_ratio = _safe_float(_row_get(row, "pe_ratio"))
    item.pb_ratio = _safe_float(_row_get(row, "pb_ratio"))
    item.dividend_ratio_ttm = _safe_float(_row_get(row, "dividend_ratio_ttm"))
    item.highest52weeks_price = _safe_float(_row_get(row, "highest52weeks_price"))
    item.lowest52weeks_price = _safe_float(_row_get(row, "lowest52weeks_price"))
    bid_price = _safe_float(_row_get(row, "bid_price"))
    ask_price = _safe_float(_row_get(row, "ask_price"))
    item.spread_bps = _spread_bps(bid_price, ask_price)
    item.score_breakdown = _candidate_score_breakdown(
        50.0,
        0.0,
        item.turnover or 0.0,
        item.change_rate or 0.0,
    )
    item.liquidity_score = item.score_breakdown.get("liquidity")
    if not item.quote_asof:
        item.warnings.append("Futu snapshot missing update_time.")
    if not item.last_price or item.last_price <= 0:
        item.quote_status = "invalid"
        item.data_quality = "unavailable"


def _fill_kline_fields(item: LightweightCandidateEvidence, rows: list[dict[str, float]]) -> None:
    if not rows:
        item.kline_status = "missing"
        item.warnings.append("Futu daily K-line returned no rows.")
        return
    rs, volatility = _market_factors(rows)
    item.kline_status = "ok"
    item.relative_strength_60d = round(rs, 4)
    item.realized_volatility = round(volatility, 6)
    item.return_20d = _return_over(rows, 20)
    item.return_60d = _return_over(rows, 60)
    item.daily_kline_rows = len(rows)
    item.trend = _trend_label(rs)
    item.score_breakdown = _candidate_score_breakdown(
        rs,
        volatility,
        item.turnover or 0.0,
        item.change_rate or 0.0,
    )
    item.liquidity_score = item.score_breakdown.get("liquidity")
    if len(_daily_returns(rows, 60)) < 20:
        item.warnings.append("Futu daily K-line has fewer than 20 recent returns.")


def _fill_option_fields(item: LightweightCandidateEvidence, options: dict[str, Any]) -> None:
    status = str(options.get("status") or "")
    item.has_option_data = bool(options.get("has_option_data"))
    item.option_contracts_sampled = options.get("contracts_sampled")
    item.option_avg_implied_volatility = options.get("avg_implied_volatility")
    item.option_avg_spread_bps = options.get("avg_spread_bps")
    if item.has_option_data and status in {"ok", ""}:
        item.option_status = "ok"
    elif status == "error":
        item.option_status = "error"
        if options.get("error"):
            item.warnings.append(str(options["error"]))
    else:
        item.option_status = "missing"


def _mark_unrequested_statuses(
    candidates: list[LightweightCandidateEvidence],
    requested: dict[str, list[str]],
) -> None:
    requested_sets = {key: set(value) for key, value in requested.items()}
    for item in candidates:
        if item.symbol not in requested_sets.get("quote_snapshot", set()):
            item.quote_status = "not_requested"
        if item.symbol not in requested_sets.get("daily_kline", set()):
            item.kline_status = "not_requested"
        if item.symbol not in requested_sets.get("owner_plate", set()):
            item.owner_plate_status = "not_requested"
        if item.symbol not in requested_sets.get("option_surface", set()):
            item.option_status = "not_requested"
        if item.quote_status == "ok" and item.kline_status in {"ok", "not_requested"}:
            item.data_quality = "fresh" if item.kline_status == "ok" else "partial"


def _check_summary(plan: LightweightEnrichmentPlan, candidates: list[LightweightCandidateEvidence]) -> dict[str, Any]:
    requested = _requests_by_type(plan)
    return {
        "requested_counts": {key: len(value) for key, value in requested.items()},
        "quote_ok": sum(1 for item in candidates if item.quote_status == "ok"),
        "quote_unavailable": sum(1 for item in candidates if item.quote_status in {"missing", "invalid", "error"}),
        "kline_ok": sum(1 for item in candidates if item.kline_status == "ok"),
        "owner_plate_ok": sum(1 for item in candidates if item.owner_plate_status == "ok"),
        "owner_plate_unsupported": sum(1 for item in candidates if item.owner_plate_status == "unsupported"),
        "option_ok": sum(1 for item in candidates if item.option_status == "ok"),
        "candidate_count": len(candidates),
    }


def _requests_by_type(plan: LightweightEnrichmentPlan) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "quote_snapshot": [],
        "daily_kline": [],
        "owner_plate": [],
        "option_surface": [],
    }
    for request in plan.check_requests:
        bucket = result.setdefault(request.check_type, [])
        for symbol in request.symbols:
            if symbol not in bucket:
                bucket.append(symbol)
    return result


def _allowed_symbols(discovery: ThemeDiscoveryPlan) -> list[str]:
    symbols: list[str] = []
    for seed in discovery.seed_symbols:
        symbol = seed.symbol
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _layers_for_symbol(discovery: ThemeDiscoveryPlan, symbol: str) -> list[str]:
    layers: list[str] = []
    for domain in discovery.domain_tree:
        for subdomain in domain.subdomains:
            if any(candidate.symbol == symbol for candidate in subdomain.candidates):
                key = subdomain.key or domain.key
                if key and key not in layers:
                    layers.append(key)
    for requirement in discovery.coverage_requirements:
        if symbol in requirement.candidate_symbols or symbol in requirement.must_consider_symbols:
            if requirement.key and requirement.key not in layers:
                layers.append(requirement.key)
    return layers


def _role_for_symbol(discovery: ThemeDiscoveryPlan, symbol: str) -> str:
    for seed in discovery.seed_symbols:
        if seed.symbol == symbol:
            return seed.role
    return ""


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default
