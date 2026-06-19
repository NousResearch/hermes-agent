"""Typed models for the investment assistant workflow."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    NEW = "NEW"
    NEEDS_POLICY_CONFIRMATION = "NEEDS_POLICY_CONFIRMATION"
    EXPANDING_THEME = "EXPANDING_THEME"
    THEME_DISCOVERY_COMPLETE = "THEME_DISCOVERY_COMPLETE"
    BUILDING_MARKET_ARTIFACTS = "BUILDING_MARKET_ARTIFACTS"
    BUILDING_UNBIASED_CANDIDATE_POOL = "BUILDING_UNBIASED_CANDIDATE_POOL"
    NEEDS_CANDIDATE_TRIAGE_STRATEGY = "NEEDS_CANDIDATE_TRIAGE_STRATEGY"
    CANDIDATE_TRIAGE_COMPLETE = "CANDIDATE_TRIAGE_COMPLETE"
    VALIDATING_EVIDENCE = "VALIDATING_EVIDENCE"
    REFLECTING_CANDIDATE_POOL = "REFLECTING_CANDIDATE_POOL"
    DRAFTING_TARGET_PORTFOLIO_MAPS = "DRAFTING_TARGET_PORTFOLIO_MAPS"
    REFLECTING_MAPS = "REFLECTING_MAPS"
    NEEDS_PORTFOLIO_MAP_SELECTION = "NEEDS_PORTFOLIO_MAP_SELECTION"
    NEEDS_PORTFOLIO_MAP_REVIEW = "NEEDS_PORTFOLIO_MAP_REVIEW"
    TARGET_PORTFOLIO_MAP_SELECTED = "TARGET_PORTFOLIO_MAP_SELECTED"
    REVISING_PORTFOLIO_MAP = "REVISING_PORTFOLIO_MAP"
    NEEDS_PORTFOLIO_REVISION_CLARIFICATION = "NEEDS_PORTFOLIO_REVISION_CLARIFICATION"
    NEEDS_PORTFOLIO_REVISION_REVIEW = "NEEDS_PORTFOLIO_REVISION_REVIEW"
    TARGET_PORTFOLIO_MAP_REVISION_SELECTED = "TARGET_PORTFOLIO_MAP_REVISION_SELECTED"
    READING_CURRENT_PORTFOLIO = "READING_CURRENT_PORTFOLIO"
    BUILDING_CONSTRUCTION_PLAN = "BUILDING_CONSTRUCTION_PLAN"
    TRACKING_PLAN = "TRACKING_PLAN"
    CANCELLED = "CANCELLED"


class WorkflowStatus(str, Enum):
    ACTIVE = "active"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class HumanActionStatus(str, Enum):
    PENDING = "pending"
    ANSWERED = "answered"
    CANCELLED = "cancelled"


class HumanActionKind(str, Enum):
    CONFIRM_POLICY = "confirm_policy"
    SELECT_CANDIDATE_TRIAGE_STRATEGY = "select_candidate_triage_strategy"
    SELECT_PORTFOLIO_MAP = "select_portfolio_map"
    REVIEW_PORTFOLIO_MAPS = "review_portfolio_maps"
    CLARIFY_PORTFOLIO_REVISION = "clarify_portfolio_revision"
    CONFIRM_PORTFOLIO_REVISION = "confirm_portfolio_revision"


class InvestmentPolicy(BaseModel):
    theme: str
    theme_description: str = ""
    required_symbols: list[str] = Field(default_factory=list)
    objective: Literal["balanced", "growth", "income"] = "balanced"
    risk_level: Literal["conservative", "moderate", "aggressive"] = "moderate"
    target_portfolio_weight: float = Field(default=0.15, ge=0, le=1)
    cash_reserve: float = Field(default=0.10, ge=0, le=1)
    single_name_limit: float = Field(default=0.15, ge=0, le=1)
    allow_options: bool = False
    notes: str = ""


class ThemeDiscoverySeed(BaseModel):
    symbol: str
    market: str = "US"
    role: str
    rationale: str = ""
    subthemes: list[str] = Field(default_factory=list)
    value_chain_stage: str = ""
    exposure_type: str = ""
    exposure_purity: str = ""
    source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    freshness: Literal["fresh", "partial", "stale", "unknown"] = "unknown"


class ThemeDomainCandidate(BaseModel):
    """Candidate named inside the theme domain tree before data enrichment."""

    symbol: str
    role: str = ""
    rationale: str = ""
    priority: Literal["must_consider", "strong_candidate", "watchlist"] = "strong_candidate"


class ThemeSubdomain(BaseModel):
    """Investable subdomain inside a broader theme value chain."""

    key: str
    name: str
    thesis: str = ""
    importance: Literal["high", "medium", "low"] = "medium"
    candidate_limit_reason: str = ""
    candidates: list[ThemeDomainCandidate] = Field(default_factory=list)
    omission_risk: str = ""


class ThemeDomain(BaseModel):
    """Top-level domain in the discovered theme map."""

    key: str
    name: str
    thesis: str = ""
    importance: Literal["core", "important", "optional"] = "important"
    subdomains: list[ThemeSubdomain] = Field(default_factory=list)


class ThemeCoverageRequirement(BaseModel):
    """LLM-authored checklist for theme coverage before market-data validation."""

    key: str
    name: str
    thesis: str = ""
    priority: Literal["required", "important", "optional"] = "important"
    min_candidates: int = Field(default=1, ge=0)
    candidate_symbols: list[str] = Field(default_factory=list)
    must_consider_symbols: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)


class ResearchSource(BaseModel):
    """Research source captured during theme discovery."""

    source_id: str
    title: str = ""
    url: str = ""
    publisher: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    source_type: Literal[
        "web",
        "filing",
        "news",
        "etf_holdings",
        "company_site",
        "other",
    ] = "web"
    summary: str = ""
    symbols: list[str] = Field(default_factory=list)
    coverage_keys: list[str] = Field(default_factory=list)


class DiscoveryFilterDecision(BaseModel):
    """One screener category decision inside a layer-level filter plan."""

    category: Literal[
        "plate",
        "market_cap",
        "liquidity",
        "valuation",
        "dividend",
        "technical",
        "financial",
        "analysis",
        "options",
        "web",
        "broad_market",
    ]
    decision: Literal["use_now", "skip", "defer_to_later_enrichment"]
    planned_fields: list[str] = Field(default_factory=list)
    planned_thresholds_or_ranking: str = ""
    rationale: str = ""


class DiscoveryFilterPlan(BaseModel):
    """Agent-authored Futu screener plan for one theme layer."""

    layer_key: str
    layer_name: str = ""
    target_candidate_profile: str = ""
    plate_search_terms: list[str] = Field(default_factory=list)
    plate_codes_to_probe: list[str] = Field(default_factory=list)
    filter_decisions: list[DiscoveryFilterDecision] = Field(default_factory=list)
    execution_plan: str = ""


class ExecutedDiscoveryProbe(BaseModel):
    """Auditable record of one Futu/web probe used during discovery."""

    layer_key: str
    probe_type: Literal[
        "broad_calibrated",
        "subdomain_plate",
        "subdomain_plate_refinement",
        "web",
        "other",
    ] = "other"
    plate_code: str | None = None
    stock_filter_specs: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    trace_id: str = ""
    result_status: str = ""
    result_count: int | None = None
    candidate_symbols: list[str] = Field(default_factory=list)


class DiscoveryFilterAuditItem(BaseModel):
    """One used/skipped/deferred filter category in a layer audit."""

    category: str
    decision: Literal["used", "skipped", "deferred_to_later_enrichment"]
    stock_fields: list[str] = Field(default_factory=list)
    filter_summary: str = ""
    rationale: str = ""
    trace_ids: list[str] = Field(default_factory=list)


class DiscoveryLayerFilterAudit(BaseModel):
    """Layer-level audit tying screener choices to resulting candidates."""

    layer_key: str
    layer_name: str = ""
    hypothesis: str = ""
    plate_codes_considered: list[str] = Field(default_factory=list)
    plate_codes_used: list[str] = Field(default_factory=list)
    used_filters: list[DiscoveryFilterAuditItem] = Field(default_factory=list)
    skipped_or_deferred_filters: list[DiscoveryFilterAuditItem] = Field(default_factory=list)
    candidate_symbols_from_probes: list[str] = Field(default_factory=list)
    result_summary: str = ""


class DiscoveryOmission(BaseModel):
    """Structured reason for excluding a discovered symbol from candidates."""

    symbol: str
    layer_key: str = ""
    source_trace_ids: list[str] = Field(default_factory=list)
    exclusion_reason: Literal[
        "invalid_symbol",
        "outside_requested_market",
        "duplicate_share_class",
        "explicit_user_exclusion",
        "failed_liquidity_gate",
        "failed_size_gate",
        "clear_theme_mismatch",
        "unsupported_security_type",
    ]
    explanation: str = ""


class ThemeDiscoveryPlan(BaseModel):
    theme: str
    market: str = "US"
    theme_description: str = ""
    initial_thesis: str = ""
    domain_tree: list[ThemeDomain] = Field(default_factory=list)
    coverage_requirements: list[ThemeCoverageRequirement] = Field(default_factory=list)
    seed_symbols: list[ThemeDiscoverySeed] = Field(default_factory=list)
    plate_keywords: list[str] = Field(default_factory=list)
    benchmark_symbols: list[str] = Field(default_factory=list)
    research_trace: list[ResearchSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    filter_plans_by_layer: list[DiscoveryFilterPlan] = Field(default_factory=list)
    executed_filter_probes: list[ExecutedDiscoveryProbe] = Field(default_factory=list)
    layer_filter_audits: list[DiscoveryLayerFilterAudit] = Field(default_factory=list)
    omissions_to_investigate: list[DiscoveryOmission] = Field(default_factory=list)
    next_enrichment_needed: list[str] = Field(default_factory=list)
    data_asof: dict[str, str] = Field(default_factory=dict)
    discovery_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class CalibrationInputProbe(BaseModel):
    """LLM-authored screener idea that should be calibrated against Futu."""

    name: str
    rationale: str = ""
    signal_type: Literal[
        "market_quote",
        "valuation",
        "dividend",
        "technical",
        "financial",
        "analysis",
        "options",
        "other",
    ] = "other"
    source_categories: list[str] = Field(default_factory=list)
    plate_code: str | None = None
    stock_filter_specs: list[dict[str, Any]] = Field(default_factory=list)
    result_limit: int = Field(default=80, ge=1, le=200)
    focus_symbols: list[str] = Field(default_factory=list)


class CalibrationTrial(BaseModel):
    """Concrete Futu get_stock_filter trial generated from an input probe."""

    trial_id: str
    probe_name: str
    rationale: str = ""
    mode: Literal["baseline", "relaxed", "threshold", "strict", "custom"] = "custom"
    plate_code: str | None = None
    stock_filter_specs: list[dict[str, Any]] = Field(default_factory=list)
    result_limit: int = Field(default=80, ge=1, le=200)


class CalibrationTrialResult(BaseModel):
    """Deterministic execution result for a single Futu filter calibration trial."""

    trial_id: str
    probe_name: str
    status: Literal["completed", "failed"] = "completed"
    diagnosis: Literal["zero_result", "too_narrow", "too_broad", "usable", "error"] = "usable"
    all_count: int = Field(default=0, ge=0)
    returned_count: int = Field(default=0, ge=0)
    sample_symbols: list[str] = Field(default_factory=list)
    sample_names: dict[str, str] = Field(default_factory=dict)
    focus_symbols_included: list[str] = Field(default_factory=list)
    focus_symbols_missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class CalibratedFilter(BaseModel):
    """Agent-selected calibration outcome for one input probe."""

    probe_name: str
    selected_mode: Literal["calibrated_filter", "rank_then_score", "skip_probe"]
    selected_trial_id: str = ""
    selected_filters: list[dict[str, Any]] = Field(default_factory=list)
    result_limit: int = Field(default=80, ge=1, le=200)
    selection_reason: str = ""
    rejected_trial_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FilterCalibrationArtifact(BaseModel):
    """Auditable artifact produced by the filter-calibration stage."""

    theme: str
    market: str = "US"
    generated_at: str = ""
    input_probe_count: int = 0
    probes: list[CalibrationInputProbe] = Field(default_factory=list)
    trials: list[CalibrationTrial] = Field(default_factory=list)
    trial_results: list[CalibrationTrialResult] = Field(default_factory=list)
    calibrated_filters: list[CalibratedFilter] = Field(default_factory=list)
    focus_symbol_audit: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class FutuQuoteData(BaseModel):
    """Normalized Futu market snapshot fields used by portfolio agents."""

    update_time: str = ""
    last_price: float | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    prev_close_price: float | None = None
    turnover: float | None = None
    volume: float | None = None
    change_rate: float | None = None
    turnover_rate: float | None = None
    amplitude: float | None = None
    volume_ratio: float | None = None
    bid_ask_ratio: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    total_market_val: float | None = None
    circular_market_val: float | None = None
    pe_ratio: float | None = None
    pe_ttm_ratio: float | None = None
    pb_ratio: float | None = None
    ey_ratio: float | None = None
    net_asset: float | None = None
    net_profit: float | None = None
    earning_per_share: float | None = None
    net_asset_per_share: float | None = None
    outstanding_shares: float | None = None
    highest52weeks_price: float | None = None
    lowest52weeks_price: float | None = None
    dividend_ttm: float | None = None
    dividend_ratio_ttm: float | None = None
    enable_margin: str = ""
    enable_short_sell: str = ""
    short_available_volume: float | None = None
    short_sell_rate: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FutuTechnicalData(BaseModel):
    """Normalized Futu k-line derived technical fields."""

    trend: str = ""
    relative_strength_60d: float | None = None
    realized_volatility: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None
    daily_returns_60d: list[float] = Field(default_factory=list)
    daily_kline_rows: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FutuLiquidityData(BaseModel):
    """Normalized liquidity and tradability fields from Futu data."""

    turnover: float | None = None
    volume: float | None = None
    volume_ratio: float | None = None
    turnover_rate: float | None = None
    bid_ask_ratio: float | None = None
    spread_bps: float | None = None
    liquidity_score: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FutuOptionsData(BaseModel):
    """Normalized option surface summary from Futu option APIs."""

    has_option_data: bool = False
    status: str = ""
    iv_rank_proxy: float | None = None
    avg_implied_volatility: float | None = None
    avg_spread_bps: float | None = None
    contracts_sampled: int | None = None
    call_count: int | None = None
    put_count: int | None = None
    expiration: dict[str, Any] = Field(default_factory=dict)
    call_candidates: list[dict[str, Any]] = Field(default_factory=list)
    put_candidates: list[dict[str, Any]] = Field(default_factory=list)
    data_asof: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class FutuData(BaseModel):
    """All Futu-derived candidate evidence grouped under one sub schema."""

    source: str = "futu_opend"
    quote: FutuQuoteData | None = None
    technical: FutuTechnicalData | None = None
    liquidity: FutuLiquidityData | None = None
    options: FutuOptionsData | None = None
    data_asof: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        last_price: float | None,
        quote: dict[str, Any],
        technical: dict[str, Any],
        liquidity: dict[str, Any],
        options: dict[str, Any],
        data_asof: dict[str, str],
        score_breakdown: dict[str, float] | None = None,
    ) -> "FutuData":
        quote_payload = {"last_price": last_price, **(quote or {})}
        options_payload = options or {}
        return cls(
            quote=FutuQuoteData(**quote_payload, raw=dict(quote or {}))
            if quote_payload
            else None,
            technical=FutuTechnicalData(**(technical or {}), raw=dict(technical or {}))
            if technical
            else None,
            liquidity=FutuLiquidityData(**(liquidity or {}), raw=dict(liquidity or {}))
            if liquidity
            else None,
            options=FutuOptionsData(
                has_option_data=bool(options_payload.get("has_option_data", False)),
                status=str(options_payload.get("status") or ""),
                iv_rank_proxy=options_payload.get("iv_rank_proxy"),
                avg_implied_volatility=options_payload.get("avg_implied_volatility"),
                avg_spread_bps=options_payload.get("avg_spread_bps"),
                contracts_sampled=options_payload.get("contracts_sampled"),
                call_count=options_payload.get("call_count"),
                put_count=options_payload.get("put_count"),
                expiration=options_payload.get("expiration") or {},
                call_candidates=options_payload.get("call_candidates") or [],
                put_candidates=options_payload.get("put_candidates") or [],
                data_asof=str(options_payload.get("data_asof") or ""),
                raw=dict(options_payload),
            )
            if options_payload
            else None,
            data_asof=data_asof or {},
            score_breakdown=score_breakdown or {},
        )


class DiscoveryData(BaseModel):
    """AI-authored discovery context that explains why a symbol entered the pool."""

    source: str = "pydantic_ai_theme_discovery"
    role: str
    rationale: str = ""
    subthemes: list[str] = Field(default_factory=list)
    value_chain_stage: str = ""
    exposure_type: str = ""
    exposure_purity: str = ""
    source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    freshness: Literal["fresh", "partial", "stale", "unknown"] = "unknown"


class SecData(BaseModel):
    """SEC filing metadata for a candidate."""

    source: str = "edgartools"
    source_status: str = "not_available"
    ticker: str | None = None
    cik: str | None = None
    company_name: str | None = None
    industry: str | None = None
    fiscal_year_end: str | None = None
    filer_category: str | None = None
    latest_10k: dict[str, Any] | None = None
    latest_10q: dict[str, Any] | None = None
    latest_8k: dict[str, Any] | None = None
    event_context: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    error: str | None = None


class FundamentalData(BaseModel):
    """Candidate-level fundamental summary from Futu snapshot and SEC data."""

    source_status: str = "not_available"
    sec_source_status: str = "not_available"
    numeric_source: str = "not_available"
    numeric_llm_generated: bool = False
    net_profit: float | None = None
    eps: float | None = None
    net_asset_per_share: float | None = None
    dividend_yield_ttm: float | None = None
    ttm_revenue: float | None = None
    ttm_net_income: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    shareholders_equity: float | None = None
    debt_to_assets: float | None = None
    roe: float | None = None
    net_margin: float | None = None
    quality_score: float | None = None


class EventData(BaseModel):
    """Candidate-level event and filing freshness summary."""

    source_status: str = "not_available"
    next_earnings_date: str | None = None
    days_to_earnings: int | None = None
    event_risk_level: str = "unknown"
    latest_10k: dict[str, Any] | None = None
    latest_10q: dict[str, Any] | None = None
    latest_8k: dict[str, Any] | None = None
    latest_periodic_filing_date: str | None = None
    periodic_filing_age_days: int | None = None
    periodic_filing_stale: bool | None = None
    known_risk_tags: list[str] = Field(default_factory=list)


class CandidateDataQuality(BaseModel):
    """Candidate-level data freshness and completeness summary."""

    freshness: Literal["fresh", "partial", "stale", "unavailable"] = "partial"
    quote_asof: str | None = None
    kline_asof: str | None = None
    sec_asof: str | None = None
    options_asof: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    symbol: str
    name: str
    theme_role: str
    source: str
    source_tags: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=100)
    candidate_status: Literal[
        "discovered",
        "futu_enriched",
        "sec_enriched",
        "quote_unavailable",
        "excluded",
    ] = "discovered"
    eligible_for_portfolio: bool = False
    exclusion_reasons: list[str] = Field(default_factory=list)
    data_quality: CandidateDataQuality = Field(default_factory=CandidateDataQuality)
    discovery_data: DiscoveryData
    futu_data: FutuData
    sec_data: SecData | None = None
    fundamental_data: FundamentalData | None = None
    event_data: EventData | None = None
    plate_memberships: list[dict[str, str]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list)


class CandidatePool(BaseModel):
    theme: str
    generated_from: list[str]
    candidates: list[Candidate]
    discovery_thesis: str = ""
    coverage_requirements: list[ThemeCoverageRequirement] = Field(default_factory=list)
    research_trace: list[ResearchSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    data_asof: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CandidatePoolReflection(BaseModel):
    coverage_ok: bool
    missing_exposure: list[str] = Field(default_factory=list)
    stale_data: list[str] = Field(default_factory=list)
    weak_sources: list[str] = Field(default_factory=list)
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateThesisAssessment(BaseModel):
    symbol: str
    role: str = ""
    thesis_fit: Literal["high", "medium", "low", "avoid"] = "medium"
    recommended_action: Literal["include", "watch", "omit"] = "watch"
    evidence_summary: list[str] = Field(default_factory=list)
    metrics_considered: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    substitute_symbols: list[str] = Field(default_factory=list)


class ThesisSynthesis(BaseModel):
    theme: str
    primary_thesis: str
    thesis_points: list[str] = Field(default_factory=list)
    key_bottlenecks: list[str] = Field(default_factory=list)
    metrics_considered: list[str] = Field(default_factory=list)
    candidate_assessments: list[CandidateThesisAssessment] = Field(default_factory=list)
    portfolio_implications: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class PortfolioHolding(BaseModel):
    symbol: str
    target_weight: float = Field(ge=0, le=1)
    role: str
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)


class PortfolioSleeve(BaseModel):
    name: str
    role: str
    target_weight: float = Field(ge=0, le=1)
    holding_symbols: list[str] = Field(default_factory=list)
    rationale: str
    risk_notes: list[str] = Field(default_factory=list)


class OmittedCandidate(BaseModel):
    symbol: str
    role: str = ""
    reason: str
    reason_category: Literal[
        "overlap",
        "weak_evidence",
        "valuation",
        "momentum",
        "data_quality",
        "weight_budget",
        "scope",
        "risk",
        "other",
    ] = "other"
    substitute_symbols: list[str] = Field(default_factory=list)
    importance: Literal["critical", "high", "medium", "low"] = "medium"


class PortfolioMap(BaseModel):
    map_id: str
    name: str
    objective: Literal["balanced", "growth", "income"]
    sleeve_weight: float = Field(ge=0, le=1)
    positioning: str = ""
    best_for: str = ""
    allocation_logic: list[str] = Field(default_factory=list)
    sleeves: list[PortfolioSleeve] = Field(default_factory=list)
    holdings: list[PortfolioHolding]
    cash_weight: float = Field(ge=0, le=1)
    thesis: str
    risks: list[str] = Field(default_factory=list)
    missing_exposure: list[str] = Field(default_factory=list)
    reflection_notes: list[str] = Field(default_factory=list)
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)


class PortfolioMaps(BaseModel):
    theme: str
    maps: list[PortfolioMap]
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CurrentHolding(BaseModel):
    symbol: str
    quantity: int = Field(ge=0)
    market_value: float = Field(ge=0)
    cost_basis: float | None = None
    can_sell_qty: int = Field(default=0, ge=0)


class CurrentPortfolio(BaseModel):
    total_assets: float = Field(gt=0)
    cash: float = Field(ge=0)
    holdings: list[CurrentHolding] = Field(default_factory=list)
    data_asof: str
    source: str
    warnings: list[str] = Field(default_factory=list)


class StockTranche(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(ge=0)
    limit_price: float = Field(gt=0)
    trigger: str
    invalidation: str
    estimated_value: float = Field(ge=0)


class SimulatedOrder(BaseModel):
    code: str
    side: Literal["BUY", "SELL"]
    quantity: int = Field(ge=0)
    price: float = Field(gt=0)
    order_type: str = "NORMAL"
    market: str = "US"
    trd_env: str = "SIMULATE"


class ConstructionPlan(BaseModel):
    plan_id: str
    selected_map_id: str
    generated_at: str
    cash_required: float = Field(ge=0)
    cash_released: float = Field(ge=0)
    post_trade_cash: float
    target_theme_weight: float = Field(ge=0, le=1)
    stock_tranches: list[StockTranche]
    simulated_orders: list[SimulatedOrder]
    invalidation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
