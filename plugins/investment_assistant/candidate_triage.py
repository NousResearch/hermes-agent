"""PydanticAI candidate triage after Futu lightweight enrichment."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .lightweight_enrichment import LightweightEnrichmentArtifact
from .pydantic_runtime import (
    create_pydantic_agent,
    ensure_pydantic_ai_available,
    pydantic_event_stream_handler,
    usage_metadata,
)
from .schemas import ThemeDiscoveryPlan
from .storage import utc_now


_TRIAGE_INSTRUCTIONS = """
You are the investment assistant's candidate triage agent.

Your job is to convert a broad discovery universe plus Futu lightweight
evidence into a research-budget allocation artifact. You are not selecting the
final portfolio. You are deciding where expensive downstream research should be
spent now, what should stay visible in watchlist, what should wait for another
data source, and what can be rejected for hard reasons. You must not produce
portfolio weights, trade plans, price targets, or orders.

Use the evidence boundary:
- Use only the discovery artifact and lightweight Futu artifact.
- Do not invent SEC, earnings, revenue, margin, customer, or news facts.
- You may request SEC/fundamental/event enrichment for candidates, but you do
  not perform it here.

Triage principles:
- Preserve the theme map. Do not collapse important layers just because one
  mega-cap looks strong.
- Treat deep_enrichment_queue as a research-spend queue, not an investable
  shortlist. A symbol enters deep enrichment because it deserves SEC,
  fundamental, event, technical, or options validation before portfolio
  construction, not because it is already approved for weight.
- Allocate research budget by layer and subdomain before selecting individual
  deep-research symbols. Record that allocation in research_budget_allocations.
- Required symbols are constraints and should normally enter deep enrichment or
  ETF-specific follow-up. Do not treat them as quality proof.
- Use Futu quote validity, market cap, turnover, 20d/60d returns, trend,
  volatility, owner plate, and option availability as lightweight evidence.
- Strong price momentum alone is not a final thesis. High-momentum names often
  need deeper validation, not automatic inclusion.
- Keep bottleneck branches visible: memory/storage, optical/networking, power,
  advanced packaging, semicap, cloud/platforms, software/security, and any
  theme-specific branch discovered by the prior agent.
- Treat high-salience bottleneck names as validation targets. If a candidate is
  flagged in high_salience_must_review_symbols, place it in
  deep_enrichment_queue unless there is a hard invalidity reason. "Needs
  validation" is a reason to deep-research it, not a reason to bury it in the
  watchlist.
- Prefer watchlist over rejection for plausible theme candidates with incomplete
  evidence.
- Keep important same-layer peers visible. If a layer has multiple subdomains,
  either assign deep-research budget to a representative of each material
  subdomain or keep the unresearched peer in watchlist/deferred with a concise
  reason. Do not hide plausible peers merely because a stronger same-layer name
  already exists.
- Reject only for hard reasons: invalid/unsupported quote, outside scope,
  clear theme mismatch, duplicate/overlap with no unique exposure, extremely
  weak liquidity/size, or explicit user exclusion.
- If selected_triage_strategy is present in the input, follow it as the user's
  confirmed strategy. Respect explicit user modifications and exclusions unless
  they conflict with required symbols or the evidence boundary.

Return CandidateTriageArtifact only.

Output compactness:
- research_budget_summary and research_budget_allocations should make clear how
  scarce research budget is being allocated across layers.
- deep_enrichment_queue decisions should be detailed.
- watchlist, deferred, rejected, and high_salience_omissions should be compact:
  one concise rationale per symbol, no long evidence lists.
"""


_TRIAGE_PLAN_INSTRUCTIONS = """
You are the investment assistant's candidate triage planning agent.

Your job is to produce a strategy plan for the candidate-triage run. Do not
produce the final deep research queue yet. The primary output is a compact
structured research dataset plan for the future Deep Research Agent.

Use the evidence boundary:
- Use only the discovery artifact and lightweight Futu artifact.
- Do not invent SEC, earnings, revenue, margin, customer, or news facts.
- You may propose what the later deep research should validate.

Planning principles:
- Offer two to four meaningfully different strategy options.
- Options may differ by selection logic, not just by candidate count.
- Let the user choose, modify, or override the strategy before final triage.
- Include layer budgets so the user can see which branches receive research
  attention.
- Keep watchlist as an intentional retained dataset, not a failure.
- Do not produce final portfolio maps, target weights, price targets, orders,
  or trade plans.

Return CandidateTriagePlanArtifact only.
"""


TriageBucket = Literal["deep_enrichment_queue", "watchlist", "defer", "reject"]
DeepResearchNeed = Literal[
    "sec_fundamentals",
    "earnings_call_or_presentation",
    "news_events",
    "etf_holdings",
    "options_surface",
    "technical_review",
    "none",
]


class LayerResearchBudget(BaseModel):
    layer_key: str
    layer_name: str = ""
    deep_research_count: int = Field(default=0, ge=0)
    watchlist_count: int = Field(default=0, ge=0)
    rationale: str = ""


class TriageStrategyOption(BaseModel):
    option_id: str
    name: str
    description: str = ""
    selection_rules: list[str] = Field(default_factory=list)
    deep_research_total: int = Field(default=0, ge=0)
    expected_watchlist_count: int = Field(default=0, ge=0)
    layer_budgets: list[LayerResearchBudget] = Field(default_factory=list)
    best_for: str = ""
    tradeoffs: list[str] = Field(default_factory=list)


class CandidateTriagePlanArtifact(BaseModel):
    artifact_type: str = "candidate_triage_plan"
    theme: str
    market: str = "US"
    generated_at: str = ""
    planning_summary: str = ""
    theme_breadth: Literal["narrow", "medium", "broad", "mega_theme"] = "medium"
    candidate_count: int = 0
    layer_count: int = 0
    strategy_options: list[TriageStrategyOption] = Field(default_factory=list)
    recommended_option_id: str = ""
    requires_user_input: bool = True
    prompt_to_user: str = ""
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class TriageStrategySelection(BaseModel):
    artifact_type: str = "triage_strategy_selection"
    selected_option_id: str = ""
    selected_option: dict[str, Any] = Field(default_factory=dict)
    raw_answer: str = ""
    modifications: str = ""
    must_include_symbols: list[str] = Field(default_factory=list)
    exclude_symbols: list[str] = Field(default_factory=list)
    unmatched_answer_needs_agent_interpretation: bool = False


class TriageCandidateDecision(BaseModel):
    symbol: str
    bucket: TriageBucket
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    layer_keys: list[str] = Field(default_factory=list)
    role: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    research_needs: list[DeepResearchNeed] = Field(default_factory=list)
    rationale: str = ""


class CompactTriageDecision(BaseModel):
    symbol: str
    bucket: Literal["watchlist", "defer", "reject"]
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    layer_keys: list[str] = Field(default_factory=list)
    reason_category: Literal[
        "lower_priority",
        "needs_more_evidence",
        "overlap",
        "weak_theme_fit",
        "weak_liquidity_or_size",
        "data_quality",
        "risk",
        "other",
    ] = "other"
    research_needs: list[DeepResearchNeed] = Field(default_factory=list)
    rationale: str = ""


class TriageLayerAudit(BaseModel):
    layer_key: str
    layer_name: str = ""
    coverage_status: Literal["covered", "thin", "missing", "deferred"] = "thin"
    selected_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)
    missing_or_weak_exposure: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchBudgetAllocation(BaseModel):
    layer_key: str
    layer_name: str = ""
    allocation_goal: Literal[
        "preserve_core_coverage",
        "validate_bottleneck",
        "compare_peers",
        "verify_controversial_or_new_listing",
        "hold_watchlist_only",
        "other",
    ] = "other"
    deep_research_budget: int = Field(default=0, ge=0)
    deep_research_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)
    deferred_symbols: list[str] = Field(default_factory=list)
    rejected_symbols: list[str] = Field(default_factory=list)
    rationale: str = ""


class CandidateTriageArtifact(BaseModel):
    artifact_type: str = "candidate_triage"
    theme: str
    market: str = "US"
    generated_at: str = ""
    triage_summary: str = ""
    research_budget_summary: str = ""
    research_budget_allocations: list[ResearchBudgetAllocation] = Field(default_factory=list)
    deep_enrichment_queue: list[TriageCandidateDecision] = Field(default_factory=list)
    watchlist: list[CompactTriageDecision] = Field(default_factory=list)
    deferred: list[CompactTriageDecision] = Field(default_factory=list)
    rejected: list[CompactTriageDecision] = Field(default_factory=list)
    layer_audits: list[TriageLayerAudit] = Field(default_factory=list)
    high_salience_omissions: list[CompactTriageDecision] = Field(default_factory=list)
    triage_criteria_used: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


def build_candidate_triage_plan(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
) -> CandidateTriagePlanArtifact:
    """Plan candidate triage strategy options and stop for user feedback."""

    allowed_symbols = _allowed_symbols(discovery, lightweight)
    if not allowed_symbols:
        raise ValueError("No candidates available for triage planning.")

    artifact, runtime = _run_triage_plan_agent(discovery, lightweight, allowed_symbols)
    artifact.theme = discovery.theme
    artifact.market = discovery.market
    artifact.generated_at = artifact.generated_at or utc_now()
    artifact.candidate_count = artifact.candidate_count or len(allowed_symbols)
    artifact.layer_count = artifact.layer_count or len(discovery.domain_tree)
    artifact.requires_user_input = True
    artifact.pydantic_ai = runtime
    return artifact


def build_candidate_triage_artifact(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    triage_strategy: dict[str, Any] | TriageStrategySelection | None = None,
) -> CandidateTriageArtifact:
    """Run the AI triage agent over discovery + Futu lightweight evidence."""

    allowed_symbols = _allowed_symbols(discovery, lightweight)
    if not allowed_symbols:
        raise ValueError("No candidates available for triage.")

    strategy_payload = (
        triage_strategy.model_dump(mode="json")
        if isinstance(triage_strategy, TriageStrategySelection)
        else (triage_strategy or {})
    )
    artifact, runtime = _run_triage_agent(discovery, lightweight, allowed_symbols, strategy_payload)
    artifact.theme = discovery.theme
    artifact.market = discovery.market
    artifact.generated_at = artifact.generated_at or utc_now()
    artifact.pydantic_ai = runtime
    return artifact


def select_triage_strategy(
    plan: CandidateTriagePlanArtifact,
    *,
    option_id: str = "",
    answer: str = "",
    modifications: str = "",
    must_include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
    market: str | None = None,
) -> TriageStrategySelection:
    """Create a typed user strategy selection from a plan and optional free text."""

    options = [option.model_dump(mode="json") for option in plan.strategy_options]
    option_ids = [str(option["option_id"]) for option in options if option.get("option_id")]
    selected = str(option_id or "").strip()
    raw_text = str(answer or "").strip()
    if not selected and raw_text:
        selected = _option_id_from_text(raw_text, option_ids)
    selected_option = next((option for option in options if option.get("option_id") == selected), None)
    selected_market = market or plan.market or "US"
    return TriageStrategySelection(
        selected_option_id=selected,
        selected_option=selected_option or {},
        raw_answer=raw_text,
        modifications=str(modifications or "").strip(),
        must_include_symbols=_normalize_symbols(must_include_symbols or [], selected_market),
        exclude_symbols=_normalize_symbols(exclude_symbols or [], selected_market),
        unmatched_answer_needs_agent_interpretation=bool(raw_text and not selected),
    )


def build_candidate_triage_from_plan(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    plan: CandidateTriagePlanArtifact,
    *,
    option_id: str = "",
    answer: str = "",
    modifications: str = "",
    must_include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
) -> tuple[CandidateTriageArtifact, TriageStrategySelection]:
    """Resume triage from an existing plan artifact without rerunning prior stages."""

    _validate_triage_plan(plan, discovery, _allowed_symbols(discovery, lightweight))
    selection = select_triage_strategy(
        plan,
        option_id=option_id,
        answer=answer,
        modifications=modifications,
        must_include_symbols=must_include_symbols,
        exclude_symbols=exclude_symbols,
        market=discovery.market or lightweight.market or plan.market,
    )
    artifact = build_candidate_triage_artifact(discovery, lightweight, triage_strategy=selection)
    return artifact, selection


def _run_triage_plan_agent(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    allowed_symbols: list[str],
) -> tuple[CandidateTriagePlanArtifact, dict[str, Any]]:
    ModelRetry = _model_retry_class()

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=CandidateTriagePlanArtifact,
        instructions=_TRIAGE_PLAN_INSTRUCTIONS,
        agent_kind="candidate_triage_plan_agent",
        output_retries=2,
        agent_skill_names=["candidate-triage"],
    )

    required_symbols = _required_symbols(discovery)
    high_salience_symbols = _high_salience_must_review_symbols(lightweight)

    @agent.output_validator
    def validate_triage_plan(data: CandidateTriagePlanArtifact) -> CandidateTriagePlanArtifact:
        try:
            _validate_triage_plan(data, discovery, allowed_symbols)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return data

    payload = _triage_plan_payload(discovery, lightweight, allowed_symbols, required_symbols, high_salience_symbols)
    result = agent.run_sync(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("candidate_triage_plan_agent"),
    )
    return result.output, {**runtime, "usage": usage_metadata(result), "input": payload}


def _run_triage_agent(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    allowed_symbols: list[str],
    triage_strategy: dict[str, Any] | None = None,
) -> tuple[CandidateTriageArtifact, dict[str, Any]]:
    ModelRetry = _model_retry_class()

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=CandidateTriageArtifact,
        instructions=_TRIAGE_INSTRUCTIONS,
        agent_kind="candidate_triage_agent",
        output_retries=2,
        agent_skill_names=["candidate-triage"],
    )

    required_symbols = _required_symbols(discovery)
    high_salience_symbols = _high_salience_must_review_symbols(lightweight)

    @agent.output_validator
    def validate_triage(data: CandidateTriageArtifact) -> CandidateTriageArtifact:
        try:
            _validate_triage(data, allowed_symbols, required_symbols, high_salience_symbols)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return data

    payload = _triage_payload(
        discovery,
        lightweight,
        allowed_symbols,
        required_symbols,
        high_salience_symbols,
        triage_strategy or {},
    )
    result = agent.run_sync(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("candidate_triage_agent"),
    )
    return result.output, {**runtime, "usage": usage_metadata(result), "input": payload}


def _model_retry_class():
    ensure_pydantic_ai_available()
    from pydantic_ai import ModelRetry

    return ModelRetry


def _triage_payload(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    allowed_symbols: list[str],
    required_symbols: list[str],
    high_salience_symbols: list[str],
    triage_strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_symbol = {item.symbol: item for item in lightweight.candidates}
    deep_target_min = _env_int("IA_TRIAGE_DEEP_MIN", 25)
    deep_target_max = _env_int("IA_TRIAGE_DEEP_MAX", 40)
    return {
        "task": "Triage candidates into research queues before SEC/fundamental deep enrichment.",
        "selected_triage_strategy": triage_strategy or {},
        "theme": discovery.theme,
        "market": discovery.market,
        "theme_description": discovery.theme_description,
        "initial_thesis": discovery.initial_thesis,
        "allowed_symbols": allowed_symbols,
        "required_symbols": required_symbols,
        "high_salience_must_review_symbols": [
            _candidate_payload(symbol, discovery, by_symbol.get(symbol)) for symbol in high_salience_symbols
        ],
        "target_queue_sizes": {
            "deep_enrichment_queue_min": deep_target_min,
            "deep_enrichment_queue_max": deep_target_max,
            "watchlist_guidance": "Keep plausible but lower-priority names visible instead of rejecting them.",
            "budget_intent": (
                "Treat deep_enrichment_queue as expensive downstream research spend, not as final stock selection. "
                "Use research_budget_allocations to show the per-layer budget and retained watchlist peers."
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
        "candidate_evidence": [
            _candidate_payload(symbol, discovery, by_symbol.get(symbol)) for symbol in allowed_symbols
        ],
        "lightweight_check_summary": lightweight.check_summary,
        "research_trace_summary": [
            {
                "source_id": source.source_id,
                "title": source.title,
                "publisher": source.publisher,
                "source_type": source.source_type,
                "symbols": source.symbols[:12],
                "coverage_keys": source.coverage_keys[:8],
                "summary": _short(source.summary, 360),
            }
            for source in discovery.research_trace[:25]
        ],
        "warnings": [*discovery.warnings, *lightweight.warnings],
        "output_contract": (
            "Return CandidateTriageArtifact only. Use only allowed_symbols. "
            "Fill research_budget_summary and research_budget_allocations before the decision buckets. "
            "research_budget_allocations must reconcile each layer's deep_research_budget with "
            "deep_research_symbols and must preserve important peers in watchlist/deferred/rejected when not deep. "
            "Make deep_enrichment_queue detailed with evidence_summary and concerns. "
            "Make watchlist, deferred, rejected, and high_salience_omissions compact: "
            "symbol, bucket, priority, layer_keys, reason_category, research_needs, rationale only. "
            "Do not create portfolio maps, target weights, trade plans, price targets, or orders."
        ),
    }


def _triage_plan_payload(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    allowed_symbols: list[str],
    required_symbols: list[str],
    high_salience_symbols: list[str],
) -> dict[str, Any]:
    by_symbol = {item.symbol: item for item in lightweight.candidates}
    return {
        "task": (
            "Plan candidate-triage strategy options for user confirmation. "
            "Do not produce the final candidate triage buckets yet."
        ),
        "theme": discovery.theme,
        "market": discovery.market,
        "theme_description": discovery.theme_description,
        "initial_thesis": discovery.initial_thesis,
        "candidate_count": len(allowed_symbols),
        "layer_count": len(discovery.domain_tree),
        "allowed_symbols": allowed_symbols,
        "required_symbols": required_symbols,
        "high_salience_must_review_symbols": [
            _candidate_payload(symbol, discovery, by_symbol.get(symbol)) for symbol in high_salience_symbols
        ],
        "domain_tree": [
            {
                "key": domain.key,
                "name": domain.name,
                "importance": domain.importance,
                "thesis": _short(domain.thesis, 500),
                "candidate_symbols": _domain_candidate_symbols(domain),
            }
            for domain in discovery.domain_tree
        ],
        "coverage_requirements": [
            requirement.model_dump(mode="json") for requirement in discovery.coverage_requirements
        ],
        "candidate_evidence": [
            _candidate_payload(symbol, discovery, by_symbol.get(symbol)) for symbol in allowed_symbols
        ],
        "lightweight_check_summary": lightweight.check_summary,
        "strategy_option_guidance": {
            "option_count": "Return 2-4 strategy options.",
            "option_examples": [
                "coverage_balanced: preserve complete value-chain coverage.",
                "bottleneck_momentum: prioritize bottleneck branches with strong Futu momentum/liquidity.",
                "quality_leaders: prioritize liquid profitable leaders and reduce speculative candidates.",
                "event_validation: prioritize controversial or high-ambiguity names that require filings/news validation.",
            ],
            "user_feedback_handling": (
                "The user may select an option, modify layer budgets, set must-include or exclude constraints, "
                "or provide natural-language preferences. The final triage run should wait for that feedback."
            ),
        },
        "warnings": [*discovery.warnings, *lightweight.warnings],
        "output_contract": (
            "Return CandidateTriagePlanArtifact only. Include 2-4 strategy_options with different "
            "selection rules, deep_research_total, expected_watchlist_count, layer_budgets, best_for, "
            "and tradeoffs. recommended_option_id must match one option_id. requires_user_input must be true. "
            "Do not produce final deep_enrichment_queue/watchlist/reject buckets."
        ),
    }


def _candidate_payload(symbol: str, discovery: ThemeDiscoveryPlan, evidence) -> dict[str, Any]:
    seed = next((item for item in discovery.seed_symbols if item.symbol == symbol), None)
    payload = {
        "symbol": symbol,
        "role": seed.role if seed else "",
        "rationale": _short(seed.rationale, 420) if seed else "",
        "subthemes": seed.subthemes if seed else [],
        "value_chain_stage": seed.value_chain_stage if seed else "",
        "confidence": seed.confidence if seed else "",
        "source_ids": seed.source_ids if seed else [],
        "layers": evidence.layers if evidence else [],
    }
    if evidence is None:
        payload["futu"] = {"status": "missing"}
        return payload
    payload["futu"] = {
        "name": evidence.name,
        "security_type": evidence.security_type,
        "quote_status": evidence.quote_status,
        "kline_status": evidence.kline_status,
        "owner_plate_status": evidence.owner_plate_status,
        "option_status": evidence.option_status,
        "last_price": evidence.last_price,
        "market_cap": evidence.total_market_val,
        "turnover": evidence.turnover,
        "pe_ttm_ratio": evidence.pe_ttm_ratio,
        "pb_ratio": evidence.pb_ratio,
        "return_20d": evidence.return_20d,
        "return_60d": evidence.return_60d,
        "trend": evidence.trend,
        "relative_strength_60d": evidence.relative_strength_60d,
        "realized_volatility": evidence.realized_volatility,
        "liquidity_score": evidence.liquidity_score,
        "plate_memberships": evidence.plate_memberships[:8],
        "has_option_data": evidence.has_option_data,
        "warnings": evidence.warnings,
    }
    return payload


def _validate_triage(
    artifact: CandidateTriageArtifact,
    allowed_symbols: list[str],
    required_symbols: list[str],
    high_salience_symbols: list[str],
) -> None:
    allowed = set(allowed_symbols)
    buckets = [
        *artifact.deep_enrichment_queue,
        *artifact.watchlist,
        *artifact.deferred,
        *artifact.rejected,
        *artifact.high_salience_omissions,
    ]
    seen: dict[str, str] = {}
    for item in buckets:
        if item.symbol not in allowed:
            raise ValueError(f"Triage referenced symbol outside allowed_symbols: {item.symbol}")
        if item.symbol in seen and item.bucket != "reject":
            raise ValueError(f"Triage duplicated {item.symbol} across decision buckets.")
        seen[item.symbol] = item.bucket
        if not item.rationale:
            raise ValueError(f"Triage decision for {item.symbol} omitted rationale.")
        if isinstance(item, TriageCandidateDecision) and not item.evidence_summary:
            raise ValueError(f"Deep triage decision for {item.symbol} omitted evidence_summary.")
    rejected_required = sorted(
        item.symbol for item in artifact.rejected if item.symbol in set(required_symbols)
    )
    if rejected_required:
        raise ValueError(f"Required symbols cannot be rejected in triage: {rejected_required}")
    deep_symbols = {item.symbol for item in artifact.deep_enrichment_queue}
    missing_high_salience = sorted(set(high_salience_symbols) - deep_symbols)
    if missing_high_salience:
        raise ValueError(
            "High-salience bottleneck candidates must be in deep_enrichment_queue "
            f"for validation: {missing_high_salience}"
        )
    deep_count = len(artifact.deep_enrichment_queue)
    deep_min = _env_int("IA_TRIAGE_DEEP_MIN", 25)
    deep_max = _env_int("IA_TRIAGE_DEEP_MAX", 40)
    if deep_count < deep_min or deep_count > deep_max:
        raise ValueError(
            f"deep_enrichment_queue size {deep_count} outside target range {deep_min}-{deep_max}."
        )
    if not artifact.research_budget_summary:
        raise ValueError("Triage must include research_budget_summary.")
    if not artifact.research_budget_allocations:
        raise ValueError("Triage must include research_budget_allocations.")
    _validate_research_budget_allocations(artifact, allowed)
    if not artifact.layer_audits:
        raise ValueError("Triage must include layer_audits.")
    watch_symbols = {item.symbol for item in artifact.watchlist}
    for audit in artifact.layer_audits:
        unknown = sorted((set(audit.selected_symbols) | set(audit.watchlist_symbols)) - allowed)
        if unknown:
            raise ValueError(f"Layer audit {audit.layer_key!r} referenced unknown symbols: {unknown[:10]}")
        if audit.coverage_status == "covered" and not (deep_symbols & set(audit.selected_symbols)):
            raise ValueError(f"Covered layer {audit.layer_key!r} has no selected deep-enrichment symbol.")
        if audit.coverage_status in {"thin", "deferred"} and not (
            deep_symbols & set(audit.selected_symbols) or watch_symbols & set(audit.watchlist_symbols)
        ):
            raise ValueError(f"Layer audit {audit.layer_key!r} lacks selected/watchlist support.")


def _validate_research_budget_allocations(artifact: CandidateTriageArtifact, allowed: set[str]) -> None:
    deep_symbols = {item.symbol for item in artifact.deep_enrichment_queue}
    watch_symbols = {item.symbol for item in artifact.watchlist}
    deferred_symbols = {item.symbol for item in artifact.deferred}
    rejected_symbols = {item.symbol for item in artifact.rejected}

    allocation_deep_symbols: list[str] = []
    for allocation in artifact.research_budget_allocations:
        if not allocation.layer_key:
            raise ValueError("Research budget allocation omitted layer_key.")
        if not allocation.rationale:
            raise ValueError(f"Research budget allocation {allocation.layer_key!r} omitted rationale.")
        if allocation.deep_research_budget != len(allocation.deep_research_symbols):
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} deep_research_budget "
                f"{allocation.deep_research_budget} does not equal deep_research_symbols count "
                f"{len(allocation.deep_research_symbols)}."
            )
        referenced = (
            set(allocation.deep_research_symbols)
            | set(allocation.watchlist_symbols)
            | set(allocation.deferred_symbols)
            | set(allocation.rejected_symbols)
        )
        unknown = sorted(referenced - allowed)
        if unknown:
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} referenced unknown symbols: {unknown[:10]}"
            )
        unexpected_deep = sorted(set(allocation.deep_research_symbols) - deep_symbols)
        if unexpected_deep:
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} has deep symbols outside "
                f"deep_enrichment_queue: {unexpected_deep[:10]}"
            )
        unexpected_watch = sorted(set(allocation.watchlist_symbols) - watch_symbols)
        if unexpected_watch:
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} has watchlist symbols outside "
                f"watchlist: {unexpected_watch[:10]}"
            )
        unexpected_deferred = sorted(set(allocation.deferred_symbols) - deferred_symbols)
        if unexpected_deferred:
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} has deferred symbols outside "
                f"deferred: {unexpected_deferred[:10]}"
            )
        unexpected_rejected = sorted(set(allocation.rejected_symbols) - rejected_symbols)
        if unexpected_rejected:
            raise ValueError(
                f"Research budget allocation {allocation.layer_key!r} has rejected symbols outside "
                f"rejected: {unexpected_rejected[:10]}"
            )
        allocation_deep_symbols.extend(allocation.deep_research_symbols)

    duplicate_deep = sorted({symbol for symbol in allocation_deep_symbols if allocation_deep_symbols.count(symbol) > 1})
    if duplicate_deep:
        raise ValueError(f"Research budget allocations duplicate deep symbols: {duplicate_deep[:10]}")
    missing_deep = sorted(deep_symbols - set(allocation_deep_symbols))
    if missing_deep:
        raise ValueError(
            "Research budget allocations must account for every deep_enrichment_queue symbol: "
            f"{missing_deep[:10]}"
        )


def _validate_triage_plan(
    artifact: CandidateTriagePlanArtifact,
    discovery: ThemeDiscoveryPlan,
    allowed_symbols: list[str],
) -> None:
    if artifact.artifact_type != "candidate_triage_plan":
        raise ValueError("Triage plan must have artifact_type='candidate_triage_plan'.")
    if not artifact.requires_user_input:
        raise ValueError("Triage plan must require user input before final triage.")
    if not artifact.prompt_to_user:
        raise ValueError("Triage plan must include prompt_to_user.")
    option_count = len(artifact.strategy_options)
    if option_count < 2 or option_count > 4:
        raise ValueError(f"Triage plan must include 2-4 strategy_options, got {option_count}.")
    option_ids = [option.option_id for option in artifact.strategy_options]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("Triage strategy option_id values must be unique.")
    if artifact.recommended_option_id not in set(option_ids):
        raise ValueError("recommended_option_id must match one strategy option_id.")
    layer_keys = {domain.key for domain in discovery.domain_tree if domain.key}
    required_layer_keys = {
        domain.key
        for domain in discovery.domain_tree
        if domain.key and domain.importance in {"core", "important"}
    }
    totals = set()
    rule_sets = set()
    for option in artifact.strategy_options:
        if not option.name:
            raise ValueError(f"Triage option {option.option_id!r} omitted name.")
        if not option.selection_rules:
            raise ValueError(f"Triage option {option.option_id!r} omitted selection_rules.")
        if option.deep_research_total <= 0:
            raise ValueError(f"Triage option {option.option_id!r} must have positive deep_research_total.")
        budget_total = sum(item.deep_research_count for item in option.layer_budgets)
        if budget_total != option.deep_research_total:
            raise ValueError(
                f"Triage option {option.option_id!r} deep_research_total {option.deep_research_total} "
                f"does not equal layer budget sum {budget_total}."
            )
        budget_layer_keys = {item.layer_key for item in option.layer_budgets}
        unknown_layers = sorted(budget_layer_keys - layer_keys)
        if unknown_layers:
            raise ValueError(f"Triage option {option.option_id!r} referenced unknown layers: {unknown_layers[:10]}")
        missing_required_layers = sorted(required_layer_keys - budget_layer_keys)
        if missing_required_layers:
            raise ValueError(
                f"Triage option {option.option_id!r} omitted core/important layer budgets: {missing_required_layers[:10]}"
            )
        if not option.best_for:
            raise ValueError(f"Triage option {option.option_id!r} omitted best_for.")
        if not option.tradeoffs:
            raise ValueError(f"Triage option {option.option_id!r} omitted tradeoffs.")
        totals.add(option.deep_research_total)
        rule_sets.add(tuple(option.selection_rules))
    if len(totals) == 1 and len(rule_sets) == 1:
        raise ValueError("Triage strategy options must differ by count or selection rules.")


def _allowed_symbols(discovery: ThemeDiscoveryPlan, lightweight: LightweightEnrichmentArtifact) -> list[str]:
    symbols: list[str] = []
    for seed in discovery.seed_symbols:
        if seed.symbol and seed.symbol not in symbols:
            symbols.append(seed.symbol)
    for item in lightweight.candidates:
        if item.symbol and item.symbol not in symbols:
            symbols.append(item.symbol)
    return symbols


def _option_id_from_text(text: str, option_ids: list[str]) -> str:
    lowered = text.lower()
    for option_id in option_ids:
        if option_id.lower() in lowered:
            return option_id
    match = re.search(r"(?:选|选择|option|方案|策略)\s*([1-4])", lowered)
    if match:
        index = int(match.group(1)) - 1
    else:
        stripped = lowered.strip()
        if stripped not in {"1", "2", "3", "4"}:
            return ""
        index = int(stripped) - 1
    if 0 <= index < len(option_ids):
        return option_ids[index]
    return ""


def _normalize_symbols(raw: Any, market: str = "US") -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = re.split(r"[,，/、\s()（）]+", raw)
    elif isinstance(raw, list | tuple | set):
        values = []
        for item in raw:
            values.extend(_normalize_symbols(item, market))
    else:
        values = [str(raw)]

    prefix = f"{str(market or 'US').upper()}."
    seen: set[str] = set()
    symbols: list[str] = []
    for value in values:
        text = str(value or "").strip().upper()
        if not text:
            continue
        symbol = text if "." in text else f"{prefix}{text}"
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _required_symbols(discovery: ThemeDiscoveryPlan) -> list[str]:
    required: list[str] = []
    for seed in discovery.seed_symbols:
        role = " ".join([seed.role, *seed.subthemes]).lower()
        if "required" in role and seed.symbol not in required:
            required.append(seed.symbol)
    return required


def _domain_candidate_symbols(domain) -> list[str]:
    symbols: list[str] = []
    for subdomain in domain.subdomains:
        for candidate in subdomain.candidates:
            if candidate.symbol and candidate.symbol not in symbols:
                symbols.append(candidate.symbol)
    return symbols


def _high_salience_must_review_symbols(lightweight: LightweightEnrichmentArtifact) -> list[str]:
    candidates = []
    for item in lightweight.candidates:
        layers = " ".join(item.layers).lower()
        plates = " ".join(
            str(plate.get("plate_name", "")) for plate in item.plate_memberships
        ).lower()
        bottleneck = any(
            key in f"{layers} {plates}"
            for key in (
                "memory",
                "storage",
                "hbm",
                "optical",
                "networking",
                "interconnect",
                "connectivity",
                "power",
                "cooling",
                "electrical",
                "grid",
                "utility",
                "energy",
                "存储",
                "光通信",
                "电气",
                "电力",
                "核电",
                "储能",
            )
        )
        if not bottleneck:
            continue
        if item.quote_status != "ok" or item.kline_status != "ok":
            continue
        if not isinstance(item.return_60d, int | float) or not isinstance(item.turnover, int | float):
            continue
        if item.return_60d < 0.40 or item.turnover < 1_000_000_000:
            continue
        candidates.append((item.return_60d, item.turnover, item.symbol))
    candidates.sort(reverse=True)
    return [symbol for _ret, _turnover, symbol in candidates[:8]]


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
