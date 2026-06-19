"""AI-scored, deterministic portfolio weight allocation experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .schemas import PortfolioMap
from .storage import new_id, utc_now

WEIGHT_FORMULA_CONTEXT_FILENAME = "weight_formula_context.json"
WEIGHT_FORMULA_SCORING_FILENAME = "weight_formula_scoring.json"
WEIGHT_FORMULA_ALLOCATION_FILENAME = "weight_formula_allocation.json"
WEIGHT_FORMULA_RUN_FILENAME = "weight_formula_run.json"
DEFAULT_PRECISION = 0.001
PortfolioStyle = Literal["balanced", "conviction", "bottleneck_barbell", "concentrated_growth"]

PORTFOLIO_STYLE_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "label": "Balanced",
        "sleeve_score_exponent": 1.0,
        "candidate_score_exponent": 1.0,
        "guidance": (
            "Use broad AI theme coverage. Keep meaningful sleeves visible when evidence supports them, "
            "and avoid turning the map into a narrow single-bottleneck bet."
        ),
    },
    "conviction": {
        "label": "Conviction",
        "sleeve_score_exponent": 1.35,
        "candidate_score_exponent": 1.45,
        "guidance": (
            "Express a stronger view. Identify the highest-conviction sleeves from the supplied evidence, "
            "score secondary sleeves materially lower, and concentrate within each sleeve around the clearest candidates."
        ),
    },
    "bottleneck_barbell": {
        "label": "Bottleneck Barbell",
        "sleeve_score_exponent": 1.55,
        "candidate_score_exponent": 1.6,
        "guidance": (
            "Favor scarce infrastructure or supply-chain bottlenecks plus a small number of core anchors. "
            "Peripheral sleeves should receive low scores unless the evidence shows a current bottleneck or clear monetization."
        ),
    },
    "concentrated_growth": {
        "label": "Concentrated Growth",
        "sleeve_score_exponent": 1.75,
        "candidate_score_exponent": 1.85,
        "guidance": (
            "Build a compact high-growth map. Give decisive score separation to the most important sleeves and candidates; "
            "use near-zero scores for merely nice-to-have diversification."
        ),
    },
}


class SleeveFormulaScore(BaseModel):
    sleeve_key: str
    sleeve_name: str = ""
    holding_symbols: list[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0, le=1)
    opportunity_score: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    risk_penalty: float = Field(ge=0, le=1)
    overlap_penalty: float = Field(ge=0, le=1)
    min_weight: float | None = Field(default=None, ge=0, le=1)
    max_weight: float | None = Field(default=None, ge=0, le=1)
    rationale: str = ""
    why_not_higher: str = ""
    why_not_lower: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class CandidateFormulaScore(BaseModel):
    symbol: str
    sleeve_key: str
    role: str = ""
    role_importance: float = Field(ge=0, le=1)
    theme_fit: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    business_quality: float = Field(ge=0, le=1)
    growth_quality: float = Field(ge=0, le=1)
    market_signal: float = Field(default=1.0, ge=0.1, le=1.2)
    valuation_adjustment: float = Field(default=1.0, ge=0.1, le=1.2)
    liquidity_score: float = Field(default=0.8, ge=0, le=1)
    risk_penalty: float = Field(ge=0, le=1)
    overlap_penalty: float = Field(ge=0, le=1)
    min_weight: float | None = Field(default=None, ge=0, le=1)
    max_weight: float | None = Field(default=None, ge=0, le=1)
    rationale: str = ""
    why_not_higher: str = ""
    why_not_lower: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class PortfolioWeightScoring(BaseModel):
    artifact_type: str = "portfolio_weight_scoring"
    scoring_id: str = Field(default_factory=lambda: new_id("pws"))
    generated_at: str = Field(default_factory=utc_now)
    theme: str = ""
    scoring_intent: str = ""
    sleeve_scores: list[SleeveFormulaScore] = Field(default_factory=list)
    candidate_scores: list[CandidateFormulaScore] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class SleeveFormulaAllocation(BaseModel):
    sleeve_key: str
    sleeve_name: str = ""
    raw_score: float
    target_weight: float = Field(ge=0, le=1)
    formula_terms: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    why_not_higher: str = ""
    why_not_lower: str = ""


class CandidateFormulaAllocation(BaseModel):
    symbol: str
    sleeve_key: str
    raw_score: float
    normalized_within_sleeve: float = Field(ge=0, le=1)
    target_weight: float = Field(ge=0, le=1)
    formula_terms: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    why_not_higher: str = ""
    why_not_lower: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class ReferenceWeightComparison(BaseModel):
    symbol: str
    reference_weight: float = Field(ge=0, le=1)
    formula_weight: float = Field(ge=0, le=1)
    delta: float


class PortfolioFormulaAllocationReport(BaseModel):
    artifact_type: str = "portfolio_formula_allocation"
    allocation_id: str = Field(default_factory=lambda: new_id("pfa"))
    generated_at: str = Field(default_factory=utc_now)
    map_id: str
    theme: str = ""
    sleeve_weight: float = Field(ge=0, le=1)
    cash_weight: float = Field(ge=0, le=1)
    formula_version: str = "v0.2"
    formula: dict[str, str] = Field(default_factory=dict)
    scoring: PortfolioWeightScoring
    sleeve_allocations: list[SleeveFormulaAllocation] = Field(default_factory=list)
    candidate_allocations: list[CandidateFormulaAllocation] = Field(default_factory=list)
    reference_comparison: list[ReferenceWeightComparison] = Field(default_factory=list)
    calculation_steps: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class PortfolioWeightFormulaRunArtifact(BaseModel):
    artifact_type: str = "portfolio_weight_formula_run"
    run_id: str = Field(default_factory=lambda: new_id("pwfr"))
    generated_at: str = Field(default_factory=utc_now)
    context_path: str = ""
    scoring_path: str = ""
    allocation_path: str = ""
    status: Literal["fresh", "partial", "error"] = "fresh"
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class ResearchCompletenessError(ValueError):
    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        symbols = ", ".join(str(issue.get("symbol")) for issue in issues)
        super().__init__(
            "Initial portfolio weighting requires more research or user confirmation; "
            f"blocking unresearched high-priority candidates: {symbols}"
        )


def build_formula_allocation_from_files(
    *,
    portfolio_map_path: str | Path,
    deep_research_path: str | Path | None = None,
    map_id: str = "",
    user_intent: str = "",
    single_name_limit: float = 0.15,
    portfolio_style: PortfolioStyle = "balanced",
    precision: float = DEFAULT_PRECISION,
    output_dir: str | Path | None = None,
    save_context: bool = True,
) -> tuple[PortfolioFormulaAllocationReport, PortfolioWeightFormulaRunArtifact]:
    raw_map = _read_json(Path(portfolio_map_path))
    portfolio_map = _extract_portfolio_map(raw_map, map_id=map_id)
    deep_research = _read_json(Path(deep_research_path)) if deep_research_path else {}
    context = build_formula_context(
        portfolio_map=portfolio_map,
        deep_research=deep_research,
        user_intent=user_intent,
        single_name_limit=single_name_limit,
        portfolio_style=portfolio_style,
        precision=precision,
    )
    run = PortfolioWeightFormulaRunArtifact()
    run_dir = Path(output_dir) if output_dir else Path(".dev") / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / WEIGHT_FORMULA_CONTEXT_FILENAME
    scoring_path = run_dir / WEIGHT_FORMULA_SCORING_FILENAME
    allocation_path = run_dir / WEIGHT_FORMULA_ALLOCATION_FILENAME
    run_path = run_dir / WEIGHT_FORMULA_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    scoring, runtime, usage = run_weight_scoring_agent(context)
    _validate_scoring(context, scoring)
    scoring.pydantic_ai = runtime
    _write_json(scoring_path, scoring.model_dump(mode="json"))
    report = allocate_from_scoring(
        context=context,
        scoring=scoring,
        single_name_limit=single_name_limit,
        precision=precision,
    )
    report.scoring = scoring
    report.pydantic_ai = runtime
    report.usage = usage
    run.status = "partial" if report.warnings or scoring.warnings or scoring.data_gaps else "fresh"
    run.warnings = _dedupe([*report.warnings, *scoring.warnings, *scoring.data_gaps])
    run.pydantic_ai = runtime
    run.usage = usage
    _write_json(allocation_path, report.model_dump(mode="json"))
    run.scoring_path = str(scoring_path)
    run.allocation_path = str(allocation_path)
    _write_json(run_path, run.model_dump(mode="json"))
    return report, run


def build_initial_formula_allocation_from_files(
    *,
    deep_research_path: str | Path,
    user_intent: str = "",
    sleeve_weight: float = 0.95,
    cash_weight: float = 0.05,
    single_name_limit: float = 0.15,
    portfolio_style: PortfolioStyle = "balanced",
    precision: float = DEFAULT_PRECISION,
    output_dir: str | Path | None = None,
    include_watchlist: bool = False,
    allow_incomplete_research: bool = False,
    save_context: bool = True,
) -> tuple[PortfolioFormulaAllocationReport, PortfolioWeightFormulaRunArtifact]:
    deep_research = _read_json(Path(deep_research_path))
    context = build_initial_formula_context(
        deep_research=deep_research,
        user_intent=user_intent,
        sleeve_weight=sleeve_weight,
        cash_weight=cash_weight,
        single_name_limit=single_name_limit,
        portfolio_style=portfolio_style,
        precision=precision,
        include_watchlist=include_watchlist,
        allow_incomplete_research=allow_incomplete_research,
    )
    run = PortfolioWeightFormulaRunArtifact()
    run_dir = Path(output_dir) if output_dir else Path(".dev") / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / WEIGHT_FORMULA_CONTEXT_FILENAME
    scoring_path = run_dir / WEIGHT_FORMULA_SCORING_FILENAME
    allocation_path = run_dir / WEIGHT_FORMULA_ALLOCATION_FILENAME
    run_path = run_dir / WEIGHT_FORMULA_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    scoring, runtime, usage = run_weight_scoring_agent(context)
    _validate_scoring(context, scoring)
    scoring.pydantic_ai = runtime
    _write_json(scoring_path, scoring.model_dump(mode="json"))
    report = allocate_from_scoring(
        context=context,
        scoring=scoring,
        single_name_limit=single_name_limit,
        precision=precision,
    )
    report.scoring = scoring
    report.pydantic_ai = runtime
    report.usage = usage
    run.status = "partial" if report.warnings or scoring.warnings or scoring.data_gaps else "fresh"
    run.warnings = _dedupe([*report.warnings, *scoring.warnings, *scoring.data_gaps])
    run.pydantic_ai = runtime
    run.usage = usage
    _write_json(allocation_path, report.model_dump(mode="json"))
    run.scoring_path = str(scoring_path)
    run.allocation_path = str(allocation_path)
    _write_json(run_path, run.model_dump(mode="json"))
    return report, run


def build_formula_context(
    *,
    portfolio_map: PortfolioMap,
    deep_research: dict[str, Any] | None = None,
    user_intent: str = "",
    single_name_limit: float = 0.15,
    portfolio_style: PortfolioStyle = "balanced",
    precision: float = DEFAULT_PRECISION,
) -> dict[str, Any]:
    deep_research = deep_research or {}
    style_profile = _portfolio_style_profile(portfolio_style)
    cards = _deep_research_cards_by_symbol(deep_research)
    sleeve_keys: dict[str, str] = {}
    holdings_by_symbol = {holding.symbol.upper(): holding for holding in portfolio_map.holdings}
    structure_sleeves: list[dict[str, Any]] = []
    for index, sleeve in enumerate(portfolio_map.sleeves, start=1):
        key = _sleeve_key(sleeve.name, index)
        sleeve_keys[sleeve.name] = key
        structure_sleeves.append(
            {
                "sleeve_key": key,
                "name": sleeve.name,
                "role": sleeve.role,
                "holding_symbols": [symbol.upper() for symbol in sleeve.holding_symbols],
                "rationale": sleeve.rationale,
                "risk_notes": sleeve.risk_notes,
            }
        )
    structure_holdings: list[dict[str, Any]] = []
    symbol_to_sleeve: dict[str, str] = {}
    for sleeve in structure_sleeves:
        for symbol in sleeve["holding_symbols"]:
            symbol_to_sleeve[symbol] = sleeve["sleeve_key"]
    for symbol, holding in holdings_by_symbol.items():
        structure_holdings.append(
            {
                "symbol": symbol,
                "sleeve_key": symbol_to_sleeve.get(symbol, ""),
                "role": holding.role,
                "rationale": holding.rationale,
                "evidence_refs": holding.evidence_refs,
            }
        )
    research_cards = {}
    for symbol in holdings_by_symbol:
        card = cards.get(symbol, {})
        if not card:
            continue
        research_cards[symbol] = {
            "symbol": symbol,
            "layer_keys": card.get("layer_keys", []),
            "candidate_decision": card.get("candidate_decision", ""),
            "confidence": card.get("confidence", ""),
            "theme_exposure": card.get("theme_exposure", ""),
            "business_quality": card.get("business_quality", ""),
            "exposure_summary": card.get("exposure_summary", ""),
            "filing_takeaways": (card.get("filing_takeaways") or [])[:5],
            "key_risks": (card.get("key_risks") or [])[:5],
            "peer_positioning": card.get("peer_positioning", ""),
            "evidence_refs": card.get("evidence_refs", []),
            "data_gaps": card.get("data_gaps", []),
        }
    return {
        "artifact_type": "portfolio_weight_formula_context",
        "generated_at": utc_now(),
        "user_intent": user_intent,
        "map_id": portfolio_map.map_id,
        "theme": deep_research.get("theme", ""),
        "policy": {
            "sleeve_weight": portfolio_map.sleeve_weight,
            "cash_weight": portfolio_map.cash_weight,
            "single_name_limit": single_name_limit,
            "portfolio_style": portfolio_style,
            "portfolio_style_profile": style_profile,
            "available_portfolio_styles": _portfolio_style_options(),
            "precision": precision,
        },
        "portfolio_structure_without_target_weights": {
            "map_id": portfolio_map.map_id,
            "name": portfolio_map.name,
            "objective": portfolio_map.objective,
            "positioning": portfolio_map.positioning,
            "best_for": portfolio_map.best_for,
            "thesis": portfolio_map.thesis,
            "sleeves": structure_sleeves,
            "holdings": structure_holdings,
        },
        "reference_weights_for_code_only": {
            symbol: holding.target_weight
            for symbol, holding in holdings_by_symbol.items()
        },
        "research_cards": research_cards,
        "formula_contract": {
            "ai_outputs_scores_only": True,
            "deterministic_code_outputs_weights": True,
            "candidate_raw_score": (
                "role_importance * theme_fit * evidence_strength * business_quality * "
                "growth_quality * market_signal * valuation_adjustment * liquidity_score * "
                "(1 - risk_penalty) * (1 - overlap_penalty)"
            ),
            "sleeve_raw_score": (
                "importance_score * opportunity_score * evidence_strength * "
                "(1 - risk_penalty) * (1 - overlap_penalty)"
            ),
            "sleeve_style_adjusted_score": (
                "sleeve_raw_score ** policy.portfolio_style_profile.sleeve_score_exponent"
            ),
            "candidate_style_adjusted_score": (
                "candidate_raw_score ** policy.portfolio_style_profile.candidate_score_exponent"
            ),
        },
    }


def build_initial_formula_context(
    *,
    deep_research: dict[str, Any],
    user_intent: str = "",
    sleeve_weight: float = 0.95,
    cash_weight: float = 0.05,
    single_name_limit: float = 0.15,
    portfolio_style: PortfolioStyle = "balanced",
    precision: float = DEFAULT_PRECISION,
    include_watchlist: bool = False,
    allow_incomplete_research: bool = False,
) -> dict[str, Any]:
    style_profile = _portfolio_style_profile(portfolio_style)
    cards = _deep_research_cards_by_symbol(deep_research)
    if not cards:
        raise ValueError("deep_research_report has no candidate_cards.")
    unresearched_cards = _deep_research_unresearched_by_symbol(deep_research)
    completeness_issues = _blocking_unresearched_candidates(unresearched_cards)
    if completeness_issues and not allow_incomplete_research:
        raise ResearchCompletenessError(completeness_issues)

    layer_by_key = {
        str(layer.get("layer_key")): layer
        for layer in deep_research.get("layer_conclusions", [])
        if isinstance(layer, dict) and layer.get("layer_key")
    }
    eligible_decisions = {
        "high_conviction_candidate",
        "core_candidate",
        "satellite_candidate",
    }
    if include_watchlist:
        eligible_decisions.add("watchlist")

    selected_cards: dict[str, dict[str, Any]] = {}
    symbol_primary_layer: dict[str, str] = {}
    for symbol, card in sorted(cards.items()):
        if str(card.get("candidate_decision") or "") not in eligible_decisions:
            continue
        layer_keys = [str(key) for key in card.get("layer_keys", []) if key]
        primary_layer = next((key for key in layer_keys if key in layer_by_key), "")
        if not primary_layer and layer_keys:
            primary_layer = layer_keys[0]
        if not primary_layer:
            primary_layer = "unclassified_candidates"
        selected_cards[symbol] = card
        symbol_primary_layer[symbol] = primary_layer
    for symbol, card in sorted(unresearched_cards.items()):
        if symbol in selected_cards:
            continue
        layer_keys = [str(key) for key in card.get("layer_keys", []) if key]
        primary_layer = next((key for key in layer_keys if key in layer_by_key), "")
        if not primary_layer and layer_keys:
            primary_layer = layer_keys[0]
        if not primary_layer:
            primary_layer = "unclassified_candidates"
        selected_cards[symbol] = {
            **card,
            "candidate_decision": "unresearched_candidate",
            "confidence": "low",
            "theme_exposure": "unknown",
            "business_quality": "unknown",
            "exposure_summary": card.get("reason", ""),
            "research_status": "unresearched_lightweight",
        }
        symbol_primary_layer[symbol] = primary_layer

    if not selected_cards:
        raise ValueError("deep_research_report has no eligible candidate_cards after watchlist filtering.")

    symbols_by_layer: dict[str, list[str]] = {}
    for symbol, layer_key in symbol_primary_layer.items():
        symbols_by_layer.setdefault(layer_key, []).append(symbol)

    structure_sleeves: list[dict[str, Any]] = []
    for index, (layer_key, symbols) in enumerate(sorted(symbols_by_layer.items()), start=1):
        layer = layer_by_key.get(layer_key, {})
        structure_sleeves.append(
            {
                "sleeve_key": _sleeve_key(layer_key, index),
                "source_layer_key": layer_key,
                "name": layer.get("layer_name") or layer_key,
                "role": layer.get("peer_tradeoff_summary", ""),
                "holding_symbols": sorted(symbols),
                "rationale": layer.get("peer_tradeoff_summary", ""),
                "risk_notes": layer.get("unresolved_questions", []),
            }
        )

    normalized_layer_key = {
        sleeve["source_layer_key"]: sleeve["sleeve_key"]
        for sleeve in structure_sleeves
    }
    structure_holdings: list[dict[str, Any]] = []
    research_cards: dict[str, dict[str, Any]] = {}
    for symbol, card in selected_cards.items():
        primary_layer = symbol_primary_layer[symbol]
        evidence_refs = card.get("evidence_refs", [])
        structure_holdings.append(
            {
                "symbol": symbol,
                "sleeve_key": normalized_layer_key.get(primary_layer, primary_layer),
                "possible_layer_keys": card.get("layer_keys", []),
                "role": card.get("theme_exposure", "") or card.get("candidate_decision", ""),
                "rationale": card.get("exposure_summary", ""),
                "evidence_refs": evidence_refs,
                "research_status": card.get("research_status", "deep_researched"),
            }
        )
        research_cards[symbol] = {
            "symbol": symbol,
            "research_status": card.get("research_status", "deep_researched"),
            "layer_keys": card.get("layer_keys", []),
            "candidate_decision": card.get("candidate_decision", ""),
            "confidence": card.get("confidence", ""),
            "theme_exposure": card.get("theme_exposure", ""),
            "business_quality": card.get("business_quality", ""),
            "exposure_summary": card.get("exposure_summary", ""),
            "filing_takeaways": (card.get("filing_takeaways") or [])[:5],
            "key_risks": (card.get("key_risks") or [])[:5],
            "peer_positioning": card.get("peer_positioning", ""),
            "available_light_materials": card.get("available_light_materials", []),
            "missing_or_stale_materials": card.get("missing_or_stale_materials", []),
            "evidence_refs": evidence_refs,
            "data_gaps": card.get("data_gaps", []),
        }

    return {
        "artifact_type": "portfolio_weight_formula_context",
        "context_mode": "initial_map_weight_generation",
        "generated_at": utc_now(),
        "user_intent": user_intent,
        "map_id": "initial_formula_map",
        "theme": deep_research.get("theme", ""),
        "policy": {
            "sleeve_weight": sleeve_weight,
            "cash_weight": cash_weight,
            "single_name_limit": single_name_limit,
            "portfolio_style": portfolio_style,
            "portfolio_style_profile": style_profile,
            "available_portfolio_styles": _portfolio_style_options(),
            "precision": precision,
            "allow_incomplete_research": allow_incomplete_research,
        },
        "portfolio_structure_without_target_weights": {
            "map_id": "initial_formula_map",
            "name": "Formula-generated initial portfolio map",
            "objective": "growth",
            "positioning": "Initial target map generated from AI scores and deterministic formula allocation.",
            "best_for": "",
            "thesis": "\n".join(deep_research.get("cross_layer_thesis", [])[:8]),
            "sleeves": structure_sleeves,
            "holdings": sorted(structure_holdings, key=lambda item: item["symbol"]),
        },
        "research_cards": research_cards,
        "source_artifacts": deep_research.get("source_artifacts", {}),
        "deep_research_summary": {
            "research_summary": deep_research.get("research_summary", ""),
            "cross_layer_thesis": deep_research.get("cross_layer_thesis", [])[:12],
            "data_gaps": deep_research.get("data_gaps", [])[:12],
            "warnings": deep_research.get("warnings", [])[:12],
        },
        "research_completeness_gate": {
            "status": "allowed_with_incomplete_research" if completeness_issues else "complete_enough",
            "blocking_if_not_overridden": completeness_issues,
        },
        "formula_contract": {
            "ai_outputs_scores_only": True,
            "deterministic_code_outputs_weights": True,
            "candidate_raw_score": (
                "role_importance * theme_fit * evidence_strength * business_quality * "
                "growth_quality * market_signal * valuation_adjustment * liquidity_score * "
                "(1 - risk_penalty) * (1 - overlap_penalty)"
            ),
            "sleeve_raw_score": (
                "importance_score * opportunity_score * evidence_strength * "
                "(1 - risk_penalty) * (1 - overlap_penalty)"
            ),
            "sleeve_style_adjusted_score": (
                "sleeve_raw_score ** policy.portfolio_style_profile.sleeve_score_exponent"
            ),
            "candidate_style_adjusted_score": (
                "candidate_raw_score ** policy.portfolio_style_profile.candidate_score_exponent"
            ),
        },
    }


def run_weight_scoring_agent(context: dict[str, Any]) -> tuple[PortfolioWeightScoring, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=PortfolioWeightScoring,
        instructions=_WEIGHT_SCORING_INSTRUCTIONS,
        agent_kind="portfolio_weight_scoring_agent",
        output_retries=2,
        agent_skill_names=["portfolio-weight-formula"],
    )
    result = agent.run_sync(
        json.dumps(_scoring_context_for_agent(context), ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("portfolio_weight_scoring_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def allocate_from_scoring(
    *,
    context: dict[str, Any],
    scoring: PortfolioWeightScoring,
    single_name_limit: float = 0.15,
    precision: float = DEFAULT_PRECISION,
) -> PortfolioFormulaAllocationReport:
    _validate_scoring(context, scoring)
    policy = context["policy"]
    style_profile = _portfolio_style_profile(str(policy.get("portfolio_style") or "balanced"))
    total_sleeve_weight = float(policy["sleeve_weight"])
    cash_weight = float(policy["cash_weight"])
    sleeve_scores = {item.sleeve_key: item for item in scoring.sleeve_scores}
    candidate_scores = {item.symbol.upper(): item for item in scoring.candidate_scores}
    candidates_by_sleeve: dict[str, list[CandidateFormulaScore]] = {}
    for score in candidate_scores.values():
        candidates_by_sleeve.setdefault(score.sleeve_key, []).append(score)

    sleeve_raw_scores: dict[str, float] = {}
    sleeve_terms: dict[str, dict[str, float]] = {}
    for key, score in sleeve_scores.items():
        terms = {
            "importance_score": score.importance_score,
            "opportunity_score": score.opportunity_score,
            "evidence_strength": score.evidence_strength,
            "risk_multiplier": 1 - score.risk_penalty,
            "overlap_multiplier": 1 - score.overlap_penalty,
        }
        raw = _positive_product(terms.values())
        sleeve_terms[key] = terms
        sleeve_raw_scores[key] = _style_adjusted_score(raw, style_profile["sleeve_score_exponent"])

    warnings: list[str] = []
    sleeve_caps: dict[str, float] = {}
    for key, score in sleeve_scores.items():
        sleeve_candidates = candidates_by_sleeve.get(key, [])
        candidate_capacity = sum(
            min(
                candidate.max_weight if candidate.max_weight is not None else single_name_limit,
                single_name_limit,
            )
            for candidate in sleeve_candidates
        )
        if candidate_capacity <= 0:
            sleeve_caps[key] = 0
            continue
        sleeve_caps[key] = min(
            score.max_weight if score.max_weight is not None else candidate_capacity,
            candidate_capacity,
        )
        if score.max_weight is not None and candidate_capacity < score.max_weight:
            warnings.append(
                f"{key}: sleeve max_weight capped by candidate capacity "
                f"{candidate_capacity:.4f}."
            )

    sleeve_alloc_map = _normalize_with_caps(
        sleeve_raw_scores,
        total=total_sleeve_weight,
        floors={key: score.min_weight for key, score in sleeve_scores.items() if score.min_weight is not None},
        caps=sleeve_caps,
    )
    sleeve_alloc_map = _round_weights_exact(sleeve_alloc_map, total_sleeve_weight, precision)
    sleeve_allocations: list[SleeveFormulaAllocation] = []
    for key in sorted(sleeve_alloc_map):
        score = sleeve_scores[key]
        sleeve_allocations.append(
            SleeveFormulaAllocation(
                sleeve_key=key,
                sleeve_name=score.sleeve_name,
                raw_score=sleeve_raw_scores[key],
                target_weight=sleeve_alloc_map[key],
                formula_terms=sleeve_terms[key],
                rationale=score.rationale,
                why_not_higher=score.why_not_higher,
                why_not_lower=score.why_not_lower,
            )
        )

    candidate_allocations: list[CandidateFormulaAllocation] = []

    for sleeve_key, sleeve_budget in sleeve_alloc_map.items():
        sleeve_candidates = candidates_by_sleeve.get(sleeve_key, [])
        if not sleeve_candidates:
            warnings.append(f"{sleeve_key}: no candidate scores; sleeve budget cannot be assigned.")
            continue
        raw_scores: dict[str, float] = {}
        terms_by_symbol: dict[str, dict[str, float]] = {}
        for score in sleeve_candidates:
            terms = {
                "role_importance": score.role_importance,
                "theme_fit": score.theme_fit,
                "evidence_strength": score.evidence_strength,
                "business_quality": score.business_quality,
                "growth_quality": score.growth_quality,
                "market_signal": score.market_signal,
                "valuation_adjustment": score.valuation_adjustment,
                "liquidity_score": score.liquidity_score,
                "risk_multiplier": 1 - score.risk_penalty,
                "overlap_multiplier": 1 - score.overlap_penalty,
            }
            raw = _positive_product(terms.values())
            raw_scores[score.symbol.upper()] = _style_adjusted_score(
                raw,
                style_profile["candidate_score_exponent"],
            )
            terms_by_symbol[score.symbol.upper()] = terms
        weights = _normalize_with_caps(
            raw_scores,
            total=sleeve_budget,
            floors={
                score.symbol.upper(): score.min_weight
                for score in sleeve_candidates
                if score.min_weight is not None
            },
            caps={
                score.symbol.upper(): min(score.max_weight, single_name_limit)
                for score in sleeve_candidates
                if score.max_weight is not None
            }
            | {
                score.symbol.upper(): single_name_limit
                for score in sleeve_candidates
                if score.max_weight is None
            },
        )
        weights = _round_weights_exact(weights, sleeve_budget, precision)
        raw_total = sum(raw_scores.values()) or 1.0
        for score in sorted(sleeve_candidates, key=lambda item: item.symbol):
            symbol = score.symbol.upper()
            candidate_allocations.append(
                CandidateFormulaAllocation(
                    symbol=symbol,
                    sleeve_key=sleeve_key,
                    raw_score=raw_scores[symbol],
                    normalized_within_sleeve=raw_scores[symbol] / raw_total,
                    target_weight=weights[symbol],
                    formula_terms=terms_by_symbol[symbol],
                    rationale=score.rationale,
                    why_not_higher=score.why_not_higher,
                    why_not_lower=score.why_not_lower,
                    evidence_refs=score.evidence_refs,
                )
            )

    reference = context.get("reference_weights_for_code_only") or {}
    formula_by_symbol = {item.symbol: item.target_weight for item in candidate_allocations}
    comparisons = [
        ReferenceWeightComparison(
            symbol=symbol,
            reference_weight=float(reference_weight),
            formula_weight=float(formula_by_symbol.get(symbol, 0.0)),
            delta=float(formula_by_symbol.get(symbol, 0.0)) - float(reference_weight),
        )
        for symbol, reference_weight in sorted(reference.items())
    ]
    report = PortfolioFormulaAllocationReport(
        map_id=str(context["map_id"]),
        theme=str(context.get("theme") or scoring.theme or ""),
        sleeve_weight=total_sleeve_weight,
        cash_weight=cash_weight,
        formula={
            "sleeve_raw_score": context["formula_contract"]["sleeve_raw_score"],
            "candidate_raw_score": context["formula_contract"]["candidate_raw_score"],
            "sleeve_style_adjusted_score": context["formula_contract"].get(
                "sleeve_style_adjusted_score",
                "sleeve_raw_score ** portfolio_style.sleeve_score_exponent",
            ),
            "candidate_style_adjusted_score": context["formula_contract"].get(
                "candidate_style_adjusted_score",
                "candidate_raw_score ** portfolio_style.candidate_score_exponent",
            ),
            "sleeve_normalization": (
                "sleeve_weight_total * sleeve_style_adjusted_score / "
                "sum(sleeve_style_adjusted_score across sleeves), after applying "
                "AI-authored sleeve min_weight/max_weight and aggregate candidate capacity caps"
            ),
            "candidate_normalization": (
                "sleeve_weight * candidate_style_adjusted_score / "
                "sum(candidate_style_adjusted_score in same sleeve), after applying "
                "AI-authored candidate min_weight/max_weight and single_name_limit caps"
            ),
            "sleeve_weight": (
                "sleeve_weight_total * sleeve_style_adjusted_score / "
                "sum(sleeve_style_adjusted_score)"
            ),
            "candidate_weight": (
                "sleeve_weight * candidate_style_adjusted_score / "
                "sum(candidate_style_adjusted_score in same sleeve)"
            ),
            "caps": "single_name_limit and optional AI-authored min/max caps are applied before rounding",
            "rounding": f"largest-remainder rounding to precision={precision}",
        },
        scoring=scoring,
        sleeve_allocations=sleeve_allocations,
        candidate_allocations=candidate_allocations,
        reference_comparison=comparisons,
        calculation_steps=_build_calculation_steps(
            policy=policy,
            style_profile=style_profile,
            precision=precision,
            sleeve_raw_scores=sleeve_raw_scores,
            sleeve_alloc_map=sleeve_alloc_map,
            candidate_allocations=candidate_allocations,
        ),
        warnings=warnings,
    )
    _validate_allocation_report(report, single_name_limit=single_name_limit, precision=precision)
    return report


def _build_calculation_steps(
    *,
    policy: dict[str, Any],
    style_profile: dict[str, Any],
    precision: float,
    sleeve_raw_scores: dict[str, float],
    sleeve_alloc_map: dict[str, float],
    candidate_allocations: list[CandidateFormulaAllocation],
) -> dict[str, Any]:
    """Return an auditable explanation of how deterministic weights were calculated."""

    candidate_groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidate_allocations:
        candidate_groups.setdefault(item.sleeve_key, []).append(
            {
                "symbol": item.symbol,
                "style_adjusted_score": item.raw_score,
                "normalized_within_sleeve": item.normalized_within_sleeve,
                "target_weight": item.target_weight,
                "formula_terms": item.formula_terms,
            }
        )
    return {
        "allocation_pipeline": [
            "score",
            "style_adjust",
            "normalize_with_floors_and_caps",
            "round_largest_remainder",
            "validate",
        ],
        "policy_inputs": {
            "sleeve_weight": float(policy["sleeve_weight"]),
            "cash_weight": float(policy["cash_weight"]),
            "single_name_limit": float(policy["single_name_limit"]),
            "portfolio_style": str(policy.get("portfolio_style") or "balanced"),
            "sleeve_score_exponent": float(style_profile["sleeve_score_exponent"]),
            "candidate_score_exponent": float(style_profile["candidate_score_exponent"]),
            "precision": float(precision),
        },
        "sleeves": [
            {
                "sleeve_key": key,
                "style_adjusted_score": sleeve_raw_scores[key],
                "target_weight": sleeve_alloc_map[key],
            }
            for key in sorted(sleeve_alloc_map)
        ],
        "candidates_by_sleeve": {
            key: sorted(items, key=lambda item: item["symbol"])
            for key, items in sorted(candidate_groups.items())
        },
    }

def _scoring_context_for_agent(context: dict[str, Any]) -> dict[str, Any]:
    """Remove reference weights before sending context to the scoring agent."""

    payload = dict(context)
    payload.pop("reference_weights_for_code_only", None)
    return payload


def _validate_scoring(context: dict[str, Any], scoring: PortfolioWeightScoring) -> None:
    structure = context["portfolio_structure_without_target_weights"]
    expected_sleeves = {sleeve["sleeve_key"] for sleeve in structure["sleeves"]}
    expected_symbols = {holding["symbol"].upper() for holding in structure["holdings"]}
    sleeve_keys = {item.sleeve_key for item in scoring.sleeve_scores}
    symbols = {item.symbol.upper() for item in scoring.candidate_scores}
    missing_sleeves = sorted(expected_sleeves - sleeve_keys)
    extra_sleeves = sorted(sleeve_keys - expected_sleeves)
    if missing_sleeves or extra_sleeves:
        raise ValueError(f"scoring sleeve mismatch; missing={missing_sleeves}, extra={extra_sleeves}")
    missing_symbols = sorted(expected_symbols - symbols)
    extra_symbols = sorted(symbols - expected_symbols)
    if missing_symbols or extra_symbols:
        raise ValueError(f"scoring symbol mismatch; missing={missing_symbols}, extra={extra_symbols}")
    for item in scoring.candidate_scores:
        if item.sleeve_key not in expected_sleeves:
            raise ValueError(f"candidate {item.symbol} references unknown sleeve_key {item.sleeve_key}")
        if not item.rationale.strip():
            raise ValueError(f"candidate score for {item.symbol} omitted rationale")
        if not item.why_not_higher.strip() or not item.why_not_lower.strip():
            raise ValueError(f"candidate score for {item.symbol} must include why_not_higher and why_not_lower")
        if not item.evidence_refs:
            raise ValueError(f"candidate score for {item.symbol} omitted evidence_refs")
    for item in scoring.sleeve_scores:
        if not item.rationale.strip():
            raise ValueError(f"sleeve score for {item.sleeve_key} omitted rationale")
        if not item.why_not_higher.strip() or not item.why_not_lower.strip():
            raise ValueError(f"sleeve score for {item.sleeve_key} must include why_not_higher and why_not_lower")


def _validate_allocation_report(
    report: PortfolioFormulaAllocationReport,
    *,
    single_name_limit: float,
    precision: float,
) -> None:
    total_candidates = sum(item.target_weight for item in report.candidate_allocations)
    total_sleeves = sum(item.target_weight for item in report.sleeve_allocations)
    tolerance = max(precision * 2, 0.0001)
    if abs(total_candidates - report.sleeve_weight) > tolerance:
        raise ValueError("formula allocation candidate weights do not sum to sleeve_weight")
    if abs(total_sleeves - report.sleeve_weight) > tolerance:
        raise ValueError("formula allocation sleeve weights do not sum to sleeve_weight")
    too_large = [
        f"{item.symbol}={item.target_weight:.4f}"
        for item in report.candidate_allocations
        if item.target_weight > single_name_limit + tolerance
    ]
    if too_large:
        raise ValueError("formula allocation exceeds single_name_limit: " + ", ".join(too_large))


def _normalize_with_caps(
    raw_scores: dict[str, float],
    *,
    total: float,
    floors: dict[str, float | None] | None = None,
    caps: dict[str, float | None] | None = None,
) -> dict[str, float]:
    keys = list(raw_scores)
    if not keys:
        return {}
    floors = {key: float(value) for key, value in (floors or {}).items() if value is not None}
    caps = {key: float(value) for key, value in (caps or {}).items() if value is not None}
    allocations = {key: max(0.0, floors.get(key, 0.0)) for key in keys}
    floor_total = sum(allocations.values())
    if floor_total > total:
        scale = total / floor_total if floor_total else 0.0
        return {key: value * scale for key, value in allocations.items()}
    active = {key for key in keys if allocations[key] < caps.get(key, math.inf) - 1e-12}
    remaining = total - floor_total
    while active and remaining > 1e-12:
        score_total = sum(max(raw_scores[key], 1e-9) for key in active)
        if score_total <= 0:
            share = remaining / len(active)
            tentative = {key: share for key in active}
        else:
            tentative = {
                key: remaining * max(raw_scores[key], 1e-9) / score_total
                for key in active
            }
        capped: set[str] = set()
        for key, add in tentative.items():
            cap = caps.get(key, math.inf)
            if allocations[key] + add > cap:
                allocations[key] = cap
                capped.add(key)
        if not capped:
            for key, add in tentative.items():
                allocations[key] += add
            remaining = 0.0
            break
        active -= capped
        remaining = total - sum(allocations.values())
    if remaining > 1e-9 and active:
        share = remaining / len(active)
        for key in active:
            allocations[key] += share
    return allocations


def _round_weights_exact(weights: dict[str, float], total: float, precision: float) -> dict[str, float]:
    if not weights:
        return {}
    unit_total = int(round(total / precision))
    raw_units = {key: value / precision for key, value in weights.items()}
    floor_units = {key: int(math.floor(value)) for key, value in raw_units.items()}
    residual = unit_total - sum(floor_units.values())
    ranked = sorted(raw_units, key=lambda key: raw_units[key] - floor_units[key], reverse=True)
    for key in ranked[: max(0, residual)]:
        floor_units[key] += 1
    return {key: round(units * precision, 10) for key, units in floor_units.items()}


def _positive_product(values) -> float:
    result = 1.0
    for value in values:
        result *= max(float(value), 1e-6)
    return result


def _style_adjusted_score(raw_score: float, exponent: float) -> float:
    return max(float(raw_score), 1e-12) ** max(float(exponent), 1e-6)


def _portfolio_style_profile(portfolio_style: str) -> dict[str, Any]:
    if portfolio_style not in PORTFOLIO_STYLE_PROFILES:
        allowed = ", ".join(sorted(PORTFOLIO_STYLE_PROFILES))
        raise ValueError(f"Unknown portfolio_style {portfolio_style!r}; allowed: {allowed}")
    return dict(PORTFOLIO_STYLE_PROFILES[portfolio_style])


def _portfolio_style_options() -> list[dict[str, Any]]:
    return [
        {"style": style, **profile}
        for style, profile in sorted(PORTFOLIO_STYLE_PROFILES.items())
    ]


def _deep_research_cards_by_symbol(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards = raw.get("candidate_cards") if isinstance(raw, dict) else None
    if not isinstance(cards, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for card in cards:
        if isinstance(card, dict) and card.get("symbol"):
            result[str(card["symbol"]).upper()] = card
    return result


def _deep_research_unresearched_by_symbol(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards = raw.get("unresearched_candidates") if isinstance(raw, dict) else None
    if not isinstance(cards, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for card in cards:
        if isinstance(card, dict) and card.get("symbol"):
            symbol = str(card["symbol"]).upper()
            if symbol and symbol not in result:
                result[symbol] = card
    return result


def _blocking_unresearched_candidates(cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for symbol, card in sorted(cards.items()):
        priority = str(card.get("original_priority") or "").lower()
        action = str(card.get("intake_action") or "").lower()
        if priority not in {"critical", "high"}:
            continue
        if action != "optional_read_filing_analysis":
            continue
        issues.append(
            {
                "symbol": symbol,
                "priority": priority,
                "intake_action": action,
                "layer_keys": card.get("layer_keys", []),
                "reason": card.get("reason", ""),
                "missing_or_stale_materials": card.get("missing_or_stale_materials", []),
                "required_resolution": (
                    "run deep research for this candidate, or obtain explicit user confirmation "
                    "to continue with incomplete research"
                ),
            }
        )
    return issues


def _extract_portfolio_map(raw: dict[str, Any], *, map_id: str = "") -> PortfolioMap:
    if isinstance(raw.get("revised_map"), dict):
        return PortfolioMap.model_validate(raw["revised_map"])
    if isinstance(raw.get("selected_map"), dict):
        return PortfolioMap.model_validate(raw["selected_map"])
    if isinstance(raw.get("selected_portfolio_map"), dict):
        return PortfolioMap.model_validate(raw["selected_portfolio_map"])
    maps = ((raw.get("portfolio_architect_result") or raw).get("portfolio_maps") or {}).get("maps") or []
    if maps:
        if map_id:
            for item in maps:
                if isinstance(item, dict) and item.get("map_id") == map_id:
                    return PortfolioMap.model_validate(item)
            raise ValueError(f"map_id {map_id!r} not found in portfolio_maps.")
        if len(maps) == 1:
            return PortfolioMap.model_validate(maps[0])
        raise ValueError("map_id is required when portfolio map file contains multiple maps.")
    if raw.get("map_id"):
        return PortfolioMap.model_validate(raw)
    raise ValueError("Could not find a PortfolioMap in the supplied JSON file.")


def _sleeve_key(name: str, index: int) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return value or f"sleeve_{index}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "allocate":
        report, run = build_formula_allocation_from_files(
            portfolio_map_path=args.portfolio_map_path,
            deep_research_path=args.deep_research_path,
            map_id=args.map_id or "",
            user_intent=args.intent or "",
            single_name_limit=args.single_name_limit,
            portfolio_style=args.portfolio_style,
            precision=args.precision,
            output_dir=args.output_dir,
            save_context=not args.no_save_context,
        )
        payload = {
            "run": run.model_dump(mode="json"),
            "formula_allocation": report.model_dump(mode="json"),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run_id: {run.run_id}")
            print(f"status: {run.status}")
            print(f"allocation_path: {run.allocation_path}")
            print(f"candidate_count: {len(report.candidate_allocations)}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    if args.command == "allocate-initial":
        try:
            report, run = build_initial_formula_allocation_from_files(
                deep_research_path=args.deep_research_path,
                user_intent=args.intent or "",
                sleeve_weight=args.sleeve_weight,
                cash_weight=args.cash_weight,
                single_name_limit=args.single_name_limit,
                portfolio_style=args.portfolio_style,
                precision=args.precision,
                output_dir=args.output_dir,
                include_watchlist=args.include_watchlist,
                allow_incomplete_research=args.allow_incomplete_research,
                save_context=not args.no_save_context,
            )
        except ResearchCompletenessError as exc:
            payload = {
                "success": False,
                "error_type": "research_completeness_blocked",
                "message": str(exc),
                "blocking_unresearched_candidates": exc.issues,
                "allowed_next_actions": [
                    "run_deep_research_for_blocking_candidates",
                    "ask_user_to_confirm_continue_with_incomplete_research",
                    "rerun_with_--allow-incomplete-research_after_user_confirmation",
                ],
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("research_completeness_blocked")
                print(str(exc))
                for issue in exc.issues:
                    print(
                        f"- {issue['symbol']}: priority={issue['priority']}, "
                        f"action={issue['intake_action']}, layers={','.join(issue.get('layer_keys') or [])}"
                    )
                print("Next: run deep research for these symbols, or confirm skipping and rerun with --allow-incomplete-research.")
            return 1
        payload = {
            "run": run.model_dump(mode="json"),
            "formula_allocation": report.model_dump(mode="json"),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run_id: {run.run_id}")
            print(f"status: {run.status}")
            print(f"allocation_path: {run.allocation_path}")
            print(f"sleeve_count: {len(report.sleeve_allocations)}")
            print(f"candidate_count: {len(report.candidate_allocations)}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-weight-formula",
        description="Score portfolio candidates with AI and allocate weights with a deterministic formula.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    allocate = subparsers.add_parser("allocate", help="Generate formula-derived target weights.")
    allocate.add_argument("--portfolio-map-path", required=True, help="Portfolio map or revision JSON path.")
    allocate.add_argument("--deep-research-path", help="Optional deep_research_report JSON path.")
    allocate.add_argument("--map-id", default="", help="Map id when input contains multiple maps.")
    allocate.add_argument("--intent", default="", help="User objective or revision intent.")
    allocate.add_argument("--single-name-limit", type=float, default=0.15)
    allocate.add_argument(
        "--portfolio-style",
        choices=sorted(PORTFOLIO_STYLE_PROFILES),
        default="balanced",
        help="Portfolio concentration style. This is usually selected by HITL before scoring.",
    )
    allocate.add_argument("--precision", type=float, default=DEFAULT_PRECISION)
    allocate.add_argument("--output-dir", help="Directory for run artifacts.")
    allocate.add_argument("--no-save-context", action="store_true")
    allocate.add_argument("--json", action="store_true")

    initial = subparsers.add_parser(
        "allocate-initial",
        help="Generate initial formula-derived target weights from deep research.",
    )
    initial.add_argument("--deep-research-path", required=True, help="deep_research_report JSON path.")
    initial.add_argument("--intent", default="", help="User objective for the initial map.")
    initial.add_argument("--sleeve-weight", type=float, default=0.95)
    initial.add_argument("--cash-weight", type=float, default=0.05)
    initial.add_argument("--single-name-limit", type=float, default=0.15)
    initial.add_argument(
        "--portfolio-style",
        choices=sorted(PORTFOLIO_STYLE_PROFILES),
        default="balanced",
        help="Portfolio concentration style. This is usually selected by HITL before scoring.",
    )
    initial.add_argument("--precision", type=float, default=DEFAULT_PRECISION)
    initial.add_argument("--output-dir", help="Directory for run artifacts.")
    initial.add_argument("--include-watchlist", action="store_true")
    initial.add_argument(
        "--allow-incomplete-research",
        action="store_true",
        help="Continue after explicit user confirmation despite high-priority unresearched optional candidates.",
    )
    initial.add_argument("--no-save-context", action="store_true")
    initial.add_argument("--json", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


_WEIGHT_SCORING_INSTRUCTIONS = """
You are the investment assistant's Portfolio Weight Scoring Agent.

Return PortfolioWeightScoring only. Your job is to score sleeves and candidates;
do not output final weights. Deterministic code will convert your scores into
weights using the formula shown in the context.

Evidence boundary:
- Use only the supplied portfolio_weight_formula_context.
- The portfolio structure intentionally excludes target weights. Do not infer or
  recreate prior target weights from memory.
- Use research_cards, holding rationales, sleeve rationales, policy, and user
  intent.
- If research_cards[].research_status is unresearched_lightweight, treat that
  candidate as visible but lower-evidence than deep_researched candidates; lower
  evidence_strength unless the supplied light materials clearly justify more.
- Do not use web search, outside market facts, current holdings, orders, or
  model memory.

Scoring behavior:
- Score every sleeve in portfolio_structure_without_target_weights.sleeves.
- Score every holding in portfolio_structure_without_target_weights.holdings.
- Each candidate score must reference exactly one sleeve_key from the supplied
  structure.
- First read policy.portfolio_style and policy.portfolio_style_profile. Use the
  chosen style to decide score separation before scoring sleeves or candidates.
  If the style is conviction, bottleneck_barbell, or concentrated_growth, do not
  give all plausible sleeves similar scores. Identify the best-evidenced primary
  sleeves, score secondary sleeves materially lower, and score nice-to-have
  diversification near zero when it is not central to the selected style.
- Use 0-1 scores consistently. market_signal and valuation_adjustment are
  bounded multipliers where 1.0 is neutral.
- risk_penalty and overlap_penalty are penalties; higher means lower eventual
  formula weight.
- Include why_not_higher and why_not_lower for every sleeve and candidate.
- Include evidence_refs for every candidate using supplied refs.
- If evidence is missing or weak, lower evidence_strength/confidence and record
  data_gaps.

Do not output:
- final target weights
- buy/sell/hold advice
- price targets
- order instructions
"""
