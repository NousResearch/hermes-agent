"""Retired downstream portfolio-map agents.

Theme discovery, candidate triage, and deep research are the current MVP
surface. Thesis synthesis and portfolio-map architecture were early experiments
and are intentionally disabled until the downstream design is reopened.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import (
    CandidatePool,
    CandidatePoolReflection,
    InvestmentPolicy,
    PortfolioMaps,
    ThesisSynthesis,
)


def pydantic_ai_status() -> dict[str, object]:
    from .pydantic_runtime import pydantic_ai_status as runtime_status

    return runtime_status()


def build_portfolio_maps(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None = None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
    thesis_synthesis: ThesisSynthesis | None = None,
) -> PortfolioMaps:
    raise NotImplementedError(
        "Portfolio-map architecture is not part of the current investment assistant MVP. "
        "Run theme discovery, candidate triage, and deep research first."
    )


def synthesize_portfolio_thesis(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None = None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
) -> ThesisSynthesis:
    raise NotImplementedError(
        "Thesis synthesis is not part of the current investment assistant MVP. "
        "Run theme discovery, candidate triage, and deep research first."
    )


def _run_pydantic_thesis_agent(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
) -> ThesisSynthesis:
    raise NotImplementedError(
        "Thesis synthesis is not part of the current investment assistant MVP. "
        "Run theme discovery, candidate triage, and deep research first."
    )


def _run_pydantic_ai_agent(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
    thesis_synthesis: ThesisSynthesis | None = None,
) -> PortfolioMaps:
    raise NotImplementedError(
        "Portfolio-map architecture is not part of the current investment assistant MVP. "
        "Run theme discovery, candidate triage, and deep research first."
    )


def _architect_prompt(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
    thesis_synthesis: ThesisSynthesis | None = None,
) -> str:
    evidence_artifacts = _architect_evidence_artifacts(market_artifacts or {})
    payload = {
        "policy": policy.model_dump(mode="json"),
        "candidate_pool": candidate_pool.model_dump(mode="json"),
        "thesis_synthesis": thesis_synthesis.model_dump(mode="json") if thesis_synthesis else None,
        "evidence_artifacts": evidence_artifacts,
        "candidate_pool_reflection": (
            reflection.model_dump(mode="json")
            if reflection
            else {
                "coverage_ok": True,
                "missing_exposure": [],
                "stale_data": [],
                "weak_sources": [],
                "recommended_actions": [],
                "checks": [],
                "warnings": [],
            }
        ),
        "output_expectations": {
            "portfolio_maps_schema": "PortfolioMaps",
            "map_count": "2-3",
            "absolute_weights": True,
            "include_sleeves": True,
            "include_positioning_and_best_for": True,
            "include_omitted_candidates": True,
            "must_use_context_layers": [
                "thesis_synthesis",
                "candidate_pool.discovery_thesis",
                "candidate_pool.coverage_requirements",
                "candidate_pool.candidates[].discovery_data",
                "candidate_pool.research_trace",
                "candidate_pool.candidates[].futu_data",
                "candidate_pool.candidates[].sec_data",
                "candidate_pool.candidates[].fundamental_data",
                "candidate_pool.candidates[].event_data",
                "evidence_artifacts.research_data",
                "evidence_artifacts.futu_data",
                "evidence_artifacts.sec_data",
            ],
            "v1_scope": "target_portfolio_map_only",
            "numeric_evidence_rule": (
                "Use only structured numeric fields with numeric_llm_generated=false "
                "for financial numbers. Treat MinerU/sub-LLM filing summaries as "
                "qualitative narrative context only."
            ),
            "omission_rule": (
                "For every eligible coverage_requirements[].must_consider_symbols "
                "candidate not selected in a map, add omitted_candidates with reason, "
                "reason_category, substitute_symbols, and importance."
            ),
        },
    }
    return (
        "Build target portfolio maps from this artifact payload. "
        "Use candidate_pool as the universe. Start from thesis_synthesis as the "
        "research brief, then cross-check candidate_pool.discovery_thesis, "
        "coverage_requirements, research_trace, candidate-level data, and aggregate evidence_artifacts. "
        "Return only a valid typed PortfolioMaps output.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _thesis_prompt(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    reflection: CandidatePoolReflection | None,
    market_artifacts: dict[str, dict[str, Any]] | None = None,
) -> str:
    evidence_artifacts = _architect_evidence_artifacts(market_artifacts or {})
    payload = {
        "policy": policy.model_dump(mode="json"),
        "candidate_pool": candidate_pool.model_dump(mode="json"),
        "candidate_pool_reflection": (
            reflection.model_dump(mode="json")
            if reflection
            else {
                "coverage_ok": True,
                "missing_exposure": [],
                "stale_data": [],
                "weak_sources": [],
                "recommended_actions": [],
                "checks": [],
                "warnings": [],
            }
        ),
        "evidence_artifacts": evidence_artifacts,
        "metric_families_to_consider": [
            "discovery_thesis_and_coverage_requirements",
            "discovery_research_trace_and_source_ids",
            "candidate_discovery_role_and_exposure_purity",
            "futu_quote_price_market_cap_turnover_valuation_profitability",
            "technical_trend_relative_strength_returns_realized_volatility",
            "liquidity_turnover_volume_spread",
            "options_iv_rank_surface_status_when_available",
            "sec_companyfacts_revenue_income_margin_roe_debt",
            "filing_freshness_recent_8k_event_risk",
            "market_regime_benchmark_context_macro_proxies",
            "correlation_diversification_and_risk_flags",
            "data_quality_missing_fields_and_warnings",
        ],
        "output_expectations": {
            "schema": "ThesisSynthesis",
            "no_weights": True,
            "no_orders": True,
            "candidate_assessments": "Assess relevant candidates before portfolio construction.",
        },
    }
    return (
        "Synthesize a data-grounded investment thesis from this artifact payload. "
        "This is an intermediate research artifact, not a portfolio map. "
        "Consider all listed metric families and explain data gaps explicitly.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _architect_evidence_artifacts(market_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    futu_keys = [
        "market_context",
        "theme_exposure_map",
        "plate_constituents",
        "market_snapshot",
        "technical_summary",
        "liquidity_context",
        "options_surface",
        "valuation_context",
        "correlation_and_diversification",
        "benchmark_context",
        "market_regime",
        "positioning_and_sentiment",
        "analyst_revision_context",
        "risk_scenario",
    ]
    sec_keys = [
        "sec_filings_context",
        "fundamental_quality",
        "earnings_event_calendar",
        "risk_flags",
    ]
    return {
        "artifact_index": sorted(market_artifacts),
        "research_data": (
            {"research_trace": market_artifacts["research_trace"]}
            if "research_trace" in market_artifacts
            else {}
        ),
        "futu_data": {
            key: market_artifacts[key]
            for key in futu_keys
            if key in market_artifacts
        },
        "sec_data": {
            key: market_artifacts[key]
            for key in sec_keys
            if key in market_artifacts
        },
    }


def _validate_thesis_synthesis(candidate_pool: CandidatePool, thesis: ThesisSynthesis) -> None:
    if thesis.theme and thesis.theme != candidate_pool.theme:
        raise ValueError(
            f"PydanticAI thesis returned theme {thesis.theme!r}, expected {candidate_pool.theme!r}."
        )
    if not thesis.primary_thesis.strip():
        raise ValueError("PydanticAI thesis synthesis omitted primary_thesis.")
    if not thesis.metrics_considered:
        raise ValueError("PydanticAI thesis synthesis must list metrics_considered.")
    if not thesis.candidate_assessments:
        raise ValueError("PydanticAI thesis synthesis must assess candidate symbols.")

    allowed = {candidate.symbol.upper(): candidate for candidate in candidate_pool.candidates}
    assessed: set[str] = set()
    for assessment in thesis.candidate_assessments:
        symbol = assessment.symbol.upper()
        if symbol not in allowed:
            raise ValueError(
                f"PydanticAI thesis assessed {assessment.symbol!r}, which is not in the candidate pool."
            )
        if symbol in assessed:
            raise ValueError(f"PydanticAI thesis duplicated assessment for {assessment.symbol!r}.")
        if not assessment.evidence_summary:
            raise ValueError(f"PydanticAI thesis assessment for {assessment.symbol!r} lacks evidence_summary.")
        if not assessment.metrics_considered:
            raise ValueError(
                f"PydanticAI thesis assessment for {assessment.symbol!r} lacks metrics_considered."
            )
        assessed.add(symbol)

    must_consider = _must_consider_symbols(candidate_pool)
    missing_must_consider = []
    for symbol in sorted(must_consider):
        candidate = allowed.get(symbol)
        if candidate and candidate.eligible_for_portfolio and symbol not in assessed:
            missing_must_consider.append(symbol)
    if missing_must_consider:
        raise ValueError(
            "PydanticAI thesis synthesis omitted must-consider candidate assessments: "
            + ", ".join(missing_must_consider)
        )


def _validate_portfolio_maps(
    policy: InvestmentPolicy,
    candidate_pool: CandidatePool,
    maps: PortfolioMaps,
) -> None:
    if not maps.maps:
        raise ValueError("PydanticAI returned no portfolio maps.")

    allowed = {candidate.symbol.upper(): candidate for candidate in candidate_pool.candidates}
    required = [symbol.upper() for symbol in policy.required_symbols if symbol.upper() in allowed]
    omission_required = _must_consider_symbols(candidate_pool)
    max_sleeve = min(policy.target_portfolio_weight, 1 - policy.cash_reserve)
    single_limit = policy.single_name_limit

    for portfolio_map in maps.maps:
        if not portfolio_map.holdings:
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} has no holdings.")
        if len(portfolio_map.sleeves) < 2:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} must include portfolio sleeves."
            )
        if not portfolio_map.positioning.strip():
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted positioning.")
        if not portfolio_map.best_for.strip():
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted best_for.")
        if abs(portfolio_map.cash_weight - policy.cash_reserve) > 0.0001:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} changed the cash reserve "
                f"from {policy.cash_reserve:.4f} to {portfolio_map.cash_weight:.4f}."
            )
        if portfolio_map.sleeve_weight > max_sleeve + 0.005:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} exceeds the allowed theme sleeve."
            )

        holding_symbols: set[str] = set()
        total_weight = 0.0
        for holding in portfolio_map.holdings:
            symbol = holding.symbol.upper()
            if symbol not in allowed:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} used symbol {holding.symbol!r}, "
                    "which is not in the candidate pool."
                )
            if symbol in holding_symbols:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} duplicated holding {holding.symbol!r}."
                )
            if holding.target_weight > single_limit + 0.0001:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} assigned {holding.symbol} "
                    f"{holding.target_weight:.4f}, above single_name_limit {single_limit:.4f}."
                )
            holding_symbols.add(symbol)
            total_weight += holding.target_weight

        omitted_symbols: set[str] = set()
        for omitted in portfolio_map.omitted_candidates:
            symbol = omitted.symbol.upper()
            if symbol not in allowed:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} omitted unknown symbol {omitted.symbol!r}."
                )
            if symbol in holding_symbols:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} both selected and omitted {omitted.symbol!r}."
                )
            if not omitted.reason.strip():
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} omitted {omitted.symbol!r} without a reason."
                )
            for substitute in omitted.substitute_symbols:
                if substitute.upper() not in holding_symbols:
                    raise ValueError(
                        f"PydanticAI map {portfolio_map.map_id!r} omission for {omitted.symbol!r} "
                        f"references substitute {substitute!r}, which is not in holdings."
                    )
            omitted_symbols.add(symbol)

        missing_omission_audit = []
        for symbol in sorted(omission_required - holding_symbols):
            candidate = allowed.get(symbol)
            if candidate and candidate.eligible_for_portfolio and symbol not in omitted_symbols:
                missing_omission_audit.append(symbol)
        if missing_omission_audit:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} omitted must-consider candidates "
                "without omitted_candidates audit: "
                + ", ".join(missing_omission_audit)
            )

        sleeve_total = 0.0
        for sleeve in portfolio_map.sleeves:
            if not sleeve.name.strip():
                raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} has an unnamed sleeve.")
            if not sleeve.holding_symbols:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} sleeve {sleeve.name!r} has no holdings."
                )
            sleeve_total += sleeve.target_weight
            for symbol in sleeve.holding_symbols:
                normalized = symbol.upper()
                if normalized not in holding_symbols:
                    raise ValueError(
                        f"PydanticAI map {portfolio_map.map_id!r} sleeve {sleeve.name!r} "
                        f"references {symbol!r}, which is not in holdings."
                    )

        missing_required = [symbol for symbol in required if symbol not in holding_symbols]
        if missing_required:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} omitted required symbols: "
                f"{', '.join(missing_required)}."
            )
        if abs(total_weight - portfolio_map.sleeve_weight) > 0.01:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} sleeve_weight does not match holdings."
            )
        if abs(sleeve_total - portfolio_map.sleeve_weight) > 0.02:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} sleeve weights do not match sleeve_weight."
            )


def _must_consider_symbols(candidate_pool: CandidatePool) -> set[str]:
    symbols: set[str] = set()
    for requirement in candidate_pool.coverage_requirements:
        if requirement.priority == "optional":
            continue
        symbols.update(symbol.upper() for symbol in requirement.must_consider_symbols)
    return {symbol for symbol in symbols if symbol}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
