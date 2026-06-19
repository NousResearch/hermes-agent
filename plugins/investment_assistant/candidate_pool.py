"""Candidate-pool generation for portfolio map workflows."""

from __future__ import annotations

import os

from .adapters import MarketDataAdapter
from .schemas import CandidatePool, InvestmentPolicy


def build_candidate_pool(
    theme: str,
    policy: InvestmentPolicy,
    market_data: MarketDataAdapter | None = None,
    discovery_mode: str | None = None,
) -> CandidatePool:
    """Build an unbiased candidate pool from theme and market sources only.

    Default production mode is Futu-assisted discovery: a PydanticAI agent
    explores the theme with a live Futu tool, then the existing Futu enrichment
    path validates quotes/K-lines/options. ``legacy`` remains available for
    direct comparison with the original web/research-first discovery path.
    """

    adapter = market_data or MarketDataAdapter()
    mode = _candidate_discovery_mode(discovery_mode)
    if mode == "legacy":
        return build_legacy_candidate_pool(theme, policy, adapter)
    if mode != "futu_assisted":
        raise ValueError(
            "Unknown investment assistant candidate discovery mode: "
            f"{mode!r}. Use 'futu_assisted' or 'legacy'."
        )
    if _supports_futu_assisted_discovery(adapter):
        from .futu_theme_discovery import build_futu_assisted_candidate_pool

        return _rank_candidate_pool(
            build_futu_assisted_candidate_pool(
                theme,
                policy,
                market_data=adapter,
            ),
            policy,
        )

    # Unit-test and third-party adapters may implement only the older
    # get_theme_universe contract. This does not affect the default production
    # path, which constructs a real MarketDataAdapter above.
    return build_legacy_candidate_pool(theme, policy, adapter)


def build_legacy_candidate_pool(
    theme: str,
    policy: InvestmentPolicy,
    market_data: MarketDataAdapter | None = None,
) -> CandidatePool:
    """Build a candidate pool with the original theme_discovery path."""

    adapter = market_data or MarketDataAdapter()
    universe = adapter.get_theme_universe(
        theme,
        required_symbols=policy.required_symbols,
        theme_description=policy.theme_description,
    )
    return _rank_candidate_pool(
        CandidatePool(
            theme=universe.canonical_theme,
            generated_from=universe.source_tags,
            candidates=universe.candidates,
            discovery_thesis=universe.discovery_thesis,
            coverage_requirements=universe.coverage_requirements,
            research_trace=universe.research_trace,
            search_queries=universe.search_queries,
            data_asof=universe.data_asof,
            warnings=list(universe.warnings),
        ),
        policy,
    )


def _candidate_discovery_mode(value: str | None = None) -> str:
    raw = value or os.getenv("IA_CANDIDATE_DISCOVERY_MODE", "futu_assisted")
    return str(raw or "futu_assisted").strip().lower().replace("-", "_")


def _supports_futu_assisted_discovery(adapter) -> bool:
    return callable(getattr(adapter, "_load_futu_candidates", None))


def _rank_candidate_pool(pool: CandidatePool, policy: InvestmentPolicy) -> CandidatePool:
    candidates = sorted(
        pool.candidates,
        key=lambda item: (item.score, _relative_strength(item)),
        reverse=True,
    )
    if policy.risk_level == "conservative":
        candidates = sorted(
            candidates,
            key=lambda item: (_realized_volatility(item), -item.score),
        )
    candidates = _prioritize_required_symbols(candidates, policy.required_symbols)
    return pool.model_copy(update={"candidates": candidates})


def _prioritize_required_symbols(candidates, required_symbols: list[str]):
    if not required_symbols:
        return candidates
    order = {symbol.upper(): index for index, symbol in enumerate(required_symbols)}
    required = sorted(
        [candidate for candidate in candidates if candidate.symbol.upper() in order],
        key=lambda candidate: order[candidate.symbol.upper()],
    )
    rest = [candidate for candidate in candidates if candidate.symbol.upper() not in order]
    return required + rest


def _relative_strength(candidate) -> float:
    technical = candidate.futu_data.technical if candidate.futu_data else None
    value = technical.relative_strength_60d if technical else None
    return float(value or 0.0)


def _realized_volatility(candidate) -> float:
    technical = candidate.futu_data.technical if candidate.futu_data else None
    value = technical.realized_volatility if technical else None
    return float(value if value is not None else 999.0)
