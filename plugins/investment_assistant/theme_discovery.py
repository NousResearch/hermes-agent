"""PydanticAI theme discovery for candidate-pool construction."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, research_settings, usage_metadata
from .schemas import ThemeCoverageRequirement, ThemeDiscoveryPlan, ThemeDiscoverySeed


_DISCOVERY_INSTRUCTIONS = """
You are the investment assistant's theme-discovery agent.

Your job is not to recommend a portfolio. Your job is to produce a candidate
discovery plan that downstream data adapters can verify. The plan should be
broad enough to let a later portfolio-map architect compare alternative
exposures.

Rules:
- First write an initial_thesis: a concise active market map of the theme, like
  a senior portfolio manager explaining the current battleground. This is not a
  final portfolio and must not include weights.
- Then build domain_tree before listing individual candidates. domain_tree is
  the primary discovery artifact: split the theme into top-level investable
  domains, each with subdomains and the strongest candidates for that
  subdomain.
- Use a layered value-chain view. For technical infrastructure themes, relevant
  layers may include compute, memory/storage, networking/interconnect, physical
  infrastructure, platforms, and applications; for other themes, derive the
  analogous layers from the user's request and your research.
- Each important subdomain should keep only the candidates that best express
  that subdomain. Use the candidate_limit_reason field to explain why those
  candidates made the cut and what was deliberately left for downstream
  validation.
- For every high-importance subdomain, actively look for recent public-market
  changes that can alter the best candidate set: spin-offs, newly listed pure
  plays, ticker changes, renamed issuers, mergers, and parent/subsidiary
  separations. If a current pure-play listing may better express the subdomain
  than an older parent-company ticker, include the pure-play candidate and note
  that downstream adapters must validate the current ticker and filings.
- Do not collapse a bottleneck subdomain into a broad legacy incumbent when
  there may be a more direct investable pure play. Include both when uncertain
  and let downstream market-data and filings enrichment decide eligibility.
- Search breadth matters: do not rely on only a theme-level overview query.
  Use targeted subdomain queries for the most important bottlenecks and
  beneficiaries before finalizing domain_tree.
- Mark priority="must_consider" only when a later portfolio architect must
  either include the candidate or explicitly explain the omission. Otherwise use
  strong_candidate or watchlist.
- Use the requested listing market from the input payload.
- Output symbols in a conventional listing-market ticker format. Bare US
  tickers are acceptable for US listings; downstream adapters can normalize
  them.
- Use only ASCII ticker characters. Do not emit visually similar non-ASCII
  letters inside tickers.
- Include ETFs only when they are useful discovery anchors for the theme.
- Include multiple value-chain exposures, not just current mega-cap leaders.
- Fill coverage_requirements with the theme-specific value-chain sleeves,
  bottlenecks, beneficiaries, or factor exposures that must be evaluated before
  portfolio construction. Derive them from domain_tree rather than from a
  hard-coded theme template.
- Each coverage requirement should list candidate_symbols and
  must_consider_symbols when a candidate is strategically important enough that
  the portfolio architect must either include it or explain why it was omitted.
- seed_symbols must be the flattened candidate list from domain_tree plus any
  user-supplied required symbols.
- For every seed, include a rationale, subthemes, value_chain_stage,
  exposure_type, and exposure_purity when you can infer them. These fields
  become candidate-level context for later portfolio construction.
- Do not include weights, trade instructions, price targets, or final
  recommendations.
- Include generic market/search keywords that downstream adapters can use to
  discover additional constituents.
- When research tools are available, use them to support the theme thesis and
  record the search_queries and research_trace sources that influenced the
  candidate list.
- If a must-consider candidate is a model hypothesis without direct source
  support, say so explicitly in warnings so downstream validators and users can
  see that it still requires external verification.
- Use current market knowledge heuristically, but treat every symbol as a
  candidate to be verified by downstream data, not as a trusted fact.
- If the user supplied required symbols, include them in seed_symbols.
""".strip()


def build_theme_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
) -> ThemeDiscoveryPlan:
    market = _normalize_market(market)
    plan = _run_pydantic_theme_agent(
        theme=theme,
        market=market,
        theme_description=theme_description,
        required_symbols=required_symbols or [],
    )
    audit_trace: list[dict[str, Any]] = []
    for iteration in range(1, 4):
        observations = _audit_theme_discovery_plan(theme, market, required_symbols or [], plan)
        audit_trace.append(
            {
                "iteration": iteration,
                "observations": observations,
                "action": "accept" if not observations else "revise",
            }
        )
        if not observations:
            plan.pydantic_ai = {
                **plan.pydantic_ai,
                "discovery_loop": "react_audit_revise",
                "audit_trace": audit_trace,
            }
            return plan
        if iteration == 3:
            raise ValueError(
                "Theme discovery failed audit after ReAct-style revisions: "
                + "; ".join(item["message"] for item in observations)
            )
        plan = _revise_pydantic_theme_agent(
            theme=theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols or [],
            previous_plan=plan,
            observations=observations,
        )
    raise AssertionError("unreachable discovery loop exit")


def _run_pydantic_theme_agent(
    *,
    theme: str,
    market: str,
    theme_description: str,
    required_symbols: list[str],
) -> ThemeDiscoveryPlan:
    settings = research_settings()
    agent, model_config, runtime = create_pydantic_agent(
        output_type=ThemeDiscoveryPlan,
        instructions=_DISCOVERY_INSTRUCTIONS,
        agent_kind="theme_discovery_research_agent",
        output_retries=2,
        enable_web_search=True,
        enable_web_fetch=True,
        agent_skill_names=["theme-discovery"],
    )

    result = agent.run_sync(
        json.dumps(
            {
                "task": "build_theme_discovery_plan",
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "output_contract": {
                    "initial_thesis": "active theme thesis before data validation; no weights",
                    "domain_tree": (
                        "top-level domains, subdomains, and strongest candidates per subdomain; "
                        "this is the primary discovery artifact and should include recent pure-play "
                        "or spin-off candidates when they may better express a subdomain"
                    ),
                    "domain_tree_fields": [
                        "domain.key",
                        "domain.name",
                        "domain.thesis",
                        "domain.importance",
                        "subdomain.key",
                        "subdomain.name",
                        "subdomain.thesis",
                        "subdomain.importance",
                        "subdomain.candidate_limit_reason",
                        "subdomain.candidates",
                        "candidate.symbol",
                        "candidate.role",
                        "candidate.rationale",
                        "candidate.priority",
                    ],
                    "coverage_requirements": "value-chain coverage checklist with candidate_symbols and must_consider_symbols",
                    "seed_symbols": "flattened candidate discovery symbols from domain_tree plus required symbols",
                    "seed_symbol_fields": [
                        "symbol",
                        "market",
                        "role",
                        "rationale",
                        "subthemes",
                        "value_chain_stage",
                        "exposure_type",
                        "exposure_purity",
                    ],
                    "plate_keywords": "generic market/search keywords for downstream discovery adapters",
                    "research_trace": "sources used during discovery, with source_id/title/url/summary/symbols/coverage_keys",
                    "search_queries": (
                        f"theme-level and targeted subdomain queries used by the research agent; "
                        f"max {settings.max_searches}"
                    ),
                    "data_asof": "timestamps for research and discovery freshness",
                    "no_weights": True,
                    "no_trade_recommendations": True,
                },
                "research_config": {
                    "web_enabled": settings.web_enabled,
                    "max_searches": settings.max_searches,
                    "max_fetches": settings.max_fetches,
                    "require_sources_for_must_consider": settings.require_sources_for_must_consider,
                    "thinking_effort": settings.thinking_effort,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        event_stream_handler=pydantic_event_stream_handler("theme_discovery_research_agent"),
    )
    plan = result.output
    _apply_research_runtime_metadata(plan)
    plan.pydantic_ai = {
        **runtime,
        "usage": usage_metadata(result),
    }
    return plan


def _revise_pydantic_theme_agent(
    *,
    theme: str,
    market: str,
    theme_description: str,
    required_symbols: list[str],
    previous_plan: ThemeDiscoveryPlan,
    observations: list[dict[str, str]],
) -> ThemeDiscoveryPlan:
    agent, model_config, runtime = create_pydantic_agent(
        output_type=ThemeDiscoveryPlan,
        instructions=(
            _DISCOVERY_INSTRUCTIONS
            + "\n\nRevise the previous discovery plan by addressing every audit observation. "
            "Keep useful candidates, add missing coverage, and return the full corrected plan."
        ),
        agent_kind="theme_discovery_research_revision",
        output_retries=2,
        enable_web_search=True,
        enable_web_fetch=True,
        agent_skill_names=["theme-discovery"],
    )
    result = agent.run_sync(
        json.dumps(
            {
                "task": "revise_theme_discovery_plan",
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "previous_plan": previous_plan.model_dump(mode="json"),
                "audit_observations": observations,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        event_stream_handler=pydantic_event_stream_handler("theme_discovery_research_revision"),
    )
    plan = result.output
    _apply_research_runtime_metadata(plan)
    plan.pydantic_ai = {
        **runtime,
        "usage": usage_metadata(result),
    }
    return plan


def _audit_theme_discovery_plan(
    theme: str,
    market: str,
    required_symbols: list[str],
    plan: ThemeDiscoveryPlan,
) -> list[dict[str, str]]:
    try:
        _validate_theme_discovery_plan(theme, market, required_symbols, plan)
    except ValueError as exc:
        return [
            {
                "severity": "blocking",
                "message": str(exc),
                "action": "revise_theme_discovery_plan",
            }
        ]
    return []


def _validate_theme_discovery_plan(
    theme: str,
    market: str,
    required_symbols: list[str],
    plan: ThemeDiscoveryPlan,
) -> None:
    market = _normalize_market(market)
    if plan.theme and plan.theme != theme:
        raise ValueError(f"Theme discovery returned theme {plan.theme!r}, expected {theme!r}.")
    plan.market = _normalize_market(plan.market or market)
    if plan.market != market:
        raise ValueError(f"Theme discovery returned market {plan.market!r}, expected {market!r}.")
    if not plan.seed_symbols and not plan.domain_tree:
        raise ValueError("Theme discovery returned no domain_tree or seed symbols.")
    plan.initial_thesis = plan.initial_thesis.strip()

    seen: set[str] = set()
    seed_by_symbol: dict[str, ThemeDiscoverySeed] = {}
    for seed in plan.seed_symbols:
        seed.market = _normalize_market(seed.market or market)
        if seed.market != market:
            raise ValueError(
                f"Theme discovery seed {seed.symbol!r} used market {seed.market!r}, expected {market!r}."
            )
        normalized = normalize_futu_symbol(seed.symbol, market)
        if not normalized:
            raise ValueError("Theme discovery returned an empty symbol.")
        _validate_symbol_ascii(normalized)
        seed.symbol = normalized
        seed.role = seed.role.strip() or "theme discovery candidate"
        seed.rationale = seed.rationale.strip()
        seed.subthemes = _dedupe_strings(seed.subthemes)
        seed.value_chain_stage = seed.value_chain_stage.strip()
        seed.exposure_type = seed.exposure_type.strip()
        seed.exposure_purity = seed.exposure_purity.strip()
        if normalized in seen:
            raise ValueError(f"Theme discovery duplicated symbol {normalized}.")
        seen.add(normalized)
        seed_by_symbol[normalized] = seed

    _normalize_domain_tree(plan, market, seen, seed_by_symbol)
    _ensure_coverage_requirements_from_domain_tree(plan, market)
    _normalize_coverage_requirements(plan, market, seen)
    _add_seed_symbols_for_coverage_references(plan, market, seen)
    _normalize_research_trace(plan, market)
    _validate_must_consider_research_sources(plan, market)

    required = {
        normalize_futu_symbol(symbol, market)
        for symbol in required_symbols
        if normalize_futu_symbol(symbol, market)
    }
    missing_required = sorted(symbol for symbol in required if symbol not in seen)
    if missing_required:
        raise ValueError(
            "Theme discovery omitted required symbols: " + ", ".join(missing_required)
        )

    plan.plate_keywords = _dedupe_strings(plan.plate_keywords)
    plan.search_queries = _dedupe_strings(plan.search_queries)[: research_settings().max_searches]
    plan.benchmark_symbols = [
        normalize_futu_symbol(symbol, market)
        for symbol in _dedupe_strings(plan.benchmark_symbols)
        if normalize_futu_symbol(symbol, market)
    ]


def normalize_futu_symbol(symbol: str, market: str = "US") -> str:
    market = _normalize_market(market)
    value = _strip_invisible_format_chars(str(symbol or "")).strip().upper()
    if not value:
        return ""
    if "." in value:
        return value
    return f"{market}.{value}"


def _normalize_market(market: str) -> str:
    return str(market or "US").strip().upper() or "US"


def _dedupe_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _strip_invisible_format_chars(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch) != "Cf")


def _normalize_domain_tree(
    plan: ThemeDiscoveryPlan,
    market: str,
    seed_symbols: set[str],
    seed_by_symbol: dict[str, ThemeDiscoverySeed],
) -> None:
    if not plan.domain_tree:
        raise ValueError("Theme discovery returned no domain_tree.")

    domain_keys: set[str] = set()
    for domain_index, domain in enumerate(plan.domain_tree, start=1):
        domain.key = _coverage_key(domain.key or domain.name or f"domain_{domain_index}")
        if domain.key in domain_keys:
            raise ValueError(f"Theme discovery duplicated domain key {domain.key!r}.")
        domain_keys.add(domain.key)
        domain.name = domain.name.strip() or domain.key
        domain.thesis = domain.thesis.strip()
        if not domain.subdomains:
            raise ValueError(f"Theme discovery domain {domain.key!r} returned no subdomains.")

        subdomain_keys: set[str] = set()
        for subdomain_index, subdomain in enumerate(domain.subdomains, start=1):
            subdomain.key = _coverage_key(
                subdomain.key or subdomain.name or f"{domain.key}_{subdomain_index}"
            )
            if subdomain.key in subdomain_keys:
                raise ValueError(
                    f"Theme discovery duplicated subdomain key {subdomain.key!r} in domain {domain.key!r}."
                )
            subdomain_keys.add(subdomain.key)
            subdomain.name = subdomain.name.strip() or subdomain.key
            subdomain.thesis = subdomain.thesis.strip()
            subdomain.candidate_limit_reason = subdomain.candidate_limit_reason.strip()
            subdomain.omission_risk = subdomain.omission_risk.strip()
            if not subdomain.candidates:
                raise ValueError(
                    f"Theme discovery subdomain {subdomain.key!r} returned no candidates."
                )

            seen_in_subdomain: set[str] = set()
            for candidate in subdomain.candidates:
                symbol = normalize_futu_symbol(candidate.symbol, market)
                if not symbol:
                    raise ValueError(
                        f"Theme discovery subdomain {subdomain.key!r} returned an empty candidate symbol."
                    )
                _validate_symbol_ascii(symbol)
                if symbol in seen_in_subdomain:
                    raise ValueError(
                        f"Theme discovery duplicated symbol {symbol} in subdomain {subdomain.key!r}."
                    )
                seen_in_subdomain.add(symbol)
                candidate.symbol = symbol
                candidate.role = candidate.role.strip() or subdomain.name
                candidate.rationale = candidate.rationale.strip()

                seed = seed_by_symbol.get(symbol)
                if seed is None:
                    seed = ThemeDiscoverySeed(
                        symbol=symbol,
                        market=market,
                        role=candidate.role,
                        rationale=candidate.rationale,
                        subthemes=[domain.name, subdomain.name],
                        value_chain_stage=subdomain.name,
                        exposure_type="domain_tree_candidate",
                        exposure_purity="unknown",
                        confidence="medium",
                        freshness="unknown",
                    )
                    plan.seed_symbols.append(seed)
                    seed_symbols.add(symbol)
                    seed_by_symbol[symbol] = seed
                    continue

                seed.subthemes = _dedupe_strings([*seed.subthemes, domain.name, subdomain.name])
                if not seed.value_chain_stage:
                    seed.value_chain_stage = subdomain.name
                if not seed.exposure_type:
                    seed.exposure_type = "domain_tree_candidate"
                if not seed.rationale and candidate.rationale:
                    seed.rationale = candidate.rationale
                if not seed.role or seed.role == "theme discovery candidate":
                    seed.role = candidate.role


def _ensure_coverage_requirements_from_domain_tree(plan: ThemeDiscoveryPlan, market: str) -> None:
    if plan.coverage_requirements:
        return
    requirements: list[ThemeCoverageRequirement] = []
    for domain in plan.domain_tree:
        for subdomain in domain.subdomains:
            candidate_symbols = _normalize_symbol_list(
                [candidate.symbol for candidate in subdomain.candidates],
                market,
            )
            if not candidate_symbols:
                continue
            must_consider_symbols = _normalize_symbol_list(
                [
                    candidate.symbol
                    for candidate in subdomain.candidates
                    if candidate.priority == "must_consider"
                ],
                market,
            )
            if domain.importance == "core" or subdomain.importance == "high":
                priority = "required"
            elif domain.importance == "optional" or subdomain.importance == "low":
                priority = "optional"
            else:
                priority = "important"
            requirements.append(
                ThemeCoverageRequirement(
                    key=subdomain.key,
                    name=subdomain.name,
                    thesis=subdomain.thesis or domain.thesis,
                    priority=priority,
                    min_candidates=min(2, len(candidate_symbols)),
                    candidate_symbols=candidate_symbols,
                    must_consider_symbols=must_consider_symbols,
                    evidence_needed=[
                        "live market data",
                        "latest financials or filings",
                        "fresh news/events",
                    ],
                )
            )
    plan.coverage_requirements = requirements


def _normalize_coverage_requirements(
    plan: ThemeDiscoveryPlan,
    market: str,
    seed_symbols: set[str],
) -> None:
    if not plan.coverage_requirements:
        raise ValueError("Theme discovery returned no coverage_requirements.")
    for requirement in plan.coverage_requirements:
        requirement.key = _coverage_key(requirement.key)
        requirement.name = requirement.name.strip() or requirement.key
        requirement.thesis = requirement.thesis.strip()
        requirement.candidate_symbols = _normalize_symbol_list(requirement.candidate_symbols, market)
        requirement.must_consider_symbols = _normalize_symbol_list(requirement.must_consider_symbols, market)
        requirement.evidence_needed = _dedupe_strings(requirement.evidence_needed)
        if not requirement.thesis:
            raise ValueError(f"Coverage requirement {requirement.key!r} omitted thesis.")
        if len(requirement.candidate_symbols) < max(1, requirement.min_candidates):
            raise ValueError(f"Coverage requirement {requirement.key!r} has no candidate symbols.")


def _add_seed_symbols_for_coverage_references(
    plan: ThemeDiscoveryPlan,
    market: str,
    seed_symbols: set[str],
) -> None:
    added: list[str] = []
    for requirement in plan.coverage_requirements:
        for symbol in requirement.candidate_symbols + requirement.must_consider_symbols:
            if symbol in seed_symbols:
                continue
            plan.seed_symbols.append(
                ThemeDiscoverySeed(
                    symbol=symbol,
                    market=market,
                    role=requirement.name or requirement.key,
                    rationale=(
                        "Added by discovery validator because the coverage requirement "
                        f"{requirement.key!r} referenced this symbol without a matching seed."
                    ),
                    subthemes=[requirement.name or requirement.key],
                    value_chain_stage=requirement.name or requirement.key,
                    exposure_type="coverage_requirement_reference",
                    exposure_purity="unknown",
                    confidence="low",
                    freshness="unknown",
                )
            )
            seed_symbols.add(symbol)
            added.append(symbol)
    if added:
        plan.warnings = _dedupe_strings(
            [
                *plan.warnings,
                "Discovery coverage referenced symbols missing from seed_symbols; "
                "validator added low-confidence seeds pending downstream validation: "
                + ", ".join(_dedupe_strings(added)),
                "Some discovery candidates are model assumptions pending downstream validation.",
            ]
        )


def _apply_research_runtime_metadata(plan: ThemeDiscoveryPlan) -> None:
    settings = research_settings()
    if not settings.web_enabled:
        plan.warnings = _dedupe_strings(
            [
                *plan.warnings,
                "No web research was run; discovery candidates are model assumptions pending downstream validation.",
            ]
        )
    if not plan.data_asof.get("research"):
        plan.data_asof["research"] = _utc_now()
    if not plan.data_asof.get("discovery"):
        plan.data_asof["discovery"] = _utc_now()
    if settings.max_fetches >= 0:
        plan.research_trace = plan.research_trace[: settings.max_fetches]


def _normalize_research_trace(plan: ThemeDiscoveryPlan, market: str) -> None:
    seen_ids: set[str] = set()
    normalized = []
    for index, source in enumerate(plan.research_trace, start=1):
        source.source_id = source.source_id.strip() or f"research_{index}"
        if source.source_id in seen_ids:
            raise ValueError(f"Theme discovery duplicated research source_id {source.source_id!r}.")
        seen_ids.add(source.source_id)
        source.title = source.title.strip()
        source.url = source.url.strip()
        source.publisher = source.publisher.strip()
        source.published_at = source.published_at.strip()
        source.retrieved_at = source.retrieved_at.strip()
        source.summary = source.summary.strip()
        source.symbols = _normalize_symbol_list(source.symbols, market)
        source.coverage_keys = [_coverage_key(value) for value in _dedupe_strings(source.coverage_keys)]
        normalized.append(source)
    plan.research_trace = normalized

    source_ids = {source.source_id for source in plan.research_trace}
    unknown_refs: list[str] = []
    for seed in plan.seed_symbols:
        seed.source_ids = _dedupe_strings(seed.source_ids)
        unknown_sources = [source_id for source_id in seed.source_ids if source_id not in source_ids]
        if unknown_sources:
            seed.source_ids = [source_id for source_id in seed.source_ids if source_id in source_ids]
            seed.confidence = "low"
            seed.freshness = "unknown"
            unknown_refs.append(f"{seed.symbol}: {', '.join(unknown_sources)}")
    if unknown_refs:
        plan.warnings = _dedupe_strings(
            [
                *plan.warnings,
                "Discovery seeds referenced source_ids missing from research_trace; "
                "removed those references and marked them as model assumptions pending downstream validation: "
                + "; ".join(unknown_refs),
                "Some discovery candidates are model assumptions pending downstream validation.",
            ]
        )


def _validate_must_consider_research_sources(plan: ThemeDiscoveryPlan, market: str) -> None:
    settings = research_settings()
    if not settings.require_sources_for_must_consider:
        return
    if _has_model_assumption_warning(plan):
        return

    seed_sources = {
        normalize_futu_symbol(seed.symbol, market): set(seed.source_ids)
        for seed in plan.seed_symbols
        if normalize_futu_symbol(seed.symbol, market)
    }
    source_support: dict[str, set[str]] = {}
    coverage_support: dict[str, set[str]] = {}
    for source in plan.research_trace:
        for symbol in source.symbols:
            source_support.setdefault(symbol, set()).add(source.source_id)
        for key in source.coverage_keys:
            coverage_support.setdefault(key, set()).add(source.source_id)

    missing: list[str] = []
    for requirement in plan.coverage_requirements:
        if requirement.priority == "optional":
            continue
        supported_by_coverage = coverage_support.get(requirement.key, set())
        for symbol in requirement.must_consider_symbols:
            normalized = normalize_futu_symbol(symbol, market)
            if seed_sources.get(normalized) or source_support.get(normalized) or supported_by_coverage:
                continue
            missing.append(normalized)
    if missing:
        raise ValueError(
            "Theme discovery must-consider candidates lack research source support "
            "or an explicit model-assumption warning: "
            + ", ".join(_dedupe_strings(missing))
        )


def _has_model_assumption_warning(plan: ThemeDiscoveryPlan) -> bool:
    patterns = ("model assumption", "pending downstream validation", "pending validation", "no web research")
    return any(any(pattern in warning.lower() for pattern in patterns) for warning in plan.warnings)


def _normalize_symbol_list(values: list[Any], market: str) -> list[str]:
    symbols: list[str] = []
    for value in values:
        symbol = normalize_futu_symbol(str(value or ""), market)
        if not symbol:
            continue
        _validate_symbol_ascii(symbol)
        symbols.append(symbol)
    return _dedupe_strings(symbols)


def _coverage_key(value: str) -> str:
    key = str(value or "").strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key or "unknown"


def _validate_symbol_ascii(symbol: str) -> None:
    if not symbol.isascii():
        raise ValueError(f"Theme discovery returned non-ASCII ticker characters: {symbol!r}.")
    if not re.fullmatch(r"[A-Z0-9.]+", symbol):
        raise ValueError(f"Theme discovery returned invalid ticker characters: {symbol!r}.")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
