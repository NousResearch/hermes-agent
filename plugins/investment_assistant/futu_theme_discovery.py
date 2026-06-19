"""Futu-assisted exploratory theme discovery.

This module is intentionally separate from ``theme_discovery.py`` so the two
approaches can be compared side by side:

1. ``theme_discovery``: research/web-first PydanticAI discovery.
2. ``futu_theme_discovery``: PydanticAI discovery with a live Futu exploration
   tool for plate and quote evidence.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .schemas import CandidatePool, DiscoveryData, InvestmentPolicy, ThemeDiscoveryPlan
from .storage import utc_now
from .theme_discovery import (
    _apply_research_runtime_metadata,
    _audit_theme_discovery_plan,
    _dedupe_strings,
    normalize_futu_symbol,
)


_FUTU_ASSISTED_DISCOVERY_INSTRUCTIONS = """
You are the investment assistant's Futu-assisted theme-discovery agent.

Your job is not to recommend a portfolio. Your job is to produce a layered
candidate discovery plan that downstream agents can validate further with
market data, filings, and events.

You have a local tool named futu_explore_theme_candidates. Use it before
returning the final ThemeDiscoveryPlan. The tool can search Futu plates by
exact offline plate codes, inspect constituents, validate must-check symbols,
run Futu stock-filter probes using the provided screener catalog, and return
compact snapshot evidence. Treat Futu as evidence and candidate discovery, not
as the sole source of judgment.

You also have futu_search_screener_catalog. Use it to search the local offline
Futu screener catalog for exact plate codes, StockField names, technical
indicator enums, valuation fields, and document paths before composing Futu
tool arguments.

Rules:
- Start with a concise initial_thesis explaining the active investable map of
  the user's theme. Do not include weights or trade instructions.
- Build domain_tree first. Split the theme into value-chain layers and
  subdomains that a portfolio manager would review before constructing a map.
- Derive the layers from the theme itself. Do not use a fixed template.
- Before the first futu_explore_theme_candidates call, call
  futu_search_screener_catalog at least once. Search the theme terms for
  plate_codes, and search candidate field needs such as market cap, valuation,
  liquidity, momentum, MA/EMA/KDJ/RSI/MACD/BOLL, or financial quality before
  adding stock_filter_specs.
- Before finalizing, call futu_explore_theme_candidates with:
  - theme and market,
  - exact plate_codes returned by futu_search_screener_catalog whenever possible,
  - plate_keywords only when the offline catalog does not contain the needed
    industry/concept,
  - must-check symbols that your domain analysis says are important even if
    Futu may not discover them by plate,
  - optional stock_filter_specs selected only from the tool's StockField catalog when a
    momentum, liquidity, valuation, technical, or financial screen would help
    discover non-obvious names.
- Do not invent Futu stock-filter field names or plate codes. Use the tool
  description, local catalog search results, and raw SDK enum values exactly.
- Prefer calling futu_search_screener_catalog over guessing. For example,
  search "市值", "市盈率", "MA", "RSI", "人工智能", or the user's theme terms,
  then pass the returned exact field enums or plate_codes.
- If an App screener dimension is marked non_stock_filter, do not pass it as a
  stock_filter_specs item. Use the listed alternate tool/source or mention the
  data gap in warnings.
- Use the Futu result to distinguish direct plate evidence from inferred
  candidates. A symbol may remain in the plan without direct Futu plate support,
  but the rationale must say why it is still worth downstream validation.
- Keep only the strongest candidates in each subdomain. Use watchlist for
  narrower, lower-liquidity, or weaker-evidence names.
- Mark priority="must_consider" only when the later architect must include the
  candidate or explicitly explain the omission.
- Fill coverage_requirements from domain_tree, including candidate_symbols and
  must_consider_symbols.
- Flatten domain_tree candidates plus any required symbols into seed_symbols.
- For seed_symbols, include role, rationale, subthemes, value_chain_stage,
  exposure_type, exposure_purity, source_ids when available, confidence, and
  freshness.
- Use only the requested listing market and ASCII ticker characters.
- Include ETFs only when useful as discovery anchors or investable baselines.
- Include research_trace entries for Futu plate/snapshot evidence you relied on.
- Add warnings for Futu evidence gaps, indirect exposures, unsupported tickers,
  or model hypotheses pending downstream validation.
- Do not emit final portfolio weights, price targets, simulated orders, or
  trading recommendations.
""".strip()


_DEFAULT_SCREENER_CATALOG_JSON = (
    Path(__file__).resolve().parent / "data" / "futu_screener_catalog.json"
)


def build_futu_assisted_theme_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
    explorer: "FutuThemeExplorer | None" = None,
) -> ThemeDiscoveryPlan:
    """Build a ThemeDiscoveryPlan using a PydanticAI agent with a Futu tool."""

    market = _normalize_market(market)
    required_symbols = required_symbols or []
    plan = _run_pydantic_futu_theme_agent(
        theme=theme,
        market=market,
        theme_description=theme_description,
        required_symbols=required_symbols,
        explorer=explorer,
        previous_plan=None,
        observations=[],
    )
    plan.theme = theme
    plan.market = market
    audit_trace: list[dict[str, Any]] = []
    for iteration in range(1, 4):
        _mark_model_assumptions(plan)
        observations = _audit_theme_discovery_plan(theme, market, required_symbols, plan)
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
                "discovery_loop": "futu_tool_audit_revise",
                "audit_trace": audit_trace,
            }
            return plan
        if iteration == 3:
            raise ValueError(
                "Futu-assisted theme discovery failed audit after revisions: "
                + "; ".join(item["message"] for item in observations)
            )
        plan = _run_pydantic_futu_theme_agent(
            theme=theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
            explorer=explorer,
            previous_plan=plan,
            observations=observations,
        )
        plan.theme = theme
        plan.market = market
    raise AssertionError("unreachable futu-assisted discovery loop exit")


def build_futu_assisted_candidate_pool(
    theme: str,
    policy: InvestmentPolicy,
    *,
    market_data: Any | None = None,
    explorer: "FutuThemeExplorer | None" = None,
) -> CandidatePool:
    """Build a candidate pool from the Futu-assisted discovery path.

    This is a comparison entry point. The default workflow still uses
    ``candidate_pool.build_candidate_pool`` and ``theme_discovery.py``.
    """

    from .adapters import FutuAdapterError, MarketDataAdapter, _seeds_from_discovery_plan

    adapter = market_data or MarketDataAdapter()
    plan = build_futu_assisted_theme_discovery_plan(
        theme,
        market=getattr(adapter.config, "market", "US"),
        theme_description=policy.theme_description,
        required_symbols=policy.required_symbols,
        explorer=explorer,
    )
    seeds = _seeds_from_discovery_plan(plan, policy.required_symbols, getattr(adapter.config, "market", "US"))
    candidates, source_tags, warnings = adapter._load_futu_candidates(  # noqa: SLF001 - comparison hook.
        plan.theme,
        seeds,
        plan.plate_keywords,
    )
    if not candidates:
        raise FutuAdapterError(
            "Futu-assisted discovery produced no usable candidate data. "
            "Check OpenD permissions and symbol coverage."
        )
    return CandidatePool(
        theme=plan.theme,
        generated_from=_dedupe_strings(
            [*source_tags, "pydantic_ai_futu_assisted_theme_discovery"]
        ),
        candidates=candidates,
        discovery_thesis=plan.initial_thesis,
        coverage_requirements=plan.coverage_requirements,
        research_trace=plan.research_trace,
        search_queries=plan.search_queries,
        data_asof=plan.data_asof,
        warnings=_dedupe_strings([*plan.warnings, *warnings]),
    )


def _run_pydantic_futu_theme_agent(
    *,
    theme: str,
    market: str,
    theme_description: str,
    required_symbols: list[str],
    explorer: "FutuThemeExplorer | None",
    previous_plan: ThemeDiscoveryPlan | None,
    observations: list[dict[str, str]],
) -> ThemeDiscoveryPlan:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=ThemeDiscoveryPlan,
        instructions=_FUTU_ASSISTED_DISCOVERY_INSTRUCTIONS,
        agent_kind="futu_assisted_theme_discovery_agent",
        output_retries=2,
        enable_web_search=False,
        enable_web_fetch=False,
        agent_skill_names=["theme-discovery"],
    )
    futu_explorer = explorer or FutuThemeExplorer()
    stock_filter_catalog = futu_stock_filter_catalog(market)
    screener_catalog = load_futu_screener_catalog_snapshot(market=market)
    tool_description = _futu_explore_tool_description(stock_filter_catalog, screener_catalog)
    tool_calls: list[dict[str, Any]] = []
    catalog_search_calls: list[dict[str, Any]] = []
    tool_sequence: list[dict[str, Any]] = []

    @agent.tool_plain(
        name="futu_search_screener_catalog",
        description=_futu_search_tool_description(screener_catalog, market),
        retries=0,
        timeout=10,
    )
    def futu_search_screener_catalog(
        query: str,
        market: str = "US",
        category: str | None = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        """Search the local offline Futu screener option catalog."""

        result = search_futu_screener_catalog(
            query=query,
            market=market,
            category=category,
            limit=limit,
        )
        catalog_search_calls.append(
            {
                "query": query,
                "market": market,
                "category": category,
                "limit": limit,
                "document_count": len(result.get("documents", [])),
                "choice_count": len(result.get("choices", [])),
                "field_count": len(result.get("stock_fields", [])),
                "plate_count": len(result.get("plates", [])),
            }
        )
        tool_sequence.append(
            {
                "tool": "futu_search_screener_catalog",
                "query": query,
                "market": market,
                "category": category,
            }
        )
        return result

    @agent.tool_plain(
        name="futu_explore_theme_candidates",
        description=tool_description,
        retries=0,
        timeout=120,
    )
    def futu_explore_theme_candidates(
        theme: str,
        market: str = "US",
        plate_keywords: list[str] | None = None,
        plate_codes: list[str] | None = None,
        must_check_symbols: list[str] | None = None,
        stock_filter_specs: list[dict[str, Any]] | None = None,
        plate_code: str | None = None,
        max_candidates: int = 160,
    ) -> dict[str, Any]:
        """Return live Futu evidence for a theme discovery step."""

        if not catalog_search_calls:
            result = {
                "error": "catalog_search_required",
                "message": (
                    "Call futu_search_screener_catalog first to query exact candidate "
                    "fields, StockField enums, technical indicators, and Futu plate_codes."
                ),
                "theme": theme,
                "market": market,
                "generated_at": utc_now(),
                "tool": "futu_explore_theme_candidates",
                "candidates": [],
                "warnings": [
                    "Futu exploration was blocked because local screener catalog search was not called first."
                ],
            }
            tool_calls.append(
                {
                    "theme": theme,
                    "market": market,
                    "plate_keywords": plate_keywords or [],
                    "plate_codes": plate_codes or [],
                    "must_check_symbols": must_check_symbols or [],
                    "stock_filter_specs": stock_filter_specs or [],
                    "plate_code": plate_code,
                    "candidate_count": 0,
                    "plate_match_count": 0,
                    "stock_filter_result_count": 0,
                    "warning_count": 1,
                    "error": "catalog_search_required",
                }
            )
            tool_sequence.append(
                {
                    "tool": "futu_explore_theme_candidates",
                    "theme": theme,
                    "market": market,
                    "blocked": True,
                    "error": "catalog_search_required",
                }
            )
            return result

        result = futu_explorer.explore(
            theme=theme,
            market=market,
            plate_keywords=plate_keywords or [],
            plate_codes=plate_codes or [],
            must_check_symbols=must_check_symbols or [],
            stock_filter_specs=stock_filter_specs or [],
            plate_code=plate_code,
            max_candidates=max_candidates,
        )
        tool_calls.append(
            {
                "theme": theme,
                "market": market,
                "plate_keywords": plate_keywords or [],
                "plate_codes": plate_codes or [],
                "must_check_symbols": must_check_symbols or [],
                "stock_filter_specs": stock_filter_specs or [],
                "plate_code": plate_code,
                "candidate_count": len(result.get("candidates", [])),
                "plate_match_count": len(result.get("plate_matches", [])),
                "stock_filter_result_count": len(result.get("stock_filter_results", [])),
                "warning_count": len(result.get("warnings", [])),
            }
        )
        tool_sequence.append(
            {
                "tool": "futu_explore_theme_candidates",
                "theme": theme,
                "market": market,
                "blocked": False,
            }
        )
        return result

    payload: dict[str, Any] = {
        "task": "build_futu_assisted_theme_discovery_plan",
        "theme": theme,
        "market": market,
        "theme_description": theme_description,
        "required_symbols": required_symbols,
        "tool_contract": {
            "must_call_sequence": [
                "futu_search_screener_catalog",
                "futu_explore_theme_candidates",
            ],
            "must_search_catalog_before_futu_explore": True,
            "catalog_search_purpose": (
                "Query exact candidate fields and Futu plate codes before composing "
                "plate_codes, plate_code, or stock_filter_specs."
            ),
            "candidate_field_query_examples": [
                theme,
                "市值",
                "流通市值",
                "市盈率",
                "成交额",
                "MA",
                "EMA",
                "KDJ",
                "RSI",
                "MACD",
                "BOLL",
                "ROE",
                "ROIC",
            ],
            "tool_result_usage": (
                "Use direct Futu plate/snapshot evidence where present, but keep "
                "important model-hypothesis candidates with explicit validation warnings."
            ),
        },
        "output_contract": {
            "initial_thesis": "active theme thesis; no weights",
            "domain_tree": "layered value-chain map with subdomains and candidates",
            "coverage_requirements": "coverage checklist derived from domain_tree",
            "seed_symbols": "flattened domain_tree candidates plus required symbols",
            "plate_keywords": "keywords used or suitable for Futu plate probing",
            "research_trace": "Futu plate/snapshot evidence relied on by source_id",
            "no_weights": True,
            "no_trade_recommendations": True,
        },
    }
    if previous_plan is not None:
        payload["previous_plan"] = previous_plan.model_dump(mode="json")
        payload["audit_observations"] = observations

    result = agent.run_sync(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        event_stream_handler=pydantic_event_stream_handler("futu_assisted_theme_discovery_agent"),
    )
    if not tool_calls:
        raise ValueError(
            "Futu-assisted discovery agent did not call futu_explore_theme_candidates."
        )
    if not catalog_search_calls:
        raise ValueError(
            "Futu-assisted discovery agent did not call futu_search_screener_catalog "
            "before exploring candidates."
        )
    if not any(not call.get("error") for call in tool_calls):
        raise ValueError(
            "Futu-assisted discovery agent did not complete an unblocked "
            "futu_explore_theme_candidates call after catalog search."
        )
    plan = result.output
    _apply_research_runtime_metadata(plan)
    plan.pydantic_ai = {
        **runtime,
        "usage": usage_metadata(result),
        "futu_tool_calls": tool_calls,
        "futu_catalog_search_calls": catalog_search_calls,
        "futu_tool_sequence": tool_sequence,
        "stock_filter_catalog_summary": _stock_filter_catalog_summary(stock_filter_catalog),
        "offline_screener_catalog_summary": _offline_screener_catalog_summary(screener_catalog),
    }
    return plan


@dataclass
class FutuThemeExplorer:
    """Live Futu exploration tool used by the comparison discovery agent."""

    config: Any | None = None

    def explore(
        self,
        *,
        theme: str,
        market: str = "US",
        plate_keywords: list[str] | None = None,
        plate_codes: list[str] | None = None,
        must_check_symbols: list[str] | None = None,
        stock_filter_specs: list[dict[str, Any]] | None = None,
        plate_code: str | None = None,
        max_candidates: int = 160,
    ) -> dict[str, Any]:
        from .adapters import (
            FutuOpenDConfig,
            MarketDataAdapter,
            _check_ret,
            _import_futu,
            _iter_rows,
            _parse_quote_market,
            _row_get,
            _safe_float,
            _safe_str,
        )

        config = self.config or FutuOpenDConfig.from_env()
        adapter = MarketDataAdapter(config)
        adapter._check_opend()  # noqa: SLF001 - this class is part of the same adapter boundary.
        futu = _import_futu()
        market = _normalize_market(market or getattr(config, "market", "US"))
        keywords = _dedupe_strings(plate_keywords or [])
        explicit_plate_codes = _dedupe_strings(plate_codes or [])
        must_check = [
            normalize_futu_symbol(symbol, market)
            for symbol in _dedupe_strings(must_check_symbols or [])
            if normalize_futu_symbol(symbol, market)
        ]
        limit = _clamp_int(max_candidates, minimum=5, maximum=240)
        warnings: list[str] = []
        plate_matches: list[dict[str, Any]] = []
        stock_filter_results: list[dict[str, Any]] = []
        stock_filter_total_count: int | None = None
        stock_filter_specs = stock_filter_specs or []
        candidates: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        static_plate_lookup = _offline_plate_lookup(market)

        quote_ctx = futu.OpenQuoteContext(host=config.host, port=config.port)
        try:
            seen_plates: set[str] = set()
            plate_lookup: dict[str, dict[str, Any]] = dict(static_plate_lookup)
            if keywords:
                if not plate_lookup:
                    try:
                        ret, plate_data = adapter._quote_call(  # noqa: SLF001
                            quote_ctx.get_plate_list,
                            _parse_quote_market(futu, market),
                            futu.Plate.ALL,
                        )
                        _check_ret(futu, ret, plate_data, "get_plate_list")
                    except Exception as exc:
                        plate_data = []
                        warnings.append(f"Futu get_plate_list failed: {exc}")
                    for _, row in _iter_rows(plate_data):
                        plate_item = {
                            "plate_code": _safe_str(_row_get(row, "code")),
                            "plate_name": _safe_str(_row_get(row, "plate_name", _row_get(row, "stock_name"))),
                            "plate_type": _safe_str(_row_get(row, "plate_type")) or "futu_plate",
                        }
                        if plate_item["plate_code"]:
                            plate_lookup[plate_item["plate_code"]] = plate_item
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    keyword_matches = []
                    for plate_item in plate_lookup.values():
                        plate_code = plate_item["plate_code"]
                        plate_name = plate_item["plate_name"]
                        if not plate_code or plate_code in seen_plates:
                            continue
                        if keyword_lower not in plate_name.lower():
                            continue
                        match = {
                            "keyword": keyword,
                            "plate_code": plate_code,
                            "plate_name": plate_name,
                            "plate_type": plate_item["plate_type"],
                        }
                        keyword_matches.append(match)
                        seen_plates.add(plate_code)
                        if len(keyword_matches) >= 3:
                            break
                    if not keyword_matches:
                        warnings.append(f"No Futu plate matched keyword: {keyword}")
                    plate_matches.extend(keyword_matches)

            for code in explicit_plate_codes:
                normalized_plate = str(code or "").strip().upper()
                if not normalized_plate:
                    continue
                if "." not in normalized_plate:
                    normalized_plate = f"{market}.{normalized_plate}"
                if normalized_plate in seen_plates:
                    continue
                plate_item = plate_lookup.get(normalized_plate)
                if not plate_item:
                    plate_item = {
                        "plate_code": normalized_plate,
                        "plate_name": normalized_plate,
                        "plate_type": "explicit_plate_code",
                    }
                    warnings.append(
                        f"Requested Futu plate_code not found in offline {market} catalog; "
                        f"will still try get_plate_stock: {code}"
                    )
                plate_matches.append(
                    {
                        "keyword": "explicit_plate_code",
                        "plate_code": normalized_plate,
                        "plate_name": plate_item["plate_name"],
                        "plate_type": plate_item["plate_type"],
                    }
                )
                seen_plates.add(normalized_plate)

            for match in plate_matches[:8]:
                try:
                    ret, data = adapter._quote_call(quote_ctx.get_plate_stock, match["plate_code"])  # noqa: SLF001
                    _check_ret(futu, ret, data, f"get_plate_stock({match['plate_code']})")
                except Exception as exc:
                    warnings.append(f"Futu get_plate_stock failed for {match['plate_code']}: {exc}")
                    continue
                sample_symbols = []
                for _, row in list(_iter_rows(data))[: max(5, config.plate_stock_limit)]:
                    code = _safe_str(_row_get(row, "code"))
                    if not code or not code.startswith(f"{market}."):
                        continue
                    sample_symbols.append(code)
                    item = candidates.setdefault(
                        code,
                        {
                            "symbol": code,
                            "name": _safe_str(_row_get(row, "stock_name")) or code,
                            "sources": [],
                            "plate_memberships": [],
                            "must_check": False,
                        },
                    )
                    source_id = f"futu_plate:{match['plate_code']}"
                    if source_id not in item["sources"]:
                        item["sources"].append(source_id)
                    item["plate_memberships"].append(
                        {
                            "plate_code": match["plate_code"],
                            "plate_name": match["plate_name"],
                            "keyword": match["keyword"],
                        }
                    )
                match["sample_symbols"] = sample_symbols[:20]
                match["sample_count"] = len(sample_symbols)

            for code in must_check:
                item = candidates.setdefault(
                    code,
                    {
                        "symbol": code,
                        "name": code,
                        "sources": [],
                        "plate_memberships": [],
                        "must_check": True,
                    },
                )
                item["must_check"] = True
                if "llm_must_check" not in item["sources"]:
                    item["sources"].append("llm_must_check")

            if stock_filter_specs:
                try:
                    filters = _build_futu_stock_filters(futu, stock_filter_specs)
                    ret, data = adapter._quote_call(  # noqa: SLF001
                        quote_ctx.get_stock_filter,
                        _parse_quote_market(futu, market),
                        filters,
                        plate_code=plate_code,
                        begin=0,
                        num=limit,
                    )
                    _check_ret(futu, ret, data, "get_stock_filter")
                    _last_page, all_count, stock_list = data
                    stock_filter_total_count = int(all_count)
                    source_id = "futu_stock_filter"
                    for item in list(stock_list)[:limit]:
                        code = _safe_str(getattr(item, "stock_code", ""))
                        if not code:
                            continue
                        code = normalize_futu_symbol(code, market)
                        if not code or not code.startswith(f"{market}."):
                            continue
                        stock_filter_results.append(
                            {
                                "symbol": code,
                                "name": _safe_str(getattr(item, "stock_name", "")),
                                "price": _safe_float(getattr(item, "cur_price", 0)),
                                "change_rate": _safe_float(getattr(item, "change_rate", 0)),
                                "market_val": _safe_float(getattr(item, "market_val", 0)),
                                "volume": _safe_float(getattr(item, "volume", 0)),
                                "turnover_rate": _safe_float(getattr(item, "turnover_rate", 0)),
                                "pe_ttm": _safe_float(getattr(item, "pe_ttm", 0)),
                                "pb_rate": _safe_float(getattr(item, "pb_rate", 0)),
                            }
                        )
                        candidate = candidates.setdefault(
                            code,
                            {
                                "symbol": code,
                                "name": _safe_str(getattr(item, "stock_name", "")) or code,
                                "sources": [],
                                "plate_memberships": [],
                                "must_check": False,
                            },
                        )
                        if source_id not in candidate["sources"]:
                            candidate["sources"].append(source_id)
                    if not stock_filter_results:
                        warnings.append(
                            "Futu get_stock_filter returned no symbols for the selected filter specs."
                        )
                except Exception as exc:
                    warnings.append(f"Futu get_stock_filter failed: {exc}")

            prioritized_codes = [code for code in must_check if code in candidates]
            prioritized_code_set = set(prioritized_codes)
            prioritized_codes.extend(code for code in candidates if code not in prioritized_code_set)
            codes = prioritized_codes[:limit]
            snapshot_by_code, quote_errors = adapter._get_market_snapshot_result(  # noqa: SLF001
                quote_ctx,
                futu,
                codes,
                warnings,
            )
            valid_count = 0
            unsupported_symbols = []
            for code in codes:
                item = candidates[code]
                row = snapshot_by_code.get(code)
                if row is None:
                    reason = quote_errors.get(code) or "Futu snapshot missing for candidate."
                    item["quote_available"] = False
                    item["quote_error"] = reason
                    unsupported_symbols.append({"symbol": code, "reason": reason})
                    continue
                last_price = _safe_float(_row_get(row, "last_price"))
                item["name"] = _safe_str(_row_get(row, "name")) or item["name"]
                item["quote_available"] = last_price > 0
                item["snapshot"] = {
                    "last_price": last_price,
                    "change_rate": _safe_float(_row_get(row, "change_rate")),
                    "turnover": _safe_float(_row_get(row, "turnover")),
                    "volume": _safe_float(_row_get(row, "volume")),
                    "pe_ttm_ratio": _safe_float(_row_get(row, "pe_ttm_ratio")),
                    "pb_ratio": _safe_float(_row_get(row, "pb_ratio")),
                    "total_market_val": _safe_float(_row_get(row, "total_market_val")),
                    "update_time": _safe_str(_row_get(row, "update_time")),
                }
                if last_price > 0:
                    valid_count += 1

            return {
                "theme": theme,
                "market": market,
                "generated_at": utc_now(),
                "tool": "futu_explore_theme_candidates",
                "plate_keywords": keywords,
                "plate_codes": explicit_plate_codes,
                "must_check_symbols": must_check,
                "plate_matches": plate_matches,
                "stock_filter_specs": stock_filter_specs or [],
                "stock_filter_plate_code": plate_code,
                "stock_filter_results": stock_filter_results,
                "stock_filter_total_count": stock_filter_total_count,
                "candidate_count": len(codes),
                "valid_candidate_count": valid_count,
                "candidates": [candidates[code] for code in codes],
                "unsupported_symbols": unsupported_symbols,
                "data_quality": {
                    "plate_keyword_count": len(keywords),
                    "plate_match_count": len(plate_matches),
                    "stock_filter_result_count": len(stock_filter_results),
                    "snapshot_checked_count": len(codes),
                    "snapshot_valid_count": valid_count,
                },
                "warnings": _dedupe_strings(warnings),
            }
        finally:
            quote_ctx.close()


def _mark_model_assumptions(plan: ThemeDiscoveryPlan) -> None:
    plan.warnings = _dedupe_strings(
        [
            *plan.warnings,
            (
                "Some Futu-assisted discovery candidates may be model hypotheses "
                "pending downstream Futu/SEC validation."
            ),
        ]
    )


def futu_stock_filter_catalog(market: str = "US") -> dict[str, Any]:
    """Return the current SDK-supported stock-filter option catalog.

    Futu exposes these as FtEnum-style class attributes rather than standard
    Python Enum classes. Reflecting the installed SDK keeps the agent prompt in
    sync with the local OpenAPI version and avoids hand-maintained field lists.
    """

    from .adapters import _import_futu

    futu = _import_futu()
    simple, accumulate, financial, pattern, indicator = _stock_field_groups(futu)
    raw_sdk_filter_catalog = {
        "source": "futu_sdk_reflection",
        "sdk_version": getattr(futu, "__version__", "unknown"),
        "market": _normalize_market(market),
        "api": "OpenQuoteContext.get_stock_filter",
        "request_limits": {
            "max_per_page": 200,
            "note": "Futu documents stock-filter pagination and request-rate limits; callers should rate-limit probes.",
        },
        "filter_types": {
            "simple": {
                "fields": simple,
                "params": ["stock_field", "filter_min", "filter_max", "sort", "is_no_filter"],
                "description": "Point-in-time quote/valuation fields.",
            },
            "accumulate": {
                "fields": accumulate,
                "params": ["stock_field", "days", "filter_min", "filter_max", "sort", "is_no_filter"],
                "description": "Accumulated market fields over a day window.",
            },
            "financial": {
                "fields": financial,
                "params": ["stock_field", "quarter", "filter_min", "filter_max", "sort", "is_no_filter"],
                "description": "Financial-statement fields.",
            },
            "pattern": {
                "fields": pattern,
                "params": ["stock_field", "ktype", "consecutive_period", "is_no_filter"],
                "description": "Technical pattern fields.",
            },
            "custom_indicator": {
                "fields": indicator,
                "params": [
                    "stock_field1",
                    "stock_field2",
                    "relative_position",
                    "value",
                    "ktype",
                    "stock_field1_para",
                    "stock_field2_para",
                    "consecutive_period",
                    "is_no_filter",
                ],
                "description": "Indicator-to-indicator or indicator-to-value comparisons.",
            },
        },
        "sort_dir": _enum_values(futu.SortDir),
        "financial_quarter": _enum_values(futu.FinancialQuarter),
        "supported_pattern_ktype": ["K_60M", "K_DAY", "K_WEEK", "K_MON"],
        "relative_position": _enum_values(futu.RelativePosition),
        "stock_filter_spec_shape": {
            "type": "simple | accumulate | financial | pattern | custom_indicator",
            "stock_field": "one of filter_types[type].fields",
            "filter_min": "optional number",
            "filter_max": "optional number",
            "sort": "optional ASCEND | DESCEND",
            "days": "accumulate only; default 1",
            "quarter": "financial only; default ANNUAL",
            "ktype": "pattern/custom_indicator only; default K_DAY",
        },
    }
    return {
        **raw_sdk_filter_catalog,
        "app_screener_catalog": _app_screener_catalog(raw_sdk_filter_catalog),
        "raw_sdk_filter_catalog": raw_sdk_filter_catalog,
    }


def load_futu_screener_catalog_snapshot(
    *,
    market: str = "US",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the pre-generated Futu screener option catalog.

    Runtime theme discovery must not connect to Futu just to enumerate screener
    choices. The snapshot is refreshed manually by
    ``plugins/investment_assistant/scripts/export_futu_screener_catalog.py``.
    """

    catalog_path = Path(path) if path is not None else _DEFAULT_SCREENER_CATALOG_JSON
    if not catalog_path.exists():
        return {
            "source": "offline_snapshot_missing",
            "path": str(catalog_path),
            "market": _normalize_market(market),
            "markets": {},
            "warnings": [
                "Offline Futu screener catalog is missing. Run "
                "plugins/investment_assistant/scripts/export_futu_screener_catalog.py "
                "to refresh the full option document."
            ],
        }
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "source": "offline_snapshot_unreadable",
            "path": str(catalog_path),
            "market": _normalize_market(market),
            "markets": {},
            "warnings": [f"Offline Futu screener catalog could not be read: {exc}"],
        }
    if not isinstance(data, dict):
        return {
            "source": "offline_snapshot_invalid",
            "path": str(catalog_path),
            "market": _normalize_market(market),
            "markets": {},
            "warnings": ["Offline Futu screener catalog root must be an object."],
        }
    data.setdefault("path", str(catalog_path))
    data.setdefault("warnings", [])
    data["market"] = _normalize_market(market)
    return data


def _offline_screener_catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    markets = catalog.get("markets", {})
    market_counts = {}
    if isinstance(markets, dict):
        for market, market_catalog in markets.items():
            if not isinstance(market_catalog, dict):
                continue
            plate_types = market_catalog.get("plates", {})
            market_counts[market] = {
                key: len(value)
                for key, value in plate_types.items()
                if isinstance(value, list)
            }
    return {
        "source": catalog.get("source"),
        "path": catalog.get("path"),
        "generated_at": catalog.get("generated_at"),
        "sdk_version": catalog.get("sdk_version"),
        "market_plate_counts": market_counts,
        "markdown_tree": {
            "root": (catalog.get("markdown_tree") or {}).get("root"),
            "index": (catalog.get("markdown_tree") or {}).get("index"),
            "document_count": len((catalog.get("markdown_tree") or {}).get("documents", []) or []),
        },
        "warning_count": len(catalog.get("warnings", []) or []),
    }


def _offline_plate_lookup(market: str) -> dict[str, dict[str, Any]]:
    catalog = load_futu_screener_catalog_snapshot(market=market)
    market_catalog = (catalog.get("markets") or {}).get(_normalize_market(market), {})
    plates_by_type = market_catalog.get("plates", {}) if isinstance(market_catalog, dict) else {}
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(plates_by_type, dict):
        return lookup
    for plate_type, entries in plates_by_type.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            code = str(entry.get("code") or "").strip().upper()
            if not code:
                continue
            lookup.setdefault(
                code,
                {
                    "plate_code": code,
                    "plate_name": str(entry.get("name") or code).strip() or code,
                    "plate_type": str(entry.get("plate_type") or plate_type).strip() or plate_type,
                },
            )
    return lookup


def search_futu_screener_catalog(
    *,
    query: str,
    market: str = "US",
    category: str | None = None,
    limit: int = 40,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Search the offline Futu screener catalog without calling OpenD."""

    selected_market = _normalize_market(market)
    catalog = load_futu_screener_catalog_snapshot(market=selected_market, path=path)
    query = str(query or "").strip()
    terms = [term.lower() for term in query.replace("/", " ").replace("|", " ").split() if term.strip()]
    category_filter = str(category or "").strip().lower()
    max_results = max(1, min(120, int(limit or 40)))
    market_catalog = (catalog.get("markets") or {}).get(selected_market, {})
    stock_filter_catalog = market_catalog.get("stock_filter_catalog", {}) if isinstance(market_catalog, dict) else {}

    documents = _search_catalog_documents(catalog, selected_market, terms, category_filter, max_results)
    choices = _search_app_choices(stock_filter_catalog, terms, category_filter, max_results)
    stock_fields = _search_stock_fields(stock_filter_catalog, terms, category_filter, max_results)
    plates = _search_plates(market_catalog.get("plates", {}) if isinstance(market_catalog, dict) else {}, terms, category_filter, max_results)

    return {
        "query": query,
        "market": selected_market,
        "category": category,
        "generated_at": catalog.get("generated_at"),
        "catalog_path": catalog.get("path"),
        "markdown_tree": {
            "root": (catalog.get("markdown_tree") or {}).get("root"),
            "index": (catalog.get("markdown_tree") or {}).get("index"),
        },
        "documents": documents,
        "choices": choices,
        "stock_fields": stock_fields,
        "plates": plates,
        "warnings": catalog.get("warnings", []),
    }


def _search_catalog_documents(
    catalog: dict[str, Any],
    market: str,
    terms: list[str],
    category_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    docs = ((catalog.get("markdown_tree") or {}).get("documents") or [])
    results = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("market") not in (None, market):
            continue
        if category_filter and category_filter not in _search_blob(doc):
            continue
        score = _match_score(_search_blob(doc), terms)
        if score <= 0 and terms:
            continue
        if not terms and doc.get("kind") not in {"root_index", "market_index"}:
            continue
        results.append((score, doc))
    results.sort(key=lambda item: (-item[0], str(item[1].get("path", ""))))
    return [item[1] for item in results[:limit]]


def _search_app_choices(
    stock_filter_catalog: dict[str, Any],
    terms: list[str],
    category_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    for category in stock_filter_catalog.get("app_screener_catalog", []) or []:
        category_key = str(category.get("key") or "")
        if category_filter and category_filter not in category_key.lower() and category_filter not in str(category.get("name", "")).lower():
            continue
        for choice in category.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            blob = _search_blob({"category": category, "choice": choice})
            score = _match_score(blob, terms)
            if score <= 0 and terms:
                continue
            item = {
                "category_key": category_key,
                "category_name": category.get("name"),
                "label": choice.get("label"),
                "capability": choice.get("capability"),
                "type": choice.get("type"),
                "fields": choice.get("fields", []),
                "alternate_source": choice.get("alternate_source"),
                "llm_hint": choice.get("llm_hint"),
            }
            results.append((score, item))
    results.sort(key=lambda item: (-item[0], str(item[1].get("category_key", "")), str(item[1].get("label", ""))))
    return [item[1] for item in results[:limit]]


def _search_stock_fields(
    stock_filter_catalog: dict[str, Any],
    terms: list[str],
    category_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    for filter_type, spec in (stock_filter_catalog.get("filter_types") or {}).items():
        if category_filter and category_filter not in str(filter_type).lower():
            continue
        if not isinstance(spec, dict):
            continue
        for field in spec.get("fields", []) or []:
            blob = " ".join([str(filter_type), str(field), _field_search_label(str(field))]).lower()
            score = _match_score(blob, terms)
            if score <= 0 and terms:
                continue
            results.append(
                (
                    score,
                    {
                        "filter_type": filter_type,
                        "field": field,
                        "label": _field_search_label(str(field)),
                        "params": spec.get("params", []),
                        "description": spec.get("description"),
                    },
                )
            )
    results.sort(key=lambda item: (-item[0], str(item[1].get("filter_type", "")), str(item[1].get("field", ""))))
    return [item[1] for item in results[:limit]]


def _search_plates(
    plates_by_type: dict[str, list[dict[str, str]]],
    terms: list[str],
    category_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    seen_codes: set[str] = set()
    if not isinstance(plates_by_type, dict):
        return []
    for plate_type, entries in plates_by_type.items():
        if category_filter and category_filter not in str(plate_type).lower():
            continue
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            blob = _search_blob({"plate_type": plate_type, **entry})
            score = _match_score(blob, terms)
            if score <= 0 and terms:
                continue
            code = str(entry.get("code") or "")
            if code and code in seen_codes:
                continue
            if code:
                seen_codes.add(code)
            results.append(
                (
                    score,
                    {
                        "plate_type": plate_type,
                        "code": code,
                        "name": entry.get("name"),
                        "plate_id": entry.get("plate_id"),
                    },
                )
            )
    results.sort(key=lambda item: (-item[0], str(item[1].get("plate_type", "")), str(item[1].get("name", ""))))
    return [item[1] for item in results[:limit]]


def _match_score(blob: str, terms: list[str]) -> int:
    if not terms:
        return 1
    return sum(1 for term in terms if _contains_search_term(blob, term))


def _contains_search_term(blob: str, term: str) -> bool:
    if len(term) <= 3 and term.isascii() and term.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", blob) is not None
    return term in blob


def _search_blob(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    except Exception:
        return str(value).lower()


_FIELD_SEARCH_LABELS = {
    "MARKET_VAL": "总市值 市值 market cap",
    "FLOAT_MARKET_VAL": "流通市值 float market cap",
    "TOTAL_SHARE": "总股本",
    "FLOAT_SHARE": "流通股本",
    "PE_ANNUAL": "市盈率 静态 年度 PE static",
    "PE_TTM": "市盈率 TTM PE",
    "PB_RATE": "市净率 PB",
    "PS_TTM": "市销率 PS",
    "PCF_TTM": "市现率 PCF",
    "MA_ALIGNMENT_LONG": "均线 MA 多头排列",
    "MA_ALIGNMENT_SHORT": "均线 MA 空头排列",
    "EMA_ALIGNMENT_LONG": "EMA 多头排列",
    "EMA_ALIGNMENT_SHORT": "EMA 空头排列",
    "RSI_GOLD_CROSS_LOW": "RSI 低位金叉",
    "RSI_DEATH_CROSS_HIGH": "RSI 高位死叉",
    "KDJ_GOLD_CROSS_LOW": "KDJ 低位金叉",
    "KDJ_DEATH_CROSS_HIGH": "KDJ 高位死叉",
    "MACD_GOLD_CROSS_LOW": "MACD 低位金叉",
    "MACD_DEATH_CROSS_HIGH": "MACD 高位死叉",
    "BOLL_BREAK_UPPER": "BOLL 突破上轨",
    "BOLL_BREAK_LOWER": "BOLL 跌破下轨",
}


def _field_search_label(field: str) -> str:
    if field in _FIELD_SEARCH_LABELS:
        return _FIELD_SEARCH_LABELS[field]
    parts = [field.replace("_", " ").lower()]
    if field == "MA" or field.startswith("MA_ALIGNMENT") or (field.startswith("MA") and field[2:].isdigit()):
        parts.append("MA")
    for token in ("EMA", "KDJ", "RSI", "MACD", "BOLL", "ROE", "ROIC", "EPS"):
        if token in field:
            parts.append(token)
    return " ".join(parts)


def _stock_filter_catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    filter_types = catalog.get("filter_types", {})
    return {
        "source": catalog.get("source"),
        "sdk_version": catalog.get("sdk_version"),
        "api": catalog.get("api"),
        "field_counts": {
            key: len(value.get("fields", []))
            for key, value in filter_types.items()
            if isinstance(value, dict)
        },
        "sort_dir": catalog.get("sort_dir", []),
        "financial_quarter": catalog.get("financial_quarter", []),
        "supported_pattern_ktype": catalog.get("supported_pattern_ktype", []),
        "relative_position": catalog.get("relative_position", []),
        "app_categories": [
            category.get("key")
            for category in catalog.get("app_screener_catalog", [])
            if isinstance(category, dict)
        ],
    }


def _futu_explore_tool_description(
    catalog: dict[str, Any],
    offline_catalog: dict[str, Any] | None = None,
) -> str:
    """Build the tool-facing capability description shown to the LLM."""

    filter_types = catalog.get("filter_types", {})
    offline_catalog = offline_catalog or {}
    selected_market = _normalize_market(str(offline_catalog.get("market") or catalog.get("market") or "US"))
    markets = offline_catalog.get("markets", {})
    selected_market_catalog = markets.get(selected_market, {}) if isinstance(markets, dict) else {}
    offline_plates = selected_market_catalog.get("plates", {}) if isinstance(selected_market_catalog, dict) else {}
    compact_catalog = {
        "offline_snapshot": {
            "source": offline_catalog.get("source", "offline_snapshot_missing"),
            "path": offline_catalog.get("path"),
            "generated_at": offline_catalog.get("generated_at"),
            "market": selected_market,
            "warnings": offline_catalog.get("warnings", []),
        },
        "offline_documents": _catalog_document_summary(offline_catalog, selected_market),
        "app_screener_categories": [
            {
                "key": "market_quote",
                "name": "行情",
                "choices": [
                    "交易所/市场: market selector, supported US/HK/SH/SZ",
                    "所属行业/概念/板块: prefer exact plate_codes from offline_futu_plate_catalog; not a StockField",
                    {
                        "quote_fields": filter_types.get("simple", {}).get("fields", []),
                    },
                    {
                        "accumulated_market_fields": filter_types.get("accumulate", {}).get("fields", []),
                    },
                ],
            },
            {
                "key": "valuation",
                "name": "估值",
                "stock_filter_fields": [
                    "MARKET_VAL",
                    "FLOAT_MARKET_VAL",
                    "TOTAL_SHARE",
                    "FLOAT_SHARE",
                    "PE_ANNUAL",
                    "PE_TTM",
                    "PB_RATE",
                    "PS_TTM",
                    "PCF_TTM",
                ],
            },
            {
                "key": "dividend",
                "name": "分红",
                "capability": "non_stock_filter",
                "alternate_source": "market snapshot dividend_ttm/dividend_ratio_ttm",
            },
            {
                "key": "technical",
                "name": "技术",
                "pattern_fields": filter_types.get("pattern", {}).get("fields", []),
                "custom_indicator_fields": filter_types.get("custom_indicator", {}).get("fields", []),
                "supported_ktype": catalog.get("supported_pattern_ktype", []),
                "relative_position": catalog.get("relative_position", []),
            },
            {
                "key": "financial",
                "name": "财务",
                "financial_fields": filter_types.get("financial", {}).get("fields", []),
                "quarters": catalog.get("financial_quarter", []),
            },
            {
                "key": "analysis",
                "name": "分析",
                "capability": "external_or_future_adapter",
                "note": "Analyst ratings, target prices, and estimate revisions are not get_stock_filter fields.",
            },
            {
                "key": "options",
                "name": "期权",
                "capability": "option_chain_enrichment",
                "note": "IV, option volume, and expiration context use option-chain enrichment, not stock_filter_specs.",
            },
        ],
        "stock_filter_spec_shape": catalog.get("stock_filter_spec_shape", {}),
        "sort_dir": catalog.get("sort_dir", []),
        "offline_futu_plate_counts": {
            "market": selected_market,
            "counts": {
                plate_type: len(entries)
                for plate_type, entries in (offline_plates if isinstance(offline_plates, dict) else {}).items()
                if isinstance(entries, list)
            },
        },
        "rules": [
            "Use stock_filter_specs only for choices marked as stock_filter fields.",
            "Do not pass dividend, analyst, or options dimensions as stock_filter_specs.",
            "For industry/concept discovery, call futu_search_screener_catalog first and prefer returned plate_codes.",
            "Use plate_keywords only if local catalog search does not contain the needed concept.",
        ],
    }
    return (
        "Explore a portfolio theme through Futu plates, plate constituents, "
        "optional stock-filter probes, must-check symbols, and market snapshots. "
        "Returns compact evidence for layered theme discovery. "
        "Use this App-style screener catalog when choosing tool arguments: "
        + json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"))
    )


def _futu_search_tool_description(offline_catalog: dict[str, Any], market: str) -> str:
    selected_market = _normalize_market(market)
    compact_catalog = {
        "purpose": "Search the local offline Futu screener option catalog.",
        "required_before": "futu_explore_theme_candidates",
        "use_before_composing": ["plate_codes", "plate_code", "stock_filter_specs"],
        "market": selected_market,
        "offline_snapshot": {
            "source": offline_catalog.get("source", "offline_snapshot_missing"),
            "path": offline_catalog.get("path"),
            "generated_at": offline_catalog.get("generated_at"),
            "warnings": offline_catalog.get("warnings", []),
        },
        "offline_documents": _catalog_document_summary(offline_catalog, selected_market),
        "query_examples": [
            "人工智能",
            "半导体",
            "所属行业",
            "所属概念",
            "市值",
            "流通市值",
            "市盈率",
            "估值分位",
            "成交额",
            "MA",
            "EMA",
            "KDJ",
            "RSI",
            "MACD",
            "BOLL",
            "ROE",
            "ROIC",
            "期权",
        ],
        "returns": [
            "documents: local Markdown files to read/search conceptually",
            "choices: App-style screener category choices",
            "stock_fields: exact StockField enums usable in stock_filter_specs",
            "plates: exact Futu plate codes usable as plate_codes",
        ],
    }
    return (
        "Search local cached Futu screener options before choosing Futu tool "
        "arguments. Call this before futu_explore_theme_candidates whenever you "
        "need candidate fields, StockField enums, technical indicators, valuation "
        "fields, or Futu plate_codes. This does not call Futu OpenD. "
        + json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"))
    )


def _catalog_document_summary(catalog: dict[str, Any], market: str) -> dict[str, Any]:
    tree = catalog.get("markdown_tree") or {}
    docs = []
    for doc in tree.get("documents", []) or []:
        if not isinstance(doc, dict):
            continue
        if doc.get("market") not in (None, market):
            continue
        docs.append(
            {
                "path": doc.get("path"),
                "title": doc.get("title"),
                "kind": doc.get("kind"),
                "category": doc.get("category"),
                "count": doc.get("count"),
            }
        )
    return {
        "root": tree.get("root"),
        "index": tree.get("index"),
        "documents": docs,
    }


def _app_screener_catalog(raw_catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Human-oriented screener choices for LLM planning.

    The Futu App exposes a richer screener UI than the single OpenAPI
    ``get_stock_filter`` call. This catalog mirrors that mental model while
    explicitly marking whether a dimension is executable as stock_filter,
    plate/market selection, snapshot enrichment, option-chain enrichment, or a
    currently unavailable/external data source.
    """

    return [
        {
            "key": "market_quote",
            "name": "行情",
            "how_to_use_for_theme_discovery": (
                "Use market/plate to constrain the universe, then quote and accumulated "
                "fields to find liquid, tradable, strong or weak candidates."
            ),
            "choices": [
                {
                    "label": "交易所/市场",
                    "capability": "market_selector",
                    "tool": "get_stock_filter.market",
                    "supported_values": ["US", "HK", "SH", "SZ"],
                    "llm_hint": "Select the listing market. This is not a StockField.",
                },
                {
                    "label": "所属行业/概念/板块",
                    "capability": "plate_selector",
                    "tools": ["get_plate_list", "get_plate_stock", "get_stock_filter.plate_code"],
                    "llm_hint": (
                        "Probe theme words as plate_keywords. If a useful plate_code is found, "
                        "stock_filter can filter within that plate_code."
                    ),
                },
                {
                    "label": "价格/52周位置/量比/委比/每手价格",
                    "capability": "stock_filter",
                    "type": "simple",
                    "fields": [
                        "CUR_PRICE",
                        "CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO",
                        "CUR_PRICE_TO_LOWEST52_WEEKS_RATIO",
                        "HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO",
                        "LOW_PRICE_TO_LOWEST52_WEEKS_RATIO",
                        "VOLUME_RATIO",
                        "BID_ASK_RATIO",
                        "LOT_PRICE",
                    ],
                    "llm_hint": "Use these for tradability, momentum context, and avoiding illiquid tails.",
                },
                {
                    "label": "涨跌幅/振幅/成交量/成交额/换手率",
                    "capability": "stock_filter",
                    "type": "accumulate",
                    "fields": ["CHANGE_RATE", "AMPLITUDE", "VOLUME", "TURNOVER", "TURNOVER_RATE"],
                    "llm_hint": (
                        "Use days plus sort/filter bounds to discover strong momentum or high "
                        "liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days."
                    ),
                },
            ],
        },
        {
            "key": "valuation",
            "name": "估值",
            "how_to_use_for_theme_discovery": (
                "Use valuation as a secondary screen after thematic relevance; do not let cheapness "
                "replace theme fit."
            ),
            "choices": [
                {
                    "label": "市值/流通市值/股本",
                    "capability": "stock_filter",
                    "type": "simple",
                    "fields": ["MARKET_VAL", "FLOAT_MARKET_VAL", "TOTAL_SHARE", "FLOAT_SHARE"],
                    "llm_hint": "Use market cap to separate mega-cap anchors from small high-beta satellites.",
                },
                {
                    "label": "市盈率（静态/TTM）/市净率/市销率/市现率",
                    "capability": "stock_filter",
                    "type": "simple",
                    "fields": ["PE_ANNUAL", "PE_TTM", "PB_RATE", "PS_TTM", "PCF_TTM"],
                    "llm_hint": "Useful for valuation risk flags and relative comparison, not theme discovery alone.",
                },
                {
                    "label": "估值分位/行业估值分位",
                    "capability": "derived_or_future_adapter",
                    "alternate_source": "derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField",
                    "llm_hint": (
                        "Do not pass valuation percentile as stock_filter_specs. Mark it as desired "
                        "downstream enrichment if needed."
                    ),
                },
            ],
        },
        {
            "key": "dividend",
            "name": "分红",
            "how_to_use_for_theme_discovery": "Dividend data is enrichment, not a get_stock_filter field in this SDK.",
            "choices": [
                {
                    "label": "TTM 分红/股息率",
                    "capability": "non_stock_filter",
                    "alternate_source": "get_market_snapshot fields dividend_ttm/dividend_ratio_ttm",
                    "llm_hint": "Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.",
                }
            ],
        },
        {
            "key": "technical",
            "name": "技术",
            "how_to_use_for_theme_discovery": (
                "Use technical filters to discover trend candidates or confirm that a theme is active; "
                "avoid using them as the only reason a name belongs to a theme."
            ),
            "choices": [
                {
                    "label": "指标解读",
                    "capability": "derived_interpretation",
                    "alternate_source": "derive from pattern/custom_indicator StockFields and K-line enrichment",
                    "llm_hint": (
                        "The app-style interpretation layer is not a single OpenAPI enum. Use the "
                        "specific MA/EMA/KDJ/RSI/MACD/BOLL fields below."
                    ),
                },
                {
                    "label": "MA/EMA 均线形态",
                    "capability": "stock_filter",
                    "type": "pattern",
                    "fields": [
                        "MA_ALIGNMENT_LONG",
                        "MA_ALIGNMENT_SHORT",
                        "EMA_ALIGNMENT_LONG",
                        "EMA_ALIGNMENT_SHORT",
                    ],
                    "supported_ktype": raw_catalog["supported_pattern_ktype"],
                    "llm_hint": "Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.",
                },
                {
                    "label": "RSI/KDJ/MACD/BOLL 形态",
                    "capability": "stock_filter",
                    "type": "pattern",
                    "fields": [
                        field
                        for field in raw_catalog["filter_types"]["pattern"]["fields"]
                        if not field.startswith(("MA_", "EMA_"))
                    ],
                    "supported_ktype": raw_catalog["supported_pattern_ktype"],
                    "llm_hint": "Use for oscillator and Bollinger-band signals when discovering active candidates.",
                },
                {
                    "label": "MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较",
                    "capability": "stock_filter",
                    "type": "custom_indicator",
                    "fields": raw_catalog["filter_types"]["custom_indicator"]["fields"],
                    "relative_position": raw_catalog["relative_position"],
                    "supported_ktype": raw_catalog["supported_pattern_ktype"],
                    "llm_hint": "Use only when a simple pattern field is insufficient.",
                },
            ],
        },
        {
            "key": "financial",
            "name": "财务",
            "how_to_use_for_theme_discovery": (
                "Use fundamentals to filter quality and growth after the theme map is drafted."
            ),
            "choices": [
                {
                    "label": "利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS",
                    "capability": "stock_filter",
                    "type": "financial",
                    "fields": raw_catalog["filter_types"]["financial"]["fields"],
                    "quarters": raw_catalog["financial_quarter"],
                    "llm_hint": (
                        "Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, "
                        "debt ratio. Exact numbers still need downstream SEC/fundamental validation."
                    ),
                }
            ],
        },
        {
            "key": "analysis",
            "name": "分析",
            "how_to_use_for_theme_discovery": "Analyst ratings/revisions are not exposed by get_stock_filter here.",
            "choices": [
                {
                    "label": "评级/目标价/一致预期/盈利修正",
                    "capability": "external_or_future_adapter",
                    "alternate_source": "analyst_estimate_revision_adapter",
                    "llm_hint": "Do not invent these facts. Mark as needed evidence if useful.",
                }
            ],
        },
        {
            "key": "options",
            "name": "期权",
            "how_to_use_for_theme_discovery": (
                "Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery."
            ),
            "choices": [
                {
                    "label": "期权活跃度/IV/到期日/Put-Call context",
                    "capability": "option_chain_enrichment",
                    "alternate_source": "get_option_expiration_date + get_option_chain + option snapshots",
                    "llm_hint": (
                        "Use later to assess tradability and income-strategy readiness, not as stock_filter_specs."
                    ),
                }
            ],
        },
    ]


def _stock_field_groups(futu: Any) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    groups: dict[str, list[str]] = {
        "simple": [],
        "accumulate": [],
        "financial": [],
        "pattern": [],
        "indicator": [],
    }
    for name, value in futu.StockField.__dict__.items():
        if name.startswith("_") or name == "NONE" or name.endswith("enum_begin"):
            continue
        if callable(value) or not isinstance(value, str):
            continue
        ok, number = futu.StockField.to_number(value)
        if not ok:
            continue
        if futu.StockField.simple_enum_begin < number < futu.StockField.acc_enum_begin:
            groups["simple"].append(value)
        elif futu.StockField.acc_enum_begin < number < futu.StockField.financial_enum_begin:
            groups["accumulate"].append(value)
        elif futu.StockField.financial_enum_begin < number < futu.StockField.pattern_enum_begin:
            groups["financial"].append(value)
        elif futu.StockField.pattern_enum_begin < number < futu.StockField.indicator_enum_begin:
            groups["pattern"].append(value)
        elif futu.StockField.indicator_enum_begin < number:
            groups["indicator"].append(value)
    return (
        groups["simple"],
        groups["accumulate"],
        groups["financial"],
        groups["pattern"],
        groups["indicator"],
    )


def _enum_values(enum_cls: Any) -> list[str]:
    values: list[str] = []
    for name, value in enum_cls.__dict__.items():
        if name.startswith("_") or name == "NONE" or callable(value):
            continue
        if isinstance(value, str):
            values.append(value)
    return values


def _build_futu_stock_filters(futu: Any, specs: list[dict[str, Any]]) -> list[Any]:
    filters: list[Any] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each stock_filter_specs item must be an object.")
        filter_type = str(spec.get("type") or "simple").strip().lower()
        if filter_type == "simple":
            item = futu.SimpleFilter()
            item.stock_field = _stock_field(futu, spec, "stock_field")
            _apply_range_sort(item, futu, spec)
        elif filter_type == "accumulate":
            item = futu.AccumulateFilter()
            item.stock_field = _stock_field(futu, spec, "stock_field")
            item.days = _clamp_int(spec.get("days", 1), minimum=1, maximum=250)
            _apply_range_sort(item, futu, spec)
        elif filter_type == "financial":
            item = futu.FinancialFilter()
            item.stock_field = _stock_field(futu, spec, "stock_field")
            item.quarter = _enum_value(
                futu.FinancialQuarter,
                spec.get("quarter") or futu.FinancialQuarter.ANNUAL,
                "quarter",
            )
            _apply_range_sort(item, futu, spec)
        elif filter_type == "pattern":
            item = futu.PatternFilter()
            item.stock_field = _stock_field(futu, spec, "stock_field")
            item.ktype = _enum_value(futu.KLType, spec.get("ktype") or futu.KLType.K_DAY, "ktype")
            if spec.get("consecutive_period") is not None:
                item.consecutive_period = _clamp_int(
                    spec.get("consecutive_period"),
                    minimum=1,
                    maximum=250,
                )
            item.is_no_filter = bool(spec.get("is_no_filter", False))
        elif filter_type == "custom_indicator":
            item = futu.CustomIndicatorFilter()
            item.stock_field1 = _stock_field(futu, spec, "stock_field1")
            item.stock_field2 = _stock_field(futu, spec, "stock_field2")
            item.relative_position = _enum_value(
                futu.RelativePosition,
                spec.get("relative_position"),
                "relative_position",
            )
            item.ktype = _enum_value(futu.KLType, spec.get("ktype") or futu.KLType.K_DAY, "ktype")
            if spec.get("value") is not None:
                item.value = float(spec["value"])
            item.stock_field1_para = _numeric_list(spec.get("stock_field1_para", []))
            item.stock_field2_para = _numeric_list(spec.get("stock_field2_para", []))
            if spec.get("consecutive_period") is not None:
                item.consecutive_period = _clamp_int(
                    spec.get("consecutive_period"),
                    minimum=1,
                    maximum=250,
                )
            item.is_no_filter = bool(spec.get("is_no_filter", False))
        else:
            raise ValueError(f"Unsupported Futu stock filter type: {filter_type}")
        filters.append(item)
    if not filters:
        default_filter = futu.SimpleFilter()
        default_filter.stock_field = futu.StockField.MARKET_VAL
        default_filter.is_no_filter = True
        default_filter.sort = futu.SortDir.DESCEND
        filters.append(default_filter)
    return filters


def _apply_range_sort(item: Any, futu: Any, spec: dict[str, Any]) -> None:
    has_range = False
    if spec.get("filter_min") is not None:
        item.filter_min = float(spec["filter_min"])
        has_range = True
    if spec.get("filter_max") is not None:
        item.filter_max = float(spec["filter_max"])
        has_range = True
    if spec.get("sort") is not None:
        item.sort = _enum_value(futu.SortDir, spec.get("sort"), "sort")
    item.is_no_filter = bool(spec.get("is_no_filter", False))
    if not has_range and item.sort is not None:
        item.is_no_filter = True
    elif has_range:
        item.is_no_filter = False


def _stock_field(futu: Any, spec: dict[str, Any], key: str) -> str:
    return _enum_value(futu.StockField, spec.get(key), key)


def _enum_value(enum_cls: Any, value: Any, label: str) -> str:
    if value is None:
        raise ValueError(f"Missing Futu enum field: {label}")
    candidate = str(value).strip().upper()
    if hasattr(enum_cls, candidate):
        candidate = getattr(enum_cls, candidate)
    ok, _number = enum_cls.to_number(candidate)
    if not ok:
        raise ValueError(f"Unsupported Futu enum value for {label}: {value}")
    return candidate


def _numeric_list(value: Any) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Indicator parameter lists must be arrays.")
    return [float(item) for item in value]


def _normalize_market(market: str) -> str:
    return str(market or "US").strip().upper() or "US"


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = maximum
    return max(minimum, min(maximum, parsed))
