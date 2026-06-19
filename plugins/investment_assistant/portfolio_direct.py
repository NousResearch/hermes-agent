"""PydanticAI direct portfolio map generation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .portfolio_weight_formula import PORTFOLIO_STYLE_PROFILES, PortfolioStyle
from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .storage import new_id, utc_now

DIRECT_PORTFOLIO_CONTEXT_FILENAME = "direct_portfolio_context.json"
DIRECT_PORTFOLIO_MAP_FILENAME = "direct_portfolio_map.json"
DIRECT_PORTFOLIO_RUN_FILENAME = "direct_portfolio_run.json"
DEFAULT_PRECISION = 0.001


class DirectPortfolioHolding(BaseModel):
    symbol: str
    sleeve_key: str
    target_weight: float = Field(ge=0, le=1)
    suggested_weight_band: tuple[float, float] | None = None
    role: str = ""
    rationale: str = ""
    why_this_weight: str = ""
    why_not_higher: str = ""
    why_not_lower: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)


class DirectPortfolioSleeve(BaseModel):
    sleeve_key: str
    sleeve_name: str = ""
    target_weight: float = Field(ge=0, le=1)
    holding_symbols: list[str] = Field(default_factory=list)
    rationale: str = ""
    why_this_weight: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class DirectPortfolioMap(BaseModel):
    artifact_type: str = "direct_portfolio_map"
    map_id: str = Field(default_factory=lambda: new_id("dpm"))
    generated_at: str = Field(default_factory=utc_now)
    theme: str = ""
    portfolio_style: PortfolioStyle = "balanced"
    name: str = ""
    thesis: str = ""
    sleeve_weight: float = Field(ge=0, le=1)
    cash_weight: float = Field(ge=0, le=1)
    sleeves: list[DirectPortfolioSleeve] = Field(default_factory=list)
    holdings: list[DirectPortfolioHolding] = Field(default_factory=list)
    allocation_logic: list[str] = Field(default_factory=list)
    omission_audit: list[str] = Field(default_factory=list)
    risk_summary: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class DirectPortfolioRunArtifact(BaseModel):
    artifact_type: str = "direct_portfolio_run"
    run_id: str = Field(default_factory=lambda: new_id("dpr"))
    generated_at: str = Field(default_factory=utc_now)
    context_path: str = ""
    map_path: str = ""
    status: Literal["fresh", "partial", "error"] = "fresh"
    eligible_symbols: list[str] = Field(default_factory=list)
    researched_symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


def build_direct_portfolio_from_files(
    *,
    deep_research_path: str | Path,
    user_intent: str = "",
    portfolio_style: PortfolioStyle = "balanced",
    sleeve_weight: float = 0.95,
    cash_weight: float = 0.05,
    single_name_limit: float = 0.15,
    precision: float = DEFAULT_PRECISION,
    output_dir: str | Path | None = None,
    save_context: bool = True,
) -> tuple[DirectPortfolioMap, DirectPortfolioRunArtifact]:
    deep_research = _read_json(Path(deep_research_path))
    context = build_direct_portfolio_context(
        deep_research=deep_research,
        user_intent=user_intent,
        portfolio_style=portfolio_style,
        sleeve_weight=sleeve_weight,
        cash_weight=cash_weight,
        single_name_limit=single_name_limit,
        precision=precision,
    )
    run = DirectPortfolioRunArtifact(
        eligible_symbols=sorted(context["eligible_symbols"]),
        researched_symbols=sorted(context["researched_symbols"]),
    )
    run_dir = Path(output_dir) if output_dir else Path(".dev") / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / DIRECT_PORTFOLIO_CONTEXT_FILENAME
    map_path = run_dir / DIRECT_PORTFOLIO_MAP_FILENAME
    run_path = run_dir / DIRECT_PORTFOLIO_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    portfolio_map, runtime, usage = run_direct_portfolio_agent(context)
    _validate_direct_portfolio_map(context, portfolio_map)
    portfolio_map.pydantic_ai = runtime
    portfolio_map.usage = usage
    run.status = "partial" if portfolio_map.warnings or portfolio_map.data_gaps else "fresh"
    run.warnings = _dedupe([*portfolio_map.warnings, *portfolio_map.data_gaps])
    run.pydantic_ai = runtime
    run.usage = usage
    _write_json(map_path, portfolio_map.model_dump(mode="json"))
    run.map_path = str(map_path)
    _write_json(run_path, run.model_dump(mode="json"))
    return portfolio_map, run


def build_direct_portfolio_context(
    *,
    deep_research: dict[str, Any],
    user_intent: str = "",
    portfolio_style: PortfolioStyle = "balanced",
    sleeve_weight: float = 0.95,
    cash_weight: float = 0.05,
    single_name_limit: float = 0.15,
    precision: float = DEFAULT_PRECISION,
) -> dict[str, Any]:
    if portfolio_style not in PORTFOLIO_STYLE_PROFILES:
        allowed = ", ".join(sorted(PORTFOLIO_STYLE_PROFILES))
        raise ValueError(f"Unknown portfolio_style {portfolio_style!r}; allowed: {allowed}")
    cards = _cards_by_symbol(deep_research.get("candidate_cards", []))
    unresearched = _cards_by_symbol(deep_research.get("unresearched_candidates", []))
    if not cards:
        raise ValueError("deep_research_report has no candidate_cards.")

    eligible_decisions = {"high_conviction_candidate", "core_candidate", "satellite_candidate"}
    candidate_cards = {
        symbol: card
        for symbol, card in cards.items()
        if str(card.get("candidate_decision") or "") in eligible_decisions
    }
    for symbol, card in unresearched.items():
        candidate_cards.setdefault(
            symbol,
            {
                **card,
                "candidate_decision": "unresearched_lightweight",
                "confidence": "low",
                "theme_exposure": "unknown",
                "business_quality": "unknown",
                "exposure_summary": card.get("reason", ""),
            },
        )
    if not candidate_cards:
        raise ValueError("deep_research_report has no eligible candidates.")

    layer_conclusions = deep_research.get("layer_conclusions", [])
    layers = [
        {
            "layer_key": str(layer.get("layer_key") or ""),
            "layer_name": str(layer.get("layer_name") or ""),
            "selected_symbols": [str(symbol).upper() for symbol in layer.get("selected_symbols", [])],
            "watchlist_symbols": [str(symbol).upper() for symbol in layer.get("watchlist_symbols", [])],
            "peer_tradeoff_summary": str(layer.get("peer_tradeoff_summary") or ""),
            "unresolved_questions": (layer.get("unresolved_questions") or [])[:5],
        }
        for layer in layer_conclusions
        if isinstance(layer, dict) and layer.get("layer_key")
    ]

    compact_cards = {}
    for symbol, card in sorted(candidate_cards.items()):
        compact_cards[symbol] = {
            "symbol": symbol,
            "research_status": "deep_researched" if symbol in cards else "unresearched_lightweight",
            "candidate_decision": card.get("candidate_decision", ""),
            "confidence": card.get("confidence", ""),
            "layer_keys": card.get("layer_keys", []),
            "theme_exposure": card.get("theme_exposure", ""),
            "business_quality": card.get("business_quality", ""),
            "exposure_summary": card.get("exposure_summary", ""),
            "filing_takeaways": (card.get("filing_takeaways") or [])[:6],
            "key_risks": (card.get("key_risks") or [])[:6],
            "peer_positioning": card.get("peer_positioning", ""),
            "evidence_refs": card.get("evidence_refs", []),
            "data_gaps": (card.get("data_gaps") or [])[:6],
        }

    return {
        "artifact_type": "direct_portfolio_context",
        "generated_at": utc_now(),
        "user_intent": user_intent,
        "theme": deep_research.get("theme", ""),
        "policy": {
            "sleeve_weight": sleeve_weight,
            "cash_weight": cash_weight,
            "single_name_limit": single_name_limit,
            "portfolio_style": portfolio_style,
            "portfolio_style_profile": PORTFOLIO_STYLE_PROFILES[portfolio_style],
            "precision": precision,
        },
        "eligible_symbols": sorted(candidate_cards),
        "researched_symbols": sorted(cards),
        "unresearched_symbols": sorted(set(candidate_cards) - set(cards)),
        "deep_research_summary": {
            "research_summary": deep_research.get("research_summary", ""),
            "cross_layer_thesis": (deep_research.get("cross_layer_thesis") or [])[:12],
            "data_gaps": (deep_research.get("data_gaps") or [])[:12],
            "warnings": (deep_research.get("warnings") or [])[:12],
        },
        "layers": layers,
        "candidate_cards": compact_cards,
        "output_contract": {
            "output_type": "DirectPortfolioMap",
            "llm_authors_final_weights": True,
            "deterministic_code_only_validates": True,
            "holdings_must_be_subset_of": "eligible_symbols",
            "cash_weight_must_equal_policy_cash_weight": True,
            "holdings_sum_must_equal_policy_sleeve_weight": True,
            "single_name_limit_must_be_respected": True,
            "do_not_output_orders_or_trade_actions": True,
        },
    }


def run_direct_portfolio_agent(context: dict[str, Any]) -> tuple[DirectPortfolioMap, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=DirectPortfolioMap,
        instructions=_DIRECT_PORTFOLIO_INSTRUCTIONS,
        agent_kind="direct_portfolio_agent",
        output_retries=2,
        agent_skill_names=["portfolio-architect"],
    )
    result = agent.run_sync(
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("direct_portfolio_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _validate_direct_portfolio_map(context: dict[str, Any], portfolio_map: DirectPortfolioMap) -> None:
    policy = context["policy"]
    precision = float(policy["precision"])
    tolerance = max(precision * 2, 0.0001)
    eligible = {str(symbol).upper() for symbol in context["eligible_symbols"]}
    if portfolio_map.portfolio_style != policy["portfolio_style"]:
        raise ValueError(
            f"direct portfolio style mismatch: {portfolio_map.portfolio_style!r} "
            f"!= {policy['portfolio_style']!r}"
        )
    if abs(portfolio_map.cash_weight - float(policy["cash_weight"])) > tolerance:
        raise ValueError("direct portfolio cash_weight does not match policy.")
    if abs(portfolio_map.sleeve_weight - float(policy["sleeve_weight"])) > tolerance:
        raise ValueError("direct portfolio sleeve_weight does not match policy.")
    total_holdings = sum(holding.target_weight for holding in portfolio_map.holdings)
    if abs(total_holdings - float(policy["sleeve_weight"])) > tolerance:
        raise ValueError("direct portfolio holdings do not sum to sleeve_weight.")
    if not portfolio_map.holdings:
        raise ValueError("direct portfolio holdings are empty.")
    seen: set[str] = set()
    for holding in portfolio_map.holdings:
        symbol = holding.symbol.upper()
        if symbol not in eligible:
            raise ValueError(f"direct portfolio used symbol outside eligible universe: {holding.symbol!r}")
        if symbol in seen:
            raise ValueError(f"direct portfolio duplicated holding: {holding.symbol!r}")
        seen.add(symbol)
        if holding.target_weight > float(policy["single_name_limit"]) + tolerance:
            raise ValueError(f"direct portfolio exceeds single_name_limit: {symbol}")
        if not holding.sleeve_key.strip():
            raise ValueError(f"direct portfolio holding {symbol} omitted sleeve_key.")
        if not holding.rationale.strip() or not holding.why_this_weight.strip():
            raise ValueError(f"direct portfolio holding {symbol} omitted rationale or why_this_weight.")
        if not holding.evidence_refs:
            raise ValueError(f"direct portfolio holding {symbol} omitted evidence_refs.")
        if holding.suggested_weight_band is not None:
            low, high = holding.suggested_weight_band
            if low < 0 or high < 0 or low > high or high > 1:
                raise ValueError(f"direct portfolio holding {symbol} has invalid suggested_weight_band.")
    sleeve_symbols = set()
    for sleeve in portfolio_map.sleeves:
        if not sleeve.sleeve_key.strip():
            raise ValueError("direct portfolio sleeve omitted sleeve_key.")
        if not sleeve.holding_symbols:
            raise ValueError(f"direct portfolio sleeve {sleeve.sleeve_key!r} has no holding_symbols.")
        if not sleeve.rationale.strip() or not sleeve.why_this_weight.strip():
            raise ValueError(f"direct portfolio sleeve {sleeve.sleeve_key!r} omitted rationale or why_this_weight.")
        sleeve_symbols.update(symbol.upper() for symbol in sleeve.holding_symbols)
        sleeve_sum = sum(
            holding.target_weight
            for holding in portfolio_map.holdings
            if holding.sleeve_key == sleeve.sleeve_key
        )
        if abs(sleeve_sum - sleeve.target_weight) > tolerance:
            raise ValueError(f"direct portfolio sleeve {sleeve.sleeve_key!r} weight does not match holdings.")
    if sleeve_symbols != seen:
        raise ValueError("direct portfolio sleeve holding_symbols do not match holdings.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        portfolio_map, run = build_direct_portfolio_from_files(
            deep_research_path=args.deep_research_path,
            user_intent=args.intent or "",
            portfolio_style=args.portfolio_style,
            sleeve_weight=args.sleeve_weight,
            cash_weight=args.cash_weight,
            single_name_limit=args.single_name_limit,
            precision=args.precision,
            output_dir=args.output_dir,
            save_context=not args.no_save_context,
        )
        payload = {
            "run": run.model_dump(mode="json"),
            "direct_portfolio_map": portfolio_map.model_dump(mode="json"),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run_id: {run.run_id}")
            print(f"status: {run.status}")
            print(f"map_path: {run.map_path}")
            print(f"holding_count: {len(portfolio_map.holdings)}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-direct-portfolio",
        description="Generate direct LLM-authored portfolio weights from deep research artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Generate one direct LLM-authored portfolio map.")
    build.add_argument("--deep-research-path", required=True, help="deep_research_report JSON path.")
    build.add_argument("--intent", default="", help="User objective for the direct map.")
    build.add_argument("--portfolio-style", choices=sorted(PORTFOLIO_STYLE_PROFILES), default="balanced")
    build.add_argument("--sleeve-weight", type=float, default=0.95)
    build.add_argument("--cash-weight", type=float, default=0.05)
    build.add_argument("--single-name-limit", type=float, default=0.15)
    build.add_argument("--precision", type=float, default=DEFAULT_PRECISION)
    build.add_argument("--output-dir", help="Directory for run artifacts.")
    build.add_argument("--no-save-context", action="store_true")
    build.add_argument("--json", action="store_true")
    return parser


def _cards_by_symbol(cards: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(cards, list):
        return result
    for card in cards:
        if isinstance(card, dict) and card.get("symbol"):
            result[str(card["symbol"]).upper()] = card
    return result


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


_DIRECT_PORTFOLIO_INSTRUCTIONS = """
You are the investment assistant's Direct Portfolio Map Agent.

Return DirectPortfolioMap only. In this experiment, you directly author the
final target weights. Deterministic code will validate mechanical constraints
but will not calculate weights for you.

Evidence boundary:
- Use only the supplied direct_portfolio_context.
- Do not use web search, model memory, current holdings, outside market facts,
  price targets, or trade actions.
- candidate_cards are the evidence surface. Use deep_researched candidates as
  higher-confidence than unresearched_lightweight candidates.
- Do not introduce symbols outside eligible_symbols.

Construction process:
- First read policy.portfolio_style and policy.portfolio_style_profile.
- Make a global cross-sleeve judgment before assigning weights. Do not
  mechanically allocate a budget to every sleeve and then fill each sleeve.
- Choose the most important theme engines, bottlenecks, anchors, and satellites
  for the selected style.
- A candidate may be more important than a sleeve peer even if it competes with
  more symbols. Compare candidates globally when sizing final weights.
- For bottleneck_barbell, emphasize scarce infrastructure/supply-chain
  bottlenecks plus a few core anchors. Keep weak or peripheral sleeves small or
  zero.
- For concentrated_growth, use a compact map and avoid false precision across
  many small positions.
- For balanced, use broader coverage but still avoid equal-weight behavior.

Sizing requirements:
- cash_weight must equal policy.cash_weight.
- sleeve_weight must equal policy.sleeve_weight.
- The sum of holding target_weight values must equal policy.sleeve_weight.
- No holding may exceed policy.single_name_limit.
- Use decimal weights, e.g. 0.10 for 10%.
- Every holding must include role, rationale, why_this_weight, why_not_higher,
  why_not_lower, evidence_refs, and key_risks where available.
- Every sleeve must include holding_symbols and target_weight equal to the sum
  of its holdings.
- Include allocation_logic explaining the high-level weighting philosophy.
- Include omission_audit for important candidates that are deliberately not
  held or held very small.

Do not output:
- buy/sell/hold advice
- entry prices
- trigger prices
- simulated orders
- options strategies
- current-position adjustments
"""
