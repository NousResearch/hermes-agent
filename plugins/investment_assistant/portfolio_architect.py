"""PydanticAI portfolio-map architect over triaged investment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .candidate_triage import (
    CandidateTriageArtifact,
    CompactTriageDecision,
    TriageCandidateDecision,
)
from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .schemas import InvestmentPolicy, OmittedCandidate, PortfolioMap, PortfolioMaps
from .storage import new_id, utc_now

DEFAULT_DATA_ROOT = Path("data/investment_assistant")
PORTFOLIO_MAPS_FILENAME = "portfolio_maps.json"
ARCHITECT_CONTEXT_FILENAME = "portfolio_architect_context.json"
ARCHITECT_RUN_FILENAME = "portfolio_architect_run.json"


class PortfolioArchitectRunArtifact(BaseModel):
    artifact_type: str = "portfolio_architect_run"
    run_id: str = Field(default_factory=lambda: new_id("par"))
    generated_at: str = Field(default_factory=utc_now)
    root: str
    triage_path: str = ""
    deep_research_path: str = ""
    context_path: str = ""
    portfolio_maps_path: str = ""
    eligible_symbols: list[str] = Field(default_factory=list)
    researched_symbols: list[str] = Field(default_factory=list)
    status: Literal["fresh", "partial", "error"] = "fresh"
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class SelectedPortfolioCandidate(BaseModel):
    symbol: str
    conviction: Literal["core", "high", "medium", "satellite"] = "medium"
    layer_keys: list[str] = Field(default_factory=list)
    role: str = ""
    why_selected: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    suggested_weight_band: tuple[float, float] | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class PostEnrichmentSelectionDecision(BaseModel):
    symbol: str
    decision: Literal["watchlist", "defer", "reject"] = "watchlist"
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    layer_keys: list[str] = Field(default_factory=list)
    reason: str = ""
    substitute_symbols: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class LayerSelectionDecision(BaseModel):
    layer_key: str
    layer_name: str = ""
    selected_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)
    deferred_symbols: list[str] = Field(default_factory=list)
    rationale: str = ""


class PeerTradeoff(BaseModel):
    layer_key: str
    comparable_symbols: list[str] = Field(default_factory=list)
    selected_symbols: list[str] = Field(default_factory=list)
    non_selected_symbols: list[str] = Field(default_factory=list)
    rationale: str = ""


class PostEnrichmentSelection(BaseModel):
    selected_for_portfolio: list[SelectedPortfolioCandidate] = Field(default_factory=list)
    watchlist_after_enrichment: list[PostEnrichmentSelectionDecision] = Field(default_factory=list)
    deferred_after_enrichment: list[PostEnrichmentSelectionDecision] = Field(default_factory=list)
    rejected_after_enrichment: list[PostEnrichmentSelectionDecision] = Field(default_factory=list)
    layer_decisions: list[LayerSelectionDecision] = Field(default_factory=list)
    peer_tradeoffs: list[PeerTradeoff] = Field(default_factory=list)
    selection_summary: str = ""
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PortfolioMapWeightRationale(BaseModel):
    map_id: str
    holding_count_rationale: str = ""
    sleeve_weight_rationale: list[str] = Field(default_factory=list)
    high_beta_position_sizing: list[str] = Field(default_factory=list)
    selected_but_unheld_explanations: list[str] = Field(default_factory=list)
    risk_budget_notes: list[str] = Field(default_factory=list)


class PortfolioArchitectResult(BaseModel):
    artifact_type: str = "portfolio_architect_result"
    theme: str
    selection: PostEnrichmentSelection
    portfolio_maps: PortfolioMaps
    map_weight_rationales: list[PortfolioMapWeightRationale] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


def build_portfolio_maps_from_files(
    *,
    triage_path: str | Path,
    root: str | Path = DEFAULT_DATA_ROOT,
    deep_research_path: str | Path | None = None,
    policy: InvestmentPolicy | None = None,
    output_dir: str | Path | None = None,
    save_context: bool = True,
) -> tuple[PortfolioArchitectResult, PortfolioArchitectRunArtifact]:
    """Build and persist portfolio maps from a saved candidate-triage artifact."""

    data_root = Path(root)
    raw_package = _read_json(Path(triage_path))
    deep_research = _read_json(Path(deep_research_path)) if deep_research_path else {}
    triage = _extract_candidate_triage(raw_package)
    lightweight = _extract_lightweight(raw_package)
    effective_policy = policy or _policy_from_package(raw_package, triage)
    return build_portfolio_maps_from_triage(
        policy=effective_policy,
        triage=triage,
        root=data_root,
        lightweight=lightweight,
        deep_research=deep_research,
        triage_path=Path(triage_path),
        deep_research_path=Path(deep_research_path) if deep_research_path else None,
        output_dir=output_dir,
        save_context=save_context,
    )


def build_portfolio_maps_from_triage(
    *,
    policy: InvestmentPolicy,
    triage: CandidateTriageArtifact,
    root: str | Path = DEFAULT_DATA_ROOT,
    lightweight: dict[str, Any] | None = None,
    deep_research: dict[str, Any] | None = None,
    triage_path: str | Path | None = None,
    deep_research_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_context: bool = True,
) -> tuple[PortfolioArchitectResult, PortfolioArchitectRunArtifact]:
    """Build portfolio maps from triage, filing summaries, SEC data, and Futu data."""

    data_root = Path(root)
    context, warnings = build_architect_context(
        policy=policy,
        triage=triage,
        root=data_root,
        lightweight=lightweight or {},
        deep_research=deep_research or {},
        deep_research_path=Path(deep_research_path) if deep_research_path else None,
    )
    run = PortfolioArchitectRunArtifact(
        root=str(data_root),
        triage_path=str(triage_path or ""),
        deep_research_path=str(deep_research_path or ""),
        eligible_symbols=context["eligible_symbols"],
        researched_symbols=context.get("researched_symbols", []),
        status="partial" if warnings else "fresh",
        warnings=_dedupe(warnings),
    )
    run_dir = Path(output_dir) if output_dir else data_root / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / ARCHITECT_CONTEXT_FILENAME
    maps_path = run_dir / PORTFOLIO_MAPS_FILENAME
    run_path = run_dir / ARCHITECT_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    validation_errors: list[str] = []
    architect_result: PortfolioArchitectResult | None = None
    runtime: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    for attempt in range(1, 3):
        attempt_context = dict(context)
        if validation_errors:
            attempt_context["previous_validation_error"] = validation_errors[-1]
            attempt_context["retry_instruction"] = (
                "Revise the PortfolioMaps output so it passes deterministic validation. "
                "Do not change the input universe. Fix mechanical issues such as "
                "holding totals, sleeve totals, missing required symbols, and omission audits."
                "Every selection.selected_for_portfolio[] item must include a non-empty "
                "role, non-empty why_selected list, and non-empty evidence_refs list. "
                "Every portfolio_maps.maps[].sleeves[] item must include non-empty "
                "holding_symbols and each holding symbol listed there must also appear "
                "in that same map's holdings list."
                "selection.selection_summary, selection.peer_tradeoffs, and "
                "map_weight_rationales are required audit surfaces."
            )
        architect_result, runtime, usage = _run_architect_agent(attempt_context)
        raw_path = run_dir / f"portfolio_maps.attempt_{attempt}.raw.json"
        _write_json(raw_path, architect_result.model_dump(mode="json"))
        try:
            _validate_architect_result(policy, triage, architect_result, context)
            break
        except ValueError as exc:
            validation_errors.append(str(exc))
            if attempt >= 2:
                run.status = "error"
                run.warnings = _dedupe([*run.warnings, *validation_errors])
                run.pydantic_ai = runtime
                run.usage = usage
                run.portfolio_maps_path = str(raw_path)
                _write_json(run_path, run.model_dump(mode="json"))
                raise

    assert architect_result is not None
    result_warnings = [
        *architect_result.warnings,
        *architect_result.selection.warnings,
        *architect_result.portfolio_maps.warnings,
    ]
    run.status = "partial" if warnings or result_warnings or validation_errors else "fresh"
    run.warnings = _dedupe([*warnings, *result_warnings, *validation_errors])
    run.pydantic_ai = runtime
    run.usage = usage
    _write_json(maps_path, architect_result.model_dump(mode="json"))
    run.portfolio_maps_path = str(maps_path)
    _write_json(run_path, run.model_dump(mode="json"))
    return architect_result, run


def build_architect_context(
    *,
    policy: InvestmentPolicy,
    triage: CandidateTriageArtifact,
    root: str | Path = DEFAULT_DATA_ROOT,
    lightweight: dict[str, Any] | None = None,
    deep_research: dict[str, Any] | None = None,
    deep_research_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Assemble architect context without passing discovery output."""

    data_root = Path(root)
    deep_research = deep_research or {}
    deep_research_cards = _deep_research_cards_by_symbol(deep_research)
    unresearched_cards = _deep_research_unresearched_by_symbol(deep_research)
    if deep_research and not deep_research_cards and not unresearched_cards:
        raise ValueError("deep_research_report has no candidate_cards; no architect research surface available.")

    if deep_research_cards or unresearched_cards:
        research_surface_symbols = set(deep_research_cards) | set(unresearched_cards)
        eligible_decisions = [
            decision
            for decision in triage.deep_enrichment_queue
            if decision.symbol.upper() in research_surface_symbols
        ]
    else:
        eligible_decisions = list(triage.deep_enrichment_queue)
    if not eligible_decisions:
        raise ValueError("no eligible architect universe available after applying deep-research report.")

    lightweight_by_symbol = _lightweight_by_symbol(lightweight or {})
    warnings: list[str] = []
    if deep_research_cards or unresearched_cards:
        triage_symbols = {decision.symbol.upper() for decision in triage.deep_enrichment_queue}
        research_surface_symbols = set(deep_research_cards) | set(unresearched_cards)
        extra_research_symbols = sorted(research_surface_symbols - triage_symbols)
        if extra_research_symbols:
            warnings.append(
                "deep_research_report included symbols outside candidate_triage.deep_enrichment_queue and they were ignored: "
                + ", ".join(extra_research_symbols)
            )
        missing_research_symbols = sorted(triage_symbols - research_surface_symbols)
        if missing_research_symbols:
            warnings.append(
                "candidate_triage symbols without deep_research candidate cards or unresearched candidate cards are excluded from architect eligible universe: "
                + ", ".join(missing_research_symbols)
            )
        if unresearched_cards:
            warnings.append(
                "deep_research_report includes unresearched lightweight candidates; architect may consider them, "
                "but should treat them as lower-evidence than candidate_cards: "
                + ", ".join(sorted(set(unresearched_cards) & triage_symbols))
            )

    symbol_materials: dict[str, Any] = {}
    for decision in eligible_decisions:
        symbol = decision.symbol.upper()
        if deep_research_cards or unresearched_cards:
            if symbol in deep_research_cards:
                material, material_warnings = _symbol_material_from_deep_research(
                    symbol=symbol,
                    decision=decision,
                    deep_research_card=deep_research_cards[symbol],
                    futu_enrichment=lightweight_by_symbol.get(symbol, {}),
                )
            else:
                material, material_warnings = _symbol_material_from_unresearched_candidate(
                    symbol=symbol,
                    decision=decision,
                    unresearched_card=unresearched_cards[symbol],
                    futu_enrichment=lightweight_by_symbol.get(symbol, {}),
                )
        else:
            material, material_warnings = _symbol_material(
                data_root=data_root,
                symbol=symbol,
                decision=decision,
                futu_enrichment=lightweight_by_symbol.get(symbol, {}),
            )
        symbol_materials[symbol] = material
        warnings.extend(material_warnings)

    context = {
        "artifact_type": "portfolio_architect_context",
        "generated_at": utc_now(),
        "excluded_upstream_discovery": True,
        "input_boundary": {
            "uses_candidate_triage": True,
            "uses_deep_research_report": bool(deep_research_cards or unresearched_cards),
            "uses_filing_summaries": not bool(deep_research_cards or unresearched_cards),
            "uses_sec_companyfacts": not bool(deep_research_cards or unresearched_cards),
            "uses_futu_enrichment": True,
            "uses_theme_discovery_directly": False,
            "uses_current_portfolio": False,
            "uses_orders_or_trades": False,
        },
        "policy": policy.model_dump(mode="json"),
        "candidate_triage": triage.model_dump(mode="json"),
        "deep_research_report": _compact_deep_research_report(deep_research) if (deep_research_cards or unresearched_cards) else {},
        "deep_research_path": str(deep_research_path or ""),
        "eligible_symbols": [decision.symbol.upper() for decision in eligible_decisions],
        "researched_symbols": sorted(deep_research_cards) if deep_research_cards else [],
        "unresearched_symbols": sorted(unresearched_cards) if unresearched_cards else [],
        "watchlist_symbols": [item.symbol.upper() for item in triage.watchlist],
        "deferred_symbols": [item.symbol.upper() for item in triage.deferred],
        "rejected_symbols": [item.symbol.upper() for item in triage.rejected],
        "symbol_materials": symbol_materials,
        "output_contract": {
            "output_type": "PortfolioArchitectResult",
            "map_count": "2-3",
            "must_perform_post_research_selection_before_weights": bool(deep_research_cards),
            "must_perform_post_enrichment_selection_before_weights": not bool(deep_research_cards),
            "selected_for_portfolio_must_be_subset_of": "eligible_symbols",
            "selected_holdings_must_be_subset_of": "selection.selected_for_portfolio",
            "cash_weight_must_equal_policy_cash_reserve": True,
            "single_name_limit_must_be_respected": True,
            "no_trade_orders_or_current_position_adjustments": True,
            "required_symbols_must_be_included_when_eligible": True,
            "important_unselected_candidates_require_omission_audit": (
                "deep_research high_conviction_candidate/core_candidate"
                if deep_research_cards
                else "candidate_triage critical/high"
            ),
        },
        "warnings": warnings,
    }
    return context, warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        triage_raw = _read_json(Path(args.triage_path))
        triage = _extract_candidate_triage(triage_raw)
        policy = InvestmentPolicy(
            theme=args.theme or triage.theme,
            theme_description=args.theme_description or "",
            required_symbols=_normalize_symbols(args.required_symbols or [], args.market or triage.market),
            objective=args.objective,
            risk_level=args.risk_level,
            target_portfolio_weight=args.target_portfolio_weight,
            cash_reserve=args.cash_reserve,
            single_name_limit=args.single_name_limit,
            allow_options=args.allow_options,
            notes=args.notes or "",
        )
        architect_result, run = build_portfolio_maps_from_files(
            triage_path=args.triage_path,
            root=args.root,
            deep_research_path=args.deep_research_path,
            policy=policy,
            output_dir=args.output_dir,
            save_context=not args.no_save_context,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "run": run.model_dump(mode="json"),
                        "portfolio_architect_result": architect_result.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"run_id: {run.run_id}")
            print(f"status: {run.status}")
            print(f"eligible_symbols: {len(run.eligible_symbols)}")
            print(f"portfolio_maps_path: {run.portfolio_maps_path}")
            if run.context_path:
                print(f"context_path: {run.context_path}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _run_architect_agent(context: dict[str, Any]) -> tuple[PortfolioArchitectResult, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=PortfolioArchitectResult,
        instructions=_ARCHITECT_INSTRUCTIONS,
        agent_kind="portfolio_architect_agent",
        output_retries=2,
        agent_skill_names=["portfolio-architect"],
    )
    result = agent.run_sync(
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("portfolio_architect_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _validate_architect_result(
    policy: InvestmentPolicy,
    triage: CandidateTriageArtifact,
    result: PortfolioArchitectResult,
    context: dict[str, Any] | None = None,
) -> None:
    if result.theme and result.theme != triage.theme:
        raise ValueError(f"PydanticAI returned theme {result.theme!r}, expected {triage.theme!r}.")
    deep_research_decisions = _deep_research_decision_by_symbol(context)
    selected_symbols = _validate_post_enrichment_selection(triage, result.selection, deep_research_decisions)
    _validate_required_symbols_selected(policy, triage, selected_symbols)
    _validate_portfolio_maps(policy, triage, result.portfolio_maps, selected_symbols, deep_research_decisions)
    _validate_weight_rationales(result.portfolio_maps, result.map_weight_rationales)


def _validate_post_enrichment_selection(
    triage: CandidateTriageArtifact,
    selection: PostEnrichmentSelection,
    deep_research_decisions: dict[str, str] | None = None,
) -> set[str]:
    allowed_decisions = {decision.symbol.upper(): decision for decision in triage.deep_enrichment_queue}
    allowed_symbols = set(allowed_decisions)
    if not selection.selected_for_portfolio:
        raise ValueError("PydanticAI selection.selected_for_portfolio is empty.")
    if not selection.selection_summary.strip():
        raise ValueError("PydanticAI selection.selection_summary is required.")
    if not selection.peer_tradeoffs:
        selection.warnings.append("selection.peer_tradeoffs is empty; layer-level candidate tradeoff audit is incomplete.")

    selected: set[str] = set()
    for item in selection.selected_for_portfolio:
        symbol = item.symbol.upper()
        if symbol not in allowed_symbols:
            raise ValueError(f"PydanticAI selection used {item.symbol!r}, which is not in deep_enrichment_queue.")
        if symbol in selected:
            raise ValueError(f"PydanticAI selection duplicated {item.symbol!r}.")
        if item.suggested_weight_band is not None:
            low, high = item.suggested_weight_band
            if low < 0 or high < 0 or low > high or high > 1:
                raise ValueError(f"PydanticAI selection has invalid weight band for {item.symbol!r}.")
        if not item.role.strip():
            raise ValueError(f"PydanticAI selection omitted role for selected candidate {item.symbol!r}.")
        if not item.why_selected:
            raise ValueError(f"PydanticAI selection omitted why_selected for selected candidate {item.symbol!r}.")
        if not item.evidence_refs:
            raise ValueError(f"PydanticAI selection omitted evidence_refs for selected candidate {item.symbol!r}.")
        selected.add(symbol)

    explained: set[str] = set(selected)
    for bucket_name, bucket in [
        ("watchlist_after_enrichment", selection.watchlist_after_enrichment),
        ("deferred_after_enrichment", selection.deferred_after_enrichment),
        ("rejected_after_enrichment", selection.rejected_after_enrichment),
    ]:
        for item in bucket:
            symbol = item.symbol.upper()
            if symbol not in allowed_symbols:
                raise ValueError(f"PydanticAI {bucket_name} used unknown symbol {item.symbol!r}.")
            if symbol in selected:
                raise ValueError(f"PydanticAI both selected and non-selected {item.symbol!r}.")
            if not item.reason.strip():
                raise ValueError(f"PydanticAI {bucket_name} omitted reason for {item.symbol!r}.")
            for substitute in item.substitute_symbols:
                if substitute.upper() not in selected:
                    raise ValueError(
                        f"PydanticAI {bucket_name} for {item.symbol!r} references substitute "
                        f"{substitute!r}, which is not selected_for_portfolio."
                    )
            explained.add(symbol)

    meaningful_tradeoff_layers: set[str] = set()
    for tradeoff in selection.peer_tradeoffs:
        for symbol in [*tradeoff.comparable_symbols, *tradeoff.selected_symbols, *tradeoff.non_selected_symbols]:
            if symbol.upper() not in allowed_symbols:
                raise ValueError(f"PydanticAI peer_tradeoffs used unknown symbol {symbol!r}.")
        explained.update(symbol.upper() for symbol in tradeoff.non_selected_symbols)
        if (tradeoff.comparable_symbols or tradeoff.non_selected_symbols) and not tradeoff.rationale.strip():
            raise ValueError(f"PydanticAI peer_tradeoff {tradeoff.layer_key!r} omitted rationale.")
        if tradeoff.layer_key and tradeoff.rationale.strip() and (
            tradeoff.comparable_symbols or tradeoff.non_selected_symbols
        ):
            meaningful_tradeoff_layers.add(tradeoff.layer_key)

    important_symbols = _important_symbols_for_architect(allowed_decisions, deep_research_decisions or {})
    missing_tradeoff_layers = _missing_peer_tradeoff_layers(
        allowed_decisions,
        selected,
        meaningful_tradeoff_layers,
        important_symbols,
    )
    if missing_tradeoff_layers:
        selection.warnings.append(
            "peer_tradeoffs did not include an explicit selected-vs-non-selected layer audit for: "
            + ", ".join(sorted(missing_tradeoff_layers))
        )

    missing_important = []
    for symbol, decision in allowed_decisions.items():
        if symbol not in important_symbols:
            continue
        if symbol not in explained:
            missing_important.append(symbol)
    if missing_important:
        selection.warnings.append(
            "post-enrichment selection did not explain important triage candidates: "
            + ", ".join(sorted(missing_important))
        )
    return selected


def _validate_required_symbols_selected(
    policy: InvestmentPolicy,
    triage: CandidateTriageArtifact,
    selected: set[str],
) -> None:
    eligible = {decision.symbol.upper() for decision in triage.deep_enrichment_queue}
    missing_required = [
        normalized
        for symbol in policy.required_symbols
        if (normalized := _normalize_symbol(symbol, triage.market)) in eligible and normalized not in selected
    ]
    if missing_required:
        raise ValueError(
            "PydanticAI selection omitted user-required eligible symbols: "
            + ", ".join(sorted(missing_required))
        )


def _missing_peer_tradeoff_layers(
    decisions_by_symbol: dict[str, TriageCandidateDecision],
    selected: set[str],
    meaningful_tradeoff_layers: set[str],
    important_symbols: set[str],
) -> set[str]:
    symbols_by_layer: dict[str, set[str]] = {}
    important_by_layer: dict[str, set[str]] = {}
    for symbol, decision in decisions_by_symbol.items():
        for layer_key in decision.layer_keys:
            if not layer_key:
                continue
            symbols_by_layer.setdefault(layer_key, set()).add(symbol)
            if symbol in important_symbols:
                important_by_layer.setdefault(layer_key, set()).add(symbol)

    missing: set[str] = set()
    for layer_key, symbols in symbols_by_layer.items():
        selected_in_layer = symbols & selected
        important_unselected = important_by_layer.get(layer_key, set()) - selected
        if selected_in_layer and important_unselected and layer_key not in meaningful_tradeoff_layers:
            missing.add(layer_key)
    return missing


def _validate_weight_rationales(
    maps: PortfolioMaps,
    rationales: list[PortfolioMapWeightRationale],
) -> None:
    if not rationales:
        raise ValueError("PydanticAI map_weight_rationales is required.")
    rationale_by_map = {item.map_id: item for item in rationales}
    for portfolio_map in maps.maps:
        rationale = rationale_by_map.get(portfolio_map.map_id)
        if rationale is None:
            raise ValueError(
                f"PydanticAI omitted map_weight_rationales entry for map {portfolio_map.map_id!r}."
            )
        if not rationale.holding_count_rationale.strip():
            raise ValueError(
                f"PydanticAI map_weight_rationales for {portfolio_map.map_id!r} omitted holding_count_rationale."
            )
        if not rationale.sleeve_weight_rationale:
            raise ValueError(
                f"PydanticAI map_weight_rationales for {portfolio_map.map_id!r} omitted sleeve_weight_rationale."
            )
        if not rationale.risk_budget_notes:
            raise ValueError(
                f"PydanticAI map_weight_rationales for {portfolio_map.map_id!r} omitted risk_budget_notes."
            )


def _validate_portfolio_maps(
    policy: InvestmentPolicy,
    triage: CandidateTriageArtifact,
    maps: PortfolioMaps,
    selected_symbols: set[str],
    deep_research_decisions: dict[str, str] | None = None,
) -> None:
    if not maps.maps:
        raise ValueError("PydanticAI returned no portfolio maps.")
    if maps.theme and maps.theme != triage.theme:
        raise ValueError(f"PydanticAI returned theme {maps.theme!r}, expected {triage.theme!r}.")

    allowed_decisions = {decision.symbol.upper(): decision for decision in triage.deep_enrichment_queue}
    important_symbols = _important_symbols_for_architect(allowed_decisions, deep_research_decisions or {})
    required = [
        _normalize_symbol(symbol, triage.market)
        for symbol in policy.required_symbols
        if _normalize_symbol(symbol, triage.market) in selected_symbols
    ]
    max_sleeve = min(policy.target_portfolio_weight, 1 - policy.cash_reserve)

    for portfolio_map in maps.maps:
        map_warnings = _validate_single_map(
            policy=policy,
            portfolio_map=portfolio_map,
            allowed_symbols=selected_symbols,
            allowed_decisions=allowed_decisions,
            important_symbols=important_symbols,
            required_symbols=required,
            max_sleeve=max_sleeve,
        )
        maps.warnings.extend(map_warnings)


def _validate_single_map(
    *,
    policy: InvestmentPolicy,
    portfolio_map: PortfolioMap,
    allowed_symbols: set[str],
    allowed_decisions: dict[str, TriageCandidateDecision],
    important_symbols: set[str],
    required_symbols: list[str],
    max_sleeve: float,
) -> list[str]:
    if not portfolio_map.holdings:
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} has no holdings.")
    if len(portfolio_map.sleeves) < 2:
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} must include at least two sleeves.")
    if abs(portfolio_map.cash_weight - policy.cash_reserve) > 0.0001:
        raise ValueError(
            f"PydanticAI map {portfolio_map.map_id!r} changed cash reserve "
            f"from {policy.cash_reserve:.4f} to {portfolio_map.cash_weight:.4f}."
        )
    if portfolio_map.sleeve_weight > max_sleeve + 0.005:
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} exceeds allowed theme sleeve.")
    if not portfolio_map.positioning.strip():
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted positioning.")
    if not portfolio_map.best_for.strip():
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted best_for.")

    holding_symbols: set[str] = set()
    total_weight = 0.0
    for holding in portfolio_map.holdings:
        symbol = holding.symbol.upper()
        if symbol not in allowed_symbols:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} used symbol {holding.symbol!r}, "
                "which is not in selection.selected_for_portfolio."
            )
        if symbol in holding_symbols:
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} duplicated {holding.symbol!r}.")
        if holding.target_weight > policy.single_name_limit + 0.0001:
            raise ValueError(
                f"PydanticAI map {portfolio_map.map_id!r} assigned {holding.symbol} "
                f"{holding.target_weight:.4f}, above single_name_limit {policy.single_name_limit:.4f}."
            )
        if not holding.rationale.strip():
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted rationale for {holding.symbol!r}.")
        holding_symbols.add(symbol)
        total_weight += holding.target_weight

    missing_required = [symbol for symbol in required_symbols if symbol not in holding_symbols]
    if missing_required:
        raise ValueError(
            f"PydanticAI map {portfolio_map.map_id!r} omitted required symbols: "
            + ", ".join(missing_required)
        )

    sleeve_total = 0.0
    for sleeve in portfolio_map.sleeves:
        if not sleeve.name.strip():
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} has unnamed sleeve.")
        if not sleeve.holding_symbols:
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} sleeve {sleeve.name!r} is empty.")
        sleeve_total += sleeve.target_weight
        for symbol in sleeve.holding_symbols:
            normalized = symbol.upper()
            if normalized not in holding_symbols:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} sleeve {sleeve.name!r} "
                    f"references non-holding {symbol!r}."
                )

    if abs(total_weight - portfolio_map.sleeve_weight) > 0.01:
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} holding weights do not match sleeve_weight.")
    if abs(sleeve_total - portfolio_map.sleeve_weight) > 0.02:
        raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} sleeve weights do not match sleeve_weight.")

    omitted_by_symbol = {item.symbol.upper(): item for item in portfolio_map.omitted_candidates}
    for omitted in portfolio_map.omitted_candidates:
        symbol = omitted.symbol.upper()
        if symbol not in allowed_decisions:
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted unknown symbol {omitted.symbol!r}.")
        if symbol in holding_symbols:
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} both selected and omitted {omitted.symbol!r}.")
        if not omitted.reason.strip():
            raise ValueError(f"PydanticAI map {portfolio_map.map_id!r} omitted {omitted.symbol!r} without reason.")
        for substitute in omitted.substitute_symbols:
            if substitute.upper() not in holding_symbols:
                raise ValueError(
                    f"PydanticAI map {portfolio_map.map_id!r} omission for {omitted.symbol!r} "
                    f"references substitute {substitute!r}, which is not selected."
                )

    missing_important = []
    for symbol in allowed_symbols:
        if symbol not in important_symbols:
            continue
        if symbol in holding_symbols or symbol in omitted_by_symbol:
            continue
        missing_important.append(symbol)
    if missing_important:
        return [
            f"PydanticAI map {portfolio_map.map_id!r} omitted important selected candidates "
            "without map-level omitted_candidates audit: "
            + ", ".join(sorted(missing_important))
        ]
    return []


def _important_symbols_for_architect(
    allowed_decisions: dict[str, TriageCandidateDecision],
    deep_research_decisions: dict[str, str],
) -> set[str]:
    if deep_research_decisions:
        return {
            symbol
            for symbol, decision in deep_research_decisions.items()
            if symbol in allowed_decisions and decision in {"high_conviction_candidate", "core_candidate"}
        }
    return {
        symbol
        for symbol, decision in allowed_decisions.items()
        if decision.priority in {"critical", "high"}
    }


def _symbol_material(
    *,
    data_root: Path,
    symbol: str,
    decision: TriageCandidateDecision,
    futu_enrichment: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    symbol_dir = data_root / "symbols" / symbol
    warnings: list[str] = []
    if not symbol_dir.exists():
        warnings.append(f"{symbol}: symbol directory not found: {symbol_dir}")

    filing_summary_path = symbol_dir / "filing_summary.md"
    filing_summary = _read_text(filing_summary_path)
    if not filing_summary:
        warnings.append(f"{symbol}: missing filing_summary.md")

    filing_summary_meta = _read_json(symbol_dir / "filing_summary.meta.json")
    sec_companyfacts = _read_json(symbol_dir / "sec_companyfacts.json")
    if not sec_companyfacts:
        warnings.append(f"{symbol}: missing sec_companyfacts.json")

    return (
        {
            "symbol": symbol,
            "triage_decision": decision.model_dump(mode="json"),
            "futu_enrichment": futu_enrichment,
            "filing_summary_path": _relative_or_str(filing_summary_path, data_root),
            "filing_summary": filing_summary,
            "filing_summary_meta": filing_summary_meta,
            "sec_companyfacts_path": _relative_or_str(symbol_dir / "sec_companyfacts.json", data_root),
            "sec_companyfacts": sec_companyfacts,
            "filing_metadata": _read_json(symbol_dir / "filing_metadata.json"),
            "manifest_layers": (_read_json(symbol_dir / "manifest.json") or {}).get("layers", {}),
        },
        warnings,
    )


def _symbol_material_from_deep_research(
    *,
    symbol: str,
    decision: TriageCandidateDecision,
    deep_research_card: dict[str, Any],
    futu_enrichment: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if deep_research_card.get("candidate_decision") in {"watchlist", "defer", "reject"}:
        warnings.append(
            f"{symbol}: deep_research candidate_decision is {deep_research_card.get('candidate_decision')}; "
            "architect should usually keep it out of target weights unless explicitly justified."
        )
    if not deep_research_card.get("evidence_refs"):
        warnings.append(f"{symbol}: deep_research card has no evidence_refs.")
    return (
        {
            "symbol": symbol,
            "triage_decision": decision.model_dump(mode="json"),
            "deep_research_card": deep_research_card,
            "futu_enrichment": futu_enrichment,
            "research_evidence_refs": deep_research_card.get("evidence_refs", []),
        },
        warnings,
    )


def _symbol_material_from_unresearched_candidate(
    *,
    symbol: str,
    decision: TriageCandidateDecision,
    unresearched_card: dict[str, Any],
    futu_enrichment: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings = [
        f"{symbol}: no deep_research candidate_card; using unresearched lightweight candidate surface."
    ]
    return (
        {
            "symbol": symbol,
            "triage_decision": decision.model_dump(mode="json"),
            "research_status": "unresearched_lightweight",
            "unresearched_candidate": unresearched_card,
            "futu_enrichment": futu_enrichment,
            "research_evidence_refs": unresearched_card.get("evidence_refs", []),
            "data_gaps": unresearched_card.get("data_gaps", []),
        },
        warnings,
    )


def _deep_research_cards_by_symbol(deep_research: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards = deep_research.get("candidate_cards") if isinstance(deep_research, dict) else None
    if not isinstance(cards, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        symbol = str(card.get("symbol") or "").upper()
        if symbol and symbol not in result:
            result[symbol] = card
    return result


def _deep_research_unresearched_by_symbol(deep_research: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards = deep_research.get("unresearched_candidates") if isinstance(deep_research, dict) else None
    if not isinstance(cards, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        symbol = str(card.get("symbol") or "").upper()
        if symbol and symbol not in result:
            result[symbol] = card
    return result


def _compact_deep_research_report(deep_research: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(deep_research, dict):
        return {}
    return {
        "artifact_type": deep_research.get("artifact_type", "deep_research_report"),
        "theme": deep_research.get("theme", ""),
        "generated_at": deep_research.get("generated_at", ""),
        "source_artifacts": deep_research.get("source_artifacts", {}),
        "research_summary": deep_research.get("research_summary", ""),
        "candidate_cards": deep_research.get("candidate_cards", []),
        "unresearched_candidates": deep_research.get("unresearched_candidates", []),
        "layer_conclusions": deep_research.get("layer_conclusions", []),
        "cross_layer_thesis": deep_research.get("cross_layer_thesis", []),
        "architect_inputs": deep_research.get("architect_inputs", {}),
        "data_gaps": deep_research.get("data_gaps", []),
        "warnings": deep_research.get("warnings", []),
    }


def _deep_research_decision_by_symbol(context_like: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(context_like, dict):
        return {}
    report = context_like.get("deep_research_report")
    if not isinstance(report, dict):
        return {}
    result: dict[str, str] = {}
    for card in report.get("candidate_cards", []) or []:
        if isinstance(card, dict) and card.get("symbol"):
            result[str(card["symbol"]).upper()] = str(card.get("candidate_decision") or "")
    return result


def _extract_candidate_triage(raw: dict[str, Any]) -> CandidateTriageArtifact:
    payload = raw.get("candidate_triage") if isinstance(raw.get("candidate_triage"), dict) else raw
    return CandidateTriageArtifact.model_validate(payload)


def _extract_lightweight(raw: dict[str, Any]) -> dict[str, Any]:
    lightweight = raw.get("lightweight")
    return lightweight if isinstance(lightweight, dict) else {}


def _policy_from_package(raw: dict[str, Any], triage: CandidateTriageArtifact) -> InvestmentPolicy:
    policy_payload = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    if policy_payload:
        return InvestmentPolicy.model_validate(policy_payload)
    return InvestmentPolicy(
        theme=triage.theme,
        theme_description="",
        required_symbols=[],
        objective="balanced",
        risk_level="moderate",
        target_portfolio_weight=0.95,
        cash_reserve=0.05,
        single_name_limit=0.15,
        allow_options=False,
        notes="Default policy inferred by portfolio architect CLI.",
    )


def _lightweight_by_symbol(lightweight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = lightweight.get("candidates") if isinstance(lightweight, dict) else None
    if not isinstance(candidates, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if symbol:
            result[symbol] = item
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-portfolio-architect",
        description="Build AI-authored portfolio maps from candidate-triage artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Generate portfolio maps from saved triage artifacts.")
    build.add_argument("--triage-path", required=True, help="Candidate triage JSON path.")
    build.add_argument("--deep-research-path", help="Deep-research report JSON path.")
    build.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Data root containing symbols/.")
    build.add_argument("--output-dir", help="Directory for context/run/portfolio output.")
    build.add_argument("--theme", default="", help="Theme override; defaults to triage theme.")
    build.add_argument("--theme-description", default="", help="Human-readable theme description.")
    build.add_argument("--market", default="US", help="Market prefix for required symbol normalization.")
    build.add_argument("--objective", choices=["balanced", "growth", "income"], default="balanced")
    build.add_argument("--risk-level", choices=["conservative", "moderate", "aggressive"], default="moderate")
    build.add_argument("--target-portfolio-weight", type=float, default=0.95)
    build.add_argument("--cash-reserve", type=float, default=0.05)
    build.add_argument("--single-name-limit", type=float, default=0.15)
    build.add_argument("--required-symbols", nargs="*", default=[])
    build.add_argument("--allow-options", action="store_true")
    build.add_argument("--notes", default="")
    build.add_argument("--no-save-context", action="store_true")
    build.add_argument("--json", action="store_true", help="Print run and portfolio maps as JSON.")
    return parser


def _normalize_symbols(symbols: list[str], market: str) -> list[str]:
    result: list[str] = []
    for symbol in symbols:
        normalized = _normalize_symbol(symbol, market)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _normalize_symbol(symbol: str, market: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        return ""
    if "." in value:
        return value
    return f"{str(market or 'US').strip().upper()}.{value}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


_ARCHITECT_INSTRUCTIONS = """
You are the investment assistant's Portfolio Architect Agent.

Build target portfolio maps from the supplied portfolio_architect_context.
The context already contains the refined inputs. Do not ask for or rely on the
theme discovery artifact directly.

Evidence boundary:
- Use candidate_triage as the universe authority.
- When deep_research_report is present, use it as the primary research evidence.
- Use deep_research_report.candidate_cards for symbol-level quality, exposure,
  risks, confidence, and evidence_refs.
- Use deep_research_report.unresearched_candidates only as a lower-confidence
  visibility surface for optional/profile candidates that were not deeply read;
  they may be selected only with explicit uncertainty and data-gap rationale.
- Use deep_research_report.layer_conclusions for same-layer peer tradeoffs.
- Use symbol_materials[].futu_enrichment for market/technical/liquidity context.
- If deep_research_report is present, do not reinterpret raw filing summaries or
  SEC/companyfacts; those have already been summarized upstream.
- If deep_research_report is absent, use symbol_materials[].filing_summary as
  qualitative filing evidence and symbol_materials[].sec_companyfacts as
  structured numeric evidence.
- Do not introduce symbols outside eligible_symbols.
- Do not infer current holdings, order parameters, or trading actions.

Output requirements:
- Return PortfolioArchitectResult only.
- Inside the same answer, perform two steps:
  1. post-research selection
  2. portfolio map construction
- Do not assign weights until after selection.selected_for_portfolio is defined.
- selection.selection_summary must explain the overall narrowing logic.
- Every selection.selected_for_portfolio item must include:
  - role: the candidate's construction role
  - why_selected: one or more evidence-backed reasons
  - evidence_refs: stable refs from deep_research candidate cards or supplied context
- For each layer or economic exposure, compare candidates with overlapping or
  substitutable exposure. Select representative candidates for final
  construction, keep plausible but lower-conviction candidates in
  watchlist_after_enrichment, and defer/reject candidates only with supplied
  evidence.
- selection.peer_tradeoffs must record important same-layer or substitutable
  candidate tradeoffs. Do not leave peer_tradeoffs empty when a layer has
  multiple plausible candidates.
- A peer_tradeoff is not a layer inventory. For each layer where some
  high-priority candidates are selected and some are not selected, include:
  comparable_symbols, selected_symbols, non_selected_symbols, and rationale.
- Do not assume every deep-enrichment candidate deserves a portfolio weight.
- When deep_research candidate_decision is watchlist, defer, or reject, do not
  assign a target weight unless you give an explicit exception reason from the
  supplied report.
- Prefer high_conviction_candidate and core_candidate as construction inputs;
  use satellite_candidate for style expression, completeness, or controlled
  higher-beta exposure.
- holdings in portfolio_maps must be a subset of selection.selected_for_portfolio.
- map_weight_rationales must explain, for each map:
  - why the holding count is appropriate
  - why each major sleeve receives its weight
  - how high-beta exposures are sized
  - any major selected candidates intentionally omitted from a particular map
    when that omission is central to the map's positioning
  - how risk budget shaped the allocation
- Produce two or three distinct maps.
- Each map must preserve cash_weight exactly from policy.cash_reserve.
- Each map's sleeve_weight must equal the sum of selected holding weights.
- Each holding must have a role, rationale, and evidence_refs.
- Each sleeve must include a non-empty holding_symbols list.
- Every sleeve.holding_symbols entry must match a symbol in that same map's
  holdings list.
- The sum of holdings assigned to a sleeve should be consistent with that
  sleeve's target_weight.
- Evidence refs should use stable labels like triage:US.SNDK,
  futu:US.SNDK, filing_summary:US.SNDK, sec_companyfacts:US.SNDK, or the
  evidence_refs already present in deep_research candidate cards.
- Include selection-level omission audit for researched high-conviction/core
  candidates that are not selected_for_portfolio. A selected candidate does not
  have to appear in every map; maps may express different styles, but major
  omissions should be explained when they materially affect positioning.
- Do not produce buy/sell/hold recommendations, price targets, trigger prices,
  simulated orders, options strategies, or current-position adjustments.
"""
