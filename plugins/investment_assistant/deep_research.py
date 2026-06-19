"""PydanticAI deep-research agent over selected filing summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .storage import new_id, utc_now

DEFAULT_DATA_ROOT = Path("data/investment_assistant")
DEEP_RESEARCH_REPORT_FILENAME = "deep_research_report.json"
DEEP_RESEARCH_CONTEXT_FILENAME = "deep_research_context.json"
DEEP_RESEARCH_RUN_FILENAME = "deep_research_run.json"


class DeepResearchRunArtifact(BaseModel):
    artifact_type: str = "deep_research_run"
    run_id: str = Field(default_factory=lambda: new_id("drr"))
    generated_at: str = Field(default_factory=utc_now)
    root: str
    intake_path: str = ""
    triage_path: str = ""
    context_path: str = ""
    report_path: str = ""
    researched_symbols: list[str] = Field(default_factory=list)
    batch_count: int = 1
    batch_paths: list[str] = Field(default_factory=list)
    status: Literal["fresh", "partial", "error"] = "fresh"
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class FundamentalSnapshot(BaseModel):
    revenue_signal: str = ""
    profitability_signal: str = ""
    balance_sheet_signal: str = ""
    numeric_highlights: list[str] = Field(default_factory=list)
    metric_gaps: list[str] = Field(default_factory=list)


class CandidateResearchCard(BaseModel):
    symbol: str
    layer_keys: list[str] = Field(default_factory=list)
    intake_action: str = ""
    original_priority: Literal["critical", "high", "medium", "low"] = "medium"
    theme_exposure: Literal["direct", "strong", "mixed", "indirect", "weak", "unknown"] = "unknown"
    exposure_summary: str = ""
    business_quality: Literal["excellent", "strong", "mixed", "weak", "unknown"] = "unknown"
    fundamental_snapshot: FundamentalSnapshot = Field(default_factory=FundamentalSnapshot)
    filing_takeaways: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    peer_positioning: str = ""
    candidate_decision: Literal[
        "core_candidate",
        "high_conviction_candidate",
        "satellite_candidate",
        "watchlist",
        "defer",
        "reject",
    ] = "watchlist"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_refs: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class LayerResearchConclusion(BaseModel):
    layer_key: str
    layer_name: str = ""
    selected_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)
    deferred_symbols: list[str] = Field(default_factory=list)
    rejected_symbols: list[str] = Field(default_factory=list)
    peer_tradeoff_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)


class UnresearchedCandidateCard(BaseModel):
    symbol: str
    layer_keys: list[str] = Field(default_factory=list)
    intake_action: str = ""
    original_priority: Literal["critical", "high", "medium", "low"] = "medium"
    reason: str = ""
    available_light_materials: list[str] = Field(default_factory=list)
    missing_or_stale_materials: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class DeepResearchReport(BaseModel):
    artifact_type: str = "deep_research_report"
    theme: str = ""
    generated_at: str = Field(default_factory=utc_now)
    source_artifacts: dict[str, Any] = Field(default_factory=dict)
    research_summary: str = ""
    candidate_cards: list[CandidateResearchCard] = Field(default_factory=list)
    unresearched_candidates: list[UnresearchedCandidateCard] = Field(default_factory=list)
    layer_conclusions: list[LayerResearchConclusion] = Field(default_factory=list)
    cross_layer_thesis: list[str] = Field(default_factory=list)
    architect_inputs: dict[str, Any] = Field(default_factory=dict)
    data_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


_DEEP_RESEARCH_INSTRUCTIONS = """
You are the investment assistant's Deep Research Agent.

Use the deep-research skill. Read the selected filing summaries and compact
numeric/profile artifacts. Produce a source-grounded research dataset for a
future portfolio architect.

Evidence boundary:
- Use only the JSON payload supplied by the caller.
- Do not invent market facts, financial numbers, customer names, news, target
  prices, or trade actions.
- Exact numeric facts must come from sec_companyfacts or profile fields in the
  payload, not from filing_summary prose.
- Filing summaries are qualitative evidence: management commentary, business
  mix, demand signals, margin/cost signals, AI/data-center relevance, risks,
  and data-quality notes.

Decision boundary:
- Do not assign portfolio weights.
- Do not write buy/sell/hold recommendations.
- Do not produce orders or construction plans.
- You may classify each candidate for the next architect stage as
  core_candidate, high_conviction_candidate, satellite_candidate, watchlist,
  defer, or reject.

Research process:
- Compare candidates within their layers.
- Surface same-layer tradeoffs and unresolved questions.
- If evidence is partial, stale, truncated, or missing, lower confidence and
  record data_gaps.
- Preserve evidence_refs using the provided reference ids such as
  filing_summary:US.NVDA, sec_companyfacts:US.NVDA, fmp_profile:US.NVDA,
  intake:US.NVDA, and triage:US.NVDA.

Return DeepResearchReport only.
"""


def build_deep_research_report_from_files(
    *,
    intake_path: str | Path,
    root: str | Path = DEFAULT_DATA_ROOT,
    triage_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_optional: bool = False,
    max_symbols: int | None = None,
    batch_size: int | None = None,
    save_context: bool = True,
) -> tuple[DeepResearchReport, DeepResearchRunArtifact]:
    """Build and persist a deep-research report from a research-intake artifact."""

    data_root = Path(root)
    raw_intake = _read_json(Path(intake_path))
    raw_triage = _read_json(Path(triage_path)) if triage_path else {}
    selected_items = _selected_intake_items(raw_intake, include_optional=include_optional)
    if max_symbols is not None:
        selected_items = selected_items[: max(0, max_symbols)]
    if not selected_items:
        raise ValueError("research intake artifact has no must-read symbols for deep research.")

    context, warnings = build_deep_research_context(
        intake=raw_intake,
        triage=raw_triage,
        root=data_root,
        intake_path=Path(intake_path),
        triage_path=Path(triage_path) if triage_path else None,
        include_optional=include_optional,
        max_symbols=max_symbols,
    )
    run = DeepResearchRunArtifact(
        root=str(data_root),
        intake_path=str(intake_path),
        triage_path=str(triage_path or ""),
        researched_symbols=context["researched_symbols"],
        status="partial" if warnings else "fresh",
        warnings=_dedupe(warnings),
    )
    run_dir = Path(output_dir) if output_dir else data_root / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / DEEP_RESEARCH_CONTEXT_FILENAME
    report_path = run_dir / DEEP_RESEARCH_REPORT_FILENAME
    run_path = run_dir / DEEP_RESEARCH_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    if batch_size and len(context["researched_symbols"]) > batch_size:
        report, runtime, usage = _run_batched_deep_research_agent(
            full_context=context,
            selected_items=selected_items,
            raw_intake=raw_intake,
            raw_triage=raw_triage,
            root=data_root,
            intake_path=Path(intake_path),
            triage_path=Path(triage_path) if triage_path else None,
            include_optional=include_optional,
            batch_size=batch_size,
            run_dir=run_dir,
            save_context=save_context,
        )
        run.batch_count = len(report.source_artifacts.get("batch_reports", []))
        run.batch_paths = list(report.source_artifacts.get("batch_reports", []))
    else:
        report, runtime, usage = _run_deep_research_agent(context)
    report.unresearched_candidates = _merge_unresearched_candidates(
        report.unresearched_candidates,
        _unresearched_candidate_cards(context),
    )
    _validate_deep_research_report(context, report)
    report.source_artifacts = {
        **report.source_artifacts,
        "intake_path": str(intake_path),
        "triage_path": str(triage_path or ""),
        "context_path": str(context_path) if save_context else "",
    }
    report.pydantic_ai = runtime
    result_warnings = _dedupe([*warnings, *report.warnings, *report.data_gaps])
    run.status = "partial" if result_warnings else "fresh"
    run.warnings = result_warnings
    run.pydantic_ai = runtime
    run.usage = usage
    _write_json(report_path, report.model_dump(mode="json"))
    run.report_path = str(report_path)
    _write_json(run_path, run.model_dump(mode="json"))
    return report, run


def build_deep_research_context(
    *,
    intake: dict[str, Any],
    triage: dict[str, Any] | None,
    root: str | Path,
    intake_path: Path | None = None,
    triage_path: Path | None = None,
    include_optional: bool = False,
    max_symbols: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    data_root = Path(root)
    intake_items = _selected_intake_items(intake, include_optional=include_optional)
    if max_symbols is not None:
        intake_items = intake_items[: max(0, max_symbols)]
    if not intake_items:
        raise ValueError("research intake artifact has no must-read symbols for deep research.")

    triage_by_symbol = _triage_by_symbol(triage or {})
    symbols: list[str] = []
    warnings: list[str] = []
    materials: dict[str, Any] = {}
    for item in intake_items:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        if symbol in symbols:
            continue
        symbols.append(symbol)
        material, material_warnings = _symbol_deep_material(data_root, symbol, item, triage_by_symbol.get(symbol, {}))
        materials[symbol] = material
        warnings.extend(material_warnings)

    context = {
        "artifact_type": "deep_research_context",
        "generated_at": utc_now(),
        "input_boundary": {
            "uses_research_intake": True,
            "uses_candidate_triage": bool(triage),
            "uses_filing_summary_markdown": True,
            "uses_sec_companyfacts": True,
            "uses_fmp_profile": True,
            "uses_discovery_directly": False,
            "forbids_portfolio_weights": True,
            "forbids_trade_actions": True,
        },
        "source_artifacts": {
            "intake_path": str(intake_path or ""),
            "triage_path": str(triage_path or ""),
            "data_root": str(data_root),
        },
        "theme": _theme_from_artifacts(intake, triage or {}),
        "intake_summary": intake.get("intake_summary", ""),
        "layer_read_budgets": intake.get("layer_read_budgets", []),
        "selected_read_actions": {
            "include_optional": include_optional,
            "symbols": symbols,
        },
        "researched_symbols": symbols,
        "unresearched_candidates": _unresearched_candidate_cards_from_intake(
            intake,
            researched_symbols=set(symbols),
        ),
        "symbol_materials": materials,
        "output_contract": (
            "Return DeepResearchReport. Do not include target weights, price targets, "
            "trade actions, orders, or construction plans."
        ),
        "warnings": _dedupe(warnings),
    }
    return context, _dedupe(warnings)


def _selected_intake_items(intake: dict[str, Any], *, include_optional: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("must_read_filing_analysis", "promote_from_watchlist_for_reading"):
        items.extend(item for item in intake.get(key, []) or [] if isinstance(item, dict))
    if include_optional:
        items.extend(item for item in intake.get("optional_read_filing_analysis", []) or [] if isinstance(item, dict))
    return items


def _unresearched_candidate_cards_from_intake(
    intake: dict[str, Any],
    *,
    researched_symbols: set[str],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in (
        "optional_read_filing_analysis",
        "profile_metrics_only",
        "do_not_read_yet",
        "skipped_deep_candidate_audit",
        "stale_or_missing_materials",
    ):
        for item in intake.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol or symbol in researched_symbols or symbol in seen:
                continue
            seen.add(symbol)
            read_action = str(item.get("read_action") or key)
            cards.append(
                {
                    "symbol": symbol,
                    "layer_keys": item.get("layer_keys", []),
                    "intake_action": read_action,
                    "original_priority": item.get("priority", "medium"),
                    "reason": item.get("reason", ""),
                    "available_light_materials": item.get("available_light_materials", []),
                    "missing_or_stale_materials": item.get("missing_or_stale_materials", []),
                    "evidence_refs": [f"intake:{symbol}"],
                    "data_gaps": [
                        f"{symbol}: not deeply researched in this pass; read_action={read_action}.",
                        *[str(value) for value in item.get("missing_or_stale_materials", [])],
                    ],
                }
            )
    return cards


def _unresearched_candidate_cards(context: dict[str, Any]) -> list[UnresearchedCandidateCard]:
    return [
        UnresearchedCandidateCard.model_validate(card)
        for card in context.get("unresearched_candidates", [])
        if isinstance(card, dict)
    ]


def _merge_unresearched_candidates(
    authored: list[UnresearchedCandidateCard],
    generated: list[UnresearchedCandidateCard],
) -> list[UnresearchedCandidateCard]:
    result: dict[str, UnresearchedCandidateCard] = {}
    for card in [*authored, *generated]:
        symbol = card.symbol.upper()
        if symbol and symbol not in result:
            card.symbol = symbol
            result[symbol] = card
    return list(result.values())


def _symbol_deep_material(
    data_root: Path,
    symbol: str,
    intake_item: dict[str, Any],
    triage_item: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    symbol_dir = data_root / "symbols" / symbol
    warnings: list[str] = []
    if not symbol_dir.exists():
        warnings.append(f"{symbol}: symbol directory not found.")

    filing_summary_path = symbol_dir / "filing_summary.md"
    filing_summary = _read_text(filing_summary_path)
    if not filing_summary:
        warnings.append(f"{symbol}: missing filing_summary.md")

    sec_companyfacts = _read_json(symbol_dir / "sec_companyfacts.json")
    if not sec_companyfacts:
        warnings.append(f"{symbol}: missing sec_companyfacts.json")

    fmp_profile = _read_json(symbol_dir / "fmp_company_profile.json")
    if not fmp_profile:
        warnings.append(f"{symbol}: missing fmp_company_profile.json")

    refs = {
        "filing_summary": f"filing_summary:{symbol}",
        "sec_companyfacts": f"sec_companyfacts:{symbol}",
        "fmp_profile": f"fmp_profile:{symbol}",
        "intake": f"intake:{symbol}",
        "triage": f"triage:{symbol}",
    }
    return (
        {
            "symbol": symbol,
            "evidence_refs": refs,
            "intake_decision": intake_item,
            "triage_decision": triage_item,
            "manifest": _compact_manifest(_read_json(symbol_dir / "manifest.json")),
            "filing_summary_path": _relative_or_str(filing_summary_path, data_root),
            "filing_summary": filing_summary,
            "filing_summary_meta": _read_json(symbol_dir / "filing_summary.meta.json"),
            "sec_companyfacts_path": _relative_or_str(symbol_dir / "sec_companyfacts.json", data_root),
            "sec_companyfacts": _compact_sec_companyfacts(sec_companyfacts),
            "filing_metadata": _compact_filing_metadata(_read_json(symbol_dir / "filing_metadata.json")),
            "fmp_profile_path": _relative_or_str(symbol_dir / "fmp_company_profile.json", data_root),
            "fmp_company_profile": _compact_fmp_profile(fmp_profile),
        },
        warnings,
    )


def _run_deep_research_agent(context: dict[str, Any]) -> tuple[DeepResearchReport, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=DeepResearchReport,
        instructions=_DEEP_RESEARCH_INSTRUCTIONS,
        agent_kind="deep_research_agent",
        output_retries=1,
        enable_web_search=False,
        enable_web_fetch=False,
        agent_skill_names=["deep-research"],
    )
    result = agent.run_sync(
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("deep_research_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _run_batched_deep_research_agent(
    *,
    full_context: dict[str, Any],
    selected_items: list[dict[str, Any]],
    raw_intake: dict[str, Any],
    raw_triage: dict[str, Any],
    root: Path,
    intake_path: Path,
    triage_path: Path | None,
    include_optional: bool,
    batch_size: int,
    run_dir: Path,
    save_context: bool,
) -> tuple[DeepResearchReport, dict[str, Any], dict[str, Any]]:
    batch_reports: list[DeepResearchReport] = []
    batch_runtimes: list[dict[str, Any]] = []
    batch_usages: list[dict[str, Any]] = []
    batch_report_paths: list[str] = []

    for index, batch_items in enumerate(_chunks(selected_items, max(1, batch_size)), start=1):
        batch_intake = _subset_intake(raw_intake, batch_items)
        batch_context, _warnings = build_deep_research_context(
            intake=batch_intake,
            triage=raw_triage,
            root=root,
            intake_path=intake_path,
            triage_path=triage_path,
            include_optional=include_optional,
        )
        batch_context["batch"] = {
            "index": index,
            "total": (len(selected_items) + max(1, batch_size) - 1) // max(1, batch_size),
            "batch_size": batch_size,
            "full_symbol_count": len(full_context.get("researched_symbols", [])),
        }
        if save_context:
            batch_context_path = run_dir / f"deep_research_context_batch_{index:02d}.json"
            _write_json(batch_context_path, batch_context)
        batch_report, batch_runtime, batch_usage = _run_deep_research_agent(batch_context)
        _validate_deep_research_report(batch_context, batch_report)
        batch_report.source_artifacts = {
            **batch_report.source_artifacts,
            "batch_index": str(index),
            "context_path": str(batch_context_path) if save_context else "",
        }
        batch_report_path = run_dir / f"deep_research_report_batch_{index:02d}.json"
        _write_json(batch_report_path, batch_report.model_dump(mode="json"))
        batch_report_paths.append(str(batch_report_path))
        batch_reports.append(batch_report)
        batch_runtimes.append(batch_runtime)
        batch_usages.append(batch_usage)

    merged = _merge_deep_research_reports(full_context, batch_reports, batch_report_paths)
    runtime = {
        "mode": "batched_deep_research_agent",
        "batch_size": batch_size,
        "batch_count": len(batch_reports),
        "batches": batch_runtimes,
    }
    usage = _merge_usage(batch_usages)
    return merged, runtime, usage


def _subset_intake(raw_intake: dict[str, Any], selected_items: list[dict[str, Any]]) -> dict[str, Any]:
    subset = dict(raw_intake)
    subset["must_read_filing_analysis"] = selected_items
    subset["promote_from_watchlist_for_reading"] = []
    subset["optional_read_filing_analysis"] = []
    return subset


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _merge_deep_research_reports(
    full_context: dict[str, Any],
    batch_reports: list[DeepResearchReport],
    batch_report_paths: list[str],
) -> DeepResearchReport:
    by_symbol: dict[str, CandidateResearchCard] = {}
    for report in batch_reports:
        for card in report.candidate_cards:
            by_symbol.setdefault(card.symbol.upper(), card)

    ordered_cards = [
        by_symbol[symbol]
        for symbol in full_context.get("researched_symbols", [])
        if symbol in by_symbol
    ]
    layer_conclusions = _merge_layer_conclusions(ordered_cards, batch_reports)
    warnings: list[str] = []
    data_gaps: list[str] = []
    cross_layer_thesis: list[str] = []
    summaries: list[str] = []
    for report in batch_reports:
        warnings.extend(report.warnings)
        data_gaps.extend(report.data_gaps)
        cross_layer_thesis.extend(report.cross_layer_thesis)
        if report.research_summary.strip():
            summaries.append(report.research_summary.strip())

    decision_counts: dict[str, int] = {}
    for card in ordered_cards:
        decision_counts[card.candidate_decision] = decision_counts.get(card.candidate_decision, 0) + 1

    return DeepResearchReport(
        theme=str(full_context.get("theme") or ""),
        source_artifacts={
            **full_context.get("source_artifacts", {}),
            "batch_reports": batch_report_paths,
        },
        research_summary="\n\n".join(summaries),
        candidate_cards=ordered_cards,
        unresearched_candidates=_merge_unresearched_candidates(
            [],
            _unresearched_candidate_cards(full_context),
        ),
        layer_conclusions=layer_conclusions,
        cross_layer_thesis=_dedupe(cross_layer_thesis),
        architect_inputs={
            "researched_symbols": full_context.get("researched_symbols", []),
            "decision_counts": decision_counts,
            "candidate_decision_scale": [
                "core_candidate",
                "high_conviction_candidate",
                "satellite_candidate",
                "watchlist",
                "defer",
                "reject",
            ],
            "batching_note": (
                "Candidate cards were authored by the same deep-research agent in batches; "
                "batching is an execution detail to keep context size reliable."
            ),
        },
        data_gaps=_dedupe(data_gaps),
        warnings=_dedupe(warnings),
    )


def _merge_layer_conclusions(
    cards: list[CandidateResearchCard],
    batch_reports: list[DeepResearchReport],
) -> list[LayerResearchConclusion]:
    layer_order: list[str] = []
    layer_names: dict[str, str] = {}
    layer_tradeoffs: dict[str, list[str]] = {}
    layer_questions: dict[str, list[str]] = {}
    for report in batch_reports:
        for layer in report.layer_conclusions:
            key = layer.layer_key
            if key and key not in layer_order:
                layer_order.append(key)
            if key and layer.layer_name:
                layer_names.setdefault(key, layer.layer_name)
            if key and layer.peer_tradeoff_summary:
                layer_tradeoffs.setdefault(key, []).append(layer.peer_tradeoff_summary)
            if key:
                layer_questions.setdefault(key, []).extend(layer.unresolved_questions)

    for card in cards:
        for key in card.layer_keys:
            if key not in layer_order:
                layer_order.append(key)

    conclusions: list[LayerResearchConclusion] = []
    for key in layer_order:
        selected: list[str] = []
        watchlist: list[str] = []
        deferred: list[str] = []
        rejected: list[str] = []
        for card in cards:
            if key not in card.layer_keys:
                continue
            if card.candidate_decision in {"core_candidate", "high_conviction_candidate", "satellite_candidate"}:
                selected.append(card.symbol)
            elif card.candidate_decision == "watchlist":
                watchlist.append(card.symbol)
            elif card.candidate_decision == "reject":
                rejected.append(card.symbol)
            else:
                deferred.append(card.symbol)
        conclusions.append(
            LayerResearchConclusion(
                layer_key=key,
                layer_name=layer_names.get(key, ""),
                selected_symbols=selected,
                watchlist_symbols=watchlist,
                deferred_symbols=deferred,
                rejected_symbols=rejected,
                peer_tradeoff_summary=" | ".join(_dedupe(layer_tradeoffs.get(key, []))),
                unresolved_questions=_dedupe(layer_questions.get(key, [])),
            )
        )
    return conclusions


def _merge_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {"batches": usages}
    numeric_keys: set[str] = set()
    for usage in usages:
        numeric_keys.update(key for key, value in usage.items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    for key in sorted(numeric_keys):
        total[key] = sum(usage.get(key, 0) for usage in usages if isinstance(usage.get(key), (int, float)))

    detail_keys: set[str] = set()
    for usage in usages:
        details = usage.get("details")
        if isinstance(details, dict):
            detail_keys.update(key for key, value in details.items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    if detail_keys:
        total["details"] = {
            key: sum(
                usage.get("details", {}).get(key, 0)
                for usage in usages
                if isinstance(usage.get("details"), dict)
            )
            for key in sorted(detail_keys)
        }
    return total


def _validate_deep_research_report(context: dict[str, Any], report: DeepResearchReport) -> None:
    allowed = {symbol.upper() for symbol in context.get("researched_symbols", [])}
    seen: set[str] = set()
    errors: list[str] = []
    for card in report.candidate_cards:
        symbol = card.symbol.upper()
        if symbol not in allowed:
            errors.append(f"unknown candidate card symbol {card.symbol!r}")
            continue
        seen.add(symbol)
        if not card.layer_keys:
            errors.append(f"{symbol}: missing layer_keys")
        if not card.exposure_summary.strip():
            errors.append(f"{symbol}: missing exposure_summary")
        if not card.filing_takeaways:
            errors.append(f"{symbol}: missing filing_takeaways")
        if not card.evidence_refs:
            errors.append(f"{symbol}: missing evidence_refs")
        for ref in card.evidence_refs:
            if not _valid_evidence_ref(ref, symbol):
                errors.append(f"{symbol}: invalid evidence_ref {ref!r}")
    missing = sorted(allowed - seen)
    if missing:
        errors.append("missing candidate cards for: " + ", ".join(missing))
    for card in report.unresearched_candidates:
        symbol = card.symbol.upper()
        if symbol in allowed:
            errors.append(f"{symbol}: unresearched candidate duplicates researched_symbols")
        if symbol in seen:
            errors.append(f"{symbol}: unresearched candidate duplicates candidate_cards")
        if not card.layer_keys:
            errors.append(f"{symbol}: unresearched candidate missing layer_keys")
        if not card.intake_action.strip():
            errors.append(f"{symbol}: unresearched candidate missing intake_action")
    layer_symbols = set()
    for layer in report.layer_conclusions:
        for symbol in [
            *layer.selected_symbols,
            *layer.watchlist_symbols,
            *layer.deferred_symbols,
            *layer.rejected_symbols,
        ]:
            normalized = symbol.upper()
            if normalized not in allowed:
                errors.append(f"layer {layer.layer_key!r} references unknown symbol {symbol!r}")
            layer_symbols.add(normalized)
    if not report.layer_conclusions:
        errors.append("missing layer_conclusions")
    if errors:
        raise ValueError("Deep research report failed validation: " + "; ".join(errors))


def _valid_evidence_ref(ref: str, symbol: str) -> bool:
    prefixes = ("filing_summary", "sec_companyfacts", "fmp_profile", "filing_metadata", "intake", "triage")
    return any(ref == f"{prefix}:{symbol}" for prefix in prefixes)


def _triage_by_symbol(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = raw.get("candidate_triage") if isinstance(raw.get("candidate_triage"), dict) else raw
    result: dict[str, dict[str, Any]] = {}
    for key in ("deep_enrichment_queue", "watchlist", "deferred", "rejected"):
        for item in payload.get(key, []) or []:
            if isinstance(item, dict) and item.get("symbol"):
                result[str(item["symbol"]).upper()] = item
    return result


def _theme_from_artifacts(intake: dict[str, Any], triage: dict[str, Any]) -> str:
    payload = triage.get("candidate_triage") if isinstance(triage.get("candidate_triage"), dict) else triage
    return str(payload.get("theme") or intake.get("theme") or "")


def _compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    layers = manifest.get("layers") if isinstance(manifest.get("layers"), dict) else {}
    return {
        "source_status": manifest.get("source_status", ""),
        "layers": {
            key: {
                "status": value.get("status"),
                "updated_at": value.get("updated_at"),
                "path": value.get("path"),
                "meta_path": value.get("meta_path"),
                "warnings": value.get("warnings", []),
            }
            for key, value in layers.items()
            if isinstance(value, dict)
        },
        "warnings": manifest.get("warnings", []),
    }


def _compact_sec_companyfacts(payload: dict[str, Any]) -> dict[str, Any]:
    fundamentals = payload.get("fundamentals") if isinstance(payload.get("fundamentals"), dict) else {}
    metric_provenance = payload.get("metric_provenance") if isinstance(payload.get("metric_provenance"), dict) else {}
    return {
        "source_status": payload.get("source_status", ""),
        "company_name": payload.get("company_name", ""),
        "industry": payload.get("industry", ""),
        "filer_category": payload.get("filer_category", ""),
        "fundamentals": fundamentals,
        "metric_provenance_keys": sorted(metric_provenance.keys()),
        "numeric_evidence": payload.get("numeric_evidence", {}),
        "event_context": payload.get("event_context", {}),
        "risk_flags": payload.get("risk_flags", []),
        "warnings": payload.get("warnings", []),
    }


def _compact_filing_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    filings = payload.get("filings") if isinstance(payload.get("filings"), dict) else {}
    return {
        "source_status": payload.get("source_status", ""),
        "filings": {
            key: {
                field: item.get(field)
                for field in (
                    "form",
                    "filing_date",
                    "period_of_report",
                    "accession_number",
                    "primary_document",
                    "filing_url",
                    "homepage_url",
                    "text_url",
                )
            }
            for key, item in filings.items()
            if isinstance(item, dict)
        },
        "event_context": payload.get("event_context", {}),
        "risk_flags": payload.get("risk_flags", []),
        "warnings": payload.get("warnings", []),
    }


def _compact_fmp_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    fields = (
        "symbol",
        "companyName",
        "sector",
        "industry",
        "exchange",
        "country",
        "currency",
        "marketCap",
        "beta",
        "averageVolume",
        "isActivelyTrading",
        "isEtf",
        "isFund",
        "isAdr",
        "description",
    )
    compact = {key: profile.get(key) for key in fields if key in profile}
    if isinstance(compact.get("description"), str) and len(compact["description"]) > 1_200:
        compact["description"] = compact["description"][:1_200] + "..."
    return {
        "source_status": payload.get("source_status", ""),
        "data_asof": payload.get("data_asof", {}),
        "profile": compact,
        "warnings": payload.get("warnings", []),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-deep-research",
        description="Read selected filing summaries and produce candidate research cards.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Generate deep-research report from intake artifact.")
    build.add_argument("--intake-path", required=True, help="Research intake triage JSON path.")
    build.add_argument("--triage-path", help="Candidate triage JSON path.")
    build.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Data root containing symbols/.")
    build.add_argument("--output-dir", help="Directory for context/run/report output.")
    build.add_argument("--include-optional", action="store_true", help="Also read optional filing analyses.")
    build.add_argument("--max-symbols", type=int, help="Maximum selected symbols to read.")
    build.add_argument("--batch-size", type=int, help="Run selected symbols in batches of this size.")
    build.add_argument("--no-save-context", action="store_true")
    build.add_argument("--json", action="store_true", help="Print run and report paths as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        report, run = build_deep_research_report_from_files(
            intake_path=args.intake_path,
            root=args.root,
            triage_path=args.triage_path,
            output_dir=args.output_dir,
            include_optional=args.include_optional,
            max_symbols=args.max_symbols,
            batch_size=args.batch_size,
            save_context=not args.no_save_context,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "run": run.model_dump(mode="json"),
                        "report_path": run.report_path,
                        "candidate_count": len(report.candidate_cards),
                        "layer_count": len(report.layer_conclusions),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"run_id: {run.run_id}")
            print(f"report_path: {run.report_path}")
            print(f"symbols: {', '.join(run.researched_symbols)}")
            print(f"status: {run.status}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
