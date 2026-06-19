"""PydanticAI portfolio-map revision agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .portfolio_architect import PortfolioArchitectResult
from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .schemas import InvestmentPolicy, PortfolioMap
from .storage import new_id, utc_now

PORTFOLIO_REVISION_CONTEXT_FILENAME = "portfolio_revision_context.json"
PORTFOLIO_REVISION_PATCH_FILENAME = "portfolio_revision_patch.json"
PORTFOLIO_REVISION_FILENAME = "portfolio_map_revision.json"
PORTFOLIO_REVISION_RUN_FILENAME = "portfolio_revision_run.json"


class RevisionTarget(BaseModel):
    kind: Literal["symbol", "sleeve", "portfolio_map", "constraint", "style", "unknown"] = "unknown"
    id: str = ""
    label: str = ""


class RevisionMagnitude(BaseModel):
    kind: Literal["unspecified", "small", "moderate", "large", "specific", "ai_decide"] = "ai_decide"
    target_weight: float | None = Field(default=None, ge=0, le=1)
    delta_weight: float | None = Field(default=None, ge=-1, le=1)
    description: str = ""


class PortfolioRevisionEdit(BaseModel):
    edit_type: Literal[
        "adjust_weight",
        "adjust_sleeve",
        "add_symbol",
        "remove_symbol",
        "replace_symbol",
        "set_constraint",
        "change_style",
        "preserve_position",
        "request_research",
        "unknown",
    ] = "unknown"
    target: RevisionTarget = Field(default_factory=RevisionTarget)
    direction: Literal[
        "increase",
        "decrease",
        "add",
        "remove",
        "replace",
        "preserve",
        "neutral",
        "unknown",
    ] = "unknown"
    magnitude: RevisionMagnitude = Field(default_factory=RevisionMagnitude)
    source_symbol: str = ""
    replacement_symbol: str = ""
    rationale: str = ""
    requires_research: bool = False
    hard_constraint: bool = False


class PortfolioRevisionPatch(BaseModel):
    artifact_type: str = "portfolio_revision_patch"
    patch_id: str = Field(default_factory=lambda: new_id("prp"))
    generated_at: str = Field(default_factory=utc_now)
    base_map_id: str
    user_request: str
    revision_intent: str = ""
    edits: list[PortfolioRevisionEdit] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    preserve: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class RevisionWeightChange(BaseModel):
    symbol: str
    old_weight: float | None = Field(default=None, ge=0, le=1)
    new_weight: float | None = Field(default=None, ge=0, le=1)
    direction: Literal["increase", "decrease", "added", "removed", "unchanged"] = "unchanged"
    reason: str = ""


class PortfolioMapRevision(BaseModel):
    artifact_type: str = "portfolio_map_revision"
    revision_id: str = Field(default_factory=lambda: new_id("pmr"))
    generated_at: str = Field(default_factory=utc_now)
    base_map_id: str
    patch_id: str
    revised_map: PortfolioMap
    change_summary: list[str] = Field(default_factory=list)
    weight_changes: list[RevisionWeightChange] = Field(default_factory=list)
    funding_sources: list[str] = Field(default_factory=list)
    tradeoff_explanation: list[str] = Field(default_factory=list)
    risk_delta: list[str] = Field(default_factory=list)
    reduced_or_removed: list[str] = Field(default_factory=list)
    needs_user_confirmation: bool = True
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)


class PortfolioRevisionRunArtifact(BaseModel):
    artifact_type: str = "portfolio_revision_run"
    run_id: str = Field(default_factory=lambda: new_id("prr"))
    generated_at: str = Field(default_factory=utc_now)
    status: Literal["fresh", "needs_clarification", "error"] = "fresh"
    base_map_id: str = ""
    context_path: str = ""
    patch_path: str = ""
    revision_path: str = ""
    allowed_symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pydantic_ai: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


def build_portfolio_revision_from_files(
    *,
    portfolio_architect_path: str | Path,
    user_request: str,
    base_map_id: str = "",
    selected_map_path: str | Path | None = None,
    deep_research_path: str | Path | None = None,
    policy: InvestmentPolicy | None = None,
    output_dir: str | Path | None = None,
    save_context: bool = True,
) -> tuple[PortfolioRevisionPatch, PortfolioMapRevision | None, PortfolioRevisionRunArtifact]:
    raw_architect = _read_json(Path(portfolio_architect_path))
    architect_result = _extract_architect_result(raw_architect)
    selected_payload = _read_json(Path(selected_map_path)) if selected_map_path else {}
    selected_map = _extract_selected_map(selected_payload)
    deep_research = _read_json(Path(deep_research_path)) if deep_research_path else {}
    base_map = _resolve_base_map(
        architect_result=architect_result,
        selected_map=selected_map,
        base_map_id=base_map_id,
    )
    effective_policy = policy or _default_policy_from_architect(architect_result, base_map)
    run = PortfolioRevisionRunArtifact(
        base_map_id=base_map.map_id,
        allowed_symbols=sorted(_allowed_symbols(architect_result, deep_research, base_map)),
    )
    run_dir = Path(output_dir) if output_dir else Path(".dev") / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context, warnings = build_portfolio_revision_context(
        user_request=user_request,
        base_map=base_map,
        architect_result=architect_result,
        policy=effective_policy,
        deep_research=deep_research,
    )
    run.warnings = _dedupe(warnings)
    context_path = run_dir / PORTFOLIO_REVISION_CONTEXT_FILENAME
    patch_path = run_dir / PORTFOLIO_REVISION_PATCH_FILENAME
    revision_path = run_dir / PORTFOLIO_REVISION_FILENAME
    run_path = run_dir / PORTFOLIO_REVISION_RUN_FILENAME
    if save_context:
        _write_json(context_path, context)
        run.context_path = str(context_path)

    patch, patch_runtime, patch_usage = parse_portfolio_revision_patch(context)
    patch.warnings = _dedupe([*patch.warnings, *warnings])
    _validate_revision_patch(context, patch)
    _write_json(patch_path, patch.model_dump(mode="json"))
    run.patch_path = str(patch_path)
    run.pydantic_ai = {"patch_agent": patch_runtime}
    run.usage = {"patch_agent": patch_usage}
    if patch.needs_clarification:
        run.status = "needs_clarification"
        _write_json(run_path, run.model_dump(mode="json"))
        return patch, None, run

    revision, revision_runtime, revision_usage = build_portfolio_map_revision(context, patch)
    _validate_portfolio_revision(context, patch, revision)
    _write_json(revision_path, revision.model_dump(mode="json"))
    run.revision_path = str(revision_path)
    run.pydantic_ai = {
        "patch_agent": patch_runtime,
        "revision_agent": revision_runtime,
    }
    run.usage = {
        "patch_agent": patch_usage,
        "revision_agent": revision_usage,
    }
    run.status = "fresh"
    run.warnings = _dedupe([*run.warnings, *patch.warnings, *revision.warnings])
    _write_json(run_path, run.model_dump(mode="json"))
    return patch, revision, run


def build_portfolio_revision_from_artifacts(
    *,
    user_request: str,
    base_map: PortfolioMap,
    architect_result: PortfolioArchitectResult,
    policy: InvestmentPolicy,
    deep_research: dict[str, Any] | None = None,
) -> tuple[PortfolioRevisionPatch, PortfolioMapRevision | None, PortfolioRevisionRunArtifact]:
    """Build a portfolio revision directly from workflow artifacts."""

    context, warnings = build_portfolio_revision_context(
        user_request=user_request,
        base_map=base_map,
        architect_result=architect_result,
        policy=policy,
        deep_research=deep_research or {},
    )
    run = PortfolioRevisionRunArtifact(
        base_map_id=base_map.map_id,
        allowed_symbols=sorted(_allowed_symbols(architect_result, deep_research or {}, base_map)),
        warnings=_dedupe(warnings),
    )
    patch, patch_runtime, patch_usage = parse_portfolio_revision_patch(context)
    patch.warnings = _dedupe([*patch.warnings, *warnings])
    _validate_revision_patch(context, patch)
    run.pydantic_ai = {"patch_agent": patch_runtime}
    run.usage = {"patch_agent": patch_usage}
    if patch.needs_clarification:
        run.status = "needs_clarification"
        return patch, None, run

    revision, revision_runtime, revision_usage = build_portfolio_map_revision(context, patch)
    _validate_portfolio_revision(context, patch, revision)
    run.status = "fresh"
    run.pydantic_ai = {
        "patch_agent": patch_runtime,
        "revision_agent": revision_runtime,
    }
    run.usage = {
        "patch_agent": patch_usage,
        "revision_agent": revision_usage,
    }
    run.warnings = _dedupe([*run.warnings, *patch.warnings, *revision.warnings])
    return patch, revision, run


def build_portfolio_revision_context(
    *,
    user_request: str,
    base_map: PortfolioMap,
    architect_result: PortfolioArchitectResult,
    policy: InvestmentPolicy,
    deep_research: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    deep_research = deep_research or {}
    allowed_symbols = sorted(_allowed_symbols(architect_result, deep_research, base_map))
    base_symbols = [holding.symbol.upper() for holding in base_map.holdings]
    selected_symbols = [
        item.symbol.upper()
        for item in architect_result.selection.selected_for_portfolio
        if item.symbol
    ]
    context = {
        "artifact_type": "portfolio_revision_context",
        "generated_at": utc_now(),
        "user_request": user_request,
        "base_map": base_map.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "portfolio_architect_result": architect_result.model_dump(mode="json"),
        "deep_research_report": _compact_deep_research(deep_research),
        "base_symbols": base_symbols,
        "selected_for_portfolio_symbols": selected_symbols,
        "allowed_symbols": allowed_symbols,
        "input_boundary": {
            "uses_current_portfolio": False,
            "uses_orders_or_trades": False,
            "may_revise_target_map_only": True,
            "may_add_only_researched_or_architect_symbols": True,
        },
        "output_contract": {
            "patch_output_type": "PortfolioRevisionPatch",
            "revision_output_type": "PortfolioMapRevision",
            "cash_weight_must_equal_base_map_unless_explicitly_requested": True,
            "sleeve_weight_must_equal_base_map": True,
            "holdings_sum_must_equal_sleeve_weight": True,
            "single_name_limit_must_be_respected": True,
            "new_symbols_must_be_in_allowed_symbols_or_require_research": True,
            "no_trade_orders_or_current_position_adjustments": True,
        },
    }
    warnings: list[str] = []
    if not allowed_symbols:
        warnings.append("portfolio revision context has no allowed symbols.")
    return context, warnings


def parse_portfolio_revision_patch(
    context: dict[str, Any],
) -> tuple[PortfolioRevisionPatch, dict[str, Any], dict[str, Any]]:
    patch, runtime, usage = _run_revision_patch_agent(context)
    runtime = dict(runtime)
    patch.pydantic_ai = runtime
    return patch, runtime, usage


def build_portfolio_map_revision(
    context: dict[str, Any],
    patch: PortfolioRevisionPatch,
) -> tuple[PortfolioMapRevision, dict[str, Any], dict[str, Any]]:
    revision_context = {
        **context,
        "portfolio_revision_patch": patch.model_dump(mode="json"),
    }
    revision, runtime, usage = _run_revision_architect_agent(revision_context)
    runtime = dict(runtime)
    revision.pydantic_ai = runtime
    return revision, runtime, usage


def _run_revision_patch_agent(
    context: dict[str, Any],
) -> tuple[PortfolioRevisionPatch, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=PortfolioRevisionPatch,
        instructions=_REVISION_PATCH_INSTRUCTIONS,
        agent_kind="portfolio_revision_patch_agent",
        output_retries=2,
        agent_skill_names=["portfolio-revision"],
    )
    result = agent.run_sync(
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("portfolio_revision_patch_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _run_revision_architect_agent(
    context: dict[str, Any],
) -> tuple[PortfolioMapRevision, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=PortfolioMapRevision,
        instructions=_REVISION_ARCHITECT_INSTRUCTIONS,
        agent_kind="portfolio_revision_architect_agent",
        output_retries=2,
        agent_skill_names=["portfolio-revision"],
    )
    result = agent.run_sync(
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("portfolio_revision_architect_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _validate_revision_patch(context: dict[str, Any], patch: PortfolioRevisionPatch) -> None:
    base_map_id = str((context.get("base_map") or {}).get("map_id") or "")
    if patch.base_map_id != base_map_id:
        raise ValueError(f"revision patch base_map_id {patch.base_map_id!r} != base map {base_map_id!r}.")
    if not patch.revision_intent.strip():
        raise ValueError("revision patch must include revision_intent.")
    if not patch.edits and not patch.needs_clarification:
        raise ValueError("revision patch has no edits and does not ask for clarification.")
    allowed_symbols = {str(symbol).upper() for symbol in context.get("allowed_symbols") or []}
    for edit in patch.edits:
        target_symbol = edit.target.id.upper() if edit.target.kind == "symbol" else ""
        replacement_symbol = edit.replacement_symbol.upper()
        for symbol in [target_symbol, replacement_symbol]:
            if not symbol:
                continue
            if symbol not in allowed_symbols and not edit.requires_research:
                raise ValueError(
                    f"revision patch references {symbol}, which is outside allowed_symbols; "
                    "mark the edit requires_research or ask for clarification."
                )


def _validate_portfolio_revision(
    context: dict[str, Any],
    patch: PortfolioRevisionPatch,
    revision: PortfolioMapRevision,
) -> None:
    base_map = PortfolioMap.model_validate(context["base_map"])
    policy = InvestmentPolicy.model_validate(context["policy"])
    revised = revision.revised_map
    if revision.base_map_id != base_map.map_id:
        raise ValueError(f"portfolio revision base_map_id {revision.base_map_id!r} != {base_map.map_id!r}.")
    if revision.patch_id != patch.patch_id:
        raise ValueError(f"portfolio revision patch_id {revision.patch_id!r} != {patch.patch_id!r}.")
    if not revision.change_summary:
        raise ValueError("portfolio revision must include change_summary.")
    if not revision.funding_sources:
        raise ValueError("portfolio revision must explain funding_sources.")
    if not revision.tradeoff_explanation:
        raise ValueError("portfolio revision must include tradeoff_explanation.")
    if abs(revised.sleeve_weight - base_map.sleeve_weight) > 0.0001:
        raise ValueError("portfolio revision changed sleeve_weight; V1 must preserve the selected map sleeve size.")
    if not _patch_explicitly_changes_cash(patch) and abs(revised.cash_weight - base_map.cash_weight) > 0.0001:
        raise ValueError("portfolio revision changed cash_weight without an explicit cash edit.")
    if revised.cash_weight + revised.sleeve_weight > 1.0001:
        raise ValueError("portfolio revision cash_weight + sleeve_weight exceeds 100%.")

    allowed_symbols = {str(symbol).upper() for symbol in context.get("allowed_symbols") or []}
    holding_symbols: set[str] = set()
    total = 0.0
    for holding in revised.holdings:
        symbol = holding.symbol.upper()
        if symbol not in allowed_symbols:
            raise ValueError(f"portfolio revision used unresearched or disallowed symbol {symbol!r}.")
        if symbol in holding_symbols:
            raise ValueError(f"portfolio revision duplicated holding {symbol!r}.")
        if holding.target_weight > policy.single_name_limit + 0.0001:
            raise ValueError(
                f"portfolio revision assigned {symbol} {holding.target_weight:.4f}, "
                f"above single_name_limit {policy.single_name_limit:.4f}."
            )
        if not holding.rationale.strip():
            raise ValueError(f"portfolio revision omitted rationale for holding {symbol!r}.")
        holding_symbols.add(symbol)
        total += holding.target_weight
    if abs(total - revised.sleeve_weight) > 0.01:
        raise ValueError("portfolio revision holding weights do not match sleeve_weight.")

    sleeve_total = 0.0
    for sleeve in revised.sleeves:
        if not sleeve.holding_symbols:
            raise ValueError(f"portfolio revision sleeve {sleeve.name!r} has no holding_symbols.")
        sleeve_total += sleeve.target_weight
        for symbol in sleeve.holding_symbols:
            if symbol.upper() not in holding_symbols:
                raise ValueError(
                    f"portfolio revision sleeve {sleeve.name!r} references non-holding {symbol!r}."
                )
    if abs(sleeve_total - revised.sleeve_weight) > 0.02:
        raise ValueError("portfolio revision sleeve weights do not match sleeve_weight.")

    required = {symbol.upper() for symbol in policy.required_symbols}
    missing_required = sorted(symbol for symbol in required if symbol in allowed_symbols and symbol not in holding_symbols)
    if missing_required:
        raise ValueError("portfolio revision omitted required symbols: " + ", ".join(missing_required))

    base_weights = {holding.symbol.upper(): holding.target_weight for holding in base_map.holdings}
    revised_weights = {holding.symbol.upper(): holding.target_weight for holding in revised.holdings}
    for edit in patch.edits:
        if edit.edit_type not in {"adjust_weight", "add_symbol"}:
            continue
        symbol = edit.target.id.upper() if edit.target.kind == "symbol" else ""
        if not symbol:
            continue
        if edit.direction in {"increase", "add"}:
            old_weight = base_weights.get(symbol, 0.0)
            new_weight = revised_weights.get(symbol, 0.0)
            if new_weight <= old_weight + 0.0001:
                raise ValueError(f"portfolio revision did not increase requested symbol {symbol}.")

    decreased = [
        symbol
        for symbol, old_weight in base_weights.items()
        if revised_weights.get(symbol, 0.0) < old_weight - 0.0001
    ]
    if decreased and not revision.reduced_or_removed:
        raise ValueError("portfolio revision lowered existing holdings but omitted reduced_or_removed audit.")


def _patch_explicitly_changes_cash(patch: PortfolioRevisionPatch) -> bool:
    for edit in patch.edits:
        if edit.direction in {"preserve", "neutral"}:
            continue
        target = edit.target.id.lower()
        if edit.edit_type == "set_constraint" and target in {"cash", "cash_weight", "cash_reserve"}:
            return True
        if edit.target.kind == "constraint" and target in {"cash", "cash_weight", "cash_reserve"}:
            return True
    return False


def _extract_architect_result(raw: dict[str, Any]) -> PortfolioArchitectResult:
    payload = raw.get("portfolio_architect_result") if isinstance(raw.get("portfolio_architect_result"), dict) else raw
    return PortfolioArchitectResult.model_validate(payload)


def _extract_selected_map(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    selected = raw.get("selected_portfolio_map")
    if isinstance(selected, dict):
        return selected
    payload = raw.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("selected_map"), dict):
        return payload["selected_map"]
    if isinstance(raw.get("selected_map"), dict):
        return raw["selected_map"]
    return raw if raw.get("map_id") else {}


def _resolve_base_map(
    *,
    architect_result: PortfolioArchitectResult,
    selected_map: dict[str, Any] | None = None,
    base_map_id: str = "",
) -> PortfolioMap:
    maps = {item.map_id: item for item in architect_result.portfolio_maps.maps}
    if base_map_id:
        if base_map_id not in maps:
            raise ValueError(f"Unknown base_map_id {base_map_id!r}; available map ids: {', '.join(maps)}")
        return maps[base_map_id]
    if selected_map and selected_map.get("map_id"):
        selected_id = str(selected_map["map_id"])
        if selected_id in maps:
            return maps[selected_id]
        return PortfolioMap.model_validate(selected_map)
    if len(maps) == 1:
        return next(iter(maps.values()))
    raise ValueError("base_map_id is required when portfolio_architect_result contains multiple maps.")


def _default_policy_from_architect(
    architect_result: PortfolioArchitectResult,
    base_map: PortfolioMap,
) -> InvestmentPolicy:
    max_holding = max((holding.target_weight for holding in base_map.holdings), default=0.15)
    return InvestmentPolicy(
        theme=architect_result.theme,
        theme_description="",
        required_symbols=[],
        objective=base_map.objective,
        risk_level="moderate",
        target_portfolio_weight=base_map.sleeve_weight,
        cash_reserve=base_map.cash_weight,
        single_name_limit=max(0.15, max_holding),
        allow_options=False,
        notes="Default policy inferred by portfolio revision CLI.",
    )


def _allowed_symbols(
    architect_result: PortfolioArchitectResult,
    deep_research: dict[str, Any],
    base_map: PortfolioMap,
) -> set[str]:
    symbols: set[str] = {holding.symbol.upper() for holding in base_map.holdings if holding.symbol}
    selection = architect_result.selection
    for item in selection.selected_for_portfolio:
        if item.symbol:
            symbols.add(item.symbol.upper())
    for bucket in (
        selection.watchlist_after_enrichment,
        selection.deferred_after_enrichment,
        selection.rejected_after_enrichment,
    ):
        for item in bucket:
            if item.symbol:
                symbols.add(item.symbol.upper())
    for portfolio_map in architect_result.portfolio_maps.maps:
        for omitted in portfolio_map.omitted_candidates:
            if omitted.symbol:
                symbols.add(omitted.symbol.upper())
    for card in (deep_research.get("candidate_cards") if isinstance(deep_research, dict) else []) or []:
        if isinstance(card, dict) and card.get("symbol"):
            symbols.add(str(card["symbol"]).upper())
    return symbols


def _compact_deep_research(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "artifact_type": raw.get("artifact_type", "deep_research_report"),
        "theme": raw.get("theme", ""),
        "generated_at": raw.get("generated_at", ""),
        "research_summary": raw.get("research_summary", ""),
        "candidate_cards": raw.get("candidate_cards", []),
        "layer_conclusions": raw.get("layer_conclusions", []),
        "cross_layer_thesis": raw.get("cross_layer_thesis", []),
        "architect_inputs": raw.get("architect_inputs", {}),
        "data_gaps": raw.get("data_gaps", []),
        "warnings": raw.get("warnings", []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "revise":
        policy = None
        if args.policy_path:
            policy = InvestmentPolicy.model_validate(_read_json(Path(args.policy_path)))
        patch, revision, run = build_portfolio_revision_from_files(
            portfolio_architect_path=args.portfolio_architect_path,
            selected_map_path=args.selected_map_path,
            deep_research_path=args.deep_research_path,
            user_request=args.request,
            base_map_id=args.base_map_id or "",
            policy=policy,
            output_dir=args.output_dir,
            save_context=not args.no_save_context,
        )
        payload = {
            "run": run.model_dump(mode="json"),
            "portfolio_revision_patch": patch.model_dump(mode="json"),
            "portfolio_map_revision": revision.model_dump(mode="json") if revision else None,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run_id: {run.run_id}")
            print(f"status: {run.status}")
            print(f"base_map_id: {run.base_map_id}")
            print(f"patch_path: {run.patch_path}")
            if run.revision_path:
                print(f"revision_path: {run.revision_path}")
            if patch.needs_clarification:
                print("clarification:")
                print(patch.clarification_question)
                for option in patch.clarification_options:
                    print(f"  - {option}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-portfolio-revision",
        description="Revise an AI-authored target portfolio map with a PydanticAI agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    revise = subparsers.add_parser("revise", help="Generate a reviewable map revision.")
    revise.add_argument("--portfolio-architect-path", required=True, help="portfolio_maps.json path.")
    revise.add_argument("--request", required=True, help="Natural-language revision request.")
    revise.add_argument("--base-map-id", default="", help="Base map id to revise.")
    revise.add_argument("--selected-map-path", help="Optional selected_portfolio_map artifact JSON.")
    revise.add_argument("--deep-research-path", help="Optional deep_research_report JSON.")
    revise.add_argument("--policy-path", help="Optional InvestmentPolicy JSON.")
    revise.add_argument("--output-dir", help="Directory for revision artifacts.")
    revise.add_argument("--no-save-context", action="store_true")
    revise.add_argument("--json", action="store_true", help="Print full JSON output.")
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


_REVISION_PATCH_INSTRUCTIONS = """
You are the investment assistant's Portfolio Revision Intent Agent.

Read portfolio_revision_context and convert the user's natural-language request
into PortfolioRevisionPatch only. Do not revise weights in this step.

Evidence boundary:
- The base_map is the only map being revised.
- portfolio_architect_result and deep_research_report define the researched
  symbol universe and evidence boundary.
- Do not use current holdings, orders, prices, or outside market memory.
- Do not introduce new facts or new symbols outside allowed_symbols. If the user
  asks for a symbol outside allowed_symbols, set requires_research=true and ask
  for clarification or research rather than pretending it is validated.

Parsing behavior:
- Use generic edits, not hardcoded workflows:
  adjust_weight, adjust_sleeve, add_symbol, remove_symbol, replace_symbol,
  set_constraint, change_style, preserve_position, request_research.
- If the user says "提高一些", "降低一点", "偏向", or similar qualitative
  language, do not ask for exact percentages. Set magnitude.kind to ai_decide
  or small/moderate and let the revision architect choose a risk-budget-aware
  number.
- Ask for clarification only when the target is ambiguous, the base map cannot
  be identified from context, the request conflicts with policy constraints, or
  the user requests a new unresearched symbol.
- revision_intent must explain the user's requested change in plain language.
- constraints and preserve should capture user constraints and things that must
  not move.
- Return PortfolioRevisionPatch only.
"""


_REVISION_ARCHITECT_INSTRUCTIONS = """
You are the investment assistant's Portfolio Revision Architect Agent.

Read portfolio_revision_context plus portfolio_revision_patch and produce
PortfolioMapRevision only.

Evidence boundary:
- Revise only the supplied base_map.
- Use portfolio_architect_result and deep_research_report as evidence.
- Do not use web search, model memory, current holdings, order history, or new
  market facts.
- Do not create trade orders, buy/sell instructions, trigger prices, options,
  or current-position advice.

Revision rules:
- Preserve base_map.sleeve_weight exactly.
- Preserve base_map.cash_weight unless the patch explicitly changes cash.
- Respect policy.single_name_limit.
- Holdings must be allowed_symbols only.
- If the patch asks to increase a symbol, increase its target weight and fund it
  by reducing one or more lower-conviction, overlapping, or lower-priority
  holdings.
- Explain funding_sources: which holdings or sleeves gave up weight and why.
- Explain tradeoff_explanation and risk_delta.
- Keep required_symbols when they are available.
- Keep the map internally consistent: holding weights sum to sleeve_weight,
  sleeve weights sum to sleeve_weight, and sleeve.holding_symbols exactly
  reference holdings in the revised map.
- Use evidence_refs already present in the supplied artifacts.
- Return a reviewable revision, not a final accepted map. Set
  needs_user_confirmation=true.
"""
