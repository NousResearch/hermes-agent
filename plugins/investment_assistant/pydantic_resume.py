"""Pydantic-only HITL helpers for investment-assistant stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .candidate_triage import (
    CandidateTriageArtifact,
    CandidateTriagePlanArtifact,
    TriageStrategySelection,
    build_candidate_triage_plan,
    build_candidate_triage_from_plan,
)
from .lightweight_enrichment import LightweightEnrichmentArtifact
from .schemas import (
    ThemeCoverageRequirement,
    ThemeDiscoveryPlan,
    ThemeDiscoverySeed,
    ThemeDomain,
    ThemeDomainCandidate,
    ThemeSubdomain,
)
from .storage import new_id, utc_now


class CandidateTriageHitlState(BaseModel):
    """Self-contained Pydantic HITL state for candidate triage."""

    artifact_type: str = "pydantic_candidate_triage_hitl_state"
    workflow_type: Literal["candidate_triage"] = "candidate_triage"
    session_id: str = Field(default_factory=lambda: new_id("pyhitl"))
    status: Literal["waiting_for_human", "completed", "failed"] = "waiting_for_human"
    state: Literal["candidate_triage_plan", "candidate_triage_complete", "failed"] = "candidate_triage_plan"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    theme: str = ""
    market: str = "US"
    discovery: ThemeDiscoveryPlan
    lightweight: LightweightEnrichmentArtifact
    candidate_triage_plan: CandidateTriagePlanArtifact
    selection: TriageStrategySelection | None = None
    candidate_triage: CandidateTriageArtifact | None = None
    prompt_to_user: str = ""
    allowed_actions: list[str] = Field(default_factory=lambda: ["answer", "resume", "cancel"])
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, str] = Field(default_factory=dict)


def start_candidate_triage_hitl(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
) -> CandidateTriageHitlState:
    """Run the Pydantic triage planner and return a waiting HITL state."""

    plan = build_candidate_triage_plan(discovery, lightweight)
    return create_candidate_triage_hitl_state(discovery, lightweight, plan)


def create_candidate_triage_hitl_state(
    discovery: ThemeDiscoveryPlan,
    lightweight: LightweightEnrichmentArtifact,
    plan: CandidateTriagePlanArtifact,
) -> CandidateTriageHitlState:
    """Create a waiting HITL state from existing typed artifacts."""

    return CandidateTriageHitlState(
        status="waiting_for_human",
        state="candidate_triage_plan",
        theme=discovery.theme or plan.theme,
        market=discovery.market or lightweight.market or plan.market,
        discovery=discovery,
        lightweight=lightweight,
        candidate_triage_plan=plan,
        prompt_to_user=plan.prompt_to_user,
        warnings=[*discovery.warnings, *lightweight.warnings, *plan.warnings],
    )


def resume_candidate_triage_hitl(
    state: CandidateTriageHitlState,
    *,
    option_id: str = "",
    answer: str = "",
    modifications: str = "",
    must_include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
) -> CandidateTriageHitlState:
    """Resume a waiting Pydantic HITL state and run final candidate triage."""

    if state.status != "waiting_for_human" or state.state != "candidate_triage_plan":
        raise ValueError("Candidate triage HITL state is not waiting for user input.")
    triage, selection = build_candidate_triage_from_plan(
        state.discovery,
        state.lightweight,
        state.candidate_triage_plan,
        option_id=option_id,
        answer=answer,
        modifications=modifications,
        must_include_symbols=must_include_symbols,
        exclude_symbols=exclude_symbols,
    )
    return state.model_copy(
        update={
            "status": "completed",
            "state": "candidate_triage_complete",
            "updated_at": utc_now(),
            "selection": selection,
            "candidate_triage": triage,
            "prompt_to_user": "",
            "allowed_actions": [],
            "warnings": [*state.warnings, *triage.warnings],
        },
        deep=True,
    )


def start_candidate_triage_hitl_from_files(
    *,
    discovery_path: str | Path,
    lightweight_path: str | Path,
    output_path: str | Path | None = None,
) -> CandidateTriageHitlState:
    """Run triage planning from saved prior artifacts and optionally save state."""

    discovery = _load_theme_discovery(discovery_path)
    lightweight = LightweightEnrichmentArtifact.model_validate(_read_json(lightweight_path))
    state = start_candidate_triage_hitl(discovery, lightweight)
    if output_path:
        save_candidate_triage_hitl_state(state, output_path)
    return state


def create_candidate_triage_hitl_state_from_files(
    *,
    discovery_path: str | Path,
    lightweight_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path | None = None,
) -> CandidateTriageHitlState:
    """Create waiting HITL state from saved discovery, lightweight, and plan artifacts."""

    discovery, lightweight, plan = load_candidate_triage_resume_artifacts(
        discovery_path=discovery_path,
        lightweight_path=lightweight_path,
        plan_path=plan_path,
    )
    state = create_candidate_triage_hitl_state(discovery, lightweight, plan)
    if output_path:
        save_candidate_triage_hitl_state(state, output_path)
    return state


def resume_candidate_triage_hitl_from_file(
    *,
    state_path: str | Path,
    output_path: str | Path | None = None,
    option_id: str = "",
    answer: str = "",
    modifications: str = "",
    must_include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
) -> CandidateTriageHitlState:
    """Load a waiting HITL state, resume it, and optionally save completed state."""

    state = load_candidate_triage_hitl_state(state_path)
    completed = resume_candidate_triage_hitl(
        state,
        option_id=option_id,
        answer=answer,
        modifications=modifications,
        must_include_symbols=must_include_symbols,
        exclude_symbols=exclude_symbols,
    )
    if output_path:
        save_candidate_triage_hitl_state(completed, output_path)
    return completed


def save_candidate_triage_hitl_state(state: CandidateTriageHitlState, path: str | Path) -> None:
    """Persist a self-contained Pydantic HITL state to JSON."""

    Path(path).write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_candidate_triage_hitl_state(path: str | Path) -> CandidateTriageHitlState:
    """Load a self-contained Pydantic HITL state from JSON."""

    return CandidateTriageHitlState.model_validate(_read_json(path))


def load_candidate_triage_resume_artifacts(
    *,
    discovery_path: str | Path,
    lightweight_path: str | Path,
    plan_path: str | Path,
) -> tuple[ThemeDiscoveryPlan, LightweightEnrichmentArtifact, CandidateTriagePlanArtifact]:
    """Load typed artifacts needed to resume candidate triage."""

    discovery = _load_theme_discovery(discovery_path)
    lightweight = LightweightEnrichmentArtifact.model_validate(_read_json(lightweight_path))
    plan = CandidateTriagePlanArtifact.model_validate(_read_json(plan_path))
    return discovery, lightweight, plan


def resume_candidate_triage_from_files(
    *,
    discovery_path: str | Path,
    lightweight_path: str | Path,
    plan_path: str | Path,
    option_id: str = "",
    answer: str = "",
    modifications: str = "",
    must_include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
) -> tuple[CandidateTriageArtifact, TriageStrategySelection]:
    """Run candidate triage from saved Pydantic artifacts without prior stages."""

    discovery, lightweight, plan = load_candidate_triage_resume_artifacts(
        discovery_path=discovery_path,
        lightweight_path=lightweight_path,
        plan_path=plan_path,
    )
    return build_candidate_triage_from_plan(
        discovery,
        lightweight,
        plan,
        option_id=option_id,
        answer=answer,
        modifications=modifications,
        must_include_symbols=must_include_symbols,
        exclude_symbols=exclude_symbols,
    )


def _read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_theme_discovery(path: str | Path) -> ThemeDiscoveryPlan:
    """Load a full discovery artifact or a discovery-v1 preview wrapper."""

    raw = _read_json(path)
    try:
        return ThemeDiscoveryPlan.model_validate(raw)
    except ValidationError:
        if isinstance(raw, dict) and isinstance(raw.get("preview"), dict):
            return _theme_discovery_from_preview(raw)
        raise


def _theme_discovery_from_preview(raw: dict) -> ThemeDiscoveryPlan:
    """Convert discovery-v1 preview JSON into the canonical discovery schema."""

    preview = raw["preview"]
    theme = str(preview.get("theme") or "")
    market = str(preview.get("market") or "US")
    raw_layers = preview.get("layers") or []
    raw_candidates = preview.get("candidates") or []

    candidates_by_layer: dict[str, list[dict]] = {}
    first_seed_by_symbol: dict[str, ThemeDiscoverySeed] = {}
    subthemes_by_symbol: dict[str, list[str]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        symbol = str(candidate.get("symbol") or "").strip()
        if not symbol:
            continue
        layer_key = str(candidate.get("layer_key") or "uncategorized")
        candidates_by_layer.setdefault(layer_key, []).append(candidate)
        role = str(candidate.get("role") or "")
        rationale = str(candidate.get("rationale") or "")
        subthemes_by_symbol.setdefault(symbol, [])
        if layer_key not in subthemes_by_symbol[symbol]:
            subthemes_by_symbol[symbol].append(layer_key)
        first_seed_by_symbol.setdefault(
            symbol,
            ThemeDiscoverySeed(
                symbol=symbol,
                market=market,
                role=role or "preview discovery candidate",
                rationale=rationale,
                subthemes=[layer_key],
                value_chain_stage=layer_key,
                exposure_type=str(candidate.get("exposure_type") or "preview_candidate"),
                exposure_purity=str(candidate.get("exposure_purity") or "unknown"),
                source_ids=[str(item) for item in candidate.get("source_ids") or []],
                confidence="medium",
                freshness="unknown",
            ),
        )

    layer_keys = [str(layer.get("key") or "") for layer in raw_layers if isinstance(layer, dict)]
    missing_layer_keys = [key for key in candidates_by_layer if key not in set(layer_keys)]
    layer_records = [layer for layer in raw_layers if isinstance(layer, dict)]
    layer_records.extend({"key": key, "name": key, "importance": "important"} for key in missing_layer_keys)

    domain_tree: list[ThemeDomain] = []
    coverage_requirements: list[ThemeCoverageRequirement] = []
    for layer in layer_records:
        layer_key = str(layer.get("key") or "uncategorized")
        layer_name = str(layer.get("name") or layer_key)
        layer_thesis = str(layer.get("economic_mechanism") or layer.get("thesis") or "")
        layer_importance = _preview_layer_importance(layer.get("importance"))
        layer_candidates = candidates_by_layer.get(layer_key, [])
        domain_candidates = [
            ThemeDomainCandidate(
                symbol=str(candidate.get("symbol") or "").strip(),
                role=str(candidate.get("role") or ""),
                rationale=str(candidate.get("rationale") or ""),
                priority=_preview_candidate_priority(candidate.get("priority")),
            )
            for candidate in layer_candidates
            if str(candidate.get("symbol") or "").strip()
        ]
        domain_tree.append(
            ThemeDomain(
                key=layer_key,
                name=layer_name,
                thesis=layer_thesis,
                importance=layer_importance,
                subdomains=[
                    ThemeSubdomain(
                        key=layer_key,
                        name=layer_name,
                        thesis=layer_thesis,
                        importance=_preview_subdomain_importance(layer_importance),
                        candidates=domain_candidates,
                    )
                ],
            )
        )
        symbols = [candidate.symbol for candidate in domain_candidates]
        coverage_requirements.append(
            ThemeCoverageRequirement(
                key=layer_key,
                name=layer_name,
                thesis=layer_thesis,
                priority=_preview_coverage_priority(layer_importance),
                min_candidates=1 if symbols else 0,
                candidate_symbols=symbols,
                must_consider_symbols=[
                    candidate.symbol
                    for candidate in domain_candidates
                    if candidate.priority == "must_consider"
                ],
                evidence_needed=["lightweight_futu_enrichment", "candidate_triage"],
            )
        )

    seed_symbols: list[ThemeDiscoverySeed] = []
    for symbol, seed in first_seed_by_symbol.items():
        seed_symbols.append(seed.model_copy(update={"subthemes": subthemes_by_symbol.get(symbol, seed.subthemes)}))

    preview_warnings = [str(item) for item in preview.get("warnings") or []]
    return ThemeDiscoveryPlan(
        theme=theme,
        market=market,
        theme_description=str(preview.get("theme_description") or ""),
        initial_thesis=str(preview.get("initial_thesis") or ""),
        domain_tree=domain_tree,
        coverage_requirements=coverage_requirements,
        seed_symbols=seed_symbols,
        next_enrichment_needed=[str(item) for item in preview.get("next_enrichment_needed") or []],
        warnings=[
            *preview_warnings,
            "Loaded from discovery-v1 preview wrapper; converted to ThemeDiscoveryPlan for resume.",
        ],
        pydantic_ai={
            **(raw.get("pydantic_ai") or {}),
            "source_format": "discovery_v1_preview",
        },
    )


def _preview_layer_importance(value: object) -> str:
    if value in {"core", "important", "optional"}:
        return str(value)
    return "important"


def _preview_subdomain_importance(layer_importance: str) -> str:
    return "high" if layer_importance in {"core", "important"} else "medium"


def _preview_coverage_priority(layer_importance: str) -> str:
    if layer_importance == "core":
        return "required"
    if layer_importance == "optional":
        return "optional"
    return "important"


def _preview_candidate_priority(value: object) -> str:
    if value in {"must_consider", "strong_candidate", "watchlist"}:
        return str(value)
    return "strong_candidate"
