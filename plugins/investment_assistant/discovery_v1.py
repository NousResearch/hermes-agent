"""Production filter-planning discovery agent for investment themes."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .filter_calibration import (
    diagnose_trial_result,
    validate_stock_filter_specs_against_catalog,
)
from .futu_theme_discovery import (
    _build_futu_stock_filters,
    _normalize_market,
    load_futu_screener_catalog_snapshot,
)
from .pydantic_runtime import create_pydantic_agent, usage_metadata
from .schemas import (
    DiscoveryFilterPlan,
    DiscoveryLayerFilterAudit,
    DiscoveryOmission,
    ExecutedDiscoveryProbe,
    ResearchSource,
    ThemeCoverageRequirement,
    ThemeDiscoveryPlan,
    ThemeDiscoverySeed,
    ThemeDomain,
    ThemeDomainCandidate,
    ThemeSubdomain,
)
from .storage import utc_now
from .theme_discovery import normalize_futu_symbol


_CATALOG_ROOT = Path(__file__).resolve().parent / "docs" / "futu_screener_catalog"
LOGGER = logging.getLogger(__name__)
_COMPACT_PROBE_FULL_PROMOTION_MAX_RESULTS = 6
_BROAD_PLATE_REFINEMENT_MIN_RESULTS = 20
_DISCOVERY_SYMBOL_RE = re.compile(r"^(?:[A-Z]{1,5}\.)?[A-Z0-9][A-Z0-9.\-]{0,24}$")


def _validate_discovery_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise ValueError("Candidate symbol must not be empty")
    if not value.isascii() or not _DISCOVERY_SYMBOL_RE.fullmatch(value):
        raise ValueError(f"Invalid candidate symbol: {symbol!r}")
    return value


_DISCOVERY_V1_INSTRUCTIONS = """
You are a theme-discovery agent focused on filter design and candidate discovery.

Your job is not to recommend a portfolio. Your job is to turn a user's
investment theme into subdomain-specific screener filter plans, execute those
filters with Futu tools, and return candidates with clear filter rationale.

Available local tools:
1. list_futu_screener_catalog
   List local Futu screener catalog directories/files.
2. read_futu_screener_catalog
   Read exact catalog files returned by list_futu_screener_catalog.
3. run_futu_stock_filter
   Execute live Futu get_stock_filter probes.

Core workflow:

1. Build an economic exposure map first.
   Before using Futu filters or listing stocks, decompose the user's theme into
   investable economic exposures. Ask what different business models can
   benefit or be repriced by the theme:
   - direct product or service sellers,
   - bottleneck component or capacity suppliers,
   - scarce asset owners,
   - operators, builders, maintainers, and service providers,
   - platforms, distribution channels, and application owners,
   - upstream inputs, enabling infrastructure, and second-order beneficiaries,
   - proxies with indirect exposure, and concept names with weak revenue
     exposure that need later validation.
   For each important exposure, explain the economic mechanism and what type of
   company should be discovered. Keep materially different economic exposures
   separate. Do not merge a distinct exposure into a broader layer just because
   it is adjacent in the value chain. If one layer only captures a single
   company type, actively search adjacent catalog plates to check whether asset
   owners, equipment suppliers, operators, service providers, upstream
   suppliers, downstream platforms, or other relevant exposure types are being
   missed.
   Keep materially different bottleneck branches separate. For infrastructure
   or platform themes, explicitly decide whether memory/storage,
   power/cooling, optical/networking, compute, software, asset ownership, and
   service/operator branches are distinct enough to deserve separate probes.

2. Design filters before executing tools.
   For every core or important economic exposure, create a filter plan first. Each
   filter plan must decide whether to use:
   - plate / industry / concept filter
   - market cap filter
   - liquidity filter
   - valuation filter
   - technical filter
   - financial filter
   - analyst / analysis filter
   - options filter
   In target_candidate_profile, name the exposure type being searched, such as
   direct_seller, bottleneck_supplier, asset_owner, operator, infra_builder,
   platform, application_owner, upstream_input, or proxy.
   For each category, choose exactly one decision: use_now, skip, or
   defer_to_later_enrichment. Explain every decision. If a category is not
   suitable for initial discovery, defer it and name the later enrichment source.

3. Use the catalog to find exact available filters.
   Start with list_futu_screener_catalog. Then read specific files such as
   market index, plates, market_quote, valuation, technical, financial,
   analysis, and options. The catalog may contain Chinese labels, but enum
   values and plate_code values are exact. Do not invent Futu field names or
   plate codes.
   For each important exposure, search the catalog with both:
   - theme words, and
   - business-model words implied by the exposure map.
   Record candidate plates considered in plate_codes_to_probe and
   plate_codes_considered. If you skip a plausible adjacent plate, explain why in
   skipped_or_deferred_filters or result_summary. The goal is not to use every
   plate; it is to make the exploration auditable.

4. Execute filters as probes.
   Use run_futu_stock_filter only after the filter plan is clear.

   Probe type A: Broad calibrated probe
   - No plate_code.
   - Use real filter_min / filter_max thresholds.
   - Purpose: discover cross-subdomain names the plate system may miss.

   Probe type B: Subdomain plate probe
   - Use plate_code.
   - Use market cap, liquidity, growth, financial, or technical filters
     appropriate to the subdomain.
   - Purpose: discover candidates inside that subdomain.

   Probe type C: Subdomain plate refinement probe
   - Use the same plate_code as a broad subdomain plate probe.
   - Use multiple stock_filter_specs in the same run_futu_stock_filter call.
     Treat this as a Futu-executed intersection, not an LLM-computed set
     operation.
   - Use probe_type="subdomain_plate_refinement" in executed_filter_probes.
   - Purpose: narrow a broad plate universe before selecting candidates.
   - Example: storage concept ∩ MARKET_VAL > 5B ∩ 20-day TURNOVER > 100M
     ∩ recent revenue growth > 0.

   If a core or important subdomain plate probe returns many names, do not
   finalize candidates directly from the raw plate universe. Run at least one
   refinement probe on the same layer/plate using size, liquidity, growth,
   financial quality, valuation, or technical filters that fit the layer thesis.
   Prefer one multi-filter Futu call over asking the LLM to compute list
   intersections manually.

5. Do not confuse ranking with filtering.
   is_no_filter=true means ranking only. It does not apply a threshold. Use
   filter_min / filter_max when you actually want to filter. Do not put sort on
   multiple fields in the same Futu filter call. If multiple sorts are needed,
   run separate probes or choose the most important sort field.

6. Audit every core or important subdomain.
   For each core or important layer, include a layer_filter_audit with:
   - plate codes considered and used,
   - exact filter categories and StockField enums used,
   - numeric thresholds or ranking settings used,
   - tool_trace_ids that executed those probes,
   - why those filters fit the layer's investment hypothesis,
   - candidate symbols that came from probes,
   - relevant filters skipped/deferred and why.
   The audit should make clear whether the layer covered only one business model
   or multiple adjacent exposure types. If only one business model was covered,
   state what adjacent exposure types still need later exploration.

7. Candidate output rules.
   The `candidates` array is the authoritative downstream candidate source.
   `layers[].candidate_symbols` is only a layer map and will not by itself
   become a downstream seed.
   Include a candidate if:
   - it appears in a relevant plate probe, or
   - it appears in a broad calibrated probe and matches the subdomain thesis, or
   - it is user-required, or
   - it is an important model-prior candidate that needs later validation.
   If a symbol appears in a relevant subdomain plate probe and you list it in
   that layer's candidate_symbols, also include it in candidates. Use
   priority="watchlist" when the symbol still needs SEC, filing, liquidity,
   valuation, or current-financial validation. "Needs later validation" is not
   a reason to omit a discovered candidate from candidates.
   If a core or important subdomain plate probe returns a compact result set
   of 6 or fewer symbols, every returned symbol must be included in
   candidates[] or omitted with a structured hard exclusion reason.
   Discovery is not portfolio construction. Do not omit a plausible symbol
   solely because the same layer already has stronger or more representative
   candidates. Same-layer coverage is only a later portfolio-construction
   consideration, not a discovery-stage exclusion reason.
   If a symbol is discovered but not promoted to candidates, put it in
   omissions_to_investigate as a structured object with one hard exclusion
   reason:
   - invalid_symbol
   - outside_requested_market
   - duplicate_share_class
   - explicit_user_exclusion
   - failed_liquidity_gate
   - failed_size_gate
   - clear_theme_mismatch
   - unsupported_security_type
   "Needs SEC validation", "needs current financial validation",
   "post-spinoff financials need review", and "same-layer exposure is already
   covered" are not valid omission reasons. Put those concerns in the
   candidate rationale or next_enrichment_needed and keep the symbol in
   candidates[] as priority="watchlist".

8. Final output.
   Return a structured discovery package in English. The output must include:
   - theme
   - market
   - initial_thesis
   - layers
   - filter_plans_by_layer
   - executed_filter_probes
   - layer_filter_audits
   - search_queries
   - research_trace
   - candidates
   - omissions_to_investigate
   - next_enrichment_needed
   - warnings

Keep ticker symbols, Futu enum values, plate codes, and tool ids exactly as
returned by tools. Do not output portfolio weights, trade plans, target prices,
or orders.

Keep the artifact compact enough for downstream review:
- target 8-12 layers unless the theme is genuinely broader,
- target 40-90 candidates,
- keep each rationale to one concise sentence,
- put detailed facts in probe/audit trace references rather than repeating long
  explanations in every candidate,
- prefer short next_enrichment_needed items over long prose.
""".strip()


_DISCOVERY_V1_PREVIEW_INSTRUCTIONS = """
You are a theme-discovery agent focused on filter design and candidate discovery.

Your job is not to recommend a portfolio. Your job is to explore a user's
investment theme, design Futu screener filters, run live Futu probes, and return
a compact discovery preview for human review.

Available local tools:
1. list_futu_screener_catalog
2. read_futu_screener_catalog
3. run_futu_stock_filter

Workflow:

1. Build an economic exposure map before naming stocks.
   Decompose the theme into investable business-model exposures: direct sellers,
   bottleneck suppliers, scarce asset owners, operators/builders/maintainers,
   platforms/distribution/application owners, upstream inputs, enabling
   infrastructure, second-order beneficiaries, and weaker proxies.

2. Use the Futu catalog before filtering.
   Start by listing the catalog. Read the market index, plate files, and the
   filter category files needed for your plan. Use exact plate_code and
   stock_field enum values from the catalog.

3. For every important exposure, decide the filters first.
   Consider plate/industry/concept, market cap, liquidity, valuation, technical,
   financial, analyst/analysis, dividend, and options filters. Use a category
   only when it helps initial discovery; otherwise explain why it should be
   deferred to later enrichment. If a plate is broad, run a refinement probe on
   the same plate using multiple stock_filter_specs in a single Futu call.

4. Execute live Futu probes.
   Use run_futu_stock_filter for broad discovery and subdomain plate probes.
   Use filter_min/filter_max with is_no_filter=false for real thresholds.
   Use is_no_filter=true only for ranking. Do not invent Futu fields.

5. Return only a compact preview.
   Include 8-12 layers and roughly 30-70 candidates. Keep rationales to one
   concise sentence. Include trace ids from the Futu probes that support each
   layer/candidate. Do not output portfolio weights, target prices, trade plans,
   or orders.
   The candidates[] array is the only candidate source of truth. Do not put
   candidate lists inside layers. Every required symbol from required_symbols
   must appear in candidates[] with priority="must_consider"; required symbols
   are user constraints, not proof of investment quality.
""".strip()


class DiscoveryLayer(BaseModel):
    key: str
    name: str
    thesis: str = ""
    importance: Literal["core", "important", "optional"] = "important"
    candidate_symbols: list[str] = Field(default_factory=list)


class DiscoveryCandidate(BaseModel):
    symbol: str
    name: str = ""
    layer_key: str
    subdomain: str = ""
    role: str = ""
    rationale: str = ""
    priority: Literal["must_consider", "strong_candidate", "watchlist"] = "strong_candidate"
    source_trace_ids: list[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return _validate_discovery_symbol(value)


class DiscoveryPackage(BaseModel):
    theme: str
    market: str = "US"
    initial_thesis: str
    layers: list[DiscoveryLayer] = Field(default_factory=list)
    filter_plans_by_layer: list[DiscoveryFilterPlan] = Field(default_factory=list)
    executed_filter_probes: list[ExecutedDiscoveryProbe] = Field(default_factory=list)
    layer_filter_audits: list[DiscoveryLayerFilterAudit] = Field(default_factory=list)
    research_trace: list[ResearchSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    tool_trace_ids_used: list[str] = Field(default_factory=list)
    filter_trial_ids_used: list[str] = Field(default_factory=list)
    omissions_to_investigate: list[DiscoveryOmission] = Field(default_factory=list)
    next_enrichment_needed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_package(self) -> "DiscoveryPackage":
        layer_keys = {layer.key for layer in self.layers}
        missing_layers = [item.symbol for item in self.candidates if item.layer_key not in layer_keys]
        if missing_layers:
            raise ValueError(f"Candidates reference missing layer_key: {missing_layers}")
        planned_layers = {item.layer_key for item in self.filter_plans_by_layer}
        audited_layers = {item.layer_key for item in self.layer_filter_audits}
        missing_filter_context = [
            layer.key
            for layer in self.layers
            if layer.importance in {"core", "important"}
            and layer.key not in planned_layers
            and layer.key not in audited_layers
        ]
        if missing_filter_context:
            raise ValueError(f"Missing filter plan/audit for important layers: {missing_filter_context}")
        missing_probe_candidates = _missing_probe_candidate_promotions(self)
        if missing_probe_candidates:
            raise ValueError(
                "Compact important subdomain plate probe hits must be included in candidates[] "
                "as at least watchlist candidates, unless omitted with a structured hard "
                "exclusion reason; later validation belongs in rationale/next_enrichment_needed, "
                "not omissions_to_investigate: "
                + "; ".join(missing_probe_candidates)
            )
        return self


class DiscoveryPreviewLayer(BaseModel):
    key: str
    name: str
    importance: Literal["core", "important", "optional"] = "important"
    economic_mechanism: str
    exposure_types: list[str] = Field(default_factory=list)
    filters_used: list[str] = Field(default_factory=list)
    filters_deferred: list[str] = Field(default_factory=list)
    plate_codes_used: list[str] = Field(default_factory=list)
    tool_trace_ids: list[str] = Field(default_factory=list)


class DiscoveryPreviewCandidate(BaseModel):
    symbol: str
    name: str = ""
    layer_key: str
    role: str
    priority: Literal["must_consider", "strong_candidate", "watchlist"] = "strong_candidate"
    rationale: str
    source_trace_ids: list[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return _validate_discovery_symbol(value)


class DiscoveryPreviewFilterTrace(BaseModel):
    trace_id: str
    layer_key: str = ""
    probe_goal: str
    plate_code: str | None = None
    filters_summary: list[str] = Field(default_factory=list)
    selected_symbols: list[str] = Field(default_factory=list)


class DiscoveryPreviewPackage(BaseModel):
    theme: str
    market: str = "US"
    initial_thesis: str
    layers: list[DiscoveryPreviewLayer] = Field(default_factory=list)
    filter_trace_summary: list[DiscoveryPreviewFilterTrace] = Field(default_factory=list)
    candidates: list[DiscoveryPreviewCandidate] = Field(default_factory=list)
    omissions_or_deferred: list[str] = Field(default_factory=list)
    next_enrichment_needed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_preview(self) -> "DiscoveryPreviewPackage":
        layer_keys = {layer.key for layer in self.layers}
        missing = [candidate.symbol for candidate in self.candidates if candidate.layer_key not in layer_keys]
        if missing:
            raise ValueError(f"Candidates reference missing layer_key: {missing}")
        return self


def _missing_probe_candidate_promotions(package: DiscoveryPackage) -> list[str]:
    """Find important plate-probe hits that were neither candidates nor hard omissions."""

    layer_importance = {layer.key: layer.importance for layer in package.layers}
    candidate_symbols = {
        symbol
        for symbol in (normalize_futu_symbol(candidate.symbol, package.market) for candidate in package.candidates)
        if symbol
    }
    omitted_symbols = {
        symbol
        for symbol in (
            normalize_futu_symbol(omission.symbol, package.market)
            for omission in package.omissions_to_investigate
        )
        if symbol
    }

    missing_by_layer: dict[str, set[str]] = {}
    for probe in package.executed_filter_probes:
        if probe.probe_type != "subdomain_plate" or not probe.plate_code:
            continue
        if layer_importance.get(probe.layer_key) not in {"core", "important"}:
            continue
        if str(probe.result_status or "").lower() not in {"ok", "usable"}:
            continue

        normalized_probe_symbols = {
            symbol
            for symbol in (normalize_futu_symbol(item, package.market) for item in probe.candidate_symbols)
            if symbol
        }
        required_candidates = (
            normalized_probe_symbols
            if probe.result_count <= _COMPACT_PROBE_FULL_PROMOTION_MAX_RESULTS
            else set()
        )
        missing = required_candidates - candidate_symbols
        missing -= omitted_symbols
        if missing:
            missing_by_layer.setdefault(probe.layer_key, set()).update(missing)

    return [
        f"{layer_key}: {', '.join(sorted(symbols))}"
        for layer_key, symbols in sorted(missing_by_layer.items())
    ]


@dataclass
class DiscoveryRecorder:
    max_catalog_reads: int
    max_filter_runs: int
    progress: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    catalog_reads: int = 0
    filter_runs: int = 0

    def add_tool_call(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
        trace_id = f"tool_{len(self.tool_calls) + 1:03d}"
        self.tool_calls.append(
            {
                "trace_id": trace_id,
                "time": utc_now(),
                "tool": tool,
                "args": args,
                "result": result,
            }
        )
        return trace_id

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append({"time": utc_now(), **event})

    def progress_log(self, stage: str, **fields: Any) -> None:
        if not self.progress:
            return
        record = {
            "time": utc_now(),
            "stage": stage,
            **fields,
        }
        message = f"IA_DISCOVERY_V1_PROGRESS {_preview(record, _env_int('IA_DISCOVERY_V1_PROGRESS_MAX_CHARS', 2400))}"
        LOGGER.info(message)
        print(message, file=sys.stderr, flush=True)


def build_ai_discovery_v1_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
) -> ThemeDiscoveryPlan:
    """Run the production discovery-v1 agent and return a ThemeDiscoveryPlan."""

    selected_market = _normalize_market(market)
    normalized_required = _normalize_symbols(required_symbols or [], selected_market)
    max_catalog_reads = _env_int("IA_DISCOVERY_V1_MAX_CATALOG_READS", 0)
    max_filter_runs = _env_int("IA_DISCOVERY_V1_MAX_FILTER_RUNS", 0)
    enable_web = _env_bool("IA_DISCOVERY_V1_WEB_ENABLED", True)
    require_web_research = _env_bool("IA_DISCOVERY_V1_REQUIRE_WEB_RESEARCH", False)
    recorder = DiscoveryRecorder(
        max_catalog_reads=max_catalog_reads,
        max_filter_runs=max_filter_runs,
        progress=_env_bool("IA_DISCOVERY_V1_PROGRESS", False),
    )
    overall_started = time.monotonic()
    recorder.progress_log(
        "build_start",
        theme=theme,
        market=selected_market,
        required_symbols=normalized_required,
        max_catalog_reads=max_catalog_reads,
        max_filter_runs=max_filter_runs,
        enable_web=enable_web,
        require_web_research=require_web_research,
    )

    agent, _model_config, runtime = create_pydantic_agent(
        output_type=DiscoveryPackage,
        instructions=_DISCOVERY_V1_INSTRUCTIONS,
        agent_kind="theme_discovery_v1_filter_planning_agent",
        output_retries=1,
        enable_web_search=enable_web,
        enable_web_fetch=enable_web,
        agent_skill_names=["theme-discovery"],
    )

    _register_catalog_tools(agent, recorder)
    _register_futu_filter_tool(agent, recorder)

    payload = {
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "tool_budget": {
            "max_catalog_reads": "unlimited" if max_catalog_reads == 0 else max_catalog_reads,
            "max_filter_runs": "unlimited" if max_filter_runs == 0 else max_filter_runs,
        },
        "require_web_research": require_web_research,
        "web_research_contract": (
            "If require_web_research is true and web capabilities are enabled, use web_search "
            "to update or challenge the theme map before finalizing. Use web_fetch only when "
            "a fetched source is necessary."
        ),
        "output_contract": "Return DiscoveryPackage only. No portfolio weights, orders, or trade plan.",
    }
    recorder.progress_log(
        "agent_run_start",
        payload_keys=sorted(payload),
        runtime_mode=runtime.get("mode"),
        capabilities=runtime.get("capabilities"),
    )
    try:
        result = agent.run_sync(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            event_stream_handler=_event_stream_handler(recorder),
        )
    except Exception as exc:
        recorder.progress_log(
            "agent_run_error",
            elapsed_s=round(time.monotonic() - overall_started, 3),
            error_type=type(exc).__name__,
            error=str(exc),
            tool_calls=len(recorder.tool_calls),
            event_count=len(recorder.events),
        )
        raise

    recorder.progress_log(
        "agent_run_end",
        elapsed_s=round(time.monotonic() - overall_started, 3),
        tool_calls=len(recorder.tool_calls),
        event_count=len(recorder.events),
        catalog_reads=recorder.catalog_reads,
        filter_runs=recorder.filter_runs,
    )
    try:
        plan = _theme_discovery_plan_from_package(
            result.output,
            theme=theme,
            market=selected_market,
            theme_description=theme_description,
            required_symbols=normalized_required,
            pydantic_ai={
                **runtime,
                "usage": usage_metadata(result),
                "tool_calls": recorder.tool_calls,
                "event_count": len(recorder.events),
                "events": recorder.events,
                "input": payload,
                "catalog_access": {
                    "catalog_reads": recorder.catalog_reads,
                    "filter_runs": recorder.filter_runs,
                },
            },
        )
    except Exception as exc:
        recorder.progress_log(
            "plan_conversion_error",
            elapsed_s=round(time.monotonic() - overall_started, 3),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    recorder.progress_log(
        "build_end",
        elapsed_s=round(time.monotonic() - overall_started, 3),
        seed_count=len(plan.seed_symbols),
        domain_count=len(plan.domain_tree),
        warning_count=len(plan.warnings),
    )
    return plan


def build_discovery_v1_preview(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Run the discovery-v1 agent and return a compact human-review preview.

    This is intentionally not wired into the workflow. It is a diagnostic path
    for auditing the discovery agent's exploration behavior without forcing the
    model to emit the full production DiscoveryPackage artifact.
    """

    selected_market = _normalize_market(market)
    normalized_required = _normalize_symbols(required_symbols or [], selected_market)
    max_catalog_reads = _env_int("IA_DISCOVERY_V1_MAX_CATALOG_READS", 0)
    max_filter_runs = _env_int("IA_DISCOVERY_V1_MAX_FILTER_RUNS", 0)
    enable_web = _env_bool("IA_DISCOVERY_V1_WEB_ENABLED", True)
    recorder = DiscoveryRecorder(
        max_catalog_reads=max_catalog_reads,
        max_filter_runs=max_filter_runs,
        progress=_env_bool("IA_DISCOVERY_V1_PROGRESS", False),
    )
    overall_started = time.monotonic()
    recorder.progress_log(
        "preview_build_start",
        theme=theme,
        market=selected_market,
        required_symbols=normalized_required,
        max_catalog_reads=max_catalog_reads,
        max_filter_runs=max_filter_runs,
        enable_web=enable_web,
    )
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=DiscoveryPreviewPackage,
        instructions=_DISCOVERY_V1_PREVIEW_INSTRUCTIONS,
        agent_kind="theme_discovery_v1_preview_agent",
        output_retries=1,
        enable_web_search=enable_web,
        enable_web_fetch=enable_web,
        agent_skill_names=["theme-discovery"],
    )
    _register_catalog_tools(agent, recorder)
    _register_futu_filter_tool(agent, recorder)
    payload = {
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "tool_budget": {
            "max_catalog_reads": "unlimited" if max_catalog_reads == 0 else max_catalog_reads,
            "max_filter_runs": "unlimited" if max_filter_runs == 0 else max_filter_runs,
        },
        "output_contract": (
            "Return DiscoveryPreviewPackage only. Keep it compact and do not output "
            "portfolio weights, orders, or trade plans."
        ),
    }
    recorder.progress_log(
        "preview_agent_run_start",
        payload_keys=sorted(payload),
        runtime_mode=runtime.get("mode"),
        capabilities=runtime.get("capabilities"),
    )
    try:
        result = agent.run_sync(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            event_stream_handler=_event_stream_handler(recorder),
        )
    except Exception as exc:
        recorder.progress_log(
            "preview_agent_run_error",
            elapsed_s=round(time.monotonic() - overall_started, 3),
            error_type=type(exc).__name__,
            error=str(exc),
            tool_calls=len(recorder.tool_calls),
            event_count=len(recorder.events),
        )
        raise

    preview = _preview_payload_with_constraints(
        result.output,
        market=selected_market,
        required_symbols=normalized_required,
    )
    elapsed_s = round(time.monotonic() - overall_started, 3)
    recorder.progress_log(
        "preview_build_end",
        elapsed_s=elapsed_s,
        layer_count=len(result.output.layers),
        candidate_count=len(result.output.candidates),
        tool_calls=len(recorder.tool_calls),
        catalog_reads=recorder.catalog_reads,
        filter_runs=recorder.filter_runs,
    )
    return {
        "preview": preview,
        "pydantic_ai": {
            **runtime,
            "usage": usage_metadata(result),
            "tool_calls": recorder.tool_calls,
            "event_count": len(recorder.events),
            "events": recorder.events,
            "input": payload,
            "catalog_access": {
                "catalog_reads": recorder.catalog_reads,
                "filter_runs": recorder.filter_runs,
            },
            "elapsed_s": elapsed_s,
        },
    }


def _preview_payload_with_constraints(
    output: DiscoveryPreviewPackage,
    *,
    market: str,
    required_symbols: list[str],
) -> dict[str, Any]:
    preview = output.model_dump()
    layers = list(preview.get("layers") or [])
    candidates = list(preview.get("candidates") or [])
    warnings = list(preview.get("warnings") or [])

    layer_keys = {str(layer.get("key")) for layer in layers if layer.get("key")}
    candidate_by_symbol = {
        symbol: candidate
        for symbol, candidate in (
            (normalize_futu_symbol(str(candidate.get("symbol") or ""), market), candidate)
            for candidate in candidates
        )
        if symbol
    }

    missing_required = [
        symbol
        for symbol in (normalize_futu_symbol(item, market) for item in required_symbols)
        if symbol and symbol not in candidate_by_symbol
    ]
    if missing_required:
        required_layer_key = _preview_required_layer_key(layers)
        if required_layer_key not in layer_keys:
            layers.insert(
                0,
                {
                    "key": required_layer_key,
                    "name": "User-required constraints",
                    "importance": "core",
                    "economic_mechanism": (
                        "Symbols explicitly required by the user are preserved as input constraints; "
                        "they still need later market and fundamental validation."
                    ),
                    "exposure_types": ["user_required_constraint"],
                    "filters_used": ["required_symbols input"],
                    "filters_deferred": ["classification by discovery agent", "market/fundamental validation"],
                    "plate_codes_used": [],
                    "tool_trace_ids": [],
                },
            )
            layer_keys.add(required_layer_key)
        for symbol in missing_required:
            candidates.insert(
                0,
                {
                    "symbol": symbol,
                    "name": "",
                    "layer_key": required_layer_key,
                    "role": "user-required discovery constraint",
                    "priority": "must_consider",
                    "rationale": (
                        "User explicitly required this symbol; included as an input constraint, "
                        "not as a model-authored recommendation."
                    ),
                    "source_trace_ids": [],
                },
            )
        warnings.append(
            "One or more required_symbols were absent from the model candidate list and were "
            "preserved as user-required constraints: " + ", ".join(missing_required)
        )

    for candidate in candidates:
        symbol = normalize_futu_symbol(str(candidate.get("symbol") or ""), market)
        if not symbol:
            continue
        candidate["symbol"] = symbol
        if symbol in set(required_symbols):
            candidate["priority"] = "must_consider"

    preview["layers"] = layers
    preview["candidates"] = candidates
    preview["warnings"] = _dedupe([str(item) for item in warnings if str(item).strip()])
    preview["candidates_by_layer"] = _group_preview_candidates_by_layer(candidates)
    return preview


def _preview_required_layer_key(layers: list[dict[str, Any]]) -> str:
    for layer in layers:
        key = str(layer.get("key") or "")
        name = str(layer.get("name") or "").lower()
        exposure_types = " ".join(str(item).lower() for item in layer.get("exposure_types") or [])
        if any(marker in f"{key.lower()} {name} {exposure_types}" for marker in ("required", "constraint", "base", "beta", "底仓")):
            return key
    return "required_constraints"


def _group_preview_candidates_by_layer(candidates: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        layer_key = str(candidate.get("layer_key") or "")
        symbol = str(candidate.get("symbol") or "")
        if not layer_key or not symbol:
            continue
        grouped.setdefault(layer_key, [])
        if symbol not in grouped[layer_key]:
            grouped[layer_key].append(symbol)
    return grouped


def _register_catalog_tools(agent: Any, recorder: DiscoveryRecorder) -> None:
    @agent.tool_plain(
        name="list_futu_screener_catalog",
        description=(
            "List directories/files in the local Futu screener catalog. Use this before "
            "read_futu_screener_catalog. Args: path='', market='US', max_depth=1, limit=80."
        ),
        retries=1,
        timeout=20,
    )
    def list_futu_screener_catalog(
        path: str = "",
        market: str = "US",
        max_depth: int = 1,
        limit: int = 80,
    ) -> dict[str, Any]:
        started = time.monotonic()
        args = {"path": path, "market": market, "max_depth": max_depth, "limit": limit}
        recorder.progress_log("tool_start", tool="list_futu_screener_catalog", args=args)
        try:
            if _budget_exceeded(recorder.catalog_reads, recorder.max_catalog_reads):
                result = {
                    "status": "budget_exceeded",
                    "message": f"catalog access budget exceeded: {recorder.max_catalog_reads}",
                }
            else:
                recorder.catalog_reads += 1
                result = _list_catalog_path(path, market=market, max_depth=max_depth, limit=limit)
            trace_id = recorder.add_tool_call(
                "list_futu_screener_catalog",
                args,
                result,
            )
            recorder.progress_log(
                "tool_end",
                tool="list_futu_screener_catalog",
                trace_id=trace_id,
                elapsed_s=round(time.monotonic() - started, 3),
                status=result.get("status"),
                entry_count=len(result.get("entries") or []),
            )
            return {"trace_id": trace_id, **result}
        except Exception as exc:
            recorder.progress_log(
                "tool_error",
                tool="list_futu_screener_catalog",
                elapsed_s=round(time.monotonic() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    @agent.tool_plain(
        name="read_futu_screener_catalog",
        description=(
            "Read one exact Markdown file from the local Futu screener catalog. "
            "Use a file path returned by list_futu_screener_catalog. Args: path, market, max_chars."
        ),
        retries=1,
        timeout=20,
    )
    def read_futu_screener_catalog(
        path: str,
        market: str = "US",
        max_chars: int = 9000,
    ) -> dict[str, Any]:
        started = time.monotonic()
        args = {"path": path, "market": market, "max_chars": max_chars}
        recorder.progress_log("tool_start", tool="read_futu_screener_catalog", args=args)
        try:
            if _budget_exceeded(recorder.catalog_reads, recorder.max_catalog_reads):
                result = {
                    "status": "budget_exceeded",
                    "message": f"catalog access budget exceeded: {recorder.max_catalog_reads}",
                }
            else:
                recorder.catalog_reads += 1
                result = _read_catalog_file(path, market=market, max_chars=max_chars)
            trace_id = recorder.add_tool_call(
                "read_futu_screener_catalog",
                args,
                result,
            )
            recorder.progress_log(
                "tool_end",
                tool="read_futu_screener_catalog",
                trace_id=trace_id,
                elapsed_s=round(time.monotonic() - started, 3),
                status=result.get("status"),
                path=result.get("path"),
                chars=result.get("chars"),
                truncated=result.get("truncated"),
            )
            return {"trace_id": trace_id, **result}
        except Exception as exc:
            recorder.progress_log(
                "tool_error",
                tool="read_futu_screener_catalog",
                elapsed_s=round(time.monotonic() - started, 3),
                path=path,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise


def _register_futu_filter_tool(agent: Any, recorder: DiscoveryRecorder) -> None:
    @agent.tool_plain(
        name="run_futu_stock_filter",
        description=(
            "Run live Futu OpenQuoteContext.get_stock_filter. Validate fields against "
            "the cached screener catalog before calling Futu. Args: stock_filter_specs, "
            "market, plate_code, limit. stock_filter_specs is a list of dicts combined "
            "within one probe. Each dict should include type (simple|financial|accumulate|"
            "pattern|custom_indicator) and stock_field. Use filter_min/filter_max plus "
            "is_no_filter=false for actual threshold filtering; use is_no_filter=true only "
            "for ranking with sort. financial filters often need quarter; accumulate filters "
            "often need days; pattern filters may need ktype and consecutive_period."
        ),
        retries=1,
        timeout=240,
    )
    def run_futu_stock_filter(
        stock_filter_specs: list[dict[str, Any]],
        market: str = "US",
        plate_code: str | None = None,
        limit: int = 80,
    ) -> dict[str, Any]:
        started = time.monotonic()
        normalized_specs = _normalize_stock_filter_specs(stock_filter_specs)
        args = {
            "stock_filter_specs": stock_filter_specs,
            "normalized_stock_filter_specs": normalized_specs,
            "market": market,
            "plate_code": plate_code,
            "limit": limit,
        }
        recorder.progress_log(
            "tool_start",
            tool="run_futu_stock_filter",
            market=market,
            plate_code=plate_code,
            limit=limit,
            filter_count=len(normalized_specs),
            normalized_stock_filter_specs=normalized_specs,
        )
        try:
            if _budget_exceeded(recorder.filter_runs, recorder.max_filter_runs):
                result = {
                    "status": "budget_exceeded",
                    "message": f"filter run budget exceeded: {recorder.max_filter_runs}",
                    "sample_symbols": [],
                }
            else:
                recorder.filter_runs += 1
                result = _run_futu_filter(
                    stock_filter_specs=normalized_specs,
                    market=market,
                    plate_code=plate_code,
                    limit=limit,
                    recorder=recorder,
                )
            trace_id = recorder.add_tool_call(
                "run_futu_stock_filter",
                args,
                result,
            )
            recorder.progress_log(
                "tool_end",
                tool="run_futu_stock_filter",
                trace_id=trace_id,
                elapsed_s=round(time.monotonic() - started, 3),
                status=result.get("status"),
                diagnosis=result.get("diagnosis"),
                returned_count=result.get("returned_count"),
                all_count=result.get("all_count"),
                sample_symbols=result.get("sample_symbols"),
            )
            return {"trace_id": trace_id, **result}
        except Exception as exc:
            recorder.progress_log(
                "tool_error",
                tool="run_futu_stock_filter",
                elapsed_s=round(time.monotonic() - started, 3),
                market=market,
                plate_code=plate_code,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise


def _theme_discovery_plan_from_package(
    package: DiscoveryPackage,
    *,
    theme: str,
    market: str,
    theme_description: str,
    required_symbols: list[str],
    pydantic_ai: dict[str, Any],
) -> ThemeDiscoveryPlan:
    candidates_by_symbol: dict[str, DiscoveryCandidate] = {}
    for candidate in package.candidates:
        symbol = normalize_futu_symbol(candidate.symbol, market)
        if not symbol:
            continue
        candidates_by_symbol.setdefault(symbol, candidate)

    layer_candidates: dict[str, list[DiscoveryCandidate]] = {}
    for candidate in candidates_by_symbol.values():
        layer_candidates.setdefault(candidate.layer_key, []).append(candidate)

    domains: list[ThemeDomain] = []
    coverage: list[ThemeCoverageRequirement] = []
    for layer in package.layers:
        layer_symbols = _dedupe(
            [
                *[normalize_futu_symbol(item, market) for item in layer.candidate_symbols],
                *[
                    normalize_futu_symbol(candidate.symbol, market)
                    for candidate in layer_candidates.get(layer.key, [])
                ],
            ]
        )
        candidates = [
            ThemeDomainCandidate(
                symbol=symbol,
                role=(candidates_by_symbol.get(symbol).role if candidates_by_symbol.get(symbol) else ""),
                rationale=(candidates_by_symbol.get(symbol).rationale if candidates_by_symbol.get(symbol) else ""),
                priority=(candidates_by_symbol.get(symbol).priority if candidates_by_symbol.get(symbol) else "watchlist"),
            )
            for symbol in layer_symbols
            if symbol
        ]
        domains.append(
            ThemeDomain(
                key=layer.key,
                name=layer.name,
                thesis=layer.thesis,
                importance=layer.importance,
                subdomains=[
                    ThemeSubdomain(
                        key=layer.key,
                        name=layer.name,
                        thesis=layer.thesis,
                        importance=_subdomain_importance(layer.importance),
                        candidates=candidates,
                    )
                ],
            )
        )
        must_consider = [
            symbol
            for symbol in layer_symbols
            if (candidates_by_symbol.get(symbol) and candidates_by_symbol[symbol].priority == "must_consider")
        ]
        coverage.append(
            ThemeCoverageRequirement(
                key=layer.key,
                name=layer.name,
                thesis=layer.thesis,
                priority=_coverage_priority(layer.importance),
                min_candidates=min(2, len(layer_symbols)) if layer_symbols else 0,
                candidate_symbols=layer_symbols,
                must_consider_symbols=must_consider,
                evidence_needed=package.next_enrichment_needed,
            )
        )

    seed_symbols = _seed_symbols_from_package(
        package=package,
        market=market,
        required_symbols=required_symbols,
        candidates_by_symbol=candidates_by_symbol,
    )
    search_queries = _research_search_queries(package, pydantic_ai)
    research_trace = _research_trace(package, pydantic_ai)
    plan = ThemeDiscoveryPlan(
        theme=theme,
        market=market,
        theme_description=theme_description,
        initial_thesis=package.initial_thesis,
        domain_tree=domains,
        coverage_requirements=coverage,
        seed_symbols=seed_symbols,
        plate_keywords=_plate_keywords(package.filter_plans_by_layer),
        benchmark_symbols=[],
        research_trace=research_trace,
        search_queries=search_queries,
        filter_plans_by_layer=package.filter_plans_by_layer,
        executed_filter_probes=package.executed_filter_probes,
        layer_filter_audits=package.layer_filter_audits,
        omissions_to_investigate=package.omissions_to_investigate,
        next_enrichment_needed=package.next_enrichment_needed,
        data_asof={
            "generated_at": utc_now(),
            "futu_screener_catalog": _catalog_generated_at(market),
        },
        discovery_notes=[
            "Discovery v1 stops after theme-layer candidate discovery.",
            "Futu probes are discovery evidence, not final recommendation evidence.",
        ],
        warnings=_discovery_warnings(package),
        pydantic_ai=pydantic_ai,
    )
    return plan


def _seed_symbols_from_package(
    *,
    package: DiscoveryPackage,
    market: str,
    required_symbols: list[str],
    candidates_by_symbol: dict[str, DiscoveryCandidate],
) -> list[ThemeDiscoverySeed]:
    seeds: list[ThemeDiscoverySeed] = []
    seen: set[str] = set()
    for required in required_symbols:
        symbol = normalize_futu_symbol(required, market)
        if not symbol or symbol in seen:
            continue
        candidate = candidates_by_symbol.get(symbol)
        seeds.append(
            ThemeDiscoverySeed(
                symbol=symbol,
                market=market,
                role=candidate.role if candidate else "user-required base symbol",
                rationale=candidate.rationale if candidate else "User required this symbol as a discovery constraint.",
                subthemes=[candidate.layer_key] if candidate else ["user_required"],
                value_chain_stage=candidate.subdomain if candidate else "user_required",
                exposure_type="user_required" if candidate is None else "",
                exposure_purity="unknown" if candidate is None else "",
                source_ids=candidate.source_trace_ids if candidate else [],
                confidence="medium",
                freshness="unknown",
            )
        )
        seen.add(symbol)

    for symbol, candidate in candidates_by_symbol.items():
        if symbol in seen:
            continue
        seeds.append(
            ThemeDiscoverySeed(
                symbol=symbol,
                market=market,
                role=candidate.role,
                rationale=candidate.rationale,
                subthemes=[candidate.layer_key],
                value_chain_stage=candidate.subdomain or candidate.layer_key,
                source_ids=candidate.source_trace_ids,
                confidence="medium",
                freshness="unknown",
            )
        )
        seen.add(symbol)
    return seeds


def _list_catalog_path(path: str, *, market: str, max_depth: int, limit: int) -> dict[str, Any]:
    selected_market = _normalize_market(market)
    if not str(path or "").strip():
        rels = [
            "index.md",
            f"markets/{selected_market}/index.md",
            f"markets/{selected_market}/filters",
            f"markets/{selected_market}/filters/stock_fields",
            f"markets/{selected_market}/plates",
        ]
        entries = [_catalog_entry(rel) for rel in rels if _catalog_target_exists(rel)]
        return {
            "status": "ok",
            "operation": "list",
            "path": "",
            "market": selected_market,
            "entries": entries[: max(1, min(200, int(limit or 80)))],
            "hint": f"Read markets/{selected_market}/index.md, then list/read files under markets/{selected_market}/filters and markets/{selected_market}/plates.",
        }

    target = _safe_catalog_dir_path(path)
    root = _CATALOG_ROOT.resolve()
    base_depth = len(target.relative_to(root).parts)
    max_allowed_depth = max(0, min(4, int(max_depth or 1)))
    entries: list[dict[str, Any]] = []
    for child in sorted(target.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if child == target:
            continue
        rel_parts = child.relative_to(root).parts
        depth = len(rel_parts) - base_depth
        if depth <= 0 or depth > max_allowed_depth:
            continue
        if child.is_dir() or child.suffix == ".md":
            entries.append(_catalog_entry(child.relative_to(root).as_posix()))
        if len(entries) >= max(1, min(200, int(limit or 80))):
            break
    return {
        "status": "ok",
        "operation": "list",
        "path": target.relative_to(root).as_posix(),
        "market": selected_market,
        "entries": entries,
    }


def _read_catalog_file(path: str, *, market: str, max_chars: int) -> dict[str, Any]:
    target = _safe_catalog_path(path)
    raw_text = target.read_text(encoding="utf-8", errors="ignore")
    max_len = max(500, min(30000, int(max_chars or 9000)))
    rel = target.relative_to(_CATALOG_ROOT).as_posix()
    return {
        "status": "ok",
        "operation": "read",
        "path": rel,
        "market": _normalize_market(market),
        "title": _doc_title(raw_text, target.name),
        "language": "mixed",
        "content": raw_text[:max_len],
        "truncated": len(raw_text) > max_len,
        "chars": len(raw_text),
        "note": "Use exact ticker symbols, StockField enum values, and plate_code values from this document.",
    }


def _run_futu_filter(
    *,
    stock_filter_specs: list[dict[str, Any]],
    market: str,
    plate_code: str | None,
    limit: int,
    recorder: DiscoveryRecorder | None = None,
) -> dict[str, Any]:
    from .adapters import (
        FutuOpenDConfig,
        MarketDataAdapter,
        _check_ret,
        _import_futu,
        _parse_quote_market,
        _safe_float,
        _safe_str,
    )

    selected_market = _normalize_market(market)
    stock_filter_specs = _normalize_stock_filter_specs(stock_filter_specs)
    started = time.monotonic()
    if recorder is not None:
        recorder.progress_log(
            "futu_filter_validate_start",
            market=selected_market,
            plate_code=plate_code,
            filter_count=len(stock_filter_specs),
        )
    errors = validate_stock_filter_specs_against_catalog(stock_filter_specs, market=selected_market)
    if errors:
        if recorder is not None:
            recorder.progress_log(
                "futu_filter_validate_error",
                elapsed_s=round(time.monotonic() - started, 3),
                errors=errors,
            )
        return {
            "status": "validation_error",
            "errors": errors,
            "sample_symbols": [],
            "diagnosis": "error",
        }
    if recorder is not None:
        recorder.progress_log("futu_filter_validate_end", elapsed_s=round(time.monotonic() - started, 3))

    futu = _import_futu()
    config = FutuOpenDConfig.from_env()
    adapter = MarketDataAdapter(config)
    if recorder is not None:
        recorder.progress_log("futu_check_opend_start", host=config.host, port=config.port)
    adapter._check_opend()  # noqa: SLF001 - plugin adapter boundary.
    if recorder is not None:
        recorder.progress_log("futu_check_opend_end", elapsed_s=round(time.monotonic() - started, 3))
        recorder.progress_log("futu_quote_context_open_start", host=config.host, port=config.port)
    quote_ctx = futu.OpenQuoteContext(host=config.host, port=config.port)
    if recorder is not None:
        recorder.progress_log("futu_quote_context_open_end", elapsed_s=round(time.monotonic() - started, 3))
    try:
        filters = _build_futu_stock_filters(futu, stock_filter_specs)
        requested_limit = max(1, min(200, int(limit or 80)))
        if recorder is not None:
            recorder.progress_log(
                "futu_get_stock_filter_start",
                market=selected_market,
                plate_code=plate_code,
                requested_limit=requested_limit,
                filter_count=len(filters),
            )
        ret, data = adapter._quote_call(  # noqa: SLF001
            quote_ctx.get_stock_filter,
            _parse_quote_market(futu, selected_market),
            filters,
            plate_code=plate_code,
            begin=0,
            num=requested_limit,
        )
        if recorder is not None:
            recorder.progress_log(
                "futu_get_stock_filter_returned",
                elapsed_s=round(time.monotonic() - started, 3),
                ret=ret,
            )
        _check_ret(futu, ret, data, "get_stock_filter")
        _last_page, all_count, stock_list = data
        candidates = []
        for item in list(stock_list)[:requested_limit]:
            raw_symbol = _safe_str(getattr(item, "stock_code", ""))
            symbol = normalize_futu_symbol(raw_symbol, selected_market)
            if not symbol:
                continue
            candidates.append(
                {
                    "symbol": symbol,
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
        if recorder is not None:
            recorder.progress_log(
                "futu_filter_parse_end",
                elapsed_s=round(time.monotonic() - started, 3),
                all_count=int(all_count),
                returned_count=len(candidates),
                sample_symbols=[item["symbol"] for item in candidates[:20]],
            )
        return {
            "status": "ok",
            "market": selected_market,
            "plate_code": plate_code,
            "all_count": int(all_count),
            "returned_count": len(candidates),
            "sample_symbols": [item["symbol"] for item in candidates],
            "sample_candidates": candidates[:30],
            "diagnosis": diagnose_trial_result(
                len(candidates),
                int(all_count),
                result_limit=requested_limit,
            ),
        }
    except Exception as exc:
        if recorder is not None:
            recorder.progress_log(
                "futu_filter_error",
                elapsed_s=round(time.monotonic() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return {
            "status": "futu_error",
            "error": str(exc),
            "sample_symbols": [],
            "diagnosis": "error",
        }
    finally:
        if recorder is not None:
            recorder.progress_log("futu_quote_context_close_start", elapsed_s=round(time.monotonic() - started, 3))
        quote_ctx.close()
        if recorder is not None:
            recorder.progress_log("futu_quote_context_close_end", elapsed_s=round(time.monotonic() - started, 3))


def _normalize_stock_filter_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            normalized.append(spec)
            continue
        item = dict(spec)
        if "type" not in item and "filter_type" in item:
            item["type"] = item["filter_type"]
        if "stock_field" not in item and "field" in item:
            item["stock_field"] = item["field"]
        if "sort" not in item and "sort_dir" in item:
            item["sort"] = item["sort_dir"]
        for key in ("type", "stock_field", "stock_field1", "stock_field2", "sort", "quarter", "ktype"):
            if item.get(key) is not None:
                item[key] = str(item[key]).strip().upper()
        if item.get("type"):
            item["type"] = str(item["type"]).strip().lower()
        normalized.append(item)
    return normalized


def _event_stream_handler(recorder: DiscoveryRecorder):
    async def handler(_ctx: Any, events: Any) -> None:
        async for event in events:
            summary = _summarize_event(event)
            if summary is not None:
                recorder.add_event(summary)
                recorder.progress_log("agent_event", **summary)

    return handler


def _summarize_event(event: Any) -> dict[str, Any] | None:
    kind = getattr(event, "event_kind", type(event).__name__)
    part = getattr(event, "part", None)
    if kind in {"function_tool_call", "output_tool_call"} and part is not None:
        return {
            "event": kind,
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
            "args": _part_args(part),
        }
    if kind in {"function_tool_result", "output_tool_result"} and part is not None:
        return {
            "event": kind,
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
            "content_preview": _preview(getattr(part, "content", ""), 1400),
        }
    if kind == "builtin_tool_call":
        native_part = getattr(event, "part", None)
        return {
            "event": kind,
            "tool": getattr(native_part, "tool_name", ""),
            "call_id": getattr(native_part, "tool_call_id", ""),
            "args": _part_args(native_part),
        }
    if kind == "builtin_tool_result":
        native_result = getattr(event, "result", None)
        return {
            "event": kind,
            "tool": getattr(native_result, "tool_name", ""),
            "call_id": getattr(native_result, "tool_call_id", ""),
            "content_preview": _preview(getattr(native_result, "content", ""), 1400),
        }
    if kind in {"part_start", "part_end"} and part is not None:
        return {
            "event": kind,
            "index": getattr(event, "index", ""),
            "part_kind": getattr(part, "part_kind", ""),
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
        }
    if kind == "part_delta":
        if not _env_bool("IA_DISCOVERY_V1_PROGRESS_PART_DELTAS", False):
            return None
        delta = getattr(event, "delta", None)
        delta_kind = getattr(delta, "part_delta_kind", "")
        args_delta = getattr(delta, "args_delta", None)
        content_delta = getattr(delta, "content_delta", None)
        delta_text = args_delta if args_delta is not None else content_delta
        return {
            "event": kind,
            "index": getattr(event, "index", ""),
            "delta_kind": delta_kind,
            "tool_name_delta": getattr(delta, "tool_name_delta", ""),
            "call_id_delta": getattr(delta, "tool_call_id_delta", ""),
            "delta_len": len(str(delta_text or "")),
            "delta_preview": _preview(delta_text, _env_int("IA_DISCOVERY_V1_PROGRESS_DELTA_PREVIEW_CHARS", 240)),
        }
    if kind == "final_result":
        return {
            "event": kind,
            "tool": getattr(event, "tool_name", ""),
            "call_id": getattr(event, "tool_call_id", ""),
        }
    return None


def _safe_catalog_dir_path(path: str) -> Path:
    cleaned = str(path or "").strip().lstrip("/")
    target = (_CATALOG_ROOT / cleaned).resolve()
    root = _CATALOG_ROOT.resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"Path escapes catalog root: {path}")
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Catalog directory not found: {path}")
    return target


def _safe_catalog_path(path: str) -> Path:
    cleaned = str(path or "").strip().lstrip("/")
    target = (_CATALOG_ROOT / cleaned).resolve()
    root = _CATALOG_ROOT.resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"Path escapes catalog root: {path}")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    return target


def _catalog_target_exists(path: str) -> bool:
    return (_CATALOG_ROOT / path).exists()


def _catalog_entry(path: str) -> dict[str, Any]:
    target = _CATALOG_ROOT / path
    entry = {
        "path": path,
        "type": "directory" if target.is_dir() else "file",
        "name": target.name,
    }
    if target.is_file():
        text = target.read_text(encoding="utf-8", errors="ignore")
        entry["title"] = _doc_title(text, target.name)
        entry["chars"] = len(text)
    return entry


def _doc_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def _catalog_generated_at(market: str) -> str:
    catalog = load_futu_screener_catalog_snapshot(market=_normalize_market(market))
    return str(catalog.get("generated_at") or "")


def _plate_keywords(plans: list[DiscoveryFilterPlan]) -> list[str]:
    keywords: list[str] = []
    for plan in plans:
        keywords.extend(plan.plate_search_terms)
    return _dedupe(keywords)


def _research_search_queries(package: DiscoveryPackage, pydantic_ai: dict[str, Any]) -> list[str]:
    queries = list(package.search_queries)
    for event in _pydantic_events(pydantic_ai):
        if event.get("event") != "builtin_tool_call" or event.get("tool") != "web_search":
            continue
        args = _decode_jsonish(event.get("args"))
        if isinstance(args, dict):
            query = str(args.get("query") or "").strip()
            if query:
                queries.append(query)
            for item in args.get("queries") or []:
                query = str(item or "").strip()
                if query:
                    queries.append(query)
    return _dedupe(queries)


def _research_trace(package: DiscoveryPackage, pydantic_ai: dict[str, Any]) -> list[ResearchSource]:
    sources = list(package.research_trace)
    sources.extend(_research_sources_from_web_events(_pydantic_events(pydantic_ai)))
    sources.extend(_research_sources_from_web_probes(package.executed_filter_probes))
    return _dedupe_research_sources(sources)


def _pydantic_events(pydantic_ai: dict[str, Any]) -> list[dict[str, Any]]:
    events = pydantic_ai.get("events") if isinstance(pydantic_ai, dict) else None
    return [event for event in events or [] if isinstance(event, dict)]


def _research_sources_from_web_events(events: list[dict[str, Any]]) -> list[ResearchSource]:
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("tool") not in {"web_search", "web_fetch"}:
            continue
        call_id = str(event.get("call_id") or "").strip()
        if not call_id:
            continue
        if event.get("event") == "builtin_tool_call":
            calls[call_id] = event
        elif event.get("event") == "builtin_tool_result":
            results[call_id] = event

    sources: list[ResearchSource] = []
    for call_id, call in calls.items():
        tool = str(call.get("tool") or "")
        args = _decode_jsonish(call.get("args"))
        result = _decode_jsonish(results.get(call_id, {}).get("content_preview"))
        retrieved_at = str(call.get("time") or results.get(call_id, {}).get("time") or utc_now())
        if tool == "web_search":
            query_values: list[str] = []
            if isinstance(args, dict):
                if args.get("query"):
                    query_values.append(str(args["query"]))
                query_values.extend(str(item) for item in args.get("queries") or [] if item)
            query_values = _dedupe(query_values)
            extracted = _search_result_sources(call_id, result, retrieved_at)
            if extracted:
                sources.extend(extracted)
                continue
            for index, query in enumerate(query_values or ["web search"], start=1):
                sources.append(
                    ResearchSource(
                        source_id=f"{call_id}:query_{index}",
                        title=f"web_search: {query}",
                        publisher="provider-native web_search",
                        retrieved_at=retrieved_at,
                        source_type="web",
                        summary="Provider web search query executed during theme discovery; detailed result URLs were not exposed by the provider event stream.",
                    )
                )
        elif tool == "web_fetch":
            url = str(args.get("url") or "") if isinstance(args, dict) else ""
            sources.append(
                ResearchSource(
                    source_id=call_id,
                    title=f"web_fetch: {url or call_id}",
                    url=url,
                    publisher="provider/local web_fetch",
                    retrieved_at=retrieved_at,
                    source_type="web",
                    summary=_research_result_summary(result),
                )
            )
    return sources


def _search_result_sources(call_id: str, result: Any, retrieved_at: str) -> list[ResearchSource]:
    if not isinstance(result, dict):
        return []
    results = result.get("results") or result.get("items") or result.get("sources") or []
    if not isinstance(results, list):
        return []
    sources: list[ResearchSource] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or item.get("url") or f"web result {index}")
        url = str(item.get("url") or item.get("link") or "")
        summary = str(item.get("snippet") or item.get("summary") or item.get("content") or "")
        publisher = str(item.get("publisher") or item.get("source") or "")
        sources.append(
            ResearchSource(
                source_id=f"{call_id}:result_{index}",
                title=title,
                url=url,
                publisher=publisher,
                retrieved_at=retrieved_at,
                source_type="web",
                summary=summary,
            )
        )
    return sources


def _research_sources_from_web_probes(probes: list[ExecutedDiscoveryProbe]) -> list[ResearchSource]:
    sources: list[ResearchSource] = []
    for index, probe in enumerate(probes, start=1):
        if probe.probe_type != "web":
            continue
        source_id = str(probe.trace_id or f"web_probe_{index}").strip()
        sources.append(
            ResearchSource(
                source_id=source_id,
                title=f"web research probe: {probe.layer_key}",
                retrieved_at=utc_now(),
                source_type="web",
                summary=probe.rationale,
                coverage_keys=[probe.layer_key] if probe.layer_key else [],
            )
        )
    return sources


def _dedupe_research_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
    seen: set[str] = set()
    result: list[ResearchSource] = []
    for source in sources:
        key = source.source_id or source.url or source.title
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _research_result_summary(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("summary", "snippet", "content", "text", "status"):
            if result.get(key):
                return str(result[key])[:1000]
        return _preview(result, 1000)
    return str(result or "")[:1000]


def _decode_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for _ in range(2):
        try:
            parsed = json.loads(text)
        except Exception:
            return text
        if not isinstance(parsed, str):
            return parsed
        text = parsed.strip()
    return text


def _discovery_warnings(package: DiscoveryPackage) -> list[str]:
    warnings = list(package.warnings)
    if not package.filter_plans_by_layer:
        warnings.append("Discovery v1 returned no filter plans; downstream review should reject weak coverage.")
    if not package.executed_filter_probes:
        warnings.append("Discovery v1 returned no executed Futu filter probes.")
    warnings.extend(_broad_plate_refinement_warnings(package))
    if package.omissions_to_investigate:
        warnings.append("Discovery has omissions_to_investigate that need later enrichment/review.")
    if package.next_enrichment_needed:
        warnings.append("SEC, market-data, fundamentals, technicals, and/or options enrichment have not run yet.")
    return _dedupe([str(item) for item in warnings if str(item or "").strip()])


def _broad_plate_refinement_warnings(package: DiscoveryPackage) -> list[str]:
    layer_importance = {layer.key: layer.importance for layer in package.layers}
    refinement_keys = {
        (probe.layer_key, probe.plate_code)
        for probe in package.executed_filter_probes
        if probe.probe_type == "subdomain_plate_refinement" and probe.plate_code
    }
    warnings: list[str] = []
    for probe in package.executed_filter_probes:
        if probe.probe_type != "subdomain_plate" or not probe.plate_code:
            continue
        if layer_importance.get(probe.layer_key) not in {"core", "important"}:
            continue
        if (probe.result_count or 0) < _BROAD_PLATE_REFINEMENT_MIN_RESULTS:
            continue
        if (probe.layer_key, probe.plate_code) in refinement_keys:
            continue
        warnings.append(
            "Broad important plate probe was not followed by a same-plate refinement probe: "
            f"{probe.layer_key} {probe.plate_code} returned {probe.result_count} symbols."
        )
    return warnings


def _coverage_priority(importance: str) -> Literal["required", "important", "optional"]:
    if importance == "core":
        return "required"
    if importance == "optional":
        return "optional"
    return "important"


def _subdomain_importance(importance: str) -> Literal["high", "medium", "low"]:
    if importance == "core":
        return "high"
    if importance == "optional":
        return "low"
    return "medium"


def _budget_exceeded(current_count: int, max_count: int) -> bool:
    return max_count > 0 and current_count >= max_count


def _normalize_symbols(symbols: list[str], market: str) -> list[str]:
    return _dedupe(
        [
            normalize_futu_symbol(symbol, market)
            for symbol in symbols
            if normalize_futu_symbol(symbol, market)
        ]
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _part_args(part: Any) -> str:
    if hasattr(part, "args_as_json_str"):
        try:
            return part.args_as_json_str()
        except Exception:
            return repr(getattr(part, "args", ""))
    return repr(getattr(part, "args", ""))


def _preview(value: Any, max_chars: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default
