"""Budgeted web-search-only theme discovery for investment themes."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .pydantic_runtime import create_pydantic_agent, usage_metadata
from .pydantic_progress import PydanticAgentRecorder, run_pydantic_agent_sync
from .schemas import (
    DiscoveryOmission,
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

LOGGER = logging.getLogger(__name__)
_SYMBOL_RE = re.compile(r"^(?:[A-Z]{1,5}\.)?[A-Z0-9][A-Z0-9.\-]{0,24}$")


_WEBSEARCH_DISCOVERY_INSTRUCTIONS = """
You are a web-search-only investment theme discovery agent.

Your job is not to recommend a portfolio. Your job is to produce a compact
candidate discovery artifact that later Futu market-data, filings, and triage
agents can validate.

Available tool:
- web_search_budgeted(query, layer_key, rationale, max_results)

Rules:
- Use web_search_budgeted for current public information. Do not rely only on
  model memory.
- The search budget is hard. Plan before searching, then stop when the
  important investable layers have enough evidence.
- First form a concise thesis for the current market battleground.
- Split the theme into investable layers and subdomains before listing stocks.
- Derive layers from the theme. Do not use a fixed template.
- Before spending searches, write a mental search plan: broad market map,
  bottleneck branches, recent public-market changes, and highest omission-risk
  branches. Use the limited searches against the branches that would most
  change the candidate set.
- For infrastructure-heavy themes, do not treat working memory, persistent
  storage, network interconnect, power/cooling, asset ownership, equipment
  suppliers, and operators as interchangeable. Decide whether each is distinct
  enough to deserve its own layer or omission note.
- For each layer, search for business models and recent public-market changes:
  spin-offs, new listings, ticker changes, renamed issuers, mergers, and
  parent/subsidiary separations.
- If the search budget prevents a plausible distinct branch from being
  searched, record that branch in warnings or omissions_to_investigate rather
  than silently collapsing it into a broad proxy.
- Prefer US-listed public securities when market is US. Do not put private
  companies, unavailable tickers, or nonstandard pseudo symbols in candidates.
  Put them in omissions_to_investigate if they matter.
- Include ETFs only when they are useful discovery anchors.
- Keep candidates compact: usually 4-10 names per important layer. Use
  must_consider only for names a later architect must include or explicitly
  explain away.
- Required symbols are user constraints and must be preserved, but they are not
  proof of quality.
- Record source_trace_ids from web_search_budgeted results on candidates when
  a search influenced the candidate.
- Do not output weights, trade instructions, price targets, or final
  recommendations.
""".strip()


_TWO_STAGE_SEARCH_PLAN_INSTRUCTIONS = """
You are an investment-theme web research planner.

Your job is not to recommend a portfolio and not to list final candidates.
Your job is to decide what should be searched before candidate discovery.

Output a compact WebSearchDiscoverySearchPlan.

Planning rules:
- Start from the user's theme and market. Derive investable layers from the
  theme; do not use a fixed portfolio template.
- Think like an analyst building a candidate universe: identify the economic
  mechanism, bottlenecks, upstream/downstream branches, recent public-market
  changes, and where omission risk is highest.
- For each core or important layer, create at least one concrete web search
  task. Optional layers can be left without a task only if you explain why in
  warnings.
- Prefer search tasks that can reveal public listed candidates, ticker changes,
  spin-offs, IPOs, renamed issuers, parent/subsidiary splits, and non-obvious
  public pure plays.
- For infrastructure-heavy themes, explicitly decide whether working memory,
  persistent storage, network interconnect, power/cooling, asset ownership,
  equipment suppliers, and operators are separate economic branches. If a
  branch is plausible but not searched, put it in warnings.
- For each task, write a query that could be executed as-is. Avoid vague
  single-word searches.
- Keep the plan small enough for the search budget. Mark only the most
  omission-sensitive tasks as required.
- Do not output weights, trade instructions, price targets, or final
  recommendations.
""".strip()


_TWO_STAGE_SYNTHESIS_INSTRUCTIONS = """
You are a candidate synthesis agent for an investment-theme discovery workflow.

You receive:
- the user's theme and required symbols
- a planned search artifact
- executed web search results with trace_ids

Use only that evidence plus the user-required symbols to produce a compact
WebSearchDiscoveryArtifact. Do not recommend a portfolio.

Rules:
- Preserve the planner's layer structure unless search evidence clearly proves
  a layer is empty or non-investable.
- Candidate source_trace_ids must reference executed search trace_ids whenever
  a search result influenced the candidate.
- Include required symbols even if search evidence is weak, and label them as
  user constraints rather than proof of quality.
- Prefer US-listed public securities for US market. Private companies,
  unavailable tickers, and vague entities belong in omissions_to_investigate.
- All ticker strings in candidates and candidate_symbols must be ASCII-only
  official tickers using A-Z, 0-9, dot, and hyphen. Do not use Unicode
  lookalikes or localized letters such as Turkish dotted İ. For example,
  Super Micro Computer must be US.SMCI, not US.SMCİ. If the exact ticker is
  uncertain, put the entity in omissions_to_investigate instead of candidates.
- If a planned branch produced weak or ambiguous evidence, record that in
  warnings or omissions_to_investigate instead of silently collapsing it into a
  broad proxy.
- Keep candidates compact: usually 4-10 names per important layer.
- Do not output weights, trade instructions, price targets, or final
  recommendations.
""".strip()


_NATIVE_WEBSEARCH_DISCOVERY_INSTRUCTIONS = """
You are a native-web-search investment theme discovery agent.

Your job is not to recommend a portfolio. Your job is to produce a compact
candidate discovery artifact that later Futu market data, filings, and triage
agents can validate.

Use native web search for current public information. Keep the final output
structured, but do not prematurely compress the candidate universe.

Rules:
- Derive investable layers from the user's theme. Do not use a fixed portfolio
  template.
- Build a broad discovery-stage candidate universe, not a final allocation.
  Favor recall over precision here; downstream market-data, filings, and triage
  agents will reduce the universe later.
- For an infrastructure-heavy theme, explicitly separate economic bottlenecks
  where relevant: compute, memory, persistent storage, network/optical
  interconnect, semicap/equipment/packaging, power/cooling/electrical, cloud
  operators, and monetizable software/applications.
- Search enough to avoid obvious omission-prone public securities. Stop once
  the main omission-risk branches have representative candidates and omissions.
- Before finalizing, run a public-market-change omission sweep for high
  omission-risk layers: recent spin-offs, split-offs, carve-outs, renamed
  issuers, ticker changes, IPOs/new listings, de-SPACs, and newly public
  pure-play companies. Search with both business terms and corporate-action
  terms. Do not assume a parent company, broad ETF, or older ticker still
  captures a spun-off or newly listed pure-play exposure; include the public
  ticker or record a clear omission.
- Do not satisfy the public-market-change sweep with one global query only.
  For each high omission-risk layer, first generate a compact layer vocabulary:
  core technologies, product categories, supply-chain bottlenecks, customer or
  use-case terms, adjacent suppliers, and enabling infrastructure. Then combine
  that generated layer vocabulary with corporate-action terms. Example pattern:
  "<generated layer vocabulary> spin-off ticker change newly public IPO
  carve-out pure-play". Apply this per-layer pattern to any theme without
  hardcoding theme-specific keywords or tickers.
- Preserve user-required symbols as constraints. They are not proof of quality.
- Prefer US-listed securities when market is US. Put private companies,
  uncertain tickers, and non-US listings in omissions_to_investigate.
- All ticker strings in candidates and candidate_symbols must be ASCII-only
  official tickers using A-Z, 0-9, dot, and hyphen. Do not use Unicode
  lookalikes or localized letters such as Turkish dotted İ. For example,
  Super Micro Computer must be US.SMCI, not US.SMCİ. If the exact ticker is
  uncertain, put the entity in omissions_to_investigate instead of candidates.
- Do not compress away plausible public candidates just because another ticker
  already represents the same broad layer. If a layer has multiple subdomains,
  keep representative public securities for each subdomain or record explicit
  omissions. Roughly 8-20 candidates per important layer is acceptable when
  evidence supports them; fewer is okay only when the public universe is thin or
  evidence is weak, and that gap must be explained.
- Put source_trace_ids as ["native_websearch"] when native web search evidence
  influenced a candidate and provider source ids are not directly available.
- Do not output weights, trade instructions, price targets, or final
  recommendations.
""".strip()


class WebSearchDiscoverySearchTask(BaseModel):
    task_id: str
    layer_key: str
    layer_name: str = ""
    branch: str = ""
    query: str
    rationale: str = ""
    priority: Literal["required", "important", "optional"] = "important"
    omission_risk: Literal["high", "medium", "low"] = "medium"
    expected_candidate_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_task(self) -> "WebSearchDiscoverySearchTask":
        self.task_id = _key(self.task_id or self.query)
        self.layer_key = _key(self.layer_key or self.layer_name)
        self.query = " ".join(str(self.query or "").split())
        if not self.query:
            raise ValueError("Search task query must not be empty.")
        return self


class WebSearchDiscoverySearchPlan(BaseModel):
    theme: str
    market: str = "US"
    initial_thesis: str
    layers: list[WebSearchDiscoveryLayer] = Field(default_factory=list)
    search_tasks: list[WebSearchDiscoverySearchTask] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> "WebSearchDiscoverySearchPlan":
        for layer in self.layers:
            layer.key = _key(layer.key or layer.name)
        layer_keys = {layer.key for layer in self.layers}
        if not layer_keys:
            raise ValueError("Websearch discovery search plan returned no layers.")
        if not self.search_tasks:
            raise ValueError("Websearch discovery search plan returned no search tasks.")
        missing = [task.layer_key for task in self.search_tasks if task.layer_key not in layer_keys]
        if missing:
            raise ValueError(f"Search tasks reference missing layer_key: {_dedupe(missing)}")
        if not any(task.priority in {"required", "important"} for task in self.search_tasks):
            raise ValueError("Search plan must contain at least one required or important task.")
        return self


class WebSearchDiscoveryCandidate(BaseModel):
    symbol: str
    name: str = ""
    layer_key: str
    subdomain: str = ""
    role: str = ""
    rationale: str = ""
    priority: Literal["must_consider", "strong_candidate", "watchlist"] = "strong_candidate"
    source_trace_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        symbol = str(value or "").strip().upper()
        if not symbol:
            raise ValueError("Candidate symbol must not be empty")
        if not symbol.isascii() or not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"Invalid candidate symbol: {value!r}")
        if symbol.startswith("PRIVATE.") or symbol.startswith("UNLISTED."):
            raise ValueError(f"Unsupported non-public candidate symbol: {value!r}")
        return symbol


class WebSearchDiscoveryLayer(BaseModel):
    key: str
    name: str
    thesis: str = ""
    importance: Literal["core", "important", "optional"] = "important"
    candidate_limit_reason: str = ""
    candidate_symbols: list[str] = Field(default_factory=list)

    @field_validator("candidate_symbols")
    @classmethod
    def validate_candidate_symbols(cls, value: list[str]) -> list[str]:
        symbols: list[str] = []
        for item in value or []:
            symbol = str(item or "").strip().upper()
            if not symbol:
                continue
            if not symbol.isascii() or not _SYMBOL_RE.fullmatch(symbol):
                raise ValueError(f"Invalid layer candidate symbol: {item!r}")
            symbols.append(symbol)
        return _dedupe(symbols)


class WebSearchDiscoveryOmission(BaseModel):
    symbol_or_entity: str
    layer_key: str = ""
    reason: str = ""
    source_trace_ids: list[str] = Field(default_factory=list)


class WebSearchDiscoveryArtifact(BaseModel):
    theme: str
    market: str = "US"
    initial_thesis: str
    layers: list[WebSearchDiscoveryLayer] = Field(default_factory=list)
    candidates: list[WebSearchDiscoveryCandidate] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    omissions_to_investigate: list[WebSearchDiscoveryOmission] = Field(default_factory=list)
    next_enrichment_needed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> "WebSearchDiscoveryArtifact":
        layer_keys = {layer.key for layer in self.layers}
        if not layer_keys:
            raise ValueError("Websearch discovery returned no layers.")
        if not self.candidates:
            raise ValueError("Websearch discovery returned no candidates.")
        missing = [candidate.symbol for candidate in self.candidates if candidate.layer_key not in layer_keys]
        if missing:
            raise ValueError(f"Candidates reference missing layer_key: {missing}")
        return self


@dataclass
class WebSearchDiscoveryRecorder:
    max_searches: int
    max_results: int
    progress: bool = False
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    successful_searches: int = 0

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append({"time": utc_now(), **event})

    def progress_log(self, stage: str, **fields: Any) -> None:
        if not self.progress:
            return
        record = {"time": utc_now(), "stage": stage, **fields}
        message = f"IA_WEBSEARCH_DISCOVERY_PROGRESS {_preview(record, _env_int('IA_WEBSEARCH_DISCOVERY_PROGRESS_MAX_CHARS', 2400))}"
        LOGGER.info(message)
        print(message, file=sys.stderr, flush=True)

    def add_search_call(self, args: dict[str, Any], result: dict[str, Any]) -> str:
        trace_id = f"web_{len(self.search_calls) + 1:03d}"
        status = str(result.get("status") or "")
        if status == "ok":
            self.successful_searches += 1
        self.search_calls.append(
            {
                "trace_id": trace_id,
                "time": utc_now(),
                "tool": "web_search_budgeted",
                "args": args,
                "result": result,
            }
        )
        return trace_id


def build_websearch_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
    max_searches: int | None = None,
    max_results: int | None = None,
) -> ThemeDiscoveryPlan:
    """Run budgeted web-search-only discovery and convert it to ThemeDiscoveryPlan."""

    started = time.monotonic()
    selected_market = _normalize_market(market)
    normalized_required = _normalize_symbols(required_symbols or [], selected_market)
    recorder = WebSearchDiscoveryRecorder(
        max_searches=max(1, max_searches or _env_int("IA_WEBSEARCH_DISCOVERY_MAX_SEARCHES", 8)),
        max_results=max(1, max_results or _env_int("IA_WEBSEARCH_DISCOVERY_MAX_RESULTS", 5)),
        progress=_env_bool("IA_WEBSEARCH_DISCOVERY_PROGRESS", False),
    )
    recorder.progress_log(
        "build_start",
        theme=theme,
        market=selected_market,
        required_symbols=normalized_required,
        max_searches=recorder.max_searches,
        max_results=recorder.max_results,
    )
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=WebSearchDiscoveryArtifact,
        instructions=_WEBSEARCH_DISCOVERY_INSTRUCTIONS,
        agent_kind="websearch_theme_discovery_agent",
        output_retries=1,
        enable_web_search=False,
        enable_web_fetch=False,
        agent_skill_names=None,
    )
    _register_web_search_tool(agent, recorder)
    payload = {
        "task": "budgeted_websearch_theme_discovery",
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "search_budget": {
            "max_searches": recorder.max_searches,
            "max_results_per_search": recorder.max_results,
            "hard_limit": True,
        },
        "output_contract": (
            "Return WebSearchDiscoveryArtifact only. Keep it compact. "
            "No portfolio weights, no orders, no trade plan."
        ),
    }
    result = _run_agent_sync(agent, payload, recorder)
    artifact = result.output
    _validate_websearch_artifact(artifact, recorder, selected_market)
    elapsed_s = round(time.monotonic() - started, 3)
    pydantic_ai = {
        **runtime,
        "usage": usage_metadata(result),
        "tool_calls": recorder.search_calls,
        "event_count": len(recorder.events),
        "events": recorder.events,
        "input": payload,
        "search_budget": {
            "max_searches": recorder.max_searches,
            "max_results": recorder.max_results,
            "successful_searches": recorder.successful_searches,
            "tool_call_count": len(recorder.search_calls),
        },
        "elapsed_s": elapsed_s,
    }
    plan = _theme_discovery_plan_from_websearch_artifact(
        artifact,
        theme=theme,
        market=selected_market,
        theme_description=theme_description,
        required_symbols=normalized_required,
        pydantic_ai=pydantic_ai,
        search_calls=recorder.search_calls,
    )
    recorder.progress_log(
        "build_end",
        elapsed_s=elapsed_s,
        seed_count=len(plan.seed_symbols),
        domain_count=len(plan.domain_tree),
        search_count=recorder.successful_searches,
    )
    return plan


def build_native_websearch_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
    max_searches: int | None = None,
    max_results: int | None = None,
) -> ThemeDiscoveryPlan:
    """Run compact discovery using provider-native web search."""

    started = time.monotonic()
    selected_market = _normalize_market(market)
    normalized_required = _normalize_symbols(required_symbols or [], selected_market)
    native_max_searches = max(1, max_searches or _env_int("IA_WEBSEARCH_DISCOVERY_MAX_SEARCHES", 6))
    context_size = os.getenv("IA_WEBSEARCH_DISCOVERY_NATIVE_CONTEXT_SIZE", "low").strip().lower() or "low"
    recorder = PydanticAgentRecorder(
        label="native_websearch_discovery",
        progress=_env_bool("IA_WEBSEARCH_DISCOVERY_PROGRESS", False),
        env_prefix="IA_WEBSEARCH_DISCOVERY_PROGRESS",
        logger=LOGGER,
        include_tool_call_deltas=True,
    )
    recorder.progress_log(
        "native_build_start",
        theme=theme,
        market=selected_market,
        required_symbols=normalized_required,
        max_searches=native_max_searches,
        search_context_size=context_size,
    )
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=WebSearchDiscoveryArtifact,
        instructions=_NATIVE_WEBSEARCH_DISCOVERY_INSTRUCTIONS,
        agent_kind="native_websearch_theme_discovery_agent",
        output_retries=1,
        enable_web_search=True,
        enable_web_fetch=False,
        agent_skill_names=None,
        research_overrides={
            "web_enabled": True,
            "web_search_mode": "native",
            "web_fetch_mode": "off",
            "max_searches": native_max_searches,
            "max_fetches": 0,
            "web_search_context_size": context_size,
        },
    )
    payload = {
        "task": "native_websearch_theme_discovery",
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "native_web_search_budget": {
            "max_uses": native_max_searches,
            "search_context_size": context_size,
            "note": (
                "This is a provider-native web search limit. A single native search "
                "call may contain multiple provider-side queries."
            ),
        },
        "output_contract": (
            "Return WebSearchDiscoveryArtifact only. Keep output compact. "
            "No portfolio weights, no orders, no trade plan."
        ),
    }
    trace_events = _env_bool("IA_WEBSEARCH_DISCOVERY_TRACE_EVENTS", True)
    result = run_pydantic_agent_sync(agent, payload, recorder if trace_events else None)
    artifact = result.output
    search_calls = _native_search_calls_from_events(recorder.events)
    if not search_calls and artifact.search_queries_used:
        search_calls = _native_search_calls_from_queries(artifact.search_queries_used)
    synthetic_recorder = WebSearchDiscoveryRecorder(max_searches=native_max_searches, max_results=max_results or 0)
    synthetic_recorder.search_calls = search_calls
    synthetic_recorder.successful_searches = len(search_calls)
    _validate_websearch_artifact(artifact, synthetic_recorder, selected_market)
    elapsed_s = round(time.monotonic() - started, 3)
    pydantic_ai = {
        **runtime,
        "discovery_mode": "pydantic_ai_native_websearch_discovery",
        "usage": usage_metadata(result),
        "tool_calls": search_calls,
        "event_count": len(recorder.events),
        "events": recorder.events,
        "input": payload,
        "search_budget": {
            "max_searches": native_max_searches,
            "search_context_size": context_size,
            "native_max_uses": native_max_searches,
            "native_search_call_count": len(search_calls),
            "event_count": len(recorder.events),
        },
        "elapsed_s": elapsed_s,
    }
    plan = _theme_discovery_plan_from_websearch_artifact(
        artifact,
        theme=theme,
        market=selected_market,
        theme_description=theme_description,
        required_symbols=normalized_required,
        pydantic_ai=pydantic_ai,
        search_calls=search_calls,
    )
    recorder.progress_log(
        "native_build_end",
        elapsed_s=elapsed_s,
        seed_count=len(plan.seed_symbols),
        domain_count=len(plan.domain_tree),
        native_search_call_count=len(search_calls),
    )
    return plan


def build_two_stage_websearch_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
    max_searches: int | None = None,
    max_results: int | None = None,
) -> ThemeDiscoveryPlan:
    """Plan searches first, execute them deterministically, then synthesize candidates."""

    started = time.monotonic()
    selected_market = _normalize_market(market)
    normalized_required = _normalize_symbols(required_symbols or [], selected_market)
    recorder = WebSearchDiscoveryRecorder(
        max_searches=max(1, max_searches or _env_int("IA_WEBSEARCH_DISCOVERY_MAX_SEARCHES", 8)),
        max_results=max(1, max_results or _env_int("IA_WEBSEARCH_DISCOVERY_MAX_RESULTS", 5)),
        progress=_env_bool("IA_WEBSEARCH_DISCOVERY_PROGRESS", False),
    )
    recorder.progress_log(
        "two_stage_build_start",
        theme=theme,
        market=selected_market,
        required_symbols=normalized_required,
        max_searches=recorder.max_searches,
        max_results=recorder.max_results,
    )
    planner_agent, _planner_model_config, planner_runtime = create_pydantic_agent(
        output_type=WebSearchDiscoverySearchPlan,
        instructions=_TWO_STAGE_SEARCH_PLAN_INSTRUCTIONS,
        agent_kind="websearch_theme_discovery_search_planner",
        output_retries=1,
        enable_web_search=False,
        enable_web_fetch=False,
        agent_skill_names=None,
    )
    plan_payload = {
        "task": "two_stage_websearch_theme_discovery_search_plan",
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "search_budget": {
            "max_searches": recorder.max_searches,
            "max_results_per_search": recorder.max_results,
            "hard_limit": True,
        },
        "output_contract": (
            "Return WebSearchDiscoverySearchPlan only. Do not output final candidates, "
            "weights, orders, or trade plans."
        ),
    }
    planner_result = _run_agent_sync(planner_agent, plan_payload, recorder)
    search_plan = planner_result.output
    _validate_search_plan(search_plan, selected_market, normalized_required)
    recorder.progress_log(
        "search_plan_ready",
        layer_count=len(search_plan.layers),
        task_count=len(search_plan.search_tasks),
        required_tasks=sum(1 for task in search_plan.search_tasks if task.priority == "required"),
    )
    skipped_tasks = _execute_search_plan(search_plan, recorder)
    synthesizer_agent, _synth_model_config, synth_runtime = create_pydantic_agent(
        output_type=WebSearchDiscoveryArtifact,
        instructions=_TWO_STAGE_SYNTHESIS_INSTRUCTIONS,
        agent_kind="websearch_theme_discovery_synthesizer",
        output_retries=1,
        enable_web_search=False,
        enable_web_fetch=False,
        agent_skill_names=None,
    )
    synthesis_payload = {
        "task": "two_stage_websearch_theme_discovery_synthesis",
        "theme": theme,
        "market": selected_market,
        "theme_description": theme_description,
        "required_symbols": normalized_required,
        "search_plan": search_plan.model_dump(mode="json"),
        "executed_searches": recorder.search_calls,
        "skipped_search_tasks": skipped_tasks,
        "output_contract": (
            "Return WebSearchDiscoveryArtifact only. Use source_trace_ids from executed_searches. "
            "No portfolio weights, no orders, no trade plan."
        ),
    }
    synth_result = _run_agent_sync(synthesizer_agent, synthesis_payload, recorder)
    artifact = synth_result.output
    _validate_websearch_artifact(artifact, recorder, selected_market)
    elapsed_s = round(time.monotonic() - started, 3)
    pydantic_ai = {
        "available": True,
        "mode": "pydantic_ai_two_stage_websearch_discovery",
        "discovery_mode": "pydantic_ai_two_stage_websearch_discovery",
        "planner": {
            **planner_runtime,
            "usage": usage_metadata(planner_result),
            "input": plan_payload,
            "output": search_plan.model_dump(mode="json"),
        },
        "synthesizer": {
            **synth_runtime,
            "usage": usage_metadata(synth_result),
            "input": synthesis_payload,
        },
        "tool_calls": recorder.search_calls,
        "skipped_search_tasks": skipped_tasks,
        "event_count": len(recorder.events),
        "events": recorder.events,
        "search_budget": {
            "max_searches": recorder.max_searches,
            "max_results": recorder.max_results,
            "successful_searches": recorder.successful_searches,
            "tool_call_count": len(recorder.search_calls),
            "planned_task_count": len(search_plan.search_tasks),
            "skipped_task_count": len(skipped_tasks),
        },
        "elapsed_s": elapsed_s,
    }
    plan = _theme_discovery_plan_from_websearch_artifact(
        artifact,
        theme=theme,
        market=selected_market,
        theme_description=theme_description,
        required_symbols=normalized_required,
        pydantic_ai=pydantic_ai,
        search_calls=recorder.search_calls,
    )
    recorder.progress_log(
        "two_stage_build_end",
        elapsed_s=elapsed_s,
        seed_count=len(plan.seed_symbols),
        domain_count=len(plan.domain_tree),
        search_count=recorder.successful_searches,
        skipped_task_count=len(skipped_tasks),
    )
    return plan


def _validate_search_plan(
    search_plan: WebSearchDiscoverySearchPlan,
    market: str,
    required_symbols: list[str] | None = None,
) -> None:
    search_plan.market = _normalize_market(search_plan.market or market)
    if search_plan.market != market:
        search_plan.warnings.append(
            f"Search plan market {search_plan.market} was normalized to requested market {market}."
        )
        search_plan.market = market
    seen_tasks: set[str] = set()
    for layer in search_plan.layers:
        layer.key = _key(layer.key or layer.name)
        layer.candidate_symbols = _normalize_symbols(layer.candidate_symbols, market)
    for task in search_plan.search_tasks:
        task.task_id = _key(task.task_id or task.query)
        if task.task_id in seen_tasks:
            raise ValueError(f"Duplicate search task_id: {task.task_id}")
        seen_tasks.add(task.task_id)
        task.layer_key = _key(task.layer_key or task.layer_name)
        if len(task.query) < 12:
            raise ValueError(f"Search task query is too vague: {task.query!r}")
    core_or_important = {layer.key for layer in search_plan.layers if layer.importance in {"core", "important"}}
    searched_layers = {task.layer_key for task in search_plan.search_tasks if task.priority in {"required", "important"}}
    required_set = set(required_symbols or [])
    missing_layers = []
    for layer in search_plan.layers:
        if layer.key not in core_or_important or layer.key in searched_layers:
            continue
        layer_symbols = set(layer.candidate_symbols)
        if layer_symbols and layer_symbols <= required_set:
            search_plan.warnings.append(
                f"Layer {layer.key} has no search task because it only contains user-required symbols."
            )
            continue
        missing_layers.append(layer.key)
    missing_layers = sorted(missing_layers)
    if missing_layers:
        raise ValueError(
            "Search plan left core/important layers without required or important search tasks: "
            f"{missing_layers}"
        )


def _execute_search_plan(
    search_plan: WebSearchDiscoverySearchPlan,
    recorder: WebSearchDiscoveryRecorder,
) -> list[dict[str, Any]]:
    priority_rank = {"required": 0, "important": 1, "optional": 2}
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    indexed_tasks = list(enumerate(search_plan.search_tasks))
    ordered_tasks = sorted(
        indexed_tasks,
        key=lambda item: (
            priority_rank.get(item[1].priority, 9),
            risk_rank.get(item[1].omission_risk, 9),
            item[0],
        ),
    )
    skipped: list[dict[str, Any]] = []
    for _index, task in ordered_tasks:
        args = {
            "task_id": task.task_id,
            "query": task.query,
            "layer_key": task.layer_key,
            "layer_name": task.layer_name,
            "branch": task.branch,
            "rationale": task.rationale,
            "priority": task.priority,
            "omission_risk": task.omission_risk,
            "max_results": recorder.max_results,
        }
        if recorder.successful_searches >= recorder.max_searches:
            skipped.append({**args, "reason": "search_budget_exhausted"})
            continue
        try:
            raw_results = _run_duckduckgo_search(task.query, recorder.max_results)
            result = {
                "status": "ok",
                "query": task.query,
                "budget_remaining": max(0, recorder.max_searches - recorder.successful_searches - 1),
                "results": raw_results,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "query": task.query,
                "error": str(exc),
                "budget_remaining": max(0, recorder.max_searches - recorder.successful_searches),
                "results": [],
            }
        trace_id = recorder.add_search_call(args, result)
        result["trace_id"] = trace_id
        recorder.progress_log(
            "planned_web_search",
            trace_id=trace_id,
            status=result["status"],
            task_id=task.task_id,
            layer_key=task.layer_key,
            priority=task.priority,
            omission_risk=task.omission_risk,
            query=task.query,
            result_count=len(result.get("results") or []),
            budget_remaining=result.get("budget_remaining"),
        )
    return skipped


def _register_web_search_tool(agent: Any, recorder: WebSearchDiscoveryRecorder) -> None:
    @agent.tool_plain(
        name="web_search_budgeted",
        description=(
            "Search the web with a hard run-level budget. Args: query, layer_key, "
            "rationale, max_results. Returns trace_id, budget_remaining, and compact results."
        ),
    )
    def web_search_budgeted(
        query: str,
        layer_key: str = "",
        rationale: str = "",
        max_results: int | None = None,
    ) -> dict[str, Any]:
        cleaned_query = " ".join(str(query or "").split())
        requested_results = max(1, int(max_results or recorder.max_results))
        effective_results = min(requested_results, recorder.max_results)
        args = {
            "query": cleaned_query,
            "layer_key": str(layer_key or "").strip(),
            "rationale": str(rationale or "").strip(),
            "max_results": effective_results,
        }
        if not cleaned_query:
            result = {
                "status": "error",
                "error": "query is required",
                "budget_remaining": max(0, recorder.max_searches - recorder.successful_searches),
                "results": [],
            }
            trace_id = recorder.add_search_call(args, result)
            result["trace_id"] = trace_id
            return result
        if recorder.successful_searches >= recorder.max_searches:
            result = {
                "status": "budget_exhausted",
                "error": f"Search budget exhausted after {recorder.max_searches} successful searches.",
                "budget_remaining": 0,
                "results": [],
            }
            trace_id = recorder.add_search_call(args, result)
            result["trace_id"] = trace_id
            return result
        try:
            raw_results = _run_duckduckgo_search(cleaned_query, effective_results)
            result = {
                "status": "ok",
                "query": cleaned_query,
                "budget_remaining": max(0, recorder.max_searches - recorder.successful_searches - 1),
                "results": raw_results,
            }
        except Exception as exc:
            result = {
                "status": "error",
                "query": cleaned_query,
                "error": str(exc),
                "budget_remaining": max(0, recorder.max_searches - recorder.successful_searches),
                "results": [],
            }
        trace_id = recorder.add_search_call(args, result)
        result["trace_id"] = trace_id
        recorder.progress_log(
            "web_search",
            trace_id=trace_id,
            status=result["status"],
            query=cleaned_query,
            result_count=len(result.get("results") or []),
            budget_remaining=result.get("budget_remaining"),
        )
        return result


def _run_duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from ddgs.ddgs import DDGS
    except ImportError as exc:  # pragma: no cover - covered by dependency tests indirectly.
        raise RuntimeError(
            "ddgs is required for budgeted websearch discovery. "
            'Install pydantic-ai-slim[duckduckgo] or ddgs.'
        ) from exc

    with DDGS() as client:
        results = client.text(query, max_results=max_results)
    normalized: list[dict[str, str]] = []
    for item in results or []:
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("href") or item.get("url") or "").strip(),
                "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
            }
        )
    return normalized[:max_results]


def _validate_websearch_artifact(
    artifact: WebSearchDiscoveryArtifact,
    recorder: WebSearchDiscoveryRecorder,
    market: str,
) -> None:
    if recorder.successful_searches <= 0:
        raise ValueError("Websearch discovery did not complete any successful web_search_budgeted calls.")
    max_candidates = _env_int("IA_WEBSEARCH_DISCOVERY_MAX_CANDIDATES", 90)
    if len(artifact.candidates) > max_candidates:
        raise ValueError(
            f"Websearch discovery returned {len(artifact.candidates)} candidates, "
            f"above IA_WEBSEARCH_DISCOVERY_MAX_CANDIDATES={max_candidates}."
        )
    seen: dict[str, WebSearchDiscoveryCandidate] = {}
    deduped_candidates: list[WebSearchDiscoveryCandidate] = []
    for candidate in artifact.candidates:
        normalized = normalize_futu_symbol(candidate.symbol, market)
        _validate_ascii_symbol(normalized)
        if normalized in seen:
            existing = seen[normalized]
            existing.source_trace_ids = _dedupe([*existing.source_trace_ids, *candidate.source_trace_ids])
            if candidate.layer_key != existing.layer_key:
                artifact.warnings.append(
                    f"Duplicate candidate {normalized} appeared in layers "
                    f"{existing.layer_key} and {candidate.layer_key}; kept first layer and merged sources."
                )
            continue
        candidate.symbol = normalized
        candidate.source_trace_ids = _dedupe(candidate.source_trace_ids)
        seen[normalized] = candidate
        deduped_candidates.append(candidate)
    artifact.candidates = deduped_candidates
    for layer in artifact.layers:
        layer.key = _key(layer.key or layer.name)
        layer.candidate_symbols = _normalize_symbols(layer.candidate_symbols, market)
    artifact.search_queries_used = _dedupe(artifact.search_queries_used)


def _theme_discovery_plan_from_websearch_artifact(
    artifact: WebSearchDiscoveryArtifact,
    *,
    theme: str,
    market: str,
    theme_description: str,
    required_symbols: list[str],
    pydantic_ai: dict[str, Any],
    search_calls: list[dict[str, Any]],
) -> ThemeDiscoveryPlan:
    candidates_by_symbol: dict[str, WebSearchDiscoveryCandidate] = {}
    for candidate in artifact.candidates:
        symbol = normalize_futu_symbol(candidate.symbol, market)
        if symbol:
            candidates_by_symbol.setdefault(symbol, candidate)

    layer_candidates: dict[str, list[WebSearchDiscoveryCandidate]] = {}
    for candidate in candidates_by_symbol.values():
        layer_candidates.setdefault(candidate.layer_key, []).append(candidate)

    domains: list[ThemeDomain] = []
    coverage: list[ThemeCoverageRequirement] = []
    for layer in artifact.layers:
        layer_symbols = _dedupe(
            [
                *_normalize_symbols(layer.candidate_symbols, market),
                *[
                    normalize_futu_symbol(candidate.symbol, market)
                    for candidate in layer_candidates.get(layer.key, [])
                ],
            ]
        )
        domain_candidates = [
            ThemeDomainCandidate(
                symbol=symbol,
                role=(candidates_by_symbol[symbol].role if symbol in candidates_by_symbol else ""),
                rationale=(candidates_by_symbol[symbol].rationale if symbol in candidates_by_symbol else ""),
                priority=(candidates_by_symbol[symbol].priority if symbol in candidates_by_symbol else "watchlist"),
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
                        candidate_limit_reason=layer.candidate_limit_reason,
                        candidates=domain_candidates,
                    )
                ],
            )
        )
        must_consider = [
            symbol
            for symbol in layer_symbols
            if symbol in candidates_by_symbol and candidates_by_symbol[symbol].priority == "must_consider"
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
                evidence_needed=artifact.next_enrichment_needed or _default_next_enrichment(),
            )
        )

    seed_symbols = _seed_symbols(
        candidates_by_symbol=candidates_by_symbol,
        required_symbols=required_symbols,
        market=market,
    )
    search_queries = _dedupe(
        [
            *artifact.search_queries_used,
            *[
                str((call.get("args") or {}).get("query") or "").strip()
                for call in search_calls
                if (call.get("args") or {}).get("query")
            ],
        ]
    )
    warnings = _dedupe(
        [
            *artifact.warnings,
            "Websearch discovery is candidate generation only; Futu market data, SEC filings, and news validation are still required.",
        ]
    )
    mode = str(pydantic_ai.get("discovery_mode") or "pydantic_ai_budgeted_websearch_discovery")
    return ThemeDiscoveryPlan(
        theme=theme,
        market=market,
        theme_description=theme_description,
        initial_thesis=artifact.initial_thesis,
        domain_tree=domains,
        coverage_requirements=coverage,
        seed_symbols=seed_symbols,
        plate_keywords=[],
        benchmark_symbols=[],
        research_trace=_research_trace_from_search_calls(search_calls),
        search_queries=search_queries,
        filter_plans_by_layer=[],
        executed_filter_probes=[],
        layer_filter_audits=[],
        omissions_to_investigate=_omissions_from_websearch(artifact.omissions_to_investigate, market),
        next_enrichment_needed=artifact.next_enrichment_needed or _default_next_enrichment(),
        data_asof={
            "generated_at": utc_now(),
            "web_search": utc_now(),
        },
        discovery_notes=[
            "Budgeted websearch discovery stops after compact theme-layer candidate generation.",
            "No Futu classifier, Futu screener catalog, or stock_filter probes were used in this stage.",
        ],
        warnings=warnings,
        pydantic_ai={
            **pydantic_ai,
            "mode": mode,
            "source_artifact": artifact.model_dump(mode="json"),
        },
    )


def _seed_symbols(
    *,
    candidates_by_symbol: dict[str, WebSearchDiscoveryCandidate],
    required_symbols: list[str],
    market: str,
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
                exposure_type="websearch_candidate" if candidate else "user_required",
                exposure_purity="unknown",
                source_ids=candidate.source_trace_ids if candidate else [],
                confidence=candidate.confidence if candidate else "medium",
                freshness="fresh" if candidate and candidate.source_trace_ids else "unknown",
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
                role=candidate.role or "websearch discovery candidate",
                rationale=candidate.rationale,
                subthemes=[candidate.layer_key],
                value_chain_stage=candidate.subdomain or candidate.layer_key,
                exposure_type="websearch_candidate",
                exposure_purity="unknown",
                source_ids=candidate.source_trace_ids,
                confidence=candidate.confidence,
                freshness="fresh" if candidate.source_trace_ids else "unknown",
            )
        )
        seen.add(symbol)
    return seeds


def _research_trace_from_search_calls(search_calls: list[dict[str, Any]]) -> list[ResearchSource]:
    sources: list[ResearchSource] = []
    for call in search_calls:
        trace_id = str(call.get("trace_id") or "")
        args = call.get("args") or {}
        result = call.get("result") or {}
        query = str(args.get("query") or result.get("query") or "").strip()
        result_items = result.get("results") or []
        top_titles = [
            str(item.get("title") or "").strip()
            for item in result_items[:5]
            if isinstance(item, dict) and item.get("title")
        ]
        first_url = ""
        for item in result_items:
            if isinstance(item, dict) and item.get("url"):
                first_url = str(item["url"])
                break
        sources.append(
            ResearchSource(
                source_id=trace_id,
                title=f"web_search_budgeted: {query}",
                url=first_url,
                publisher="DuckDuckGo local search",
                retrieved_at=str(call.get("time") or utc_now()),
                source_type="web",
                summary="; ".join(top_titles) or str(result.get("error") or result.get("status") or ""),
                coverage_keys=[_key(str(args.get("layer_key") or ""))] if args.get("layer_key") else [],
            )
        )
    return sources


def _native_search_calls_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = _native_search_calls_from_event_kind(events, "builtin_tool_call")
    if calls:
        return calls
    return _native_search_calls_from_event_kind(events, "part_delta")


def _native_search_calls_from_event_kind(events: list[dict[str, Any]], event_kind: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("event") != event_kind:
            continue
        tool = str(event.get("tool") or event.get("tool_name_delta") or "web_search").strip() or "web_search"
        text = str(event.get("args") or event.get("delta_preview") or "").strip()
        parsed = _parse_native_tool_args(text)
        queries = _extract_native_queries(parsed)
        if not queries:
            continue
        key = "\n".join(queries)
        if key in seen:
            continue
        seen.add(key)
        trace_id = f"native_web_{len(calls) + 1:03d}"
        calls.append(
            {
                "trace_id": trace_id,
                "time": str(event.get("time") or utc_now()),
                "tool": tool or "web_search",
                "args": {
                    "query": queries[0],
                    "queries": queries,
                    "native": True,
                    "source_event": event.get("event"),
                },
                "result": {
                    "status": "ok",
                    "query": queries[0],
                    "queries": queries,
                    "results": [],
                    "content_preview": event.get("content_preview", ""),
                },
            }
        )
    return calls


def _native_search_calls_from_queries(queries: list[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for query in _dedupe(queries):
        trace_id = f"native_web_{len(calls) + 1:03d}"
        calls.append(
            {
                "trace_id": trace_id,
                "time": utc_now(),
                "tool": "web_search",
                "args": {"query": query, "queries": [query], "native": True},
                "result": {"status": "ok", "query": query, "queries": [query], "results": []},
            }
        )
    return calls


def _parse_native_tool_args(text: str) -> Any:
    value: Any = text
    for _ in range(3):
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
            return {}
        try:
            value = json.loads(cleaned)
        except Exception:
            return cleaned
    return value


def _extract_native_queries(parsed: Any) -> list[str]:
    if isinstance(parsed, dict):
        queries = parsed.get("queries")
        if isinstance(queries, list):
            return _dedupe([str(item) for item in queries if str(item or "").strip()])
        query = str(parsed.get("query") or "").strip()
        return [query] if query else []
    if isinstance(parsed, str):
        matches = re.findall(r'"([^"]{8,240})"', parsed)
        queryish = [
            item
            for item in matches
            if any(token in item.lower() for token in ("ai", "data center", "semiconductor", "storage", "power"))
        ]
        return _dedupe(queryish)
    return []


def _omissions_from_websearch(
    omissions: list[WebSearchDiscoveryOmission],
    market: str,
) -> list[DiscoveryOmission]:
    result: list[DiscoveryOmission] = []
    for omission in omissions:
        symbol = str(omission.symbol_or_entity or "").strip().upper()
        normalized = normalize_futu_symbol(symbol, market) if _SYMBOL_RE.fullmatch(symbol) else symbol
        reason = "unsupported_security_type"
        if normalized.startswith(f"{market}.") and not symbol.startswith(("PRIVATE.", "UNLISTED.")):
            reason = "clear_theme_mismatch"
        result.append(
            DiscoveryOmission(
                symbol=normalized,
                layer_key=omission.layer_key,
                source_trace_ids=omission.source_trace_ids,
                exclusion_reason=reason,
                explanation=omission.reason or "Omitted by websearch discovery pending later validation.",
            )
        )
    return result


def _run_agent_sync(agent: Any, payload: dict[str, Any], recorder: WebSearchDiscoveryRecorder):
    kwargs: dict[str, Any] = {}
    if _env_bool("IA_WEBSEARCH_DISCOVERY_TRACE_EVENTS", False):
        kwargs["event_stream_handler"] = _event_stream_handler(recorder)
    return agent.run_sync(json.dumps(payload, ensure_ascii=False, sort_keys=True), **kwargs)


def _event_stream_handler(recorder: WebSearchDiscoveryRecorder):
    async def handler(_ctx: Any, events: Any) -> None:
        async for event in events:
            summary = _summarize_event(event)
            if summary is not None:
                recorder.add_event(summary)
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
            "content_preview": _preview(getattr(part, "content", ""), 1200),
        }
    if kind in {"part_start", "part_end"} and part is not None:
        return {
            "event": kind,
            "index": getattr(event, "index", ""),
            "part_kind": getattr(part, "part_kind", ""),
            "tool": getattr(part, "tool_name", ""),
            "call_id": getattr(part, "tool_call_id", ""),
        }
    if kind == "final_result":
        return {
            "event": kind,
            "tool": getattr(event, "tool_name", ""),
            "call_id": getattr(event, "tool_call_id", ""),
        }
    return None


def _run_mode(payload: dict[str, Any]) -> str:
    return str(payload.get("discovery_mode") or payload.get("mode") or "").strip().lower()


def is_websearch_discovery_mode(payload: dict[str, Any]) -> bool:
    return _run_mode(payload) in {
        "websearch",
        "web_search",
        "websearch_only",
        "ai_websearch",
        "native_websearch",
        "native_web_search",
        "native_search",
    }


def is_native_websearch_discovery_mode(payload: dict[str, Any]) -> bool:
    return _run_mode(payload) in {"native_websearch", "native_web_search", "native_search"}


def _part_args(part: Any) -> str:
    if not hasattr(part, "args_as_json_str"):
        return ""
    try:
        return part.args_as_json_str()
    except Exception:
        return repr(getattr(part, "args", ""))


def _normalize_market(market: str) -> str:
    return str(market or "US").strip().upper() or "US"


def _normalize_symbols(symbols: list[str], market: str) -> list[str]:
    normalized: list[str] = []
    for symbol in symbols:
        value = normalize_futu_symbol(symbol, market)
        if not value:
            continue
        _validate_ascii_symbol(value)
        normalized.append(value)
    return _dedupe(normalized)


def _validate_ascii_symbol(symbol: str) -> None:
    if not symbol.isascii() or not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"Invalid symbol: {symbol!r}")


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


def _default_next_enrichment() -> list[str]:
    return ["Futu lightweight market data", "SEC filings and earnings evidence", "latest news/events"]


def _key(value: str) -> str:
    key = str(value or "").strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key or "unknown"


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _preview(value: Any, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
