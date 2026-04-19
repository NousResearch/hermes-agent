"""Deterministic 80/20 memory-plan policy for the meta-router."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

_HISTORY_MARKERS = (
    "before", "earlier", "previous", "previously", "last time", "past progress",
    "past work", "what did we decide", "what have we decided", "decision", "decisions",
    "status", "context", "continue from", "resume", "progress", "history", "remember",
)
_DOC_MARKERS = (
    "report", "reports", "doc", "docs", "document", "documents", "transcript",
    "transcripts", "audit", "audits", "checklist", "checklists", "prd", "spec",
    "specs", "architecture", "codebase", "runtime", "files", "logs",
)
_WIKI_MARKERS = (
    "wiki", "vault", "obsidian", "wikilink", "entity", "entities", "concept",
    "concepts", "synthesis", "syntheses", "dashboard page", "knowledge page",
)
_STRATEGIC_MARKERS = (
    "capture", "store this", "save this", "remember this", "log this", "preserve this",
    "strategic", "insight", "decision log", "write this down",
)


@dataclass(frozen=True)
class MemoryPlan:
    need: str
    authority: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    optional_tools: list[str] = field(default_factory=list)
    skip_tools: list[str] = field(default_factory=list)
    max_memory_steps: int = 0
    policy_version: str = "mr-memory-v1"
    rationale: str = ""


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in markers)


def build_memory_plan(text: str, task_type: str, mode: str, *, bypassed: bool = False) -> MemoryPlan:
    lower = (text or '').lower()
    if bypassed or not lower.strip():
        return MemoryPlan(
            need='auto',
            authority=['active-memory'],
            required_tools=[],
            optional_tools=['memory_search'],
            skip_tools=['open-brain__recall', 'open-brain__capture'],
            max_memory_steps=0,
            rationale='Bypassed or trivial turn; rely on automatic recall only.',
        )

    history = _contains_any(lower, _HISTORY_MARKERS)
    docs = _contains_any(lower, _DOC_MARKERS)
    wiki = _contains_any(lower, _WIKI_MARKERS)
    strategic = _contains_any(lower, _STRATEGIC_MARKERS)

    if wiki:
        return MemoryPlan(
            need='wiki+history',
            authority=['active-memory', 'memory-core', 'memory-wiki', 'qmd'],
            required_tools=['memory_search', 'qmd__query'],
            optional_tools=['memory_get', 'qmd__get'],
            skip_tools=['open-brain__recall'],
            max_memory_steps=2,
            rationale='Wiki/vault work needs durable history plus document-backed retrieval.',
        )

    if strategic and not docs:
        return MemoryPlan(
            need='history',
            authority=['active-memory', 'memory-core', 'open-brain'],
            required_tools=['memory_search'],
            optional_tools=['memory_get', 'open-brain__recall'],
            skip_tools=['qmd__query'],
            max_memory_steps=2,
            rationale='Strategic recall should confirm prior decisions before any new capture/update.',
        )

    if task_type in {'audit', 'research'} and (docs or history or mode in {'review', 'plan'}):
        return MemoryPlan(
            need='history+docs',
            authority=['active-memory', 'memory-core', 'qmd'],
            required_tools=['memory_search', 'qmd__query'],
            optional_tools=['memory_get', 'qmd__get'],
            skip_tools=['open-brain__capture'],
            max_memory_steps=2,
            rationale='Audit/research turns should ground both prior context and long-form artifacts.',
        )

    if history and docs:
        return MemoryPlan(
            need='history+docs',
            authority=['active-memory', 'memory-core', 'qmd'],
            required_tools=['memory_search', 'qmd__query'],
            optional_tools=['memory_get', 'qmd__get'],
            skip_tools=['open-brain__capture'],
            max_memory_steps=2,
            rationale='Task explicitly references both prior progress and documents.',
        )

    if docs:
        return MemoryPlan(
            need='docs',
            authority=['active-memory', 'qmd'],
            required_tools=['qmd__query'],
            optional_tools=['qmd__get', 'memory_search'],
            skip_tools=['open-brain__capture'],
            max_memory_steps=2,
            rationale='Task is document-heavy; start with QMD before drafting.',
        )

    if history or task_type in {'config', 'production'}:
        return MemoryPlan(
            need='history',
            authority=['active-memory', 'memory-core'],
            required_tools=['memory_search'],
            optional_tools=['memory_get'],
            skip_tools=['open-brain__capture', 'qmd__query'],
            max_memory_steps=2,
            rationale='Task depends on prior decisions, preferences, or execution context.',
        )

    return MemoryPlan(
        need='auto',
        authority=['active-memory'],
        required_tools=[],
        optional_tools=['memory_search'],
        skip_tools=['open-brain__capture', 'memory-wiki'],
        max_memory_steps=0,
        rationale='Default lightweight turn; automatic recall is enough unless the task proves otherwise.',
    )


def format_memory_plan_block(plan: MemoryPlan) -> str:
    if not isinstance(plan, MemoryPlan):
        return ''
    lines = [
        '[META-MEMORY]',
        f'need: {plan.need}',
        f'authority: {", ".join(plan.authority) if plan.authority else "active-memory"}',
        f'required_tools: {", ".join(plan.required_tools) if plan.required_tools else "(none)"}',
        f'optional_tools: {", ".join(plan.optional_tools) if plan.optional_tools else "(none)"}',
        f'skip_tools: {", ".join(plan.skip_tools) if plan.skip_tools else "(none)"}',
        f'max_memory_steps: {plan.max_memory_steps}',
        f'policy_version: {plan.policy_version}',
        'contract: Use the required retrieval tools before drafting the final answer. If exact names differ in this runtime, use the closest equivalent memory/document tools.'
    ]
    return "\n".join(lines)
