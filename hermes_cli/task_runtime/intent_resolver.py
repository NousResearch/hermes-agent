"""Intent Resolver for Task Runtime.

Substitutes a closed-set IntentClassifier with a broad, metadata-driven
resolver. Detects:
- task_type (research, code, review, archival, kanban).
- whether the intent comes from Kanban.
- whether GBrain / Obsidian / memory lookups are needed.
- which skills are likely relevant.

NO HTTP, NO LLM. Pure metadata analysis from the raw_text + source_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Keywords (lowercased) used to infer task_type and required context sources.
_CODE_KEYWORDS = ("implement", "code", "function", "class", "module", "bug", "fix", "patch", "refactor", "add")
_RESEARCH_KEYWORDS = ("explain", "research", "what is", "how does", "compare", "analyze", "describe", "documentation")
_REVIEW_KEYWORDS = ("review", "audit", "validate", "check", "verify", "find issues", "rubric")
_KANBAN_KEYWORDS = ("kanban", "task board", "worker", "dispatch", "claim", "complete", "block")
_ARCHIVAL_KEYWORDS = ("archive", "backup", "snapshot", "history", "restore")

# Skill-name hints: regex → skill name (looked up later via SkillLoader).
_SKILL_HINTS = [
    (re.compile(r"\b(producer\s*normalizer|normaliz)\w*", re.I), "producer-normalizer"),
    (re.compile(r"\b(reviewer|review)\w*", re.I), "codex-reviewer"),
    (re.compile(r"\b(kanban|board)\w*", re.I), "kanban"),
    (re.compile(r"\b(gbrain|knowledge)\w*", re.I), "gbrain-think-emulator"),
    (re.compile(r"\b(obsidian|note|diario)\w*", re.I), "obsidian"),
    (re.compile(r"\b(hermes[- ]?agent|hermes cli)\w*", re.I), "hermes-agent"),
]


@dataclass(frozen=True)
class ResolvedIntent:
    """Output of IntentResolver. Pure data; no side effects."""
    intent_id: str
    raw_text: str
    source: str           # "cli" | "task_file" | "kanban" | "mcp"
    source_id: str | None
    task_type: str        # "research" | "code" | "review" | "archival" | "kanban" | "unknown"
    needs_gbrain: bool
    needs_obsidian: bool
    needs_memory: bool
    needs_skills: bool
    suggested_skills: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _infer_task_type(text: str, source: str, source_id: str | None) -> str:
    """Infer task_type from keywords, source, and source_id."""
    lt = text.lower()
    if source == "kanban" or source_id:
        return "kanban"
    # Count keyword matches per category.
    scores = {
        "code": sum(1 for kw in _CODE_KEYWORDS if kw in lt),
        "research": sum(1 for kw in _RESEARCH_KEYWORDS if kw in lt),
        "review": sum(1 for kw in _REVIEW_KEYWORDS if kw in lt),
        "kanban": sum(1 for kw in _KANBAN_KEYWORDS if kw in lt),
        "archival": sum(1 for kw in _ARCHIVAL_KEYWORDS if kw in lt),
    }
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "unknown"
    return best[0]


def _infer_skill_hints(text: str) -> list[str]:
    """Return a stable, deduped list of suggested skill names."""
    seen: set[str] = set()
    out: list[str] = []
    for pat, name in _SKILL_HINTS:
        if pat.search(text) and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _needs_context_source(text: str, source: str, source_id: str | None) -> tuple[bool, bool, bool]:
    """Decide whether GBrain, Obsidian, or memory lookups are likely useful."""
    lt = text.lower()
    # GBrain for knowledge Qs / research.
    needs_gbrain = any(kw in lt for kw in ("what is", "how does", "explain", "compare", "research", "documentation"))
    # Obsidian for notes / diary references.
    needs_obsidian = any(kw in lt for kw in ("note", "diario", "obsidian", "vault"))
    # Memory for cross-session recall (always-on unless explicit).
    needs_memory = source in ("kanban", "mcp") or source_id is not None or "previous" in lt or "earlier" in lt
    return needs_gbrain, needs_obsidian, needs_memory


def resolve(
    raw_text: str,
    source: str = "cli",
    source_id: str | None = None,
    intent_id: str | None = None,
) -> ResolvedIntent:
    """Resolve a raw user intent into a ResolvedIntent.

    Pure function: no I/O, no HTTP, no LLM. Safe to call in dry-run.
    """
    import uuid
    rid = intent_id or f"intent-{uuid.uuid4().hex[:12]}"
    task_type = _infer_task_type(raw_text, source, source_id)
    needs_gbrain, needs_obsidian, needs_memory = _needs_context_source(raw_text, source, source_id)
    suggested_skills = _infer_skill_hints(raw_text)
    return ResolvedIntent(
        intent_id=rid,
        raw_text=raw_text,
        source=source,
        source_id=source_id,
        task_type=task_type,
        needs_gbrain=needs_gbrain,
        needs_obsidian=needs_obsidian,
        needs_memory=needs_memory,
        needs_skills=bool(suggested_skills),
        suggested_skills=suggested_skills,
        metadata={
            "raw_text_length": len(raw_text),
            "raw_text_word_count": len(raw_text.split()),
        },
    )