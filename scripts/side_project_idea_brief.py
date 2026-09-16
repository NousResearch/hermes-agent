#!/usr/bin/env python3
"""Render a Joe-style morning brief from a local side-project idea backlog.

The script is intentionally local-first: it reads JSON/YAML from disk and prints
markdown only. It does not create tasks, send messages, or touch external APIs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Optional; JSON remains supported without PyYAML.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


JOE_RELEVANT_LANES = {
    "side project",
    "side projects",
    "personal os",
    "information aggregation",
    "top-of-funnel leads",
    "investment framework",
    "health",
    "body composition",
    "personal reminders",
    "life ops",
    "ai dev tooling",
    "developer tooling",
}

JOE_RELEVANT_KEYWORDS = {
    "ai",
    "agent",
    "agents",
    "dev",
    "developer",
    "github",
    "radar",
    "investment",
    "thesis",
    "portfolio",
    "health",
    "body",
    "reminder",
    "reminders",
    "bd",
    "lead",
    "leads",
    "payments",
    "fintech",
    "movement",
}


@dataclass
class Idea:
    title: str
    lane: str = "side project"
    problem: str = ""
    action: str = ""
    impact: int = 3
    effort: int = 3
    confidence: int = 3
    source: str = ""
    notes: str = ""
    created: str = ""
    updated: str = ""
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _as_int(value: Any, default: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, parsed))


def _as_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _load_raw(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        return yaml.safe_load(text) or []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise
        return yaml.safe_load(text) or []


def _is_inactive(raw: dict[str, Any]) -> bool:
    status = str(raw.get("status", "")).strip().lower()
    return bool(raw.get("done") or raw.get("completed") or status in {"done", "completed", "archived"})


def _normalise(raw: dict[str, Any]) -> Idea:
    return Idea(
        title=str(raw.get("title", "")).strip(),
        lane=str(raw.get("lane") or raw.get("area") or "side project").strip(),
        problem=str(raw.get("problem") or raw.get("why") or "").strip(),
        action=str(raw.get("action") or raw.get("next_action") or "").strip(),
        impact=_as_int(raw.get("impact")),
        effort=_as_int(raw.get("effort")),
        confidence=_as_int(raw.get("confidence")),
        source=str(raw.get("source") or "").strip(),
        notes=str(raw.get("notes") or "").strip(),
        created=str(raw.get("created") or raw.get("created_at") or "").strip(),
        updated=str(raw.get("updated") or raw.get("updated_at") or "").strip(),
        tags=_as_tags(raw.get("tags")),
    )


def load_ideas(path: str | Path) -> list[Idea]:
    """Load active ideas from a JSON/YAML backlog.

    Accepts either a top-level list or an object with an ``ideas`` list. Ideas
    marked done/completed/archived are filtered out.
    """

    data = _load_raw(Path(path))
    if isinstance(data, dict):
        items = data.get("ideas", [])
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("Input must be a list or an object with an 'ideas' list")

    ideas: list[Idea] = []
    for item in items:
        if not isinstance(item, dict) or _is_inactive(item):
            continue
        idea = _normalise(item)
        if idea.title:
            ideas.append(idea)
    return ideas


def _days_since(date_text: str, today: str) -> int | None:
    if not date_text:
        return None
    try:
        then = dt.date.fromisoformat(date_text[:10])
        now = dt.date.fromisoformat(today)
    except ValueError:
        return None
    return (now - then).days


def _keyword_hits(idea: Idea) -> int:
    haystack = " ".join([idea.title, idea.lane, idea.problem, idea.action, " ".join(idea.tags)]).lower()
    tokens = set(re.findall(r"[a-z0-9-]+", haystack))
    return len(tokens & JOE_RELEVANT_KEYWORDS)


def score_idea(idea: Idea, today: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    lane = idea.lane.lower()
    score = idea.impact * 3.0 + idea.confidence * 2.0 - idea.effort * 1.5

    if lane in JOE_RELEVANT_LANES:
        score += 3.0
        reasons.append(idea.lane)

    hits = _keyword_hits(idea)
    if hits:
        score += min(3.0, hits * 0.75)
        reasons.append("Joe-relevant keywords")

    if idea.source:
        score += 1.5
        reasons.append("source/evidence present")

    age = _days_since(idea.updated or idea.created, today)
    if age is not None:
        if age <= 14:
            score += 1.0
            reasons.append("recently touched")
        elif age > 90:
            score -= 1.0
            reasons.append("stale backlog item")

    if idea.effort <= 2 and idea.impact >= 4:
        score += 2.0
        reasons.append("high impact / low effort")

    return round(score, 2), reasons


def rank_ideas(ideas: list[Idea], today: str | None = None) -> list[Idea]:
    today = today or dt.date.today().isoformat()
    ranked: list[Idea] = []
    for idea in ideas:
        score, reasons = score_idea(idea, today)
        ranked.append(
            Idea(
                title=idea.title,
                lane=idea.lane,
                problem=idea.problem,
                action=idea.action,
                impact=idea.impact,
                effort=idea.effort,
                confidence=idea.confidence,
                source=idea.source,
                notes=idea.notes,
                created=idea.created,
                updated=idea.updated,
                tags=list(idea.tags),
                score=score,
                reasons=reasons,
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.effort, item.title.lower()))


def _bullet(label: str, value: str) -> str:
    return f"  - {label}: {value}" if value else ""


def render_brief(
    ideas: list[Idea],
    today: str | None = None,
    top: int = 3,
    silent_if_empty: bool = False,
) -> str:
    today = today or dt.date.today().isoformat()
    if not ideas:
        if silent_if_empty:
            return "[SILENT]"
        return "## TL;DR\n- Fact / verified: 沒有可評估的主動 side-project idea。"

    selected = ideas[: max(1, top)]
    best = selected[0]
    lines = [
        "## TL;DR",
        f"- Fact / verified: 讀到 {len(ideas)} 個 active idea；今日最值得 Joe review 的是「{best.title}」。",
        f"- Hypothesis: 這批 idea 中，分數高者通常是本地優先、低努力、高複利，適合變成小 PR 或週末實驗。",
        "- Action for Joe: 選 1 個保留，其餘直接降級；不要讓 backlog 變成心理負債。",
        "",
        "## Fact / verified",
    ]

    for index, idea in enumerate(selected, 1):
        lines.extend(
            [
                f"{index}. **{idea.title}** — score {idea.score:g}",
                _bullet("Lane", idea.lane),
                _bullet("Problem", idea.problem),
                _bullet("Action", idea.action),
                f"  - ICE-ish: impact {idea.impact}/5, effort {idea.effort}/5, confidence {idea.confidence}/5",
                _bullet("Why now", ", ".join(idea.reasons)),
                _bullet("Notes", idea.notes),
                f"  - Source: {idea.source}" if idea.source else "",
            ]
        )
    lines = [line for line in lines if line]

    lines.extend(
        [
            "",
            "## Hypothesis",
            f"- 「{best.title}」若能在 30–60 分鐘內產出可見 artifact，會比再新增 5 個 idea 更有價值。",
            "- 本地優先是安全邊界：先讀檔與輸出 markdown，不接私有資料、不自動聯絡人、不部署。",
            "",
            "## Action for Joe",
            f"- 今天只需決定：要不要把「{best.title}」升級成下一個 nightly build / weekend spike。",
            "- 若答案不是明確 yes，直接把它降成 parking lot，避免反覆重看。",
        ]
    )
    return "\n".join(lines)


def build_brief(path: str | Path, today: str | None = None, top: int = 3, silent_if_empty: bool = False) -> str:
    ideas = rank_ideas(load_ideas(path), today=today)
    return render_brief(ideas, today=today, top=top, silent_if_empty=silent_if_empty)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Joe-style side-project idea brief from JSON/YAML.")
    parser.add_argument("input", help="Path to JSON/YAML backlog; top-level list or {ideas: [...]}.")
    parser.add_argument("--today", default=dt.date.today().isoformat(), help="YYYY-MM-DD date for deterministic scoring.")
    parser.add_argument("--top", type=int, default=3, help="Number of ranked ideas to include.")
    parser.add_argument("--silent-if-empty", action="store_true", help="Print exact [SILENT] when no active ideas exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(build_brief(args.input, today=args.today, top=args.top, silent_if_empty=args.silent_if_empty))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
