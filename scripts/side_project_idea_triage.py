#!/usr/bin/env python3
"""Rank side-project ideas from a local Markdown note.

Offline, deterministic helper for turning a human-maintained idea dump into a
small reviewable shortlist. It intentionally reads only local files and writes
only to stdout/stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PREFERRED_AREA_BONUSES = {
    "information aggregation": 3,
    "investment": 3,
    "investing": 3,
    "personal systems": 3,
    "health": 3,
    "side projects": 3,
    "side project": 3,
    "dev tools": 3,
    "developer tools": 3,
    "learning": 1,
    "writing": 1,
    "finance": 1,
}

VAGUE_MARKERS = ("someday", "maybe", "research more", "one day", "eventually")
SKIP_HEADINGS = {"template", "example", "examples", "instructions", "guidance"}
NUMERIC_FIELDS = {"impact", "effort", "confidence", "urgency"}
TEXT_FIELDS = {"area", "first step", "notes"}
BOOL_FIELDS = {"blocked"}
FIELD_ALIASES = {
    "first-step": "first step",
    "first_step": "first step",
    "next step": "first step",
    "next_step": "first step",
}


@dataclass
class Idea:
    title: str
    area: str = "uncategorized"
    impact: int = 3
    effort: int = 3
    confidence: int = 3
    urgency: int = 3
    first_step: str = ""
    notes: str = ""
    blocked: bool = False
    score: int = 0
    score_explanation: str = ""
    raw_fields: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "area": self.area,
            "impact": self.impact,
            "effort": self.effort,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "first_step": self.first_step,
            "notes": self.notes,
            "blocked": self.blocked,
            "score": self.score,
            "score_explanation": self.score_explanation,
        }


_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_FIELD_RE = re.compile(r"^\s*[-*]\s*([^:]{2,40}):\s*(.*?)\s*$")


def _normalize_field_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ")
    return FIELD_ALIASES.get(normalized, normalized)


def _clamp_rating(value: str, default: int = 3) -> int:
    try:
        rating = int(float(value.strip()))
    except (TypeError, ValueError):
        return default
    return max(1, min(5, rating))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "blocked"}


def _looks_like_real_idea(title: str, fields: dict[str, str]) -> bool:
    if title.strip().lower() in SKIP_HEADINGS:
        return False
    if not fields:
        return False
    return bool(({"impact", "first step", "area", "effort"} & set(fields)))


def parse_ideas(markdown: str) -> list[Idea]:
    """Parse Markdown idea sections into Idea records.

    Only level-2 and level-3 headings are considered idea candidates. Prose and
    bullets before the first candidate are ignored, and template/instruction
    sections are explicitly skipped.
    """
    candidates: list[tuple[str, dict[str, str]]] = []
    current_title: str | None = None
    current_fields: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_title, current_fields
        if current_title and _looks_like_real_idea(current_title, current_fields):
            candidates.append((current_title, dict(current_fields)))
        current_title = None
        current_fields = {}

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            current_title = heading.group(2).strip().strip("# ")
            continue
        if current_title is None:
            continue
        field_match = _FIELD_RE.match(line)
        if not field_match:
            continue
        name = _normalize_field_name(field_match.group(1))
        value = field_match.group(2).strip()
        if name in NUMERIC_FIELDS | TEXT_FIELDS | BOOL_FIELDS:
            current_fields[name] = value
    flush()

    return [_idea_from_fields(title, fields) for title, fields in candidates]


def _idea_from_fields(title: str, fields: dict[str, str]) -> Idea:
    return Idea(
        title=title,
        area=fields.get("area", "uncategorized") or "uncategorized",
        impact=_clamp_rating(fields.get("impact", "3")),
        effort=_clamp_rating(fields.get("effort", "3")),
        confidence=_clamp_rating(fields.get("confidence", "3")),
        urgency=_clamp_rating(fields.get("urgency", "3")),
        first_step=fields.get("first step", ""),
        notes=fields.get("notes", ""),
        blocked=_parse_bool(fields.get("blocked", "false")),
        raw_fields=fields,
    )


def score_idea(idea: Idea) -> Idea:
    area_bonus = PREFERRED_AREA_BONUSES.get(idea.area.strip().lower(), 0)
    penalties: list[str] = []
    penalty_points = 0

    if idea.blocked:
        penalty_points += 8
        penalties.append("blocked -8")
    if not idea.first_step.strip():
        penalty_points += 4
        penalties.append("missing first step -4")
    vague_text = f"{idea.title} {idea.notes}".lower()
    if any(marker in vague_text for marker in VAGUE_MARKERS):
        penalty_points += 2
        penalties.append("vague wording -2")

    score = (
        idea.impact * 3
        + idea.confidence * 2
        + idea.urgency
        + area_bonus
        - idea.effort * 2
        - penalty_points
    )
    idea.score = score
    reason = (
        f"impact {idea.impact}*3 + confidence {idea.confidence}*2 + urgency {idea.urgency} "
        f"+ area bonus {area_bonus} - effort {idea.effort}*2"
    )
    if penalties:
        reason += " - " + ", ".join(penalties)
    idea.score_explanation = reason
    return idea


def rank_ideas(ideas: Iterable[Idea]) -> list[Idea]:
    scored = [score_idea(idea) for idea in ideas]
    return sorted(
        scored,
        key=lambda idea: (
            -idea.score,
            idea.blocked,
            idea.effort,
            -idea.impact,
            idea.title.lower(),
        ),
    )


def _cell(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|") or "—"


def render_markdown(ideas: list[Idea], *, limit: int | None = None) -> str:
    shown = ideas[:limit] if limit else ideas
    lines = [
        "# Side Project Idea Triage",
        "",
        "| Rank | Title | Area | Score | Impact | Effort | Confidence | First step |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, idea in enumerate(shown, start=1):
        lines.append(
            "| {rank} | {title} | {area} | {score} | {impact} | {effort} | {confidence} | {first_step} |".format(
                rank=idx,
                title=_cell(idea.title),
                area=_cell(idea.area),
                score=idea.score,
                impact=idea.impact,
                effort=idea.effort,
                confidence=idea.confidence,
                first_step=_cell(idea.first_step),
            )
        )

    actionable = [idea for idea in ideas if not idea.blocked]
    top = actionable[0] if actionable else (ideas[0] if ideas else None)
    if top:
        lines.extend(
            [
                "",
                "## Top recommendation",
                f"**{top.title}** — score {top.score}. {top.score_explanation}.",
            ]
        )
        if top.first_step:
            lines.append(f"Next move: {top.first_step}")

    defer = [idea for idea in ideas if idea.blocked or idea.score <= 8]
    lines.extend(["", "## Watch / defer"])
    if defer:
        for idea in defer:
            status = "blocked" if idea.blocked else "low score"
            lines.append(f"- **{idea.title}** ({status}, score {idea.score}): {idea.score_explanation}")
    else:
        lines.append("- No blocked or low-score ideas in this input.")

    return "\n".join(lines) + "\n"


def render_json(ideas: list[Idea]) -> str:
    return json.dumps({"ideas": [idea.as_dict() for idea in ideas]}, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank side-project ideas from a local Markdown file.",
    )
    parser.add_argument("path", type=Path, help="Markdown file containing idea sections")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum ideas to show in Markdown output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        markdown = args.path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read {args.path}: {exc}", file=sys.stderr)
        return 1

    ideas = rank_ideas(parse_ideas(markdown))
    if not ideas:
        print("No side-project ideas found. Add ## idea headings with fields such as Impact and First step.", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_json(ideas))
    else:
        print(render_markdown(ideas, limit=args.limit), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
