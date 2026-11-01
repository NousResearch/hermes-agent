#!/usr/bin/env python3
"""Rank side-project ideas from a Markdown note into a compact review brief.

The script is intentionally offline: it does not fetch services, create tasks,
or contact anyone. It converts a human-maintained idea dump into a deterministic
queue for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

URL_RE = re.compile(r"https?://[^\s)>,;|]+")
BULLET_RE = re.compile(r"^[-*+]\s+(?P<body>.+?)\s*$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*$")
PRIORITY_RE = re.compile(r"^\[(?P<priority>high|medium|low)\]\s*", re.IGNORECASE)
GUIDANCE_SECTION_RE = re.compile(
    r"\b(format|instructions?|notes?|safety|permission|template|output|example|contract|verification|goal)\b",
    re.IGNORECASE,
)
FIELD_RE = re.compile(
    r"(?:^|\s)(?P<key>effort|leverage|reuse|next|status|audience)\s*:\s*(?P<value>.*?)(?=\s*(?:[;|]|\s+—\s+|\s+-\s+|\s+--\s+|$))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SideProjectIdea:
    """One side-project candidate extracted from a Markdown bullet."""

    title: str
    section: str
    priority: str
    effort: str | None
    leverage: str | None
    reuse: str | None
    next_action: str | None
    status: str | None
    audience: str | None
    urls: list[str]
    raw: str
    score: int


def _normalize_field(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" -—;|,.")
    return cleaned or None


def _extract_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(body):
        key = match.group("key").lower()
        value = _normalize_field(match.group("value"))
        if value:
            fields[key] = value
    return fields


def _title_from_body(body: str, urls: Sequence[str]) -> str:
    title_source = body
    priority_match = PRIORITY_RE.match(title_source)
    if priority_match:
        title_source = title_source[priority_match.end() :].strip()

    # Strip URLs before splitting, so a URL colon does not become a title separator.
    for url in urls:
        title_source = title_source.replace(url, "")

    # The first field marker usually starts the metadata area; keep text before it.
    field_match = re.search(r"(?:^|[;|]|\s+—\s+|\s+-\s+|\s+--\s+)\s*(effort|leverage|reuse|next|status|audience)\s*:", title_source, re.IGNORECASE)
    if field_match:
        title_source = title_source[: field_match.start()]

    candidate = re.split(r"\s*(?:—|--|\|)\s*", title_source, maxsplit=1)[0]
    candidate = re.sub(r"\s+", " ", candidate).strip(" -—:;|")
    return candidate or "Untitled idea"


def _score_idea(priority: str, fields: dict[str, str]) -> int:
    score = {"high": 3, "medium": 1, "normal": 0, "low": -1}.get(priority, 0)

    effort = (fields.get("effort") or "").lower()
    if effort in {"low", "small", "easy", "quick"}:
        score += 2
    elif effort in {"medium", "moderate"}:
        score += 0
    elif effort in {"high", "large", "hard"}:
        score -= 2

    leverage = (fields.get("leverage") or "").lower()
    if leverage in {"high", "large", "strong"}:
        score += 3
    elif leverage in {"medium", "moderate"}:
        score += 1
    elif leverage in {"low", "small"}:
        score -= 1

    reuse = (fields.get("reuse") or "").lower()
    if reuse and reuse not in {"none", "no", "n/a", "na"}:
        score += 1

    status = (fields.get("status") or "").lower()
    if any(term in status for term in ("blocked", "waiting", "parked")):
        score -= 3
    elif any(term in status for term in ("ready", "active")):
        score += 1

    if fields.get("next"):
        score += 1
    return score


def parse_ideas(path: str | Path) -> list[SideProjectIdea]:
    """Parse top-level Markdown bullets into normalized side-project ideas."""

    idea_path = Path(path).expanduser()
    text = idea_path.read_text(encoding="utf-8")
    entries: list[SideProjectIdea] = []
    section = "Unsectioned"

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            section = heading.group("title").strip()
            continue

        bullet = BULLET_RE.match(line)
        if not bullet or GUIDANCE_SECTION_RE.search(section):
            continue

        raw = bullet.group("body").strip()
        priority = "normal"
        priority_match = PRIORITY_RE.match(raw)
        if priority_match:
            priority = priority_match.group("priority").lower()

        urls = URL_RE.findall(raw)
        fields = _extract_fields(raw)
        title = _title_from_body(raw, urls)
        score = _score_idea(priority, fields)
        entries.append(
            SideProjectIdea(
                title=title,
                section=section,
                priority=priority,
                effort=fields.get("effort"),
                leverage=fields.get("leverage"),
                reuse=fields.get("reuse"),
                next_action=fields.get("next"),
                status=fields.get("status"),
                audience=fields.get("audience"),
                urls=urls,
                raw=raw,
                score=score,
            )
        )

    return entries


def rank_ideas(entries: Iterable[SideProjectIdea]) -> list[SideProjectIdea]:
    """Return ideas in deterministic review order."""

    return sorted(entries, key=lambda entry: (-entry.score, entry.section.lower(), entry.title.lower()))


def _format_idea(entry: SideProjectIdea) -> str:
    bits = [f"score {entry.score}", entry.section]
    if entry.priority != "normal":
        bits.append(f"priority {entry.priority}")
    if entry.effort:
        bits.append(f"effort {entry.effort}")
    if entry.leverage:
        bits.append(f"leverage {entry.leverage}")
    if entry.reuse:
        bits.append(f"reuse {entry.reuse}")
    if entry.status:
        bits.append(f"status {entry.status}")
    next_label = f" — next: {entry.next_action}" if entry.next_action else ""
    url_label = f" — {entry.urls[0]}" if entry.urls else ""
    return f"- {entry.title} ({'; '.join(bits)}){next_label}{url_label}"


def build_markdown_brief(entries: Iterable[SideProjectIdea], *, limit: int = 5) -> str:
    """Render a compact Markdown triage brief for manual review."""

    ranked = rank_ideas(entries)
    top = ranked[:limit]
    parking_lot = [entry for entry in ranked if entry.score < 3 or (entry.status and "blocked" in entry.status.lower())]
    missing_next = [entry for entry in ranked if not entry.next_action]

    lines = ["# Side Project Triage", "", f"Ideas: {len(ranked)}", "", "## Top picks"]
    if top:
        lines.extend(_format_idea(entry) for entry in top)
    else:
        lines.append("- No side-project ideas found.")

    lines.extend(["", "## Parking lot"])
    if parking_lot:
        lines.extend(_format_idea(entry) for entry in parking_lot)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing next actions"])
    if missing_next:
        for entry in missing_next:
            lines.append(f"- {entry.title} ({entry.section})")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def build_json_payload(entries: Iterable[SideProjectIdea], *, limit: int | None = None) -> dict[str, object]:
    ranked = rank_ideas(entries)
    if limit is not None:
        ranked = ranked[:limit]
    return {"count": len(ranked), "ideas": [asdict(entry) for entry in ranked]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ideas", help="Markdown side-project ideas path")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--limit", type=int, default=5, help="Maximum top picks to include")
    args = parser.parse_args(argv)

    entries = parse_ideas(args.ideas)
    if args.format == "json":
        print(json.dumps(build_json_payload(entries, limit=args.limit), indent=2, sort_keys=True))
    else:
        print(build_markdown_brief(entries, limit=args.limit), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke test
    raise SystemExit(main())
