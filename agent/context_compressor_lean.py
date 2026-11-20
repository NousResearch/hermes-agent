"""#136540: lean-mode deterministic summary sections accumulate across compactions.

``ContextCompressor._augment_summary_lean`` used to append its three deterministic sections only when
the heading was missing from the summary it was handed. Both things a summariser does with the
PREVIOUS SUMMARY it is given ("PRESERVE all existing information") lost real information: echoing it
back kept the headings present, so the region being compacted never got its own section and its user
words survived only as the model's paraphrase; rewriting it dropped the sections and with them the
earlier region's quoted words. Both sections were also free to compound, one stale copy per
compaction.

The sections are rebuilt here — narrative first, then anchors, quotes and footer — from the region
being compacted plus the sections of the previous augmented summary, so every compacted region stays
represented exactly once and the newest region wins the char budget.
"""

from __future__ import annotations

import re
from typing import Any

_BULLET_PREFIX = "> "


# Anchor ledger: mechanically harvested exact identifiers, no LLM, so needle facts
# (SHAs, ids, error strings) cannot be paraphrased away; also a session_search map.
_LEAN_ANCHOR_HEADING = "## Anchor Index (mechanically extracted, exact)"
_LEAN_ANCHOR_BUDGET_CHARS = 7_000
_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("PRs/issues", re.compile(r"#\d{3,6}\b"), 120),
    ("commits", re.compile(r"\b[0-9a-f]{9,40}\b"), 40),
    ("branches", re.compile(r"\b(?:fix|feat|docs|refactor|chore|salvage|ent)/[A-Za-z0-9._/-]{3,60}"), 40),
    ("files", re.compile(r"\b[\w./-]+/[\w.-]+\.(?:py|ts|tsx|js|rs|md|yaml|yml|json|toml|sh)\b"), 80),
    ("errors", re.compile(r"\b(?:[A-Z][a-zA-Z]*Error|Exception|ENOSPC|EACCES|SIGKILL|Traceback)\b[^\n]{0,90}"), 40),
    ("handles", re.compile(r"@[A-Za-z0-9-]{3,30}\b"), 40),
    ("urls", re.compile(r"https?://[^\s)\"']{10,110}"), 30),
]
_ANCHOR_NOISE = frozenset({
    "@teknium", "@teknium1",  # session owner, in every transcript
})


_ANCHOR_LINE_RE = re.compile(r"^(?P<label>[A-Za-z][^:\n]{0,39}): (?P<body>.+)$")
_ANCHOR_ITEM_COUNT_RE = re.compile(r"^(?P<val>.+?)\(x(?P<n>\d+)\)$")
# Separator between entries inside one category line. A middle dot cannot occur inside an
# identifier or an error string, so a rendered index parses back unambiguously (", " could not:
# two adjacent error entries merged into one and were cut as a single over-long value).
_ANCHOR_ITEM_SEP = " \u00b7 "


def _parse_anchor_index(section: str) -> dict[str, dict[str, int]]:
    """Parse a previously rendered Anchor Index back into ``{label: {value: occurrence_count}}``."""
    parsed: dict[str, dict[str, int]] = {}
    if not section:
        return parsed
    for line in section.splitlines()[1:]:
        if line.startswith("(Exact identifiers"):
            break
        match = _ANCHOR_LINE_RE.match(line)
        if not match:
            continue
        counts = parsed.setdefault(match.group("label").strip(), {})
        for raw in match.group("body").split(_ANCHOR_ITEM_SEP):
            item = raw.strip().rstrip(".,;:")
            if not item:
                continue
            counted = _ANCHOR_ITEM_COUNT_RE.match(item)
            if counted:
                value = counted.group("val")
                counts[value] = counts.get(value, 0) + int(counted.group("n"))
            else:
                counts[item] = counts.get(item, 0) + 1
    return parsed


def _build_anchor_index(turns: list[dict[str, Any]], *, carried_section: str = "") -> str:
    """Regex-harvest exact identifiers from the compacted region (LLM-free); per-category caps, most-frequent first.

    ``carried_section`` is the previous augmented summary's Anchor Index. Its counts are merged
    numerically with the fresh harvest rather than re-harvested as text: re-harvesting counted a
    rendered ``value(xN)`` as a single mention (losing multiplicity) and let two adjacent error
    entries merge into one value before the cap was applied.
    """
    from agent.context_compressor import _typed_user_text

    contents = (_typed_user_text(m["content"]) if m.get("role") == "user" and isinstance(m.get("content"), str) else m.get("content") for m in turns)
    text = "\n".join(c for c in contents if isinstance(c, str) and c)
    carried = _parse_anchor_index(carried_section)
    if not text and not carried:
        return ""
    sections: list[str] = []
    used = 0
    for label, pattern, cap in _ANCHOR_PATTERNS:
        counts: dict[str, int] = dict(carried.get(label, {}))
        last_seen: dict[str, int] = {}
        fresh: dict[str, int] = {}
        for n, m in enumerate(pattern.finditer(text)):
            val = m.group(0).strip().rstrip(".,;:")
            if val.lower() in _ANCHOR_NOISE:
                continue
            fresh[val] = fresh.get(val, 0) + 1
            last_seen[val] = n
        for val, hits in fresh.items():
            counts[val] = counts.get(val, 0) + hits
        if not counts:
            continue
        # Genuine mention counts across every compacted region; ties go to the newest region.
        ranked = sorted(counts, key=lambda v: (-counts[v], -last_seen.get(v, -1)))[:cap]
        line = f"{label}: " + _ANCHOR_ITEM_SEP.join(f"{v}(x{counts[v]})" if counts[v] > 1 else v for v in ranked)
        if used + len(line) > _LEAN_ANCHOR_BUDGET_CHARS:
            break
        sections.append(line)
        used += len(line)
    if not sections:
        return ""
    return (
        "\n\n" + _LEAN_ANCHOR_HEADING + "\n"
        + "\n".join(sections)
        + "\n(Exact identifiers from the compacted region — use these verbatim, "
        "and as session_search query anchors to recover their full context.)"
    )


def strip_lean_sections(summary: str, headings: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """Split *summary* into ``(narrative, {heading: section})`` for the given deterministic headings.

    A heading only counts when it is its own unquoted line — a substring match mistook the model's
    own ``## Context Recovery Plan`` for ``## Context Recovery`` and deleted its body. A section runs
    from its heading to the next unquoted level-2 heading (or end of summary), and only the section
    spans are removed from the narrative, so model prose written after an echoed appendix survives.
    Quoted ``> ## …`` lines stay quote content either way.
    """
    spans = _unquoted_section_spans(summary, headings)
    if not spans:
        return summary, {}
    found: dict[str, str] = {}
    narrative: list[str] = []
    cursor = 0
    for start, end, heading in spans:
        narrative.append(summary[cursor:start])
        found[heading] = summary[start:end]
        cursor = end
    narrative.append(summary[cursor:])
    return "".join(narrative), found


def _unquoted_section_spans(summary: str, headings: tuple[str, ...]) -> list[tuple[int, int, str]]:
    """``(start, end, heading)`` for each deterministic heading that stands alone on an unquoted line."""
    spans: list[tuple[int, int, str]] = []
    lines = summary.splitlines(keepends=True)
    offset = 0
    for index, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        if line in headings:
            end = len(summary)
            probe = offset + len(raw)
            for following in lines[index + 1:]:
                if following.rstrip("\r\n").startswith("## "):
                    end = probe
                    break
                probe += len(following)
            spans.append((offset, end, line))
        offset += len(raw)
    return spans


def quoted_user_texts(section: str) -> list[str]:
    """Raw texts of a rendered verbatim-user section, newest-first (``> `` prefixes removed)."""
    quotes: list[str] = []
    for chunk in section.split("\n", 1)[-1].split("\n\n"):
        lines = chunk.split("\n")
        first = next((i for i, line in enumerate(lines) if line.startswith(_BULLET_PREFIX)), None)
        if first is None:
            continue
        text_lines: list[str] = []
        for line in lines[first:]:
            if not line.startswith(_BULLET_PREFIX):
                break
            text_lines.append(line[len(_BULLET_PREFIX):])
        text = "\n".join(text_lines).strip()
        if text:
            quotes.append(text)
    return quotes


def _unique(texts: list[str], *more: list[str]) -> list[str]:
    """Concatenated texts, order preserved, exact duplicates dropped (newest region first)."""
    seen: set[str] = set()
    out: list[str] = []
    for text in (*texts, *[t for group in more for t in group]):
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def rebuild_lean_sections(
    summary: str, turns: list[dict[str, Any]], *, previous_summary: str, session_id: str,
) -> str:
    """Narrative plus the deterministic sections, rebuilt from this region and the previous summary."""
    from agent.context_compressor import (
        _LEAN_RECOVERY_HEADING,
        _LEAN_USER_MESSAGES_HEADING,
        _build_recovery_footer,
        _build_verbatim_user_section,
        _redact_compaction_text,
    )

    headings = (_LEAN_ANCHOR_HEADING, _LEAN_USER_MESSAGES_HEADING, _LEAN_RECOVERY_HEADING)
    narrative, _echoed = strip_lean_sections(summary, headings)
    _, carried = strip_lean_sections(previous_summary or "", headings)
    quotes = _unique(
        quoted_user_texts(_build_verbatim_user_section(turns)),
        quoted_user_texts(carried.get(_LEAN_USER_MESSAGES_HEADING, "")),
    )
    # The previous index is parsed back into counts and merged numerically with this region's
    # harvest, so a value's multiplicity survives compaction instead of decaying to one per pass.
    harvested = _build_anchor_index(list(turns), carried_section=carried.get(_LEAN_ANCHOR_HEADING, ""))
    return narrative.rstrip() + "".join((
        _redact_compaction_text(harvested),
        _redact_compaction_text(_build_verbatim_user_section([{"role": "user", "content": q} for q in quotes])),
        _build_recovery_footer(session_id, len(turns)),
    ))
