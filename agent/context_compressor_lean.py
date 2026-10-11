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

from typing import Any

_BULLET_PREFIX = "> "


def strip_lean_sections(summary: str, headings: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """Split *summary* into ``(narrative, {heading: section})`` for the given deterministic headings.

    A section runs from its heading to the next known heading (a quoted user message may itself
    contain a ``##`` line, so only the known headings bound a section). An echoed section is
    discarded in favour of the rebuild, which is authoritative.
    """
    hits = sorted((summary.find(h), h) for h in headings if h in summary)
    if not hits:
        return summary, {}
    found = {
        heading: summary[start: hits[i + 1][0] if i + 1 < len(hits) else len(summary)]
        for i, (start, heading) in enumerate(hits)
    }
    return summary[: hits[0][0]], found


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
        _LEAN_ANCHOR_HEADING,
        _LEAN_RECOVERY_HEADING,
        _LEAN_USER_MESSAGES_HEADING,
        _build_anchor_index,
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
    # The previous index is harvested first so a tie (equal mention count) goes to the newest region,
    # matching the quote ordering; the counts are genuine mentions across the compacted regions.
    harvest: list[dict[str, Any]] = []
    if carried.get(_LEAN_ANCHOR_HEADING):
        harvest.append({"role": "tool", "content": carried[_LEAN_ANCHOR_HEADING]})
    harvest += list(turns)
    return narrative.rstrip() + "".join((
        _redact_compaction_text(_build_anchor_index(harvest)),
        _redact_compaction_text(_build_verbatim_user_section([{"role": "user", "content": q} for q in quotes])),
        _build_recovery_footer(session_id, len(turns)),
    ))
