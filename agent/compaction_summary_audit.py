"""Post-generation audit of a compaction summary — LLM-free, bounded, redaction-safe.

The summarizer prompt orders exact-identifier preservation, and lean mode appends a
mechanical anchor index, but neither CHECKS the generated summary: an identifier
paraphrased away in the narrative that also fell past an anchor category cap left
the compaction boundary silently (OpenClaw's compaction carries the same strict
identifier policy; deepagents re-audits its evicted record before accepting a
summary). This module closes that seam: harvest the exact identifiers from the
compacted region, diff them against the finished summary, and mechanically append
the genuinely missing ones — one bounded pass, no extra LLM call (the one-summary-
call-per-attempt architecture, #96603, is preserved).

Comparison is redaction-safe: an identifier whose redacted form differs from its
raw form is intentionally absent from the summary, so it is excluded rather than
"repaired" back across the boundary.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# Identifier classes the lean anchor index does not cover: dashed UUIDs and
# host:port endpoints (its "commits" class is undashed hex only, its "urls" class
# requires a scheme).
EXTRA_AUDIT_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("uuids", re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), 40),
    ("hosts", re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[a-zA-Z]{2,}:\d{2,5}\b"), 30),
    ("hexids", re.compile(r"\b(?:0x)?[0-9a-f]{24,64}\b"), 30),
]

# How many distinct identifiers the audit tracks per compaction (bounded scan of a
# region that can be megabytes); ranked by frequency like the anchor index.
_HARVEST_CAP = 400
# The corrective section stays small: it is an appendix to a summary, not a second
# anchor index. The highest-frequency drops are the ones the model is most likely
# to need verbatim.
_REPAIR_MAX_IDENTS = 24
_REPAIR_MAX_CHARS = 800

REPAIR_HEADING = "## Recovered Identifiers (verbatim audit)"


def harvest_identifiers(
    region_text: str, patterns: Iterable[tuple[str, re.Pattern[str], int]]
) -> list[tuple[str, int]]:
    """``(identifier, frequency)`` ranked most-frequent-first, deduplicated."""
    counts: dict[str, int] = {}
    for _label, pattern, _cap in patterns:
        for match in pattern.finditer(region_text):
            value = match.group(0).strip().rstrip(".,;:")
            if value:
                counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:_HARVEST_CAP]


def find_missing_identifiers(
    region_text: str,
    summary_text: str,
    patterns: Sequence[tuple[str, re.Pattern[str], int]],
) -> list[str]:
    """Identifiers present in the region but absent from the summary, verbatim.

    An identifier that the boundary redactor rewrites is excluded: its absence is
    policy, not summarizer loss, and appending it would undo redaction.
    """
    from agent.context_compressor import _redact_compaction_text

    missing: list[str] = []
    for value, _freq in harvest_identifiers(region_text, patterns):
        if value in summary_text:
            continue
        if _redact_compaction_text(value) != value:
            continue
        missing.append(value)
    return missing


def audit_and_repair_summary(
    summary: str,
    region_text: str,
    patterns: Sequence[tuple[str, re.Pattern[str], int]],
    *,
    pending_ask: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Audit ``summary`` against the compacted region; append a bounded repair line
    for identifiers the summarizer dropped; return ``(summary, telemetry)``.

    ``pending_ask`` (the grounded latest-user-ask snapshot, when one exists) is only
    verified, never rewritten: task grounding already forces it mechanically, so the
    audit reports ``pending_ask_grounded`` for the eval telemetry stream.
    """
    missing = find_missing_identifiers(region_text, summary, patterns)
    repaired: list[str] = []
    used = 0
    for value in missing:
        if len(repaired) >= _REPAIR_MAX_IDENTS or used + len(value) > _REPAIR_MAX_CHARS:
            break
        repaired.append(value)
        used += len(value) + 2
    telemetry: dict[str, Any] = {
        "audit_identifiers_missing": len(missing),
        "audit_repaired": len(repaired),
        "pending_ask_grounded": (pending_ask in summary) if pending_ask else None,
    }
    if not repaired:
        return summary, telemetry
    return (
        summary
        + "\n\n"
        + REPAIR_HEADING
        + "\n"
        + ", ".join(repaired)
        + "\n(Exact identifiers the summary paraphrased away, restored verbatim.)"
    ), telemetry
