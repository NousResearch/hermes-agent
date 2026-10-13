"""Lean compaction sections are rebuilt every pass: model-written copies are dropped, and the
quotes and anchors Hermes wrote into the previous summary carry forward under the same budgets."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# A lean section runs from its heading to the next ``## `` heading or to a blank line that does
# not open another ``> `` quote (user quotes are separated by blank lines).
_SECTION_END_RE = re.compile(r"\n(?=## )|\n\n(?!> )")
_ANCHOR_COUNT_RE = re.compile(r"\(x\d+\)$")


def _section_spans(text: str, heading: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every copy of ``heading``'s section whose heading is a whole line."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (start := text.find(heading, pos)) >= 0:
        pos = start + len(heading)
        if (start and text[start - 1] != "\n") or text[pos:pos + 1] not in ("", "\n"):
            continue  # e.g. ``## Context Recovery Plan`` is the model's own section
        match = _SECTION_END_RE.search(text, pos)
        spans.append((start, match.start() if match else len(text)))
        pos = spans[-1][1]
    return spans


def strip_lean_section(summary: str, heading: str) -> str:
    """Drop every copy of a lean section from ``summary``; the caller appends the fresh one."""
    for start, end in reversed(_section_spans(summary, heading)):
        summary = summary[:start].rstrip("\n") + summary[end:]
    return summary


def previous_user_quotes(previous_summary: str, heading: str) -> list[str]:
    """Quote texts of the previous summary's verbatim user section, in their existing order. Only
    Hermes-format entries (every line ``"> "``-prefixed, the last one may close with the section
    note) are parsed; any other entry is skipped whole."""
    from agent.compaction_user_quotes import USER_QUOTES_NOTE

    quotes: list[str] = []
    for start, end in _section_spans(previous_summary or "", heading):
        for entry in previous_summary[start + len(heading):end].strip("\n").split("\n\n"):
            lines = entry.removesuffix("\n" + USER_QUOTES_NOTE).split("\n")
            if all(line.startswith("> ") for line in lines):
                quotes.append("\n".join(line[2:] for line in lines))
    return quotes


def with_pre_prune_tool_bodies(
    turns: list[dict[str, Any]], source: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Anchor-index input only: tool rows get the body ``compress()`` received (matched by
    ``tool_call_id``), so an identifier that only appeared in a pruned result still reaches the
    index. The summary prompt and the returned stubs keep the pruned rows."""
    rows = [msg for msg in source or () if msg.get("role") == "tool" and msg.get("tool_call_id")]
    reused = {cid for cid, n in Counter(msg["tool_call_id"] for msg in rows).items() if n > 1}
    bodies = {  # a reused id (resumed histories) is ambiguous: those rows keep their own content
        msg["tool_call_id"]: msg["content"] for msg in rows
        if msg["tool_call_id"] not in reused and isinstance(msg.get("content"), str)
    }
    return [
        {**msg, "content": bodies[msg["tool_call_id"]]}
        if msg.get("role") == "tool" and msg.get("tool_call_id") in bodies else msg
        for msg in turns
    ]


def _anchor_lines(section: str) -> tuple[dict[str, list[str]], str]:
    """Split an Anchor Index section into per-label tokens and its closing note."""
    from agent.context_compressor import _ANCHOR_PATTERNS

    patterns = {label: pattern for label, pattern, _ in _ANCHOR_PATTERNS}
    tokens: dict[str, list[str]] = {}
    note = ""
    for line in section.split("\n"):
        label, sep, rest = line.partition(": ")
        if not sep or label not in patterns:
            note = note or (line if line.startswith("(") else "")
            continue
        # Error values may contain ", ": split only where the next token starts a new match.
        cuts = [0] + [m.end() for m in re.finditer(", ", rest) if patterns[label].match(rest, m.end())]
        bounds = zip(cuts, [c - 2 for c in cuts[1:]] + [len(rest)])
        tokens.setdefault(label, []).extend(rest[a:b] for a, b in bounds)
    return tokens, note


def lean_anchor_index(
    turns: list[dict[str, Any]], source: list[dict[str, Any]] | None, previous_summary: str | None,
) -> str:
    """The Anchor Index for this pass: harvested from the pre-prune region, then the previous
    summary's anchors carried after it."""
    from agent.context_compressor import _build_anchor_index

    return carry_anchor_index(_build_anchor_index(with_pre_prune_tool_bodies(turns, source)), previous_summary or "")


def carry_anchor_index(fresh: str, previous_summary: str) -> str:
    """Append the previous Anchor Index's entries after the fresh ones, deduplicated, inside the
    per-category caps and the section budget. Unchanged when there is nothing to carry."""
    from agent.context_compressor import _ANCHOR_PATTERNS, _LEAN_ANCHOR_BUDGET_CHARS, _LEAN_ANCHOR_HEADING

    spans = _section_spans(previous_summary or "", _LEAN_ANCHOR_HEADING)
    carried, previous_note = _anchor_lines("\n".join(previous_summary[a:b] for a, b in spans))
    if not carried:
        return fresh
    current, note = _anchor_lines(fresh)
    lines: list[str] = []
    used = 0
    for label, _, cap in _ANCHOR_PATTERNS:
        merged: dict[str, str] = {}
        for token in current.get(label, []) + carried.get(label, []):
            merged.setdefault(_ANCHOR_COUNT_RE.sub("", token), token)
        if not merged:
            continue
        line = f"{label}: " + ", ".join(list(merged.values())[:cap])
        if used + len(line) > _LEAN_ANCHOR_BUDGET_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return "\n\n" + _LEAN_ANCHOR_HEADING + "\n" + "\n".join(lines) + "\n" + (note or previous_note)
