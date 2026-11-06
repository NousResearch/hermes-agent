"""Anchor ledger for compaction-v2 lean mode: exact identifiers, harvested without an LLM.

The ledger exists so needle facts (SHAs, ids, paths, error strings) cannot be paraphrased away
by the summary model. It also doubles as the session_search map for the compacted region. This
sibling owns the whole topic — pattern table, harvest scan, budgeted render.

Kept out of the facade on purpose (root AGENTS.md, facade + sibling layout): the compressor is
already over the 2,000-line target, so behaviour belongs beside it, never appended to it.
"""

import re
from typing import Any

# Anchor ledger: mechanically harvested exact identifiers, no LLM, so needle facts
# (SHAs, ids, error strings) cannot be paraphrased away; also a session_search map.
_LEAN_ANCHOR_HEADING = "## Anchor Index (mechanically extracted, exact)"
_LEAN_ANCHOR_BUDGET_CHARS = 7_000
_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # Exact, cheap-to-emit identifier classes first: a greedy prose section must never
    # starve them out of the budget (see _truncate_anchor_section).
    ("PRs/issues", re.compile(r"#\d{3,6}\b"), 120),
    ("session ids", re.compile(r"\b\d{8}_\d{6}_[0-9a-f]{8}\b"), 40),
    ("todo ids", re.compile(r"\[\d{6}\]"), 40),
    ("commits", re.compile(r"\b[0-9a-f]{9,40}\b"), 40),
    ("branches", re.compile(r"\b(?:fix|feat|docs|refactor|chore|salvage|ent)/[A-Za-z0-9._/-]{3,60}"), 40),
    ("handles", re.compile(r"@[A-Za-z0-9-]{3,30}\b"), 40),
    ("urls", re.compile(r"https?://[^\s)\"']{10,110}"), 30),
    # Filename segment: path characters incl. CJK names, but never quoting/CJK list punctuation —
    # unpunctuated prose runs ("`a.py`：`b.py`、`c.py`") otherwise collapse into one unusable value;
    # the 120-char cap bounds any run that still slips through. Directory part stays \w so
    # non-ASCII directory names keep working.
    ("files", re.compile(r"\b[\w./-]+/[^\s/`'\"\u3001\uff0c\u3002\uff1a\uff1b\uff01\uff1f\uff08\uff09\u3010\u3011\{\}\[\]<>*|]{1,120}\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|php|sql|md|rst|csv|tsv|xlsx|xls|ipynb|html|css|scss|vue|yaml|yml|json|jsonl|toml|ini|cfg|sh|db|sqlite|log)\b"), 80),
    ("errors", re.compile(r"\b(?:[A-Z][a-zA-Z]*Error|Exception|ENOSPC|EACCES|SIGKILL|Traceback)\b[^\n]{0,90}"), 40),
]
_ANCHOR_NOISE = frozenset({
    "@teknium", "@teknium1",  # session owner, in every transcript
})


def _truncate_anchor_section(line: str, room: int) -> str:
    """Trim a section line to ``room`` characters on a value boundary (never mid-identifier).

    Returns ``""`` when nothing fits, which the caller turns into the bare ``label:`` — a
    scanned class that lost its values must not look like a class that never matched. A
    section that overflows must cost its own tail, not every section behind it — the previous
    whole-loop ``break`` silently dropped session ids, todo ids, urls and error strings
    whenever the (greedy) file-path list ran long.
    """
    if room <= 0:
        return ""
    if len(line) <= room:
        return line
    cut = line.rfind(", ", 0, room)
    return line[:cut] if cut > 0 else ""


def _anchor_harvest_text(turns: list[dict[str, Any]]) -> str:
    """Compacted-region text the anchor ledger harvests from: message content **plus tool-call arguments**.

    Paths, commands and URLs a session depends on routinely reach the transcript only as
    tool-call arguments (``read_file(path=...)``, ``terminal(command=...)``); a content-only
    scan cannot see them, so exactly the identifiers the ledger exists to preserve were the
    ones most likely to be paraphrased away. Arguments are harvested raw, never evaluated.

    User rows go through ``_typed_user_text`` first: the inline body of a ``/skill`` turn is
    not the user's words, and harvesting it would index the skill's own text as anchors.
    """
    # Late import: the facade imports this sibling at module level, so a module-level import
    # back into it would be a cycle (root AGENTS.md, facade + sibling layout).
    from agent.context_compressor import _extract_tool_call_name_and_args, _typed_user_text

    chunks: list[str] = []
    for msg in turns:
        content = msg.get("content")
        if isinstance(content, str) and content:
            chunks.append(_typed_user_text(content) if msg.get("role") == "user" else content)
        for tool_call in msg.get("tool_calls") or []:
            _name, args = _extract_tool_call_name_and_args(tool_call)
            if args:
                chunks.append(args)
    return "\n".join(chunks)


def _build_anchor_index(turns: list[dict[str, Any]]) -> str:
    """Regex-harvest exact identifiers from the compacted region (LLM-free); per-category caps, most-frequent first."""
    text = _anchor_harvest_text(turns)
    if not text:
        return ""
    sections: list[str] = []
    used = 0
    for label, pattern, cap in _ANCHOR_PATTERNS:
        counts: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        for n, m in enumerate(pattern.finditer(text)):
            val = m.group(0).strip().rstrip(".,;:")
            if val.lower() in _ANCHOR_NOISE:
                continue
            counts[val] = counts.get(val, 0) + 1
            last_seen[val] = n
        if not counts:
            continue
        ranked = sorted(counts, key=lambda v: (-counts[v], -last_seen[v]))[:cap]
        line = f"{label}: " + ", ".join(f"{v}(x{counts[v]})" if counts[v] > 1 else v for v in ranked)
        room = _LEAN_ANCHOR_BUDGET_CHARS - used
        line = _truncate_anchor_section(line, room)
        if not line:
            # Cutting a section's values must not delete its label: a missing label reads as
            # "this class was absent from the region", which is a different and wrong claim.
            line = f"{label}:" if len(label) + 1 <= room else ""
        if not line:
            continue
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
