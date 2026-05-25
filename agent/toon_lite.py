"""
TOON-lite: Compact context encoding for LLM system prompts.

Inspired by SochDB's TOON format (Tabular Object-Oriented Notation).
NOT about saving tokens — about fitting MORE useful information in the
SAME budget.  Every byte freed from format overhead goes to actual content.

Three encoding modes, ordered by compression ratio:

  toon_table()   — structured lists  (skills, tool lists)
  toon_records() — keyed entries     (memory, user profile)
  toon_kv()      — flat key-value    (config, env vars)

Design principles:
  - LLM-readable first, compression second (this IS for LLMs to read)
  - Header declares schema once; rows carry only data
  - Tokenizer-friendly separators (comma, newline, colon)
  - No structural noise: no XML tags, no JSON braces, no markdown bullets
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


# ── Table mode ──────────────────────────────────────────────────────────
# For structured lists like skills, tools, search results.
#
#   skills[20]{name,desc}:
#   agent-safety,Safety compliance checklist for self-evolving agents
#   apple-notes,Manage Apple Notes via memo CLI
#   ...
#
# vs current XML:
#   <available_skills>
#     category:
#       - name: description
#   </available_skills>

def toon_table(
    rows: Sequence[Tuple[str, ...]],
    fields: Sequence[str],
    name: str = "table",
    *,
    max_desc_len: int = 120,
) -> str:
    """Encode a list of rows as a compact table.

    Args:
        rows: List of (col1, col2, ...) tuples.
        fields: Column names.
        name: Table name for the header.
        max_desc_len: Truncate description-like fields (last column) to this length.

    Returns:
        Compact TOON string. Empty string if no rows.
    """
    if not rows:
        return ""

    ncols = len(fields)
    header = f"{name}[{len(rows)}]{{{','.join(fields)}}}:"

    lines = [header]
    for row in rows:
        # Pad short rows
        padded = list(row) + [""] * (ncols - len(row))
        # Truncate last field if it's a description
        if max_desc_len > 0 and padded:
            padded[-1] = _truncate_desc(padded[-1], max_desc_len)
        # Escape commas in values
        escaped = [_escape_csv(v) for v in padded[:ncols]]
        lines.append(",".join(escaped))

    return "\n".join(lines)


# ── Records mode ────────────────────────────────────────────────────────
# For memory entries, user profile, and other keyed blocks.
#
#   memory[12]{fact}:
#   • output_dir=~/Documents/hermes-output/
#   • dashscope: provider=alibaba model=qwen-vl-max | bug=L4472 drops base_url
#   ...
#
# vs current:
#   ══════════════════════════════════════════════
#   MEMORY (your personal notes) [97% — 2,155/2,200 chars]
#   ══════════════════════════════════════════════
#   Content line one...
#   §
#   Content line two...

def toon_records(
    entries: Sequence[str],
    name: str = "entries",
    *,
    bullet: str = "•",
    usage_pct: int = 0,
    usage_cur: int = 0,
    usage_max: int = 0,
) -> str:
    """Encode free-text entries as a compact record block.

    Args:
        entries: List of text entries.
        name: Block name for the header.
        bullet: Bullet marker for each entry.
        usage_pct, usage_cur, usage_max: Usage stats for the header line.

    Returns:
        Compact TOON string. Empty string if no entries.
    """
    if not entries:
        return ""

    # Header with optional usage stats
    if usage_max > 0:
        header = f"{name}[{len(entries)}]{bullet} [{usage_pct}% {usage_cur}/{usage_max}]"
    else:
        header = f"{name}[{len(entries)}]{bullet}"

    lines = [header]
    for entry in entries:
        # Each entry on its own line with bullet, no blank lines between
        text = entry.strip().replace("\n", " ")
        lines.append(f"{bullet} {text}")

    return "\n".join(lines)


# ── KV mode ─────────────────────────────────────────────────────────────
# For flat key-value pairs like environment info, config snippets.
#
#   env[5]{k,v}:
#   OS=macOS 12.7.6
#   HOME=/Users/macbook
#   CWD=/Users/macbook/.hermes/hermes-agent
#   ...
#
# vs current prose:
#   Host: macOS (12.7.6)
#   User home directory: /Users/macbook
#   Current working directory: /Users/macbook/.hermes/hermes-agent

def toon_kv(
    pairs: Sequence[Tuple[str, str]],
    name: str = "kv",
) -> str:
    """Encode key-value pairs compactly.

    Args:
        pairs: List of (key, value) tuples.
        name: Block name for the header.

    Returns:
        Compact TOON string. Empty string if no pairs.
    """
    if not pairs:
        return ""

    header = f"{name}[{len(pairs)}]{{k,v}}:"
    lines = [header]
    for k, v in pairs:
        lines.append(f"{k}={_escape_kv(v)}")

    return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────────

def _escape_csv(s: str) -> str:
    """Escape a value for CSV-style output. Quote if it contains comma or newline."""
    s = s.replace("\n", " ").replace("\r", "")
    if "," in s or '"' in s:
        s = s.replace('"', '""')
        return f'"{s}"'
    return s


def _escape_kv(s: str) -> str:
    """Escape a value for KV output. Replace newlines with spaces."""
    return s.replace("\n", " ").replace("\r", "")


def _truncate_desc(s: str, max_len: int) -> str:
    """Truncate a description to max_len, preserving word boundaries."""
    if len(s) <= max_len:
        return s
    # Try to break at a word boundary
    truncated = s[:max_len]
    # Find last space within limit
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        return truncated[:last_space] + "…"
    return truncated + "…"


# ── Format a skill index block ──────────────────────────────────────────
# Takes the same data structure used by build_skills_system_prompt()
# and produces a compact TOON representation.

def format_skills_index_toon(
    skills_by_category: Dict[str, List[Tuple[str, str]]],
    category_descriptions: Dict[str, str] = None,
) -> str:
    """Format a skills index as a TOON-lite block.

    Args:
        skills_by_category: {category: [(skill_name, description), ...]}
        category_descriptions: {category: description}

    Returns:
        Compact TOON string suitable for system prompt injection.
        Empty string if no skills.
    """
    if not skills_by_category:
        return ""

    category_descriptions = category_descriptions or {}
    blocks: List[str] = []

    for category in sorted(skills_by_category.keys()):
        skills = skills_by_category[category]
        if not skills:
            continue

        cat_desc = category_descriptions.get(category, "")
        cat_header = f"{category}: {cat_desc}" if cat_desc else f"{category}:"
        cat_lines = [cat_header]

        for name, desc in sorted(set(skills), key=lambda x: x[0]):
            clean_desc = _truncate_desc(desc.strip(), 100) if desc else ""
            cat_lines.append(f"  - {name}: {clean_desc}")

        blocks.append("\n".join(cat_lines))

    return "\n".join(blocks)


# ── Format a memory/user block ──────────────────────────────────────────

def format_memory_block_toon(
    entries: List[str],
    target: str,  # "memory" or "user"
    usage_pct: int = 0,
    usage_cur: int = 0,
    usage_max: int = 0,
) -> str:
    """Format a memory or user profile block in compact TOON style.

    Unlike the full toon_records(), this keeps the category grouping
    that makes memory blocks skimmable for LLMs, while removing the
    heavy ═══════ decorative separators.

    Args:
        entries: Memory or profile entries.
        target: "memory" or "user".
        usage_pct, usage_cur, usage_max: Usage stats.

    Returns:
        Compact block string.
    """
    if not entries:
        return ""

    if target == "user":
        label = "USER PROFILE"
        subtitle = "(who the user is)"
    else:
        label = "MEMORY"
        subtitle = "(your personal notes)"

    if usage_max > 0:
        header = f"{label} {subtitle} [{usage_pct}% — {usage_cur:,}/{usage_max:,} chars]"
    else:
        header = f"{label} {subtitle}"

    lines = [header]
    for entry in entries:
        text = entry.strip().replace("\n", " ")
        lines.append(f"§ {text}")

    return "\n".join(lines)
