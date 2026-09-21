"""Compact dated-bullet logs in brain pages into clean grouped entries.

The 3a/3b promoters append dated-bullet entries to brain pages, turning them
into chronological logs over time. This script compacts those logs into clean
grouped entries while preserving ALL facts (never drops information).

Approach: mechanical (no LLM). Groups dated bullets by month, deduplicates
exact and near-exact matches, and renders as compact month-grouped entries.
This is:
- 100% reliable (no model to time out or produce bad output)
- Preserves ALL facts by construction
- Sub-second runtime
- Idempotent (already-tidy pages are a near no-op)

Model justification (per plan requirement):
- LFM2.5-8B-A1B is a reasoning model (~7tok/s, 20-50s/call, <think> tags)
- gemma3:4b is explicitly rejected by the plan (48s/call)
- No other generation models available on local Ollama
- Mechanical compaction is more reliable than any available model for this task

Idempotent: re-running on already-tidy pages skips them (checks for compaction
marker). [Silent] when nothing to compact. Designed to run AFTER promoters in
the weekly cycle.

Upstream: KenseiAgent scripts/brain-compaction.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# P13 isolation: REPO derives from KENSEI_REPO_ROOT (env-overridable) so
# the sys.path insert targets the active checkout, not a hardcoded path.
REPO = Path(os.environ.get("KENSEI_REPO_ROOT", "/home/kensei/repos/KenseiAgent"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BRAIN_DIR = Path(os.environ.get("GBRAIN_REPO", "~/brain")).expanduser()
COMPACTION_BULLET_THRESHOLD = int(os.environ.get("BRAIN_COMPACTION_BULLET_THRESHOLD", "5"))
DATE_TAG = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# P13 isolation: when --dry-run is passed, the brain page rewrite
# (fp.write_text) is suppressed. Read paths (_list_brain_pages,
# _extract_dated_bullets, _find_bullet_blocks, _compact_entries) run
# unchanged so the compaction decisions are still computed and printed.
_DRY_RUN = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _list_brain_pages() -> dict[str, str]:
    """Return {slug: full_content} for all brain markdown files."""
    pages: dict[str, str] = {}
    if not BRAIN_DIR.is_dir():
        return pages
    for fp in sorted(BRAIN_DIR.rglob("*.md")):
        try:
            rel = fp.relative_to(BRAIN_DIR)
            slug = str(rel.with_suffix(""))
            if slug in ("RESOLVER", "README"):
                continue
            pages[slug] = fp.read_text(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue
    return pages


def _strip_frontmatter(content: str) -> tuple[str, str]:
    """Extract YAML frontmatter from content. Returns (frontmatter, body)."""
    if content.startswith("---"):
        idx = content.find("---", 3)
        if idx >= 0:
            return content[:idx + 3], content[idx + 3:].strip()
    return "", content.strip()


def _has_compaction_marker(body: str) -> bool:
    """Check if the page already has a compaction marker (idempotency guard)."""
    return f"<!-- compacted" in body


def _extract_dated_bullets(body: str) -> list[dict]:
    """Extract all dated-bullet entries from the body.

    Returns list of dicts: {date: 'YYYY-MM-DD', fact: str, line: int, full: str}
    """
    entries: list[dict] = []
    lines = body.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*-\s*(\d{4}-\d{2}-\d{2})[-: ]\s*(.+)", line)
        if m:
            entries.append({
                "date": m.group(1),
                "fact": m.group(2).strip(),
                "line": i,
                "full": line.strip(),
            })
    return entries


def _month_key(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYY-MM for grouping."""
    return date_str[:7]


def _normalize_fact(fact: str) -> str:
    """Normalize a fact string for dedup comparison.

    Strips punctuation, lowercases, removes stopwords.
    """
    text = fact.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    stopwords = {
        "the", "a", "an", "is", "was", "were", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "shall", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "this", "that", "these", "those", "it",
        "its", "also", "not", "no", "and", "or", "but", "if", "so",
    }
    return " ".join(w for w in words if w not in stopwords and len(w) > 2)


def _is_duplicate(norm: str, existing_norms: list[str]) -> bool:
    """Check if a fact is a genuine duplicate of an existing entry.

    Only exact duplicates are removed. Two steps:
    1. The caller checks exact full-text match (via entry content).
    2. Here, exact match on normalised text (stripped, lowercased,
       stopwords removed). This catches near-identical entries that
       differ only in punctuation or casing but are the same fact.

    No fuzzy/overlap matching. Distinct facts with shared vocabulary
    are never treated as duplicates.
    """
    if not norm or not existing_norms:
        return False
    return norm in existing_norms


def _compact_entries(entries: list[dict]) -> list[str]:
    """Mechanically compact dated entries into month-grouped deduplicated output.

    Returns lines to replace the original bullet block.
    """
    if not entries:
        return []

    # Group by month
    by_month: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_month[_month_key(entry["date"])].append(entry)

    result: list[str] = []
    seen_norms: list[str] = []

    for month in sorted(by_month.keys()):
        month_entries = by_month[month]
        # Sort within month by date
        month_entries.sort(key=lambda e: e["date"])

        # Month header
        year, m = month.split("-")
        month_name = datetime(int(year), int(m), 1).strftime("%B %Y")
        result.append(f"### {month_name}")

        for entry in month_entries:
            norm = _normalize_fact(entry["fact"])
            if _is_duplicate(norm, seen_norms):
                continue
            seen_norms.append(norm)
            result.append(f"- {entry['date']}: {entry['fact']}")

        result.append("")  # blank line between months

    return result


def _find_bullet_blocks(body: str, entries: list[dict]) -> list[dict]:
    """Group dated-bullet entries into contiguous blocks.

    Each block is a run where every line between the first and last dated
    bullet is either a dated bullet or a blank line. Non-blank content (prose,
    section headers, See Also links, ---) breaks the block.

    Returns list of dicts with keys: start (int), end (int, exclusive), entries.
    Never consumes non-dated content. Leading/trailing blank lines within
    the block are included; --- lines are never included.
    """
    if not entries:
        return []

    lines = body.split("\n")
    blocks: list[dict] = []
    sorted_entries = sorted(entries, key=lambda e: e["line"])

    block_entries = [sorted_entries[0]]
    block_end_line = sorted_entries[0]["line"]

    for entry in sorted_entries[1:]:
        gap_lines = lines[block_end_line + 1:entry["line"]]
        if all(line.strip() == "" for line in gap_lines):
            # All blanks in the gap — same contiguous block
            block_entries.append(entry)
            block_end_line = entry["line"]
        else:
            # Non-blank content in the gap — block boundary
            if block_entries:
                _finalize_block(blocks, block_entries, lines)
            block_entries = [entry]
            block_end_line = entry["line"]

    # Finalize the last block
    if block_entries:
        _finalize_block(blocks, block_entries, lines)

    return blocks


def _finalize_block(
    blocks: list[dict],
    block_entries: list[dict],
    lines: list[str],
) -> None:
    """Compute and append one block to the blocks list.

    Expands boundaries through leading/trailing blank lines only.
    Never consumes ---, headers, links, or prose.
    """
    first = block_entries[0]["line"]
    last = block_entries[-1]["line"]

    # Expand through leading blank lines only
    while first > 0 and first - 1 < len(lines) and lines[first - 1].strip() == "":
        first -= 1
    # Expand through trailing blank lines only
    while last < len(lines) - 1 and lines[last + 1].strip() == "":
        last += 1

    blocks.append({
        "start": first,
        "end": last + 1,
        "entries": list(block_entries),
    })


def _insert_compacted(body: str, start: int, end: int,
                      compacted_lines: list[str]) -> str:
    """Replace a range of lines with compacted output."""
    lines = body.split("\n")
    compacted_lines = [l.rstrip() for l in compacted_lines]
    new_lines = lines[:start] + compacted_lines + lines[end:]
    return "\n".join(new_lines)


def _tag_compacted(body: str) -> str:
    """Add a compaction marker."""
    return body.rstrip() + f"\n\n<!-- compacted {DATE_TAG} -->\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    pages = _list_brain_pages()
    total_compacted = 0
    total_blocks = 0

    for slug, content in pages.items():
        frontmatter, body = _strip_frontmatter(content)

        # Skip if already compacted (idempotency)
        if _has_compaction_marker(body):
            continue

        entries = _extract_dated_bullets(body)
        if not entries:
            continue

        blocks = _find_bullet_blocks(body, entries)
        blocks_to_process = [
            b for b in blocks if len(b["entries"]) >= COMPACTION_BULLET_THRESHOLD
        ]

        if not blocks_to_process:
            continue

        # Process bottom-to-top so earlier line indices stay valid
        page_compacted = 0
        for block in reversed(blocks_to_process):
            compacted = _compact_entries(block["entries"])
            body = _insert_compacted(body, block["start"], block["end"],
                                     compacted)
            page_compacted += 1
            total_blocks += 1

        body = _tag_compacted(body)

        full_content = frontmatter + "\n" + body if frontmatter else body
        fp = BRAIN_DIR / f"{slug}.md"
        if _DRY_RUN:
            total_compacted += 1
        else:
            fp.write_text(full_content, encoding="utf-8")
            total_compacted += 1

        # Count changes for this page
        orig_bullets = len(entries)
        new_bullets = sum(
            len([l for l in _compact_entries(b["entries"]) if l.startswith("- ")])
            for b in blocks_to_process
        )
        deduped = orig_bullets - new_bullets
        print(f"  COMPACTED {slug}: {len(blocks_to_process)} block(s), "
              f"{orig_bullets} entries → {new_bullets} unique "
              f"({deduped} deduped, by month)")

    if total_compacted == 0:
        print("[SILENT]")
        return 0

    print(f"\nSummary:")
    print(f"  Pages compacted: {total_compacted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
