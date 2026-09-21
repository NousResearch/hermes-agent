"""Promote high-confidence Mnemosyne working memory facts into GBrain (~/brain) pages.

Mode 2 (cron auto-promotion):
- Reads working_memory WHERE veracity='stated' AND importance>=0.8
- Maps each fact to a single existing brain page using keyword matching
- Mechanical dedup: exact-text + significant-word overlap (no LLM, sub-second)
- Appends the fact mechanically as a bullet point with date and [[wikilinks]]
- Writes via tools.gbrain.gbrain_put
- Tags promoted facts in working_memory metadata_json to prevent re-promotion
- Files kanban tasks for ambiguous/no-clear-page cases
- Outputs [SILENT] when nothing to promote

SKILL.md alignment: Mode-2 auto-promotion uses targeted append (not full-file overwrite).
No LLM calls needed — fast and deterministic (sub-second for 25+ candidates).

Upstream: KenseiAgent scripts/memory-promotion-daily.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure we can import tools.gbrain from KenseiAgent
REPO = Path("/home/kensei/repos/KenseiAgent")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.gbrain import GBRAIN_REPO, gbrain_put

# ── Config ───────────────────────────────────────────────────────────────────

DB = os.environ.get(
    "MNEMOSYNE_DB",
    "/home/kensei/.hermes/mnemosyne/data/mnemosyne.db",
)
MIN_IMPORTANCE = float(os.environ.get("MEMORY_PROMOTION_MIN_IMPORTANCE", "0.8"))
VERACITY_ALLOWED = {"stated", "imported"}
BRAIN_DIR = GBRAIN_REPO

# ── Brain page index ─────────────────────────────────────────────────────────

# Mapping rules: (keyword_match, target_slug) pairs.
# First match wins. Order matters -- more specific keywords first.
PAGE_MAP: list[tuple[str, str]] = [
    # People
    ("Sahil Saghir", "people/sahil-saghir"),
    ("Sahil's Family", "people/sahil-family"),
    # Conventions
    ("branch-cleanup", "conventions/infrastructure"),
    ("Hermes branch", "conventions/infrastructure"),
    ("source checkout", "conventions/infrastructure"),
    ("Guard pin", "conventions/infrastructure"),
    ("KENSEI responses", "conventions/infrastructure"),
    ("Discord-bound cron", "conventions/infrastructure"),
    ("profile architecture", "conventions/infrastructure"),
    ("Profile architecture", "conventions/infrastructure"),
    ("Discord voice auto-join", "conventions/infrastructure"),
    ("Misa-Misa voice", "conventions/infrastructure"),
    ("Hermes salvage policy", "conventions/infrastructure"),
    ("Mnemosyne cross-venv", "conventions/infrastructure"),
    ("Discord operational", "conventions/infrastructure"),
    ("Sahil has switched", "conventions/infrastructure"),
    ("GBrain", "conventions/infrastructure"),
    ("KENSEI v6", "conventions/infrastructure"),
    ("KENSEI Denji", "conventions/infrastructure"),
    ("KENSEI orchestration", "conventions/infrastructure"),
    ("KENSEI all-work review", "conventions/infrastructure"),
    ("KENSEI organised", "conventions/infrastructure"),
    ("~/.hermes/profiles", "conventions/infrastructure"),
    ("hermes-agent source", "conventions/infrastructure"),
    ("hermes-agent fork", "conventions/infrastructure"),
    ("Never write changes", "conventions/infrastructure"),
    ("X API cost", "conventions/infrastructure"),
    ("kanban notifications", "conventions/infrastructure"),
    ("QA engineer", "conventions/infrastructure"),
    ("Design Lead", "conventions/infrastructure"),
    ("Hermes salvage policy", "conventions/infrastructure"),
    # Accounts
    ("Gmail", "accounts/connected-accounts"),
    ("accounts", "accounts/connected-accounts"),
    # Projects
    ("GitRadar", "projects/gitradar"),
    ("gitradar", "projects/gitradar"),
    ("GitRadar pipeline", "projects/gitradar"),
    ("mission-control", "projects/mission-control"),
    ("Mission Control", "projects/mission-control"),
    ("content-pipeline", "projects/content-pipeline"),
    ("Content Pipeline", "projects/content-pipeline"),
    ("job-hunt", "projects/job-hunt"),
    ("Job hunt", "projects/job-hunt"),
    ("job hunting", "projects/job-hunt"),
    ("Kinexio", "projects/job-hunt"),
    ("VPS-based development", "projects/mission-control"),
    ("KenseiDashboard", "projects/mission-control"),
    ("mission-control", "projects/mission-control"),
    ("kensei-redesign", "projects/mission-control"),
    ("redesign scratch", "projects/mission-control"),
    # Concepts
    ("Nous CLI Pet", "concepts/nous-cli-pet"),
    ("Nous CLI pet", "concepts/nous-cli-pet"),
    ("Nous CLI", "concepts/nous-cli-pet"),
    ("codec panel", "concepts/nous-cli-pet"),
    ("figurine-level", "concepts/nous-cli-pet"),
    # References
    ("component prices", "references/operating-model"),
    ("pricing", "references/operating-model"),
    ("live pricing", "references/operating-model"),
    ("Live pricing", "references/operating-model"),
    ("DO NOT quote", "references/operating-model"),
    # Generic fallbacks
    ("Sahil wants", "people/sahil-saghir"),
    ("Sahil requires", "people/sahil-saghir"),
    ("Sahil prefers", "people/sahil-saghir"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _db_connect() -> sqlite3.Connection:
    return sqlite3.connect(DB, timeout=10)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_full_page(slug: str) -> str:
    """Read full content of a brain page."""
    fp = (BRAIN_DIR / f"{slug}.md").resolve()
    if not fp.is_file():
        return ""
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _append_fact_to_page(content: str, fact: str,
                         related_slugs: list[str], date_str: str) -> str:
    """Mechanically append a fact as a bullet point to the end of the page.
    
    This is more reliable and faster than LLM-based merging. It adds the fact
    as a markdown list item with date prefix and [[wikilinks]] to related pages.
    """
    # Build the backlinks string
    backlinks = " ".join(f"[[{s}]]" for s in related_slugs if s)
    
    # Choose section: if there's a "## See Also" or "---" end marker,
    # insert before it. Otherwise append to the end.
    bullet = f"- {date_str}: {fact}"
    if backlinks:
        bullet += f" {backlinks}"
    
    # Look for section markers to insert before
    for marker in ["## See Also", "---\n\n", "## Changelog"]:
        pos = content.rfind(marker)
        if pos >= 0:
            # Insert before this marker
            before = content[:pos].rstrip()
            after = content[pos:]
            return before + "\n" + bullet + "\n\n" + after
    
    # No marker found: append to the end
    return content.rstrip() + "\n\n" + bullet + "\n"


def _list_brain_pages() -> dict[str, str]:
    """Return {slug: first_400_chars_content} for all brain markdown files."""
    pages: dict[str, str] = {}
    if not BRAIN_DIR.is_dir():
        return pages
    for fp in sorted(BRAIN_DIR.rglob("*.md")):
        try:
            rel = fp.relative_to(BRAIN_DIR)
            slug = str(rel.with_suffix(""))
            if slug in ("RESOLVER", "README"):
                continue
            content = fp.read_text(encoding="utf-8", errors="replace")[:400]
            pages[slug] = content
        except (ValueError, OSError):
            continue
    return pages


def _map_to_page(content: str, brain_pages: dict[str, str]) -> str | None:
    """Map a memory fact to the best brain page slug using keyword matching."""
    for keyword, slug in PAGE_MAP:
        if keyword.lower() in content.lower():
            return slug
    # Fallback: search brain page content for key terms
    terms = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", content)
    for term in terms[:5]:
        term_lower = term.lower()
        for slug, page_content in brain_pages.items():
            if slug in ("RESOLVER", "README"):
                continue
            if term_lower in page_content.lower():
                return slug
    return None


def _check_already_covered(slug: str, fact: str, brain_pages: dict[str, str]) -> bool:
    """Check if the fact content is already present in the target brain page.
    
    Uses mechanical dedup: checks if the fact text (or its key phrases) already
    exists in the page content. Much faster and more reliable than an LLM judge
    for the cron auto-promotion path.
    """
    if slug not in brain_pages:
        return False
    page_content = brain_pages.get(slug, "").lower()
    
    # Exact match on trimmed fact
    fact_trimmed = fact.strip().lower()[:100]
    if fact_trimmed in page_content:
        return True
    
    # Check for key noun phrases (first few significant words)
    words = re.findall(r'\b[a-z]{4,}\b', fact.lower())
    significant = set(w for w in words if w not in 
                      {"this", "that", "from", "with", "have", "been", 
                       "will", "into", "about", "than", "their", "would",
                       "there", "which", "should", "could", "what", "been"})
    if len(significant) >= 3:
        # If 60%+ of significant words appear in the page, consider it covered
        matches = sum(1 for w in significant if w in page_content)
        if matches >= len(significant) * 0.6:
            return True
    
    return False


def _find_related(slug: str, fact: str, brain_pages: dict[str, str]) -> list[str]:
    """Find related pages to link via [[wikilinks]]."""
    related = []
    slug_dir = slug.split("/")[0]
    for other in brain_pages:
        if other == slug:
            continue
        # Same directory category
        if other.startswith(slug_dir) and other != slug:
            related.append(other)
        # Fact mentions the other page
        other_name = other.split("/")[-1].replace("-", " ").replace("_", " ")
        if other_name.lower() in fact.lower():
            if other not in related:
                related.append(other)
        # The other page mentions this slug
        if slug in brain_pages.get(other, ""):
            if other not in related:
                related.append(other)
    return related[:3]


def _tag_promoted(db: sqlite3.Connection, mem_id: str, slug: str) -> None:
    """Update metadata_json in working_memory with promotion flag."""
    cur = db.execute(
        "SELECT metadata_json FROM working_memory WHERE id = ?",
        (mem_id,),
    )
    row = cur.fetchone()
    if not row:
        return
    meta = {}
    raw = row[0]
    if raw and raw.strip():
        try:
            meta = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    meta["promoted_to_gbrain"] = slug
    meta["promoted_at"] = _now_iso()
    db.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
        (json.dumps(meta), mem_id),
    )
    db.commit()


def _file_kanban(title: str, fact: str) -> None:
    """File a kanban task for ambiguous/unclear promotion cases."""
    desc = f"Memory promotion (no clear brain page): {fact[:200]}"
    try:
        subprocess.run(
            [
                "hermes", "kanban", "create",
                f"Memory promotion: {title[:80]}",
                "--triage", "--assignee", "wesker", "--priority", "3",
                "--body", desc[:500],
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# ── Main pipeline ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Memory promotion cron")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be promoted without writing")
    parser.add_argument("--debug", action="store_true",
                        help="Show LLM responses and decisions")
    parser.add_argument("--candidate-limit", type=int, default=0,
                        help="Max candidates to process (0 = unlimited)")
    args, _ = parser.parse_known_args(argv)

    db = _db_connect()
    db.row_factory = sqlite3.Row

    # ── Step 0: Query candidates from working_memory ──────────────────────
    cur = db.execute("""
        SELECT id, content, importance, veracity, recall_count,
               metadata_json, valid_until, superseded_by
        FROM working_memory
        WHERE veracity IN ('stated', 'imported')
          AND importance >= ?
          AND (valid_until IS NULL OR valid_until = '')
          AND (superseded_by IS NULL OR superseded_by = '')
          AND content NOT LIKE '%PROMOTED%'
          AND json_extract(metadata_json, '$.promoted_to_gbrain') IS NULL
        ORDER BY importance DESC, recall_count DESC
    """, (MIN_IMPORTANCE,))

    rows = cur.fetchall()
    if not rows:
        print("[SILENT]")
        db.close()
        return 0

    candidates = [
        {
            "id": r["id"],
            "content": r["content"],
            "importance": r["importance"],
            "veracity": r["veracity"],
            "recall_count": r["recall_count"],
        }
        for r in rows
    ]

    if not candidates:
        print("[SILENT]")
        db.close()
        return 0

    if args.candidate_limit > 0:
        candidates = candidates[:args.candidate_limit]

    brain_pages = _list_brain_pages()
    today = _today_str()

    if args.debug:
        print(f"[DEBUG] {len(candidates)} candidate(s) from working_memory")
        print(f"[DEBUG] {len(brain_pages)} brain page(s) loaded")

    promoted_count = 0
    kanban_count = 0
    skipped_count = 0

    for cand in candidates:
        content = cand["content"]

        # ── Step 1: Map to brain page ────────────────────────────────────
        slug = _map_to_page(content, brain_pages)
        if not slug:
            if args.debug:
                print(f"  [DEBUG] No page match: {content[:80]}")
            _file_kanban(content[:80], content)
            kanban_count += 1
            continue
        if args.debug:
            print(f"  [DEBUG] Mapped to ~/brain/{slug}.md: {content[:80]}")

        # ── Step 2: Check existing coverage via LLM judge ─────────────────
        if args.debug:
            print(f"  [DEBUG] Checking coverage for {slug}...")
        covered = _check_already_covered(slug, content, brain_pages)
        if covered:
            if args.debug:
                print(f"  [DEBUG] SKIP -- already covered in {slug}")
            skipped_count += 1
            continue

        # ── Step 3: Mechanical append (no LLM merge, faster and reliable) ─
        full_content = _read_full_page(slug)
        if not full_content:
            if not args.dry_run:
                _file_kanban(f"Page not found: {slug}", content)
            kanban_count += 1
            continue

        related = _find_related(slug, content, brain_pages)
        new_content = _append_fact_to_page(full_content, content, related, today)

        # ── Step 4: Write via gbrain_put (markdown-native) ───────────────
        if args.dry_run:
            if args.debug:
                added = len(new_content) - len(full_content)
                print(f"  [DRY-RUN] Would append {added} chars to ~/brain/{slug}.md")
        else:
            try:
                gbrain_put({"slug": slug, "content": new_content})
            except Exception as exc:
                _file_kanban(f"gbrain_put failed for {slug}",
                             f"Error: {exc}\nFact: {content}")
                kanban_count += 1
                continue

        # ── Step 5: Tag promoted fact in Mnemosyne ───────────────────────
        if not args.dry_run:
            _tag_promoted(db, cand["id"], slug)
        promoted_count += 1

    db.close()

    if promoted_count == 0 and kanban_count == 0:
        print("[SILENT]")
        return 0

    verb = "would promote" if args.dry_run else "promoted"
    filed = "would file" if args.dry_run else "filed"
    lines = []
    if promoted_count > 0:
        lines.append(f"{verb} {promoted_count} fact(s) to ~/brain")
    if kanban_count > 0:
        lines.append(f"{kanban_count} ambiguous fact(s) {filed} as kanban tasks")
    if skipped_count > 0:
        lines.append(f"{skipped_count} fact(s) already covered in target pages")
    print(" | ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
