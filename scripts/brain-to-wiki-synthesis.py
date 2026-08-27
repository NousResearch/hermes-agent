"""Promote structured ~/brain knowledge UP into ~/wiki (LLM-WIKI, the SSOT).

Reads brain pages in categories (people, projects, conventions, apps) and
writes/updates wiki pages in concepts/ and comparisons/ following
~/wiki/SCHEMA.md conventions. One-off + weekly cron safe.

Rules:
- Does NOT touch paper-derived concepts (those come from research-synthesis).
- Dedup by slug: if wiki page already exists and brain hasn't changed, skip.
- YAML frontmatter per SCHEMA.md.
- Updates wiki index.md and log.md.
- [SILENT] when nothing new to synthesise.

Upstream: KenseiAgent scripts/brain-to-wiki-synthesis.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# P13 isolation: REPO derives from KENSEI_REPO_ROOT (env-overridable) so
# the sys.path insert targets the active checkout, not a hardcoded path.
REPO = Path(os.environ.get("KENSEI_REPO_ROOT", "/home/kensei/repos/KenseiAgent"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BRAIN_DIR = Path(os.environ.get("GBRAIN_REPO", "~/brain")).expanduser()
WIKI_DIR = Path(os.environ.get("WIKI_DIR", "~/wiki")).expanduser()

# P13 isolation: when --dry-run is passed, every write path (wiki page
# write_text, index.md update, log.md append) is suppressed. Read paths
# (_read_brain_page, _brain_mtime, _wiki_mtime, _read_wiki_page) run
# unchanged so the synthesis decisions are still computed and printed.
_DRY_RUN = False
CONCEPTS_DIR = WIKI_DIR / "concepts"
COMPARISONS_DIR = WIKI_DIR / "comparisons"
SCHEMA_PATH = WIKI_DIR / "SCHEMA.md"
INDEX_PATH = WIKI_DIR / "index.md"
LOG_PATH = WIKI_DIR / "log.md"

DATE_TAG = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ── Mapping: brain category → wiki page mapping ──────────────────────────────
# Each entry: (brain_category_glob, wiki_type, slug_fn, priority)
# The slug_fn receives the full brain page content and returns a wiki page slug
# (without .md) or None to skip.
#
# For simple mappings, a (brain_slug, wiki_type, wiki_slug) tuple is used.
BRAIN_TO_WIKI: list[tuple[str, str, str]] = [
    # Simple: brain_slug → wiki_type/wiki_slug
    ("conventions/infrastructure", "concept", "kensei-infrastructure-conventions"),
    ("conventions/skill-library",   "concept", "skill-library-architecture"),
    ("apps/portfolio",              "concept", "kensei-app-portfolio"),
    ("projects/content-pipeline",   "concept", "content-pipeline-architecture"),
    ("projects/mission-control",    "concept", "mission-control-architecture"),
    ("projects/gitradar",           "concept", "gitradar-pipeline-architecture"),
    ("projects/job-hunt",           "concept", "job-hunt-strategy"),
    ("projects/audit-hardening",    "concept", "audit-hardening-strategy"),
]

# Comparison pages generated from cross-cutting brain content
COMPARISON_PAGES: list[dict] = [
    {
        "slug": "convex-vs-supabase",
        "title": "Convex vs Supabase for Sahil's App Stack",
        "sources": ["apps/portfolio"],
        "dimensions": [
            ("Hosting", "Convex-hosted, no ops", "Self-hosted or Supabase-hosted, more ops"),
            ("Type Safety", "TypeScript-native, end-to-end types", "TypeScript via pgtyped or raw SQL"),
            ("Realtime", "Built-in, seamless", "Via Supabase Realtime (websockets)"),
            ("Offline", "Convex offline support mature", "Local-first via PowerSync or custom"),
            ("Pricing", "Per-operation billing (predictable at scale)", "Billing by DB size + bandwidth (predictable at low end)"),
            ("Admin UI", "Convex dashboard (functional)", "Supabase Studio (richer)"),
            ("Sahil's projects", "Plenishd (primary preference for new RN+Expo)", "CoachOS, Kick-tionary (historic/locked-in)"),
        ],
        "verdict": "Convex is preferred for new RN+Expo projects (Plenishd). Supabase is already embedded in CoachOS and Kick-tionary; cost of migration exceeds benefit. The rule is Convex for new builds unless Supabase dependency is already locked in.",
    },
    {
        "slug": "keyword-vs-embeddings-brain-search",
        "title": "Keyword vs Embedding Search for GBrain",
        "sources": ["conventions/infrastructure"],
        "dimensions": [
            ("Method", "grep/ripgrep over ~/brain markdown files", "Ollama nomic-embed-text -> vector similarity"),
            ("Speed", "Instant (~100ms for 26 files)", "Slower (embedding generation ~2-5s, then vector search)"),
            ("Cost", "Free (no API key, no inference)", "Free (local Ollama, no paid key)"),
            ("Accuracy", "Word-match recall; misses semantic matches", "Semantic recall; catches synonym/paraphrase matches"),
            ("Dependencies", "None beyond ripgrep (optional)", "Ollama daemon running + nomic-embed-text model pulled"),
            ("When to use", "Everyday lookups, quick fact-checking", "When the keyword search misses, or for A/B testing"),
        ],
        "verdict": "Keyword mode is the default (fast, free, zero-dependency). Embeddings mode is available via BRAIN_SEARCH_MODE=embeddings for A/B testing recall quality. Both are fully wired; switch at any time.",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _read_brain_page(slug: str) -> str | None:
    """Read a brain page by slug (e.g. 'conventions/infrastructure')."""
    fp = BRAIN_DIR / f"{slug}.md"
    if not fp.is_file():
        return None
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _brain_mtime(slug: str) -> float:
    """Get modification time of a brain page."""
    fp = BRAIN_DIR / f"{slug}.md"
    return fp.stat().st_mtime if fp.is_file() else 0.0


def _wiki_mtime(slug: str, wtype: str) -> float:
    """Get modification time of a wiki page if it exists."""
    if wtype == "comparison":
        fp = COMPARISONS_DIR / f"{slug}.md"
    else:
        fp = CONCEPTS_DIR / f"{slug}.md"
    return fp.stat().st_mtime if fp.is_file() else 0.0


def _read_wiki_page(slug: str, wtype: str) -> str | None:
    """Read a wiki page if it exists."""
    if wtype == "comparison":
        fp = COMPARISONS_DIR / f"{slug}.md"
    else:
        fp = CONCEPTS_DIR / f"{slug}.md"
    if not fp.is_file():
        return None
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _frontmatter_yaml(title: str, slug: str, wtype: str, tags: list[str],
                       sources: list[str] | None = None,
                       confidence: str = "high") -> str:
    """Build YAML frontmatter per SCHEMA.md."""
    lines = [
        "---",
        f"title: {title}",
        f"created: {DATE_TAG}",
        f"updated: {DATE_TAG}",
        f"type: {wtype}",
        f"tags: [{', '.join(tags)}]",
    ]
    if sources:
        lines.append(f"sources: [{', '.join(sources)}]")
    lines.append(f"confidence: {confidence}")
    lines.append("---\n")
    return "\n".join(lines)


def _extract_tags_from_brain(content: str) -> list[str]:
    """Extract tags from brain page frontmatter."""
    tags = ["kensei", "memory"]
    m = re.search(r"tags:\s*\[([^\]]+)\]", content)
    if m:
        for t in m.group(1).split(","):
            t = t.strip().strip("'\"")
            if t and t not in tags:
                tags.append(t)
    return tags


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if content.startswith("---"):
        # Find the closing ---
        idx = content.find("---", 3)
        if idx >= 0:
            return content[idx + 3:].strip()
    return content.strip()


def _strip_dated_bullets(content: str) -> str:
    """Remove dated-bullet entries (from 3a promoter) from brain page body.
    
    Returns the core prose content without the chronological bullet log.
    """
    lines = content.split("\n")
    clean = []
    in_bullet_section = False
    for line in lines:
        # Lines starting with "- YYYY-MM-DD:" or "- YYYY-MM-DD " are bullet entries
        if re.match(r"^\s*-\s*\d{4}-\d{2}-\d{2}[-: ]", line):
            in_bullet_section = True
            continue
        # Also skip lines that are purely timestamps (no context)
        if re.match(r"^\s*-\s*\d{4}-\d{2}-\d{2}$", line):
            in_bullet_section = True
            continue
        # Also skip lines matching "---" separators used before bullet sections
        if line.strip() == "---":
            in_bullet_section = True
            continue
        if line.strip() == "":
            if in_bullet_section:
                continue
        else:
            in_bullet_section = False
        clean.append(line)
    return "\n".join(clean).strip()


def _synthesise_concept_from_brain(brain_content: str,
                                    title: str, tags: list[str]) -> str:
    """Synthesise a wiki concept page from brain content.
    
    Strips frontmatter and dated bullets, then reformats as clean
    wiki-style prose. Preserves all factual information.
    """
    body = _strip_frontmatter(brain_content)
    body = _strip_dated_bullets(body)

    if not body.strip():
        return ""

    # Get the first heading or first line as the intro
    frontmatter = _frontmatter_yaml(title, "", "concept", tags)

    # Build the page
    sections = [frontmatter]

    # Add a summary section if there's substantial content
    sections.append("## Overview\n")
    sections.append(body)

    return "\n\n".join(sections) + "\n"


def _build_comparison_page(cfg: dict) -> str:
    """Build a full comparison page from the config dict."""
    fm = _frontmatter_yaml(
        title=cfg["title"],
        slug=cfg["slug"],
        wtype="comparison",
        tags=["comparison", "technology"],
        sources=cfg.get("sources"),
    )

    lines = [fm, "## What is Being Compared\n", cfg["title"], "\n"]

    lines.append("## Comparison Dimensions\n")
    lines.append("| Dimension | Keyword/FTS | Embeddings (Ollama) |")
    lines.append("|-----------|-------------|---------------------|")
    for dim, kw, emb in cfg["dimensions"]:
        safe_kw = kw.replace("|", "\\|")
        safe_emb = emb.replace("|", "\\|")
        lines.append(f"| {dim} | {safe_kw} | {safe_emb} |")

    lines.append(f"\n## Verdict\n{cfg['verdict']}\n")
    return "\n".join(lines) + "\n"


def _update_index(new_pages: list[dict]) -> None:
    """Add new wiki pages to index.md under their sections."""
    if not INDEX_PATH.is_file():
        return

    content = INDEX_PATH.read_text(encoding="utf-8")

    for page in new_pages:
        slug = page["slug"]
        title = page["title"]
        wtype = page["type"]
        desc = page.get("desc", "")

        # Build the wikilink entry
        entry = f"- [[{slug}]] — {title}"

        # Determine section header
        if wtype == "comparison":
            section = "## Comparisons"
        else:
            section = "## Concepts"

        # Check if already in index
        if f"[[{slug}]]" in content:
            continue

        # Check if section exists, append under it
        if section in content:
            # Find the section end (next ## or end of file)
            sec_pos = content.index(section)
            after_sec = content[sec_pos + len(section):]
            next_sec = after_sec.find("\n## ")
            if next_sec < 0:
                insert_pos = len(content)
            else:
                insert_pos = sec_pos + len(section) + next_sec

            before = content[:insert_pos].rstrip()
            after = content[insert_pos:]
            content = before + "\n" + entry + "\n" + after
        else:
            # Append the section + entry at the end
            content = content.rstrip() + f"\n\n{section}\n{entry}\n"

    # Bump date in index (use lambda to avoid backreference clash with day digits >= 10)
    _new_date = datetime.now(timezone.utc).strftime('%d/%m/%y')
    content = re.sub(
        r"(Last updated: )\d{2}/\d{2}/\d{2}",
        lambda m: m.group(1) + _new_date,
        content,
    )

    if _DRY_RUN:
        return
    INDEX_PATH.write_text(content, encoding="utf-8")


def _append_to_log(entries: list[str]) -> None:
    """Append entries to wiki log.md."""
    if not LOG_PATH.is_file():
        return

    header = f"\n## [{DATE_TAG}] brain-to-wiki | Brain knowledge synthesis\n"
    body = "\n".join(f"- {e}" for e in entries)
    # Count total pages
    concept_count = len([f for f in CONCEPTS_DIR.glob("*.md") if f.is_file()])
    total_line = f"- Wiki concept count: {concept_count}"

    if _DRY_RUN:
        return
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(header + body + "\n" + total_line + "\n")


# ── Main pipeline ────────────────────────────────────────────────────────────


def main() -> int:
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    new_pages: list[dict] = []
    log_entries: list[str] = []

    # ── 1. Brain → Concept pages ──────────────────────────────────────────
    for brain_slug, wtype, wiki_slug in BRAIN_TO_WIKI:
        brain_content = _read_brain_page(brain_slug)
        if not brain_content:
            continue

        # Check mtime: skip if wiki page is newer than brain page
        if _wiki_mtime(wiki_slug, wtype) >= _brain_mtime(brain_slug):
            continue

        tags = _extract_tags_from_brain(brain_content)
        title = wiki_slug.replace("-", " ").title()
        content = _synthesise_concept_from_brain(brain_content, title, tags)
        if not content:
            continue

        # Count existing bullet points to show in log
        brain_mode = "extracted"

        # Write wiki page
        wp = CONCEPTS_DIR / f"{wiki_slug}.md"
        existed = wp.is_file()
        if not _DRY_RUN:
            wp.write_text(content, encoding="utf-8")
        action = "Updated" if existed else "Created"
        log_entries.append(f"{action} concepts/{wiki_slug}.md ({brain_slug} → wiki, {brain_mode})")
        new_pages.append({"slug": wiki_slug, "title": title, "type": "concept",
                          "desc": f"Synthesised from ~/brain/{brain_slug}.md"})

    # ── 2. Comparison pages ───────────────────────────────────────────────
    if not _DRY_RUN:
        COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in COMPARISON_PAGES:
        slug = cfg["slug"]
        comp_path = COMPARISONS_DIR / f"{slug}.md"
        existed = comp_path.is_file()
        content = _build_comparison_page(cfg)
        if not _DRY_RUN:
            comp_path.write_text(content, encoding="utf-8")
        action = "Updated" if existed else "Created"
        log_entries.append(f"{action} comparisons/{slug}.md")
        new_pages.append({
            "slug": f"comparisons/{slug}",
            "title": cfg["title"],
            "type": "comparison",
            "desc": cfg.get("verdict", "")[:80],
        })

    # ── 3. Silent if nothing changed ──────────────────────────────────────
    if not log_entries:
        return 0  # silent — nothing changed

    # ── 4. Update index.md + log.md ───────────────────────────────────────
    _update_index(new_pages)
    _append_to_log(log_entries)

    for entry in log_entries:
        print(entry)
    print(f"Wiki index updated. {len(new_pages)} page(s) added/updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
