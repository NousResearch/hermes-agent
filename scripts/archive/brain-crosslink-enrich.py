"""Add [[wikilinks]] between related brain pages and resolve dangling links.

One-off (safe to re-run). Reads all ~/brain/*.md files, parses [[wikilinks]],
adds backlinks between genuinely related pages, and fixes dangling links.

Preserve-first: only ADDS content / fixes broken links. Never deletes page
content, never drops a fact.

Rules:
- Orphan pages (never linked to): add backlinks from related pages' See Also
- Dangling links: resolve to the correct existing page or remove the broken ref
- Related pages: pages in the same directory category get cross-links
- Never delete factual content, only add/move link syntax

Upstream: KenseiAgent scripts/brain-crosslink-enrich.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path("/home/kensei/repos/KenseiAgent")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BRAIN_DIR = Path(os.environ.get("GBRAIN_REPO", "~/brain")).expanduser()

# ── Dangling link resolution map ──────────────────────────────────────────────
# Format: (broken_target, replacement_target_or_None)
# None = remove the link (page genuinely doesn't exist and no reasonable target)
DANGLING_FIXES: dict[str, str | None] = {
    # people/sahil-saghir.md links to per-app pages that don't exist; route to apps/portfolio
    "projects/plenishd": "apps/portfolio",
    "projects/coach-os": "apps/portfolio",
    "projects/matchday-maestro": "apps/portfolio",
    "projects/kick-tionary": "apps/portfolio",
    # infrastructure/kensei-vps doesn't exist; conventions/infrastructure covers VPS ops
    "infrastructure/kensei-vps": "conventions/infrastructure",
    # concepts/kick-tionary-brand-voice doesn't exist
    "concepts/kick-tionary-brand-voice": None,
    # KENSEI/ style links in references/ — these are cross-refs to pages in KenseiAgent
    # None of these exist as brain pages; the content is in conventions/infrastructure
    "KENSEI/Operating Model": "conventions/infrastructure",
    "KENSEI/Profile Implementation Checklist": "references/profile-checklist",
    "KENSEI/Profile Pilot Log": "references/profile-pilot-log",
    "KENSEI/Research Memos": "references/backlog-may-2026",
}

# ── Directory-based backlinks ─────────────────────────────────────────────────
# Pages in the same directory are automatically considered related.
# These are categories where same-dir links are added.
BACKLINK_CATEGORIES = {"projects", "concepts", "conventions", "people", "references"}

# ── Hardcoded additional cross-links ──────────────────────────────────────────
# (from_page, to_page) pairs for pages that are genuinely related but not
# automatically detected.
CROSS_LINKS: list[tuple[str, str]] = [
    # Project pages cross-link to each other via people/sahil-saghir
    ("projects/gitradar", "projects/content-pipeline"),
    ("projects/content-pipeline", "projects/gitradar"),
    ("projects/mission-control", "projects/job-hunt"),
    ("projects/job-hunt", "projects/mission-control"),
    # Concepts link to conventions
    ("concepts/nous-cli-pet", "conventions/infrastructure"),
    ("concepts/nous-cli-pet", "people/sahil-saghir"),
    # Accounts reference conventions
    ("accounts/connected-accounts", "conventions/infrastructure"),
    # Timeline references people
    ("timeline/kensei-setup", "people/sahil-saghir"),
    # Audit log references projects
    ("Audits/_audit-rotation-log", "projects/audit-hardening"),
    # Reference pages link back
    ("references/operating-model", "conventions/infrastructure"),
    ("references/backlog-may-2026", "projects/mission-control"),
    ("references/profile-pilot-log", "people/sahil-saghir"),
    # Properties reference people
    ("properties/sahil-properties", "people/sahil-saghir"),
    # Orphan page backlinks (reverse direction)
    ("conventions/infrastructure", "accounts/connected-accounts"),
    ("people/sahil-saghir", "timeline/kensei-setup"),
    ("projects/audit-hardening", "Audits/_audit-rotation-log"),
]


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


def _parse_links(content: str) -> set[str]:
    """Extract all [[wikilinks]] from content."""
    return set(re.findall(r'\[\[([^\]]+)\]\]', content))


def _clean_target(target: str) -> str:
    """Strip anchor or display text from a wikilink target.
    
    [[page#anchor]] or [[page|display]] both resolve to 'page'.
    """
    return target.split("#")[0].split("|")[0].strip()


def _resolve_dangling(target: str, all_slugs: set[str]) -> str | None:
    """Check if a target is dangling, and return the replacement if known."""
    clean = _clean_target(target)
    if clean in all_slugs:
        return None  # Not dangling
    return DANGLING_FIXES.get(clean)


def _find_see_also_section(content: str) -> int | None:
    """Find the '## See Also' section position in content."""
    m = re.search(r"^## See\s+Also", content, re.MULTILINE)
    return m.start() if m else None


def _ensure_see_also(content: str) -> tuple[str, int]:
    """Ensure a page has a '## See Also' section. Returns (content, insert_pos)."""
    pos = _find_see_also_section(content)
    if pos is not None:
        return content, pos

    # Check for closing `---` marker (before bullet sections)
    m = re.search(r"\n---\s*\n", content)
    if m:
        # Insert before the ---
        insert_pos = m.start() + 1
        content = content[:insert_pos] + "\n## See Also\n\n" + content[insert_pos:]
        return content, insert_pos

    # Append to the end (before final empty lines)
    stripped = content.rstrip()
    content = stripped + "\n\n## See Also\n\n"
    return content, len(stripped) + 2


def _add_link(content: str, target: str, see_also_pos: int | None = None) -> str:
    """Add a [[wikilink]] to the See Also section if not already present."""
    if f"[[{target}]]" in content:
        return content  # Already linked

    if see_also_pos is not None:
        # Insert into See Also
        after_see = content[see_also_pos:]
        sec_end = after_see.find("\n## ")
        if sec_end < 0:
            sec_end = len(after_see)
        # Find a blank line after See Also entries, or use next section boundary
        see_section = after_see[:sec_end]
        lines = see_section.split("\n")
        # Find the first empty line after the heading
        insert_after = 1  # Skip the heading
        while insert_after < len(lines) and lines[insert_after].strip():
            insert_after += 1
        # Insert the new link
        indent = "  " if lines[insert_after - 1].startswith("  ") else ""
        new_line = f"{indent}- [[{target}]]\n"
        abs_pos = see_also_pos
        for i in range(insert_after):
            abs_pos += len(lines[i]) + 1
        content = content[:abs_pos] + new_line + content[abs_pos:]
        return content

    # Fallback: just append
    bullet = f"- [[{target}]]\n"
    # Insert before final section markers or ---
    pos = content.rfind("\n---")
    if pos >= 0:
        content = content[:pos + 1] + "\n## See Also\n\n" + bullet + content[pos + 1:]
    else:
        content = content.rstrip() + "\n\n## See Also\n\n" + bullet
    return content


def _replace_link(content: str, old_target: str, new_target: str | None) -> str:
    """Replace or remove a dangling [[wikilink]].

    Handles all forms: [[target]], [[target#anchor]], [[target|display]],
    [[target#anchor|display]].
    """
    # Regex: [[ target (# anchor)? (| display)? ]]
    pattern = rf'\[\[{re.escape(old_target)}(?:#[^\]]*)?(?:\|[^\]]*)?\]\]'
    if new_target is None:
        content = re.sub(pattern, '', content)
        content = re.sub(r'  +', ' ', content)
    else:
        content = re.sub(pattern, f'[[{new_target}]]', content)
    return content


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    pages = _list_brain_pages()
    all_slugs = set(pages.keys())

    # ── Phase 1: Fix dangling links ───────────────────────────────────────
    fix_count = 0
    for slug, content in pages.items():
        links = _parse_links(content)
        for link in links:
            clean = _clean_target(link)
            if clean not in all_slugs:
                replacement = DANGLING_FIXES.get(clean)
                if replacement is not None or replacement is None:
                    # None means remove the link
                    new_content = _replace_link(content, clean, replacement)
                    if new_content != content:
                        pages[slug] = new_content
                        content = new_content
                        fix_count += 1
                        if replacement:
                            print(f"  FIX: {slug}: [[{clean}]] → [[{replacement}]]")
                        else:
                            print(f"  FIX: {slug}: [[{clean}]] → (removed, no valid target)")

    # ── Phase 2: Add backlinks for orphan pages ──────────────────────────
    # Find pages never linked to
    linked_to = set()
    for slug, content in pages.items():
        links = _parse_links(content)
        for link in links:
            clean = _clean_target(link)
            # Resolve dangling
            if clean in all_slugs:
                linked_to.add(clean)

    orphans = sorted(all_slugs - linked_to)
    orphan_backlinks = 0
    for orphan in orphans:
        if orphan in ("RESOLVER", "README"):
            continue
        # Find the best page to link from
        orphan_category = orphan.split("/")[0]
        # Check the CROSS_LINKS list first
        for from_page, to_page in CROSS_LINKS:
            if to_page == orphan:
                content = pages.get(from_page, "")
                if content and f"[[{orphan}]]" not in content:
                    _, pos = _ensure_see_also(content)
                    pages[from_page] = _add_link(pages[from_page], orphan, pos)
                    orphan_backlinks += 1
                    print(f"  BACKLINK: {from_page} → [[{orphan}]] (via cross-links)")
                    break
        else:
            # Auto-link from same-category pages
            for other_slug in sorted(all_slugs):
                if other_slug.startswith(orphan_category) and other_slug != orphan:
                    content = pages.get(other_slug, "")
                    if content and f"[[{orphan}]]" not in content:
                        _, pos = _ensure_see_also(content)
                        pages[other_slug] = _add_link(pages[other_slug], orphan, pos)
                        orphan_backlinks += 1
                        print(f"  BACKLINK: {other_slug} → [[{orphan}]] (same category)")
                        break

    # ── Phase 3: Add CROSS_LINKS between genuinely related pages ─────────
    cross_count = 0
    for from_page, to_page in CROSS_LINKS:
        content = pages.get(from_page, "")
        if not content:
            continue
        if f"[[{to_page}]]" in content:
            continue  # Already linked
        _, pos = _ensure_see_also(content)
        pages[from_page] = _add_link(pages[from_page], to_page, pos)
        cross_count += 1
        print(f"  CROSSLINK: {from_page} → [[{to_page}]]")

    # ── Phase 4: Write modified pages ─────────────────────────────────────
    write_count = 0
    for slug, content in pages.items():
        orig_fp = BRAIN_DIR / f"{slug}.md"
        if not orig_fp.is_file():
            continue
        original = orig_fp.read_text(encoding="utf-8", errors="replace")
        if content != original:
            orig_fp.write_text(content, encoding="utf-8")
            write_count += 1

    print(f"\nSummary:")
    print(f"  Dangling fixes:   {fix_count}")
    print(f"  Orphan backlinks: {orphan_backlinks}")
    print(f"  Cross-links:      {cross_count}")
    print(f"  Files written:    {write_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
