from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "skills" / "creative" / "popular-web-designs"
SKILL_MD = SKILL_DIR / "SKILL.md"
TEMPLATES_DIR = SKILL_DIR / "templates"
REFERENCE_MD = SKILL_DIR / "references" / "upstream-awesome-design-md-comparison.md"
ATTRIBUTION_MD = SKILL_DIR / "references" / "ATTRIBUTION.md"


def imported_template_slugs() -> set[str]:
    reference_text = REFERENCE_MD.read_text(encoding="utf-8")
    added_section = reference_text.split("## Designs added in this refresh", 1)[1]
    added_section = added_section.split("## Format differences", 1)[0]
    return set(re.findall(r"^- `([^`]+)`$", added_section, re.MULTILINE))


def test_catalog_exactly_matches_template_files():
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    catalog_files = re.findall(r"^\| `([^`]+\.md)` \|", skill_text, re.MULTILINE)
    template_files = {path.name for path in TEMPLATES_DIR.glob("*.md")}

    assert catalog_files
    assert len(catalog_files) == len(set(catalog_files))
    assert set(catalog_files) == template_files
    assert imported_template_slugs() <= {Path(name).stem for name in template_files}


def test_skill_frontmatter_and_section_order_follow_hardline_rules():
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    description = re.search(r"^description: (.+)$", skill_text, re.MULTILINE)
    author = re.search(r"^author: (.+)$", skill_text, re.MULTILINE)

    assert description is not None
    assert len(description.group(1)) <= 60
    assert description.group(1).endswith(".")
    assert author is not None
    assert author.group(1).startswith("Filipe Bezerra (@lipebez) +")
    assert "Hermes Agent" in author.group(1)
    assert "# Popular Web Designs Skill" in skill_text

    required_sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [skill_text.index(section) for section in required_sections]
    assert positions == sorted(positions)


def test_imported_templates_use_native_tools_and_valid_markdown_structure():
    imported_slugs = imported_template_slugs()
    assert imported_slugs

    for slug in sorted(imported_slugs):
        text = (TEMPLATES_DIR / f"{slug}.md").read_text(encoding="utf-8")

        assert text.startswith("# Design System: ")
        assert text.count("Hermes Agent — Implementation Notes") == 1
        assert "`write_file`" in text
        assert "`browser_navigate`" in text
        assert "`browser_vision`" in text
        assert "https://github.com/VoltAgent/awesome-design-md" in text
        assert "../references/ATTRIBUTION.md" in text
        assert "generative-widgets" not in text
        assert text.count("```") % 2 == 0
        assert len(text.encode("utf-8")) < 1024 * 1024


def test_provenance_counts_are_consistent_without_content_overclaim():
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    reference_text = REFERENCE_MD.read_text(encoding="utf-8")
    attribution_text = ATTRIBUTION_MD.read_text(encoding="utf-8")
    template_count = len(list(TEMPLATES_DIR.glob("*.md")))
    counts = {
        label.strip(): int(value)
        for label, value in re.findall(
            r"^\| ([^|]+?) \| (\d+) \|$", reference_text, re.MULTILINE
        )
    }

    hermes_count = counts["Hermes `skills/creative/popular-web-designs/templates/*.md`"]
    upstream_count = counts["Upstream `design-md/*/DESIGN.md`"]
    shared_count = counts["Shared design slugs"]

    upstream_commit_match = re.search(
        r"Upstream commit: `([0-9a-f]{40})`", reference_text
    )
    assert upstream_commit_match is not None
    upstream_commit = upstream_commit_match.group(1)

    assert hermes_count == upstream_count == shared_count == template_count
    assert counts["Upstream-only design slugs"] == 0
    assert counts["Hermes-only design slugs"] == 0
    assert f"covers all {shared_count} upstream slugs" in skill_text
    assert "54 pre-existing templates remain adaptations" in skill_text
    assert "is synchronized with upstream commit" not in skill_text
    assert upstream_commit[:8] in skill_text
    assert upstream_commit in attribution_text
    assert "Shopifi` → `Shopify" in reference_text
    assert "Slacc` → `Slack" in reference_text
    assert "https://github.com/VoltAgent/awesome-design-md" in attribution_text
    assert "Copyright (c) 2026 VoltAgent" in attribution_text
    assert "The above copyright notice and this permission notice" in attribution_text
