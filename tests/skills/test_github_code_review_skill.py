"""Hermetic tests for the github-code-review skill.

Stdlib + pytest only; NO live network. Validates frontmatter constraints,
required section order (AGENTS.md skill-authoring standards), and presence of
the core operational content.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "github" / "github-code-review" / "SKILL.md"


def _text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter() -> dict[str, str]:
    text = _text()
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter"
    fm = text.split("---", 2)[1]
    out: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line and not line.startswith((" ", "-")):
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip().strip("'\"")
    return out


def test_frontmatter_required_fields():
    fm = _frontmatter()
    assert fm.get("name") == "github-code-review"
    description = fm.get("description", "")
    assert description, "description is required"
    assert len(description) <= 60, f"description must be <=60 chars (HARDLINE), got {len(description)}"
    assert description.endswith("."), "description must end with a period"
    assert fm.get("license") == "MIT"
    assert "platforms" in fm
    assert fm.get("author", "").startswith("A-KH17"), "author must credit the human contributor first"


def test_section_order_modern():
    text = _text()
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## Review Procedure",
        "## Quick Reference",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for section in sections:
        pos = text.find(section)
        assert pos != -1, f"missing section: {section}"
        positions.append(pos)
    assert positions == sorted(positions), f"sections out of order: {sections}"


def test_core_operational_content_present():
    text = _text()
    # Auth capture chain (capture, not just detect)
    assert "gh auth token" in text
    assert ".git-credentials" in text
    assert "HERMES_HOME" in text
    # Base-branch derivation (no hardcoded main assumption)
    assert "symbolic-ref" in text
    # Worktree test runs against PR code
    assert "git worktree add" in text
    # Anti-fabrication / honesty rules
    assert re.search(r"[Nn]ever fabricate", text)
    # Leftover scan on added lines
    assert "TODO|FIXME" in text
    # Verdict discipline
    assert "Request Changes" in text
    # Inline-comment side semantics for deleted lines
    assert 'side: "LEFT"' in text
    # Empty-diff scenario handling
    assert "empty" in text.lower()
    assert "pull/N/head" in text


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
