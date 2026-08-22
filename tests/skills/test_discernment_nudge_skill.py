"""Tests for the discernment-nudge optional skill port."""
import re
from pathlib import Path

import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills/productivity/discernment-nudge"
SKILL_MD = SKILL_DIR / "SKILL.md"
LICENSE = SKILL_DIR / "LICENSE.txt"

# Permitted attribution lines: the upstream-credit footer and the author
# frontmatter field crediting the upstream source.
CREDIT_RE = re.compile(
    r"ported from anthropics/skills|^author: ['\"]?Anthropic \(upstream\)", re.IGNORECASE
)


def _split_frontmatter():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm), body


def test_frontmatter_parses():
    fm, _ = _split_frontmatter()
    assert isinstance(fm, dict)
    assert fm["name"] == "discernment-nudge"
    assert fm["version"] == "1.0.0"


def test_description_length():
    fm, _ = _split_frontmatter()
    desc = fm["description"]
    assert isinstance(desc, str)
    assert len(desc) <= 60, f"description is {len(desc)} chars (>60)"


def test_license_field():
    fm, _ = _split_frontmatter()
    assert fm["license"] == "Apache-2.0"


def test_license_file():
    assert LICENSE.exists()
    assert "Apache License" in LICENSE.read_text(encoding="utf-8")


def test_no_claude_anthropic_residue():
    text = SKILL_MD.read_text(encoding="utf-8")
    offending = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r"claude|anthropic", line, re.IGNORECASE):
            if CREDIT_RE.search(line):
                continue  # the upstream-credit line is permitted
            offending.append((i, line.strip()))
    assert not offending, f"residual claude/anthropic references: {offending}"


def test_related_skills_resolve():
    fm, _ = _split_frontmatter()
    related = (fm.get("metadata", {}) or {}).get("hermes", {}).get("related_skills", [])
    assert isinstance(related, list)
    staging_root = Path(__file__).resolve().parents[2]
    hermes_repo = Path.home() / ".hermes" / "hermes-agent"
    for name in related:
        found = False
        for root in (staging_root, hermes_repo):
            for sub in ("skills", "optional-skills"):
                base = root / sub
                if base.is_dir() and any(base.glob(f"*/{name}")) or (base / name).is_dir():
                    found = True
                    break
            if found:
                break
        assert found, f"related skill {name!r} not found in staging tree or hermes-agent"
