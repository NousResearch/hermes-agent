"""Shared helper utilities for skill contract and discipline tests."""

import re
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_frontmatter_and_body(skill_path: Path) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from a SKILL.md file."""
    content = skill_path.read_text(encoding="utf-8").lstrip("\ufeff\r\n\t ")
    assert content.startswith("---"), f"{skill_path}: SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, f"{skill_path}: unclosed frontmatter"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), f"{skill_path}: frontmatter must be a YAML mapping"
    body = content[m.end() + 3 :]
    return fm, body


def resolve_related_skills_in_repo(names: list[str]) -> list[str]:
    """Check that each related skill name exists in skills/ or optional-skills/."""
    missing = []
    for name in names:
        hits = (
            list(REPO_ROOT.glob(f"skills/**/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"optional-skills/**/{name}/SKILL.md"))
        )
        if not hits:
            missing.append(name)
    return missing
