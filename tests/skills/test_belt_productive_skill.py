"""
Smoke tests for the belt-productive optional skill.

Validates SKILL.md frontmatter against repo hardline standards and
verifies reference docs are present.  No network calls.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "belt-productive"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == SKILL_DIR.name


def test_env_var_is_infsh(frontmatter) -> None:
    env_vars = frontmatter.get("required_environment_variables", [])
    for var in env_vars:
        assert var["name"] == "INFSH_API_KEY", f"expected INFSH_API_KEY, got {var['name']}"


def test_references_exist() -> None:
    refs_dir = SKILL_DIR / "references"
    assert refs_dir.is_dir(), "references/ directory missing"
    assert any(refs_dir.iterdir()), "references/ is empty"


def test_no_belt_whoami_in_skill() -> None:
    text = (SKILL_DIR / "SKILL.md").read_text()
    assert "belt whoami" not in text, "use 'belt me' not 'belt whoami'"


def test_required_sections() -> None:
    text = (SKILL_DIR / "SKILL.md").read_text()
    for section in ["When to Use", "Prerequisites", "Pitfalls", "Verification"]:
        assert f"## {section}" in text, f"missing ## {section}"
