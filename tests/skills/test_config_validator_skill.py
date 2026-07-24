"""Smoke tests for the config-validator optional skill."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "config-validator"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "config-validator"


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "vijays365" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_script_exists() -> None:
    assert (SKILL_DIR / "scripts" / "validate-config.sh").is_file()


def test_script_has_model_checks() -> None:
    """Must check model.default and model.provider per SKILL.md docs."""
    src = (SKILL_DIR / "scripts" / "validate-config.sh").read_text()
    assert "model.default" in src, "script must check model.default"
    assert "model.provider" in src, "script must check model.provider"


def test_script_uses_if_guarded_yaml_check() -> None:
    """validate_yaml must be called inside an if guard, not bare under set -e."""
    src = (SKILL_DIR / "scripts" / "validate-config.sh").read_text()
    assert "if validate_yaml" in src, "validate_yaml must be if-guarded, not bare under set -e"


def test_skill_has_pitfalls_section() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Common Pitfalls" in src or "## Pitfalls" in src


def test_paths_use_installed_layout() -> None:
    """SKILL.md must reference installed paths, not repo paths."""
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "skills/optional-skills/devops" not in src, "paths should not include 'optional-skills' prefix"
