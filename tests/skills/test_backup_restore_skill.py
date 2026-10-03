"""Smoke tests for the backup-restore optional skill."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "backup-restore"


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
    assert frontmatter["name"] == "backup-restore"


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "vijays365" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_script_exists() -> None:
    assert (SKILL_DIR / "scripts" / "backup.sh").is_file()


def test_skill_has_pitfalls_section() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Common Pitfalls" in src or "## Pitfalls" in src


def test_paths_use_installed_layout() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "skills/optional-skills/devops" not in src, "paths should not include 'optional-skills' prefix"
