"""
Smoke tests for the pinecone-research optional skill.

Validates:
  - SKILL.md frontmatter conforms to the ≤60-char description standard
  - The skill name is distinct from the existing mlops/pinecone skill
  - Frontmatter has required fields
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "pinecone-research"
)

MLOPS_PINECONE_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mlops"
    / "pinecone"
)


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit ≤60): {desc!r}"


def test_name_is_distinct_from_mlops_pinecone(frontmatter) -> None:
    """The research skill must use a different name from mlops/pinecone."""
    mlops_src = (MLOPS_PINECONE_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", mlops_src, re.DOTALL)
    assert m, "mlops/pinecone SKILL.md missing frontmatter"
    mlops_fm = yaml.safe_load(m.group(1))
    assert frontmatter["name"] != mlops_fm["name"], (
        f"research pinecone name {frontmatter['name']!r} must differ from "
        f"mlops pinecone name {mlops_fm['name']!r}"
    )


def test_name_matches_expected(frontmatter) -> None:
    assert frontmatter["name"] == "pinecone-research"


def test_has_required_frontmatter_fields(frontmatter) -> None:
    for field in ("name", "description", "version", "license"):
        assert field in frontmatter, f"missing required field: {field}"


def test_platforms_includes_all_major(frontmatter) -> None:
    platforms = frontmatter.get("platforms", [])
    assert set(platforms) >= {"linux", "macos", "windows"}


@pytest.mark.parametrize(
    "path",
    [
        "scripts/rag_pipeline.py",
        "scripts/memory_manager.py",
    ],
)
def test_shipped_scripts_parse(path: str) -> None:
    """Shipped scripts must be valid Python (ast.parse raises SyntaxError otherwise)."""
    import ast

    src = (SKILL_DIR / path).read_text(encoding="utf-8")
    ast.parse(src)  # raises SyntaxError on broken Python


def test_scripts_use_pinecone_client() -> None:
    """Both scripts should reference the Pinecone SDK."""
    for script in ("scripts/rag_pipeline.py", "scripts/memory_manager.py"):
        src = (SKILL_DIR / script).read_text(encoding="utf-8")
        assert "PINECONE_API_KEY" in src, f"{script} should read PINECONE_API_KEY"
        assert "Pinecone" in src, f"{script} should import/use Pinecone client"
