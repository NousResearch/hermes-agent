"""
Smoke tests for the darwinian-evolver optional skill.

We can't actually run the evolution loop in CI (it needs network + a paid LLM),
so these tests verify:
  - SKILL.md frontmatter conforms to the hardline format
  - shipped scripts parse as valid Python
  - the scripts reference the right env var / module paths
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "research" / "darwinian-evolver"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"




def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"




def test_platforms_excludes_windows(frontmatter) -> None:
    # Upstream uses func_timeout (POSIX signals) and uv subprocess pipelines; the
    # skill is gated [linux, macos]. If we ever port to Windows, update this test
    # to assert ["linux", "macos", "windows"].
    assert "windows" not in frontmatter["platforms"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Bihruze" in author, f"author should credit the original contributor: {author!r}"




@pytest.mark.parametrize(
    "path",
    [
        "scripts/parrot_openrouter.py",
        "scripts/show_snapshot.py",
        "templates/custom_problem_template.py",
    ],
)
def test_shipped_scripts_parse(path: str) -> None:
    src = (SKILL_DIR / path).read_text()
    ast.parse(src)  # raises SyntaxError on broken Python


def test_parrot_script_uses_openrouter() -> None:
    src = (SKILL_DIR / "scripts" / "parrot_openrouter.py").read_text()
    assert "OPENROUTER_API_KEY" in src, "parrot driver should read OPENROUTER_API_KEY"
    assert "openrouter.ai/api/v1" in src, "parrot driver should target OpenRouter"
    assert "EVOLVER_MODEL" in src, "model should be overridable via EVOLVER_MODEL"






