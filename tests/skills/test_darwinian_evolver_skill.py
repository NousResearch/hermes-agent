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


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "darwinian-evolver"


def test_platforms_excludes_windows(frontmatter) -> None:
    # Upstream uses func_timeout (POSIX signals) and uv subprocess pipelines; the
    # skill is gated [linux, macos]. If we ever port to Windows, update this test
    # to assert ["linux", "macos", "windows"].
    assert "windows" not in frontmatter["platforms"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Bihruze" in author, f"author should credit the original contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


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


def test_parrot_script_has_error_swallowing() -> None:
    """Provider content-filter / rate-limit must not kill the run — see Pitfall 2."""
    src = (SKILL_DIR / "scripts" / "parrot_openrouter.py").read_text()
    assert "LLM_ERROR" in src, "_prompt_llm should swallow provider errors and tag them"


def test_skill_calls_out_agpl(frontmatter) -> None:
    """The upstream tool is AGPL-3.0. The skill MUST flag this so users don't
    import it into MIT-licensed code by accident."""
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "AGPL" in src, "SKILL.md must mention upstream AGPL license"


def test_skill_pitfalls_section_present() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Pitfalls" in src
    # Pitfalls we discovered during the spike — keep them in sync with reality.
    assert "Initial organism must be viable" in src
    assert "generator" in src  # loop.run() pitfall
