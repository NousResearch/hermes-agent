"""
Smoke tests for the autoreason optional skill.

We can't actually run the autoreason loop in CI (it needs network + a paid LLM),
so these tests verify:
  - SKILL.md frontmatter conforms to the hardline format
  - the shipped script parses as valid Python and is referenced correctly
  - the core pure functions (ranking parsing, Borda aggregation, blind shuffle)
    behave as documented
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "autonomous-ai-agents" / "autoreason"
SCRIPT = SKILL_DIR / "scripts" / "run_autoreason.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def mod():
    """Load scripts/run_autoreason.py without executing any LLM call."""
    spec = importlib.util.spec_from_file_location("autoreason_cli", SCRIPT)
    assert spec is not None and spec.loader is not None, "cannot load scripts/run_autoreason.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"
    assert desc.rstrip().endswith("."), "description must end with a period"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "autoreason"


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Xinyu Du" in author, f"author should credit the human contributor first: {author!r}"
    assert "@Starfie1d1272" in author, "author should include the GitHub handle"


def test_author_retains_upstream(frontmatter) -> None:
    assert "SHL0MS" in frontmatter["author"], "upstream SHL0MS attribution must be retained"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_platforms_include_major(frontmatter) -> None:
    # Pure-Python CLI over litellm — no POSIX-only primitives.
    assert set(frontmatter["platforms"]) >= {"linux", "macos", "windows"}


def test_script_exists() -> None:
    assert SCRIPT.is_file(), "scripts/run_autoreason.py must ship with the skill"


def test_script_parses() -> None:
    ast.parse(SCRIPT.read_text())  # raises SyntaxError on broken Python


def test_skill_documents_script_path() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "scripts/run_autoreason.py" in src, "SKILL.md must reference the shipped script"
    assert "terminal" in src, "SKILL.md must drive the CLI via the native `terminal` tool"


def test_skill_has_modern_sections() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    for section in ["## When to Use", "## Prerequisites", "## How to Run",
                    "## Procedure", "## Pitfalls", "## Verification"]:
        assert section in src, f"SKILL.md missing required section {section}"


# ── Core pure-function behavior ────────────────────────────────────────────

def test_parse_ranking_extracts_order(mod) -> None:
    text = "Proposal 2 is strongest.\nRANKING: 2, 1, 3"
    assert mod.parse_ranking(text) == ["2", "1", "3"]


def test_parse_ranking_ignores_suffix_noise(mod) -> None:
    text = "RANKING: [3], [1], [2] (ordered by fit)"
    assert mod.parse_ranking(text) == ["3", "1", "2"]


def test_parse_ranking_missing_returns_none(mod) -> None:
    assert mod.parse_ranking("No ranking given here") is None


def test_borda_aggregation_picks_top(mod) -> None:
    rankings = [["A", "B", "AB"], ["A", "AB", "B"]]
    winner, scores, valid = mod.aggregate_rankings(rankings, ["A", "B", "AB"], tiebreak_winner="A")
    assert winner == "A"
    assert scores["A"] > scores["B"] and scores["A"] > scores["AB"]
    assert valid == 2


def test_borda_tiebreak_prefers_incumbent(mod) -> None:
    # One judge ranks B first, the other ranks A first → tied scores, A must win.
    rankings = [["B", "A", "AB"], ["A", "B", "AB"]]
    winner, scores, _ = mod.aggregate_rankings(rankings, ["A", "B", "AB"], tiebreak_winner="A")
    assert scores["A"] == scores["B"]
    assert winner == "A", "tiebreak must prefer the incumbent"


def test_borda_ignores_malformed_judges(mod) -> None:
    rankings = [["A", "B", "AB"], None, ["AB", "A", "B"]]
    winner, _, valid = mod.aggregate_rankings(rankings, ["A", "B", "AB"], tiebreak_winner="A")
    assert valid == 2
    assert winner in {"A", "AB"}


def test_blind_shuffle_permutes_labels(mod) -> None:
    for _ in range(20):
        _, order = mod.randomize_for_judge("va", "vb", "vab")
        assert sorted(order.values()) == ["A", "AB", "B"]
        assert set(order) == {"1", "2", "3"}


def test_script_does_not_import_litellm_at_top_level(mod) -> None:
    """Top-level import must stay stdlib-only so CI can load the module."""
    tree = ast.parse(SCRIPT.read_text())
    top_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) \
                else [node.module.split(".")[0] if node.module else ""]
            top_imports.extend(names)
    assert "litellm" not in top_imports, "litellm must be imported lazily inside call_llm()"
