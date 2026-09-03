"""Hermetic contract tests for the Kubrick optional Hermes skill."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_CANDIDATES = [
    _HERE.parents[2] / "optional-skills" / "creative" / "kubrick",
    _HERE.parents[2],
]
SKILL_DIR = next((path for path in _CANDIDATES if (path / "SKILL.md").is_file()), _CANDIDATES[0])


def _frontmatter_text() -> str:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def _scalar(field: str) -> str:
    match = re.search(rf"^{re.escape(field)}: (.+)$", _frontmatter_text(), re.MULTILINE)
    assert match, f"SKILL.md missing single-line {field}"
    return match.group(1).strip()


def test_frontmatter_matches_hermes_contract() -> None:
    description = _scalar("description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert _scalar("name") == "kubrick"
    assert "Daniel Meyer" in _scalar("author")
    assert _scalar("platforms") == "[linux, macos, windows]"


def test_skill_uses_modern_section_order() -> None:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    required = [
        "# Kubrick Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(heading) for heading in required]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    "name",
    ["brief.yaml", "symbolic-ledger.yaml", "storyboard-plan.yaml"],
)
def test_storyboard_recipe_fixture_is_shipped(name: str) -> None:
    assert (SKILL_DIR / "examples" / "authority-transfer-storyboard" / name).is_file()


def test_manifest_storyboard_recipe_paths_exist() -> None:
    manifest = json.loads((SKILL_DIR / "kubrick.manifest.yaml").read_text(encoding="utf-8"))
    argv = manifest["recipes"]["storyboard-example"]
    referenced = [arg for arg in argv if arg.startswith("examples/")]
    assert len(referenced) == 3
    assert all((SKILL_DIR / path).is_file() for path in referenced)


def test_router_resolves_storyboard_recipe() -> None:
    path = SKILL_DIR / "scripts" / "intent_router.py"
    spec = importlib.util.spec_from_file_location("kubrick_intent_router", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    argv = module.resolve_recipe("storyboard-example")
    assert argv[:2] == ["do", "compile"]
    assert all((SKILL_DIR / arg).is_file() for arg in argv if arg.startswith("examples/"))


def test_shipped_python_scripts_parse() -> None:
    for path in (SKILL_DIR / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
