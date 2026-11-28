"""Format, native-tool, and safety contracts for the OpenClaw skill."""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest

from agent.skill_utils import parse_frontmatter

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "autonomous-ai-agents" / "openclaw"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    metadata, body = parse_frontmatter(skill_text)
    assert body.strip(), "Skill body must not be empty"
    return metadata


def test_frontmatter_contract(frontmatter: dict) -> None:
    assert frontmatter["name"] == SKILL_DIR.name
    assert frontmatter["license"]
    assert frontmatter["version"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}
    assert frontmatter["author"].startswith("Joey Cera (@jtc268)")
    description = frontmatter["description"]
    assert 0 < len(description) <= 60
    assert description.endswith(".")


def test_metadata_points_to_official_source(frontmatter: dict) -> None:
    hermes = frontmatter["metadata"]["hermes"]
    assert hermes["homepage"] == "https://github.com/openclaw/openclaw"
    assert "hermes-agent" in hermes["related_skills"]


def test_modern_section_order(skill_text: str) -> None:
    _, body = parse_frontmatter(skill_text)
    assert body.startswith("# OpenClaw Skill")
    headings = re.findall(r"^## (.+)$", body, re.MULTILINE)
    required = [
        "When to Use", "Prerequisites", "How to Run", "Quick Reference",
        "Procedure", "Pitfalls", "Verification",
    ]
    positions = [headings.index(heading) for heading in required]
    assert positions == sorted(positions)


def test_examples_use_native_tools_with_literal_arguments(skill_text: str) -> None:
    examples = re.findall(r"^```(\w*)\n(.*?)^```", skill_text, re.MULTILINE | re.DOTALL)
    assert examples, "Provide native tool-call examples"
    tools = set()
    for language, example in examples:
        assert language == "python", "Examples should show Hermes tool calls"
        for statement in ast.parse(example).body:
            assert isinstance(statement, ast.Expr)
            call = statement.value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            assert not call.args, "Use named tool arguments"
            tools.add(call.func.id)
            arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            if call.func.id == "terminal":
                assert isinstance(arguments["command"], str) and arguments["command"].strip()
            else:
                assert call.func.id == "web_extract"
                assert arguments["urls"]
                for url in arguments["urls"]:
                    parsed = urlparse(url)
                    assert parsed.scheme == "https" and parsed.hostname == "docs.openclaw.ai"
    assert tools == {"terminal", "web_extract"}


def test_changes_require_preview_and_recovery(skill_text: str) -> None:
    procedure = skill_text.split("## Procedure\n", 1)[1].split("## Pitfalls\n", 1)[0].lower()
    steps = re.findall(r"^\d+\. (.+)$", procedure, re.MULTILINE)
    preview = next(i for i, step in enumerate(steps) if "dry run" in step)
    backup = next(i for i, step in enumerate(steps) if "backup" in step)
    apply = next(i for i, step in enumerate(steps) if step.startswith("apply"))
    assert preview < backup < apply
    assert "validate before restarting" in procedure
    assert "rollback" in procedure and "verify" in procedure


def test_secrets_and_profile_boundaries_are_preserved(skill_text: str) -> None:
    pitfalls = skill_text.split("## Pitfalls\n", 1)[1].split("## Verification\n", 1)[0].lower()
    assert "do not print secret values" in pitfalls
    assert "redact" in pitfalls
    assert "do not copy hermes config or credentials into openclaw" in pitfalls
    assert "hermes_home" in pitfalls and "profile" in pitfalls
    assert "do not widen skill allowlists" in pitfalls


def test_migration_is_delegated(skill_text: str) -> None:
    assert "official Hermes OpenClaw migration docs" in skill_text
    for recipe in ("hermes claw migrate", "openclaw migrate apply", "--include-secrets"):
        assert recipe not in skill_text


def test_skill_loads_from_an_isolated_profile(tmp_path, monkeypatch, skill_text: str) -> None:
    from tools import skills_tool

    profile = tmp_path / "hermes-profile"
    installed = profile / "skills" / "autonomous-ai-agents" / SKILL_DIR.name
    shutil.copytree(SKILL_DIR, installed)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    result = json.loads(skills_tool.skill_view(SKILL_DIR.name, preprocess=False))
    assert result["success"] is True
    assert result["name"] == SKILL_DIR.name
    _, body = parse_frontmatter(skill_text)
    assert body.strip() in result["content"]
