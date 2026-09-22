from __future__ import annotations

from pathlib import Path

import yaml

from agent.skill_utils import parse_frontmatter


SKILL = Path(__file__).parents[2] / "skills/software-development/engineering-memory/SKILL.md"


def _frontmatter() -> tuple[dict, str]:
    content = SKILL.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)
    return yaml.safe_load(yaml.safe_dump(frontmatter)), body


def test_skill_frontmatter_and_contract_are_present() -> None:
    frontmatter, body = _frontmatter()
    assert frontmatter["name"] == "engineering-memory"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert "hermes.engineering_memory.v1" in body
    assert "Hindsight" in body and "Neural Steering" in body
    assert "terminal" in body and "read_file" in body and "search_files" in body


def test_skill_forbids_authority_and_secret_ingestion() -> None:
    _, body = _frontmatter()
    assert "Never put secrets" in body
    assert "cannot control signals" in body
    assert "Never use this index to authorize" in body
