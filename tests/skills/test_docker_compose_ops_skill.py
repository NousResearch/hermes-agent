"""Contract tests for the bundled docker-compose-ops skill."""

from __future__ import annotations

import re
from pathlib import Path
import pytest
import yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "devops"
    / "docker-compose-ops"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_docker_compose_ops_skill_exists() -> None:
    assert SKILL_MD.is_file()


def test_docker_compose_ops_frontmatter(skill_text: str) -> None:
    assert skill_text.startswith("---")
    m = re.search(r"\n---\s*\n", skill_text[3:])
    assert m is not None
    fm = yaml.safe_load(skill_text[3 : m.start() + 3])
    assert isinstance(fm, dict)

    assert fm.get("name") == "docker-compose-ops"
    desc = fm.get("description", "")
    assert len(desc) <= 60
    assert desc.endswith(".")

    assert fm.get("version") == "1.0.0"
    assert "author" in fm
    assert fm.get("license") == "MIT"
    assert "linux" in fm.get("platforms", [])
    assert "macos" in fm.get("platforms", [])
    assert "windows" in fm.get("platforms", [])

    hermes_meta = fm.get("metadata", {}).get("hermes", {})
    assert "docker" in hermes_meta.get("tags", [])
    assert "compose" in hermes_meta.get("tags", [])
    assert hermes_meta.get("category") == "devops"


def test_docker_compose_ops_content_guidelines(skill_text: str) -> None:
    # Must use modern Compose V2 syntax
    assert "docker compose" in skill_text
    # Must emphasize volume safety
    assert "docker compose down" in skill_text
    assert "docker compose config" in skill_text
