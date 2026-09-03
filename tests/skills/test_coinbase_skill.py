from __future__ import annotations

import re
from pathlib import Path

import pytest


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "payments"
    / "coinbase"
    / "SKILL.md"
)


def test_skill_frontmatter_meets_authoring_contract():
    yaml = pytest.importorskip("yaml")
    content = SKILL_MD.read_text(encoding="utf-8")
    frontmatter = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)

    assert frontmatter
    metadata = yaml.safe_load(frontmatter.group(1))
    assert metadata["name"] == "coinbase"
    assert len(metadata["description"]) <= 60
    assert metadata["description"].endswith(".")
    assert metadata["author"].split(",")[0].strip() != "Hermes Agent"
    assert {"linux", "macos", "windows"} <= set(metadata["platforms"])
    assert {
        "mcp-oauth-remote-gateway",
        "mpp-agent",
    } <= set(metadata["metadata"]["hermes"]["related_skills"])
