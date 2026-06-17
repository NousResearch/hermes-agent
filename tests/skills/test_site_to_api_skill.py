from __future__ import annotations

import re
from pathlib import Path

import yaml


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "site-to-api"
    / "SKILL.md"
)


def _load_skill() -> tuple[dict, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    match = re.search(r"\n---\s*\n", content[3:])
    assert match is not None
    frontmatter = yaml.safe_load(content[3 : match.start() + 3])
    body = content[match.end() + 3 :]
    return frontmatter, body


def test_site_to_api_skill_is_valid_bundled_skill():
    frontmatter, body = _load_skill()

    assert frontmatter["name"] == "site-to-api"
    assert "description" in frontmatter
    assert len(frontmatter["description"]) <= 1024
    assert frontmatter["metadata"]["hermes"]["tags"]
    assert body.strip()


def test_site_to_api_skill_preserves_local_first_contract():
    frontmatter, body = _load_skill()
    combined = f"{frontmatter['description']}\n{body}"

    assert "Browserbase API key" in combined
    assert "should not request a Browserbase API key" in combined
    assert "site-to-api --doctor" in combined
    assert "/Users/yu/.hermes/scripts/site-to-api-local.sh doctor" in combined
    assert "Do not print cookies" in combined
    assert "blocker.json" in combined
