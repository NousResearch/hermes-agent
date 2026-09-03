from pathlib import Path

import yaml


SKILL = Path("skills/productivity/collective-wisdom-install/SKILL.md")


def test_collective_wisdom_install_skill_contract():
    text = SKILL.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "collective-wisdom-install"
    assert metadata["description"].endswith(".")
    assert len(metadata["description"]) <= 60
    assert "--plan --json" in body
    assert "--apply-receipt" in body
    assert "`clarify`" in body
    assert "changes the active toolset" in body
