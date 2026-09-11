from pathlib import Path

import hermes_yaml as yaml


SKILL = Path("skills/productivity/collective-wisdom-install/SKILL.md")


def test_collective_wisdom_install_skill_contract():
    text = SKILL.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "collective-wisdom-install"
    assert metadata["description"].endswith(".")
    assert len(metadata["description"]) <= 60
    assert "`wisdom_inbox`" in body
    assert "`present_wisdom_consent`" in body
    assert 'A conversational "yes" prompts the control' in body
    assert "Never apply a" in body and "receipt through terminal" in body
    assert "obtain separate approval" in body
    assert "exact proposed package" in body
