from pathlib import Path

import yaml


def test_prospecting_director_skill_exists_and_has_required_frontmatter():
    path = Path("skills/marketing/prospecting-director/SKILL.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = text.split("---", 2)[1]
    data = yaml.safe_load(frontmatter)
    assert data["name"] == "prospecting-director"
    assert "lead prospecting" in data["description"]
    assert data["version"] == "1.0.0"
    assert "facebook-ads-library" in data["tags"]
    assert "obsidian" in data["tags"]


def test_prospecting_director_quality_gate_mentions_source_evidence_and_vault():
    text = Path("skills/marketing/prospecting-director/SKILL.md").read_text(encoding="utf-8")
    assert "Every prospect has evidence" in text
    assert "OBSIDIAN_VAULT_PATH" in text
    assert "country=LY" in text
