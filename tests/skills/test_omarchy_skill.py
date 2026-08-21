"""Contract tests for the bundled Omarchy skill and its references."""

from pathlib import Path

import yaml


SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "omarchy"
SKILL = SKILL_DIR / "SKILL.md"
REFERENCE_NAMES = {"capture", "hooks", "hyprland", "plugins", "theming"}


def _frontmatter():
    content = SKILL.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    end = content.index("\n---\n", 4)
    return yaml.safe_load(content[4:end]), content


def test_skill_frontmatter_and_platform():
    frontmatter, content = _frontmatter()
    assert frontmatter["name"] == "omarchy"
    assert frontmatter["platforms"] == ["linux"]
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert "## When This Skill MUST Be Used" in content
    assert "## Critical Safety Rules" in content


def test_all_topic_guides_are_present_and_linked():
    _, content = _frontmatter()
    for name in REFERENCE_NAMES:
        assert (SKILL_DIR / "references" / f"{name}.md").is_file()
        assert f"references/{name}.md" in content


def test_safety_contract_is_preserved():
    _, content = _frontmatter()
    assert "NEVER modify anything in `/usr/share/omarchy/`" in content
    assert "sudo" in content and "pkexec" in content
    assert "omarchy debug --no-sudo --print" in content
    assert "ALWAYS SEEK USER CONFIRMATION" in content
