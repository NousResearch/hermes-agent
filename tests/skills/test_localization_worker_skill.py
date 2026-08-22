"""Static contract tests for the localization-worker optional skill."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "productivity" / "localization-worker"
SKILL = SKILL_DIR / "SKILL.md"
FORMATS = SKILL_DIR / "references" / "formats.md"
PLUGIN = ROOT / "plugins" / "localization-worker" / "plugin.yaml"


def _frontmatter(text: str):
    assert text.startswith("---\n")
    _, raw, body = text.split("---", 2)
    return yaml.safe_load(raw), body


def test_skill_frontmatter_and_description_follow_repository_contract():
    metadata, body = _frontmatter(SKILL.read_text(encoding="utf-8"))

    assert metadata["name"] == "localization-worker"
    assert metadata["version"] == "0.1.0"
    assert metadata["platforms"] == ["linux", "macos"]
    assert metadata["description"].endswith(".")
    assert len(metadata["description"]) <= 60
    assert body.strip()


def test_skill_describes_only_native_plugin_workflow():
    skill = SKILL.read_text(encoding="utf-8")
    formats = FORMATS.read_text(encoding="utf-8")

    assert "scripts/localization_cli.py" not in skill
    assert "localization_create_job" in skill
    assert "localization_verify_output" in skill
    assert "## How to Run" in skill
    assert "## Quick Reference" in skill
    assert "## Procedure" in skill
    assert "## Pitfalls" in skill
    assert "## Verification" in skill
    assert "Only UTF-8 `.txt` and `.md`" in skill
    assert "all other extensions | explicit unsupported" in formats


def test_skill_and_plugin_versions_match():
    skill_metadata, _ = _frontmatter(SKILL.read_text(encoding="utf-8"))
    plugin_metadata = yaml.safe_load(PLUGIN.read_text(encoding="utf-8"))

    assert skill_metadata["name"] == plugin_metadata["name"]
    assert skill_metadata["version"] == plugin_metadata["version"]
