from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "autonomous-ai-agents" / "hermes-digital-assistant"
SKILL = SKILL_DIR / "SKILL.md"
METHOD = SKILL_DIR / "references" / "DIGITAL_ASSISTANT.md"


def _frontmatter(text: str) -> dict:
    assert text.startswith("---\n")
    _, raw, _ = text.split("---", 2)
    return yaml.safe_load(raw)


def test_skill_metadata_matches_optional_skill_contract():
    text = SKILL.read_text()
    meta = _frontmatter(text)

    assert meta["name"] == "hermes-digital-assistant"
    assert meta["version"] == "0.1.0"
    assert meta["license"] == "MIT"
    assert meta["platforms"] == ["linux", "macos", "windows"]
    assert meta["description"].endswith(".")
    assert len(meta["description"]) <= 60
    assert "Daniel Steele (keeltrace)" in meta["author"]
    assert "hermes-agent" in meta["metadata"]["hermes"]["related_skills"]


def test_skill_is_an_adaptive_self_upgrade_procedure():
    text = SKILL.read_text()

    required = [
        "Establish current reality",
        "Build a requirement-to-mechanism map",
        "Use the narrowest stable Hermes seam",
        "Preserve the installation",
        "Run the full acceptance suite",
        "Attack the completion claim independently",
        "running Hermes process",
    ]
    for phrase in required:
        assert phrase in text

    assert "Do not create a new external repository" in text
    assert "does not depend on this skill remaining loaded" in text


def test_method_reference_preserves_the_full_behavioral_spec():
    text = METHOD.read_text()

    for pillar in range(1, 6):
        assert f"# Pillar {pillar} -" in text

    for vein in range(1, 9):
        assert f"## Vein {vein} -" in text

    assert "# The pre-send gate" in text
    assert "# Implementation notes" in text
    assert "# Build order" in text
    assert "queue incoming messages instead of interrupting" in text


def test_supporting_references_exist_and_are_linked():
    text = SKILL.read_text()
    names = [
        "DIGITAL_ASSISTANT.md",
        "UPGRADE_PLAYBOOK.md",
        "ACCEPTANCE_TEST.md",
        "ENGINEERING_DISCIPLINE.md",
    ]
    for name in names:
        path = SKILL_DIR / "references" / name
        assert path.is_file()
        assert f"references/{name}" in text


def test_skill_contains_no_machine_local_paths():
    text = "\n".join(
        p.read_text()
        for p in [SKILL, *sorted((SKILL_DIR / "references").glob("*.md"))]
    )
    assert "/home/" not in text
    assert "C:\\Users\\" not in text
