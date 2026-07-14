"""Contracts for the bundled Obsidian skill's source-resolution policy."""

from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = ROOT / "skills" / "note-taking" / "obsidian" / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_source_resolution_precedes_vault_operations():
    text = _skill_text()

    assert text.index("### Source resolution") < text.index("### Quick reference")
    assert text.index("### Source resolution") < text.index("## Procedure")


def test_source_resolution_order_is_explicit_and_stable():
    text = _skill_text()
    section = text.split("### Source resolution", 1)[1].split("### Quick reference", 1)[0]
    stages = re.findall(r"^\d+\. \*\*(.+?)\*\*", section, flags=re.MULTILINE)

    assert stages == [
        "Explicit source",
        "Known canonical home",
        "Narrow source index",
        "Obsidian discovery",
        "Broad discovery",
    ]


def test_direct_source_rules_prevent_vault_authority_drift():
    text = _skill_text()

    required_rules = (
        "Never scan Obsidian merely because a task concerns system configuration.",
        "Do not repeat a vault search after resolving the direct source.",
        "search known Codex hook, config, and plugin",
        "do not silently change authority.",
        "Preserve one canonical home for each artifact.",
    )
    for rule in required_rules:
        assert rule in text


def test_validation_covers_each_resolution_case():
    text = _skill_text()
    verification = text.split("## Verification", 1)[1]

    required_cases = (
        "Exact hook file or config location known",
        "Exact GitHub repository and pull request known",
        "User asks for an Obsidian note",
        "Source genuinely unknown but likely recorded in notes",
        "Direct source unavailable",
        "Generated skill copy",
    )
    for case in required_cases:
        assert case in verification


def test_bundled_sync_regenerates_byte_identical_copy(tmp_path, monkeypatch):
    """Exercise the real sync path without touching the user's HERMES_HOME."""
    import tools.skills_sync as skills_sync

    bundled = tmp_path / "bundled"
    bundled_skill = bundled / "note-taking" / "obsidian"
    shutil.copytree(SKILL_PATH.parent, bundled_skill)

    hermes_home = tmp_path / "hermes-home"
    installed_skills = hermes_home / "skills"
    manifest = installed_skills / ".bundled_manifest"

    monkeypatch.setattr(skills_sync, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", installed_skills)
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", manifest)
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(
        skills_sync,
        "_get_optional_dir",
        lambda: tmp_path / "optional-skills",
    )
    monkeypatch.setattr(skills_sync, "_read_suppressed_names", lambda: set())
    monkeypatch.setattr(skills_sync, "_build_external_skill_index", lambda: set())
    monkeypatch.setattr(
        skills_sync,
        "_backfill_optional_provenance",
        lambda quiet=False: [],
    )

    result = skills_sync.sync_skills(quiet=True)
    generated = installed_skills / "note-taking" / "obsidian" / "SKILL.md"

    assert result["copied"] == ["obsidian"]
    assert generated.read_bytes() == SKILL_PATH.read_bytes()
    assert manifest.read_text(encoding="utf-8").startswith("obsidian:")
