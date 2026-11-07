"""Regression coverage for removal of the unsafe optional godmode skill (#118780)."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOVED_SKILL = REPO_ROOT / "optional-skills" / "security" / "godmode"
PUBLIC_REFERENCES = (
    REPO_ROOT / "website" / "sidebars.ts",
    REPO_ROOT / "website" / "docs" / "reference" / "optional-skills-catalog.md",
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "optional"
    / "security"
    / "security-godmode.md",
)


def test_godmode_skill_and_public_catalog_references_are_removed():
    """The repository must neither ship nor advertise the removed skill."""
    violations = []
    if REMOVED_SKILL.exists():
        violations.append("optional-skills/security/godmode")
    advertised_by = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in PUBLIC_REFERENCES
        if path.exists() and "godmode" in path.read_text(encoding="utf-8").lower()
    ]
    violations.extend(advertised_by)
    assert not violations, f"removed skill remains publicly advertised by: {violations}"
