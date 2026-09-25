"""Security scan contracts for the bundled GitHub skill."""
from pathlib import Path

import pytest

from tools.skills_guard import scan_skill, should_allow_install
from tools.threat_patterns import scan_for_threats

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "github"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"


# Other bundled skills have existing findings; keep this gate scoped to GitHub.
@pytest.mark.parametrize(
    "path",
    [SKILL_PATH]
    + sorted(path for path in (SKILL_DIR / "references").rglob("*") if path.is_file()),
    ids=lambda path: str(path.relative_to(SKILL_DIR)),
)
def test_skill_documents_pass_context_threat_scan(path):
    assert scan_for_threats(path.read_text(encoding="utf-8"), scope="context") == []


@pytest.fixture(scope="module")
def github_install_scan():
    # GitHub installs pass the repository identifier, not official provenance.
    return scan_skill(
        SKILL_DIR,
        source="NousResearch/hermes-agent/skills/software-development/github",
    )


def test_skill_install_scan_has_no_critical_findings(github_install_scan):
    assert [f for f in github_install_scan.findings if f.severity == "critical"] == []


def test_skill_community_install_requires_force_for_admin_examples(github_install_scan):
    assert github_install_scan.trust_level == "community"
    assert github_install_scan.verdict == "caution"
    # SSH and sudo documentation still blocks community installs by default;
    # unlike dangerous findings, these warnings permit an explicit override.
    assert should_allow_install(github_install_scan)[0] is False
    assert should_allow_install(github_install_scan, force=True)[0] is True
