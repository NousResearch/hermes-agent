"""Security scan contracts for the skills this PR clears for installs.

Scope: the github skill (#102473) and web/blocked-page-recovery — the two
corpus entries this PR owns. The repo-wide corpus gate over the remaining
bundled skills lives with the consolidation vehicle (#111334 class 2).
"""
from pathlib import Path

import pytest

from tools.skills_guard import scan_skill, should_allow_install
from tools.threat_patterns import scan_for_threats

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
SKILL_DIR = SKILLS_DIR / "software-development" / "github"
SKILL_PATH = SKILL_DIR / "SKILL.md"
RECOVERY_DIR = SKILLS_DIR / "web" / "blocked-page-recovery"


# Context-injection checks are separate from the corpus install-scan gate.
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


def test_skill_community_install_is_safe_without_force(github_install_scan):
    assert github_install_scan.trust_level == "community"
    assert github_install_scan.verdict == "safe"
    assert should_allow_install(github_install_scan)[0] is True


def test_blocked_page_recovery_scan_has_no_critical_findings():
    # #98474's original subject: the recovery skill's Bearer-token curl tripped
    # env_exfil_curl and hard-blocked installs. Reworded here; the scan must
    # stay clean so the debt entry can retire.
    result = scan_skill(RECOVERY_DIR)
    assert [f for f in result.findings if f.severity == "critical"] == []
