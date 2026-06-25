import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
START_SKILL = ROOT / "skills/productivity/company-second-brain-start/SKILL.md"
INSTALLER = ROOT / "deploy/second-brain/install-company-second-brain-skill.sh"
BUILD_SCRIPT = ROOT / "deploy/second-brain/scripts/build-install-assets.sh"
BUNDLE = ROOT / "deploy/second-brain/services/company-ai-gateway/static/company-second-brain-skill.tar.gz"


def test_starter_skill_is_agent_session_bootstrap():
    text = START_SKILL.read_text()

    assert "name: company-second-brain-start" in text
    assert "Use when" in text
    assert "second-brain query" in text
    assert "second-brain workspaces" in text
    assert "second-brain analytics" in text
    assert "Do not ask for admin key" in text


def test_installer_installs_main_and_starter_skills():
    text = INSTALLER.read_text()

    assert "company-second-brain-start" in text
    assert "SKILL_PARENT" in text
    assert "START_SKILL_ROOT" in text
    assert "Installed company-second-brain skills" in text


def test_build_script_packages_main_and_starter_skills():
    text = BUILD_SCRIPT.read_text()

    assert "START_SKILL_DIR" in text
    assert "company-second-brain-start" in text
    assert "company-second-brain company-second-brain-start" in text


def test_built_bundle_contains_main_and_starter_skills():
    with tarfile.open(BUNDLE, "r:gz") as archive:
        names = set(archive.getnames())

    assert "company-second-brain/SKILL.md" in names
    assert "company-second-brain/scripts/second-brain" in names
    assert "company-second-brain-start/SKILL.md" in names
