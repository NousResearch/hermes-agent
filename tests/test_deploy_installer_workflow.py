"""Static guard for publishing public installer script updates."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-installer.yml"


def test_installer_changes_trigger_vercel_deploy() -> None:
    text = WORKFLOW.read_text()

    assert "name: Deploy Installer" in text
    assert "workflow_dispatch:" in text
    assert "branches: [main]" in text
    assert "'scripts/install.sh'" in text
    assert "'scripts/install.ps1'" in text
    assert "github.repository == 'NousResearch/hermes-agent'" in text
    assert "secrets.VERCEL_DEPLOY_HOOK" in text
    assert "curl -fsS -X POST" in text
