"""Contract checks for the opt-in recovery merge coverage workflow."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "merge-coverage.yml"


def test_workflow_is_manifest_opt_in_and_covers_pr_and_main_post_merge_ranges():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in content
    assert "push:" in content
    assert "main" in content
    assert ".github/merge-coverage.json" in content
    assert "github.event.pull_request.base.sha" in content
    assert "POST_TARGET=\"${{ github.sha }}\"" in content
    assert "github.event.before" in content
    assert "github.sha" in content
    assert "--manifest .github/merge-coverage.json" in content
    assert "--run-policy-tests" in content
    assert "--strict" in content
