"""Regression tests for workflow ownership protections."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_github_codeowners_exists():
    codeowners_path = REPO_ROOT / ".github" / "CODEOWNERS"

    assert codeowners_path.exists()


def test_workflows_and_actions_require_maintainer_review():
    codeowners_path = REPO_ROOT / ".github" / "CODEOWNERS"
    lines = {
        line.strip()
        for line in codeowners_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".github/workflows/ @NousResearch/hermes-maintainers" in lines
    assert ".github/actions/ @NousResearch/hermes-maintainers" in lines
