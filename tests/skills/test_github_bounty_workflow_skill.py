from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "skills" / "github" / "github-bounty-workflow" / "SKILL.md"


def _read_skill() -> tuple[dict, str, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    _, frontmatter_text, body = content.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_text)
    assert isinstance(frontmatter, dict)
    return frontmatter, body, content


def test_github_bounty_workflow_skill_has_valid_metadata():
    frontmatter, body, content = _read_skill()

    assert frontmatter["name"] == "github-bounty-workflow"
    assert frontmatter["description"].startswith("Use when")
    assert len(frontmatter["description"]) <= 1024
    assert "GitHub Bounty Workflow" in body
    assert len(content) <= 100_000


def test_github_bounty_workflow_skill_requires_authorized_private_disclosure():
    _, body, _ = _read_skill()
    lowered = body.lower()

    required_terms = [
        "explicit authorization",
        "in-scope",
        "safe harbor",
        "private vulnerability reporting",
        "security advisory",
        "no payout is guaranteed",
        "do not open a public issue",
        "private keys",
        "seed phrases",
    ]

    for term in required_terms:
        assert term in lowered


def test_github_bounty_workflow_skill_outputs_submission_artifacts():
    _, body, _ = _read_skill()

    assert "Evidence Pack" in body
    assert "Private Report Template" in body
    assert "PR Description Template" in body
    assert "Verification Checklist" in body
