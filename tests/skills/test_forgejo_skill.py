"""Contract tests for the Forgejo issue-to-PR skill."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills/software-development/forgejo"


def test_forgejo_skill_has_required_files_and_boundaries():
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    reference = (SKILL / "references/issue-to-pr.md").read_text(encoding="utf-8")
    for text in (
        "`terminal`",
        "`read_file`",
        "`search_files`",
        "`patch`",
        "`forgejo_",
        "Agent Sandbox",
        "pull request",
        "fail closed",
        "no merge",
    ):
        assert text in skill or text in reference, text
    assert "credential" in reference.lower()
    assert "secret" in reference.lower()
    assert "Matrix" in reference


def test_forgejo_procedure_has_required_steps_and_no_live_cutover():
    body = (SKILL / "references/issue-to-pr.md").read_text(encoding="utf-8")
    for text in (
        "issue and comments",
        "repository instructions",
        "one disposable Agent Sandbox",
        "checkout",
        "tests",
        "branch",
        "pull request",
        "link",
        "URL",
        "changed files",
        "test result",
        "failure",
        "live deployment",
        "production",
    ):
        assert text.lower() in body.lower(), text
    assert "merge" in body.lower()
    assert "credential" in body.lower()
