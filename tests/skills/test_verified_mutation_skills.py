from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GITHUB_ISSUES = ROOT / "skills/software-development/github/references/issues.md"
APPLE_REMINDERS = ROOT / "skills/apple/apple-reminders/SKILL.md"


def test_github_issues_uses_bounded_gh_mutations_with_readback() -> None:
    text = GITHUB_ISSUES.read_text(encoding="utf-8")
    assert "gh auth status" in text
    assert "duplicate" in text.casefold()
    assert "Mutation protocol" in text
    assert "read the exact issue back" in text.casefold()
    assert "reconcile" in text.casefold()
    assert "deterministic candidate manifest" in text
    assert "Authorization: token" not in text
    assert "curl -s -X" not in text
    assert "| xargs" not in text


def test_apple_reminders_verifies_every_mutation_class() -> None:
    text = APPLE_REMINDERS.read_text(encoding="utf-8")
    assert 'description: "Use when ' in text
    assert "remindctl info <id> --json" in text
    assert "For complete/delete" in text
    assert "re-list the affected reminder or list" in text
    assert "do not ask for duplicate confirmation" in text.casefold()
