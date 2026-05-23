from pathlib import Path


SKILL_PATH = Path(".agents/skills/hermes-ops-review/SKILL.md")


def test_hermes_ops_review_skill_exists_with_required_metadata():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "name: hermes-ops-review" in text
    assert "description:" in text
    assert "# Hermes Ops Review" in text


def test_hermes_ops_review_skill_uses_redacted_status_receipts():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "hermes_cli.main ops status --markdown --no-health" in text
    assert "hermes_cli.main ops status --json --no-health" in text
    assert "json.tool /tmp/hermes-ops-status.json" in text
    assert "hermes_cli.main gateway status" in text
    assert "hermes_cli.main doctor" in text


def test_hermes_ops_review_skill_preserves_runtime_boundaries():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "ai.hermes.gateway" in text
    assert "/Users/agent1/Operator/scripts/hermes-gateway.sh" in text
    assert "Do not mutate private memory" in text
    assert "Do not dump raw `.env`" in text
    assert "raw logs" in text
    assert "hermes status --all" in text


def test_hermes_ops_review_skill_does_not_suggest_live_mutation_commands():
    text = SKILL_PATH.read_text(encoding="utf-8")
    forbidden = [
        "launchctl kickstart",
        "launchctl bootout",
        "launchctl bootstrap",
        "git reset --hard",
        "docker system prune",
        "rm -rf ~/.hermes",
        "cat ~/.hermes/.env",
        "cat /Users/agent1/.hermes/.env",
    ]

    for needle in forbidden:
        assert needle not in text
