from pathlib import Path


SKILL_PATH = Path(".agents/skills/hermes-final-report/SKILL.md")


def test_hermes_final_report_skill_exists_with_required_metadata():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "name: hermes-final-report" in text
    assert "description:" in text
    assert "# Hermes Final Report" in text


def test_hermes_final_report_skill_defines_required_report_anatomy():
    text = SKILL_PATH.read_text(encoding="utf-8")

    for section in [
        "Executive summary",
        "Upgrade map",
        "System anatomy",
        "How-to guide",
        "Validation evidence",
        "Known limitations",
        "Operator commands",
        "Next actions",
    ]:
        assert section in text


def test_hermes_final_report_skill_uses_redacted_final_evidence_commands():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "hermes_cli.main ops status --markdown" in text
    assert "hermes_cli.main gateway status" in text
    assert "hermes_cli.main doctor" in text
    assert "hermes_cli.main send --to telegram" in text
    assert "--dry-run --json --output" in text
    assert "Secret-scan the final report draft" in text


def test_hermes_final_report_skill_gates_telegram_delivery():
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "Telegram delivery is an external action" in text
    assert "Phase 8 final integration passes" in text
    assert "user has requested final Telegram delivery" in text
    assert "without printing or copying bot tokens" in text
    assert "Telegram preflight passes" in text
    assert "3000-3600 characters" in text
    assert "message IDs" in text


def test_hermes_final_report_skill_avoids_secret_dump_and_mutation_patterns():
    text = SKILL_PATH.read_text(encoding="utf-8")

    required_boundaries = [
        "Do not include raw `.env`",
        "Keychain values",
        "launchd environment",
        "private memory",
        "raw logs",
        "provider facts",
        "credentials",
    ]
    for boundary in required_boundaries:
        assert boundary in text

    forbidden = [
        "cat ~/.hermes/.env",
        "cat /Users/agent1/.hermes/.env",
        "hermes status --all",
        "launchctl kickstart",
        "curl https://api.telegram.org",
        "TELEGRAM_BOT_TOKEN",
    ]
    for needle in forbidden:
        assert needle not in text
