"""Tests for AI Office redaction-first DTO serialization."""

import json

from hermes_cli.office_redaction import REDACTION_POLICY_VERSION, redact_display_text
from hermes_cli.office_state import build_empty_office_state


def test_redact_display_text_removes_secret_like_strings():
    text = (
        "token sk-office-redaction-sentinel and Bearer bearerRedactionSentinel plus "
        "bot 123456789:AAExampleTelegramToken and /home/alice/.hermes/.env"
    )

    redacted, report = redact_display_text(text)

    assert REDACTION_POLICY_VERSION == 1
    assert "sk-test-secret" not in redacted
    assert "Bearer bearerRedactionSentinel" not in redacted
    assert "123456789:AAExampleTelegramToken" not in redacted
    assert ".env" not in redacted
    assert report.redacted_field_count >= 4
    assert report.policy_version == REDACTION_POLICY_VERSION


def test_empty_office_state_serializes_valid_read_only_dto_without_sensitive_sections():
    state = build_empty_office_state()
    payload = state.to_dict()

    assert payload["schema_version"] == 1
    assert payload["mode"] == "read_only"
    assert payload["display_mode"] == "localhost"
    assert payload["capabilities"] == {
        "read_only": True,
        "mutations_enabled": False,
        "remote_mode": "unsupported",
    }
    assert payload["redactions"]["policy_version"] == REDACTION_POLICY_VERSION
    assert isinstance(payload["data_sources"], list)
    assert {source["id"] for source in payload["data_sources"]} == {
        "kanban",
        "cron",
        "sessions",
        "topics",
        "provenance",
    }
    assert all(source["status"] == "missing" for source in payload["data_sources"])

    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden_keys = [
        "body",
        "result",
        "latest_summary",
        "prompt",
        "script",
        "context_from",
        "messages",
        "tool_calls",
        "reasoning",
    ]
    for key in forbidden_keys:
        assert key not in serialized
