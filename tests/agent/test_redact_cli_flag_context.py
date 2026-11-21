"""#153484: a dash-prefixed credential word in source or prose is not a CLI flag."""

import pytest

from agent.redact import (
    redact_for_egress,
    redact_sensitive_text,
    redact_terminal_output,
)


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)


@pytest.mark.parametrize(
    "text",
    [
        'X}-token context.", +',
        'print("a)-token bcd")',
        '{"msg": "done)-token abc"}',
        r"const re = /^x within the (\\d+)-token context/u;",
        "template `${n}-token budget`",
    ],
)
def test_a_dash_prefixed_credential_word_in_source_or_prose_is_not_a_flag(text):
    """A looser lookbehind also fired after "}", ")", and similar, so the
    following word (and any trailing quote or punctuation) was masked.
    A bare ``X}-token context.`` already survived via the placeholder
    vocabulary; these cases carry punctuation that vocabulary does not allow."""
    assert redact_sensitive_text(text, force=True) == text
    assert redact_sensitive_text(text, force=True, file_read=True) == text
    assert redact_terminal_output(text, "cat src.ts", force=True) == text
    assert redact_for_egress(text) == text


@pytest.mark.parametrize(
    "text,secret",
    [
        ("--token abc123", "abc123"),
        ("docker run --token abc123", "abc123"),
        ('tool --password "S3cretPassw0rd!x"', "S3cretPassw0rd!x"),
        ('" -token abc123"', "abc123"),
    ],
)
def test_a_command_line_secret_flag_is_still_masked(text, secret):
    for out in (
        redact_sensitive_text(text, force=True),
        redact_sensitive_text(text, force=True, file_read=True),
        redact_terminal_output(text, "ps aux", force=True),
        redact_for_egress(text),
    ):
        assert secret not in out, out
        assert "--token" in out or "-token" in out or "--password" in out
