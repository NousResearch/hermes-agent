"""SendGrid keys are ``SG.<key-id>.<key-secret>``: the secret segment must not survive (#58167)."""

import pytest

from agent.redact import redact_sensitive_text

KEY_ID = "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf"
KEY_SECRET = "Ab3dE5gH7jK9mN1pQ3sT5vX7zB9dF1hJ3kL5nP7rT9v"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _outputs(text: str) -> list[str]:
    return [redact_sensitive_text(text, force=True),
            redact_sensitive_text(text, force=True, file_read=True, secret_file=True)]


@pytest.mark.parametrize("template", ["key: {}", "curl -H 'Authorization: x {}' api", "{}"])
def test_three_part_key_masks_both_segments(template):
    for out in _outputs(template.format(f"SG.{KEY_ID}.{KEY_SECRET}")):
        assert KEY_ID not in out, out
        assert KEY_SECRET not in out, out


def test_two_part_key_still_masked():
    for out in _outputs(f"key: SG.{KEY_ID}"):
        assert KEY_ID not in out, out


def test_short_lookalike_kept():
    assert redact_sensitive_text("SG.short", force=True) == "SG.short"
