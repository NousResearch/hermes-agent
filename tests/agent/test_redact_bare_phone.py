"""Opt-in bare phone redaction from #82556; default text behavior stays exact."""

import pytest

from agent.redact import redact_sensitive_text


class TestBarePhoneRedaction:
    """Bare phone identities are masked only at explicit PII boundaries."""

    @pytest.mark.parametrize(
        ("wa_id", "expected"),
        (
            ("1234567", "12****67"),
            ("123456789", "12****89"),
            ("1234567890", "12****90"),
            ("123456789012345", "12****45"),
        ),
    )
    def test_masks_supported_lengths_when_requested(self, wa_id, expected):
        result = redact_sensitive_text(
            wa_id,
            force=True,
            redact_bare_phone_numbers=True,
        )

        assert result == expected

    def test_default_preserves_bare_numeric_identifiers(self):
        text = "build_id=15551234567"

        assert redact_sensitive_text(text, force=True) == text

    @pytest.mark.parametrize(
        "text",
        (
            "account15551234567",
            "15551234567suffix",
            "ref_15551234567",
            "+15551234567",
        ),
    )
    def test_bare_mode_does_not_select_identifier_fragments(self, text):
        assert redact_sensitive_text(
            text,
            force=True,
            redact_bare_phone_numbers=True,
        ) == redact_sensitive_text(text, force=True)

    def test_punctuation_delimited_phone_is_selected(self):
        wa_id = "15551234567"

        result = redact_sensitive_text(
            f"call {wa_id}.",
            force=True,
            redact_bare_phone_numbers=True,
        )

        assert wa_id not in result

    def test_plus_prefixed_e164_keeps_existing_format(self):
        assert redact_sensitive_text(
            "+15551234567",
            force=True,
            redact_bare_phone_numbers=True,
        ) == "+155****4567"
