"""Tests for redaction of bare E.164-like phone numbers (WhatsApp Cloud wa_id)."""

from agent.redact import redact_sensitive_text


def test_redacts_bare_wa_id_in_message_body():
    text = "call 15551234567 now"
    result = redact_sensitive_text(text)
    assert "15551234567" not in result
    # masking preserves first 4 and last 4 digits for long numbers
    assert "1555****4567" in result


def test_redacts_plus_prefixed_e164():
    text = "my number is +15551234567"
    result = redact_sensitive_text(text)
    assert "+15551234567" not in result


def test_redacts_bare_wa_id_in_gateway_log_identity():
    log_line = (
        "inbound message: platform=whatsapp_cloud user=15551234567 "
        "chat=15551234567 msg='hello there' reply_to_id=None reply_to_text=None"
    )
    result = redact_sensitive_text(log_line)
    assert "15551234567" not in result
    assert "user=1555****4567" in result
    assert "chat=1555****4567" in result
    assert "hello there" in result


def test_does_not_redact_short_numeric_runs():
    text = "count is 1234567 and code 12345"
    result = redact_sensitive_text(text)
    assert "1234567" in result
    assert "12345" in result


def test_does_not_clip_longer_numeric_identifier():
    text = "id 1234567890123456 is a 16 digit card"
    result = redact_sensitive_text(text)
    assert "1234567890123456" in result
