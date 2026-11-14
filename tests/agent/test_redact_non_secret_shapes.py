"""Non-secret shapes the redactor must leave byte-identical (#142574).

Each false positive here rewrote content an agent then copied back into files; the
positive controls pin that the narrowed rules still mask real credentials.
"""

import pytest

from agent.redact import redact_sensitive_text


@pytest.mark.parametrize("text", [
    "redact_secrets: true",
    "  redact_secrets: false",
    "api_key: null",
    "REDACT_SECRETS=False",
    '"token": "none"',
])
def test_config_literal_under_secret_named_key_is_kept(text):
    assert redact_sensitive_text(text, force=True) == text


def test_opaque_value_under_secret_named_key_still_masked():
    value = "Zq8" + "vN3pL6xR2tY9wB4m"
    out = redact_sensitive_text(f"redact_secrets: {value}", force=True)
    assert value not in out


@pytest.mark.parametrize("text", [
    "python-3.14.7+" + "20260901-linux-x64",
    "torch==2.5.1+" + "20260901.cu124",
])
def test_version_local_segment_is_not_a_phone(text):
    assert redact_sensitive_text(text, force=True) == text


@pytest.mark.parametrize("text", ["call +" + "4915123456789 now", "tel:+" + "4915123456789", "(+" + "15551234567)"])
def test_phone_number_still_masked(text):
    assert "****" in redact_sensitive_text(text, force=True)


@pytest.mark.parametrize("text", [
    '    "authorization: bearer ",',
    '    "authorization: bot ",',
    "Authorization: Basic",
    "proxy-authorization: token",
])
def test_bare_authorization_scheme_word_is_kept(text):
    assert redact_sensitive_text(text, force=True) == text


@pytest.mark.parametrize("header", ["Authorization: Bearer ", "Authorization: Bot ", "Authorization: "])
def test_authorization_credential_still_masked(header):
    credential = "opaque" + "Cred3ntialValue0123456789"
    out = redact_sensitive_text(f'curl -H "{header}{credential}"', force=True)
    assert credential not in out
    assert out.count('"') == 2
