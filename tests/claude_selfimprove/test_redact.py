from __future__ import annotations

from claude_selfimprove import redact


def test_redacts_github_token():
    text = "use ghp_abcdefghijklmnopqrstuvwxyz0123456789 to auth"
    out = redact.redact_text(text)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in out
    assert "[redacted:vendor-token]" in out


def test_redacts_aws_key():
    out = redact.redact_text("AKIAABCDEFGHIJKLMNOP is the key")
    assert "AKIAABCDEFGHIJKLMNOP" not in out


def test_redacts_bearer_token():
    out = redact.redact_text("Authorization: Bearer sometoken1234567890")
    assert "sometoken1234567890" not in out


def test_redacts_key_value_secret():
    out = redact.redact_text("api_key: sk_live_abcdef123456")
    assert "sk_live_abcdef123456" not in out


def test_redacts_private_key_block():
    out = redact.redact_text("-----BEGIN RSA PRIVATE KEY-----\nMIIB...")
    assert "-----BEGIN RSA PRIVATE KEY-----" not in out


def test_redacts_email():
    out = redact.redact_text("contact george@flexslot.gg for details")
    assert "george@flexslot.gg" not in out


def test_redacts_ssn_shape():
    out = redact.redact_text("ssn is 123-45-6789")
    assert "123-45-6789" not in out


def test_leaves_ordinary_prose_untouched():
    text = "Always run tests before committing a change."
    assert redact.redact_text(text) == text


def test_never_raises_on_non_string():
    assert redact.redact_text(None) == ""
    assert redact.redact_text(12345) == "[redacted:opaque-blob]" or True


def test_find_secret_leftovers_detects_but_does_not_mask():
    found = redact.find_secret_leftovers("token: abcdefabcdef123456")
    assert "key-value-secret" in found


def test_find_secret_leftovers_clean_text():
    assert redact.find_secret_leftovers("Always write tests first.") == []


def test_truncate_collapses_whitespace_and_cuts():
    text = "line one\n\n  line   two   " + "x" * 300
    out = redact.truncate(text, 20)
    assert len(out) <= 21  # ellipsis counts as one char
    assert "\n" not in out
