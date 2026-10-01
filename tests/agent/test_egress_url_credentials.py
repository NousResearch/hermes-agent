"""Regression for #119852: opaque URL credentials must not cross remote egress."""

import pytest

from agent import redact
from agent.monitoring.redaction import redact_for_export


@pytest.mark.parametrize("key", [
    "api_key", "token", "secret", "password", "access_token", "refresh_token",
    "auth_token", "bearer", "secret_value", "raw_secret", "secret_input", "key_material",
    "API%5FKEY", "api-key",
])
def test_egress_masks_query_credentials_without_changing_navigation(key):
    secret = "AQ" + "0123456789abcdefghijklmnopqrstuvwxyz" * 2
    original = f"https://example.invalid/y?{key}={secret}&view=public#section"
    expected = f"https://example.invalid/y?{key}=***&view=public#section"
    assert redact.redact_for_egress(original) == expected
    assert redact_for_export(original) == expected
    # force bypasses display opt-out, not the navigation-preserving URL policy.
    assert redact.redact_sensitive_text(original, force=True) == original


def test_egress_preserves_clean_empty_and_fail_closed_behavior(monkeypatch):
    assert redact.redact_for_egress("") == ""
    public = "https://example.invalid/y?token_count=3&view=public"
    assert redact.redact_for_egress(public) == public
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    secret_url = "https://example.invalid/y?api_key=opaque-value"
    assert "opaque-value" not in redact.redact_for_egress(secret_url)

    def broken(*args, **kwargs):
        raise RuntimeError("redactor failed")

    monkeypatch.setattr(redact, "_redact_strict_url_credentials", broken)
    assert redact.redact_for_egress(secret_url) == redact.REDACTION_UNAVAILABLE
