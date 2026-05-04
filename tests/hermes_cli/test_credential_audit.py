from __future__ import annotations

from hermes_cli.credential_audit import summarize_credential_audit


def test_credential_audit_flags_missing_auth_and_recovery_metadata():
    summary = summarize_credential_audit(
        {
            "EXAMPLE_API_KEY": {
                "prompt": "Example API key",
                "password": True,
                "category": "tool",
            }
        }
    )

    assert summary["status"] == "warn"
    issues = {item["issue"] for item in summary["findings"]}
    assert "missing auth_method metadata" in issues
    assert "missing recovery or rotation URL" in issues


def test_credential_audit_accepts_phishing_resistant_oauth_metadata():
    summary = summarize_credential_audit(
        {
            "EXAMPLE_OAUTH_TOKEN": {
                "prompt": "Example OAuth token",
                "password": True,
                "auth_method": "oauth_pkce",
                "phishing_resistant": True,
                "recovery": "Run `hermes auth example` to refresh the token.",
            }
        }
    )

    assert summary["status"] == "pass"
    assert summary["warnings"] == 0
