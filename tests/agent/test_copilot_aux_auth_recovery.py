"""Copilot auxiliary recovery must recognize account hosts and drop stale caches."""

import pytest

from agent import auxiliary_client as aux
from hermes_cli import copilot_auth


@pytest.mark.parametrize("host,expected", [
    ("api.githubcopilot.com", "copilot"),
    ("api.business.githubcopilot.com", "copilot"),
    ("api.enterprise.githubcopilot.com", "copilot"),
    ("api.githubcopilot.com.evil.invalid", "auto"),
    ("githubcopilot.com.attacker.invalid", "auto"),
])
def test_auto_auth_recovery_uses_account_host(host, expected):
    assert aux._auth_refresh_provider_for_route("auto", f"https://{host}") == expected
    assert aux._auth_refresh_provider_for_route("anthropic", f"https://{host}") == "anthropic"


def test_auth_recovery_does_not_reuse_durable_rejected_token(monkeypatch):
    raw = "gho_fixture_only"
    monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: (raw, "fixture"))
    # Exercise actual durable storage and cache eviction, not a mock eviction call.
    fp = copilot_auth._token_fingerprint(raw)
    copilot_auth._save_jwt_to_disk(fp, "rejected", 9999999999, None)
    assert copilot_auth._load_jwt_from_disk(fp)[0] == "rejected"
    monkeypatch.setattr(copilot_auth, "exchange_copilot_token", lambda token: (
        pytest.fail("Rejected durable token survived refresh")
        if copilot_auth._load_jwt_from_disk(fp) else ("fresh", None, None)
    ))
    assert aux._refresh_provider_credentials("copilot")
