"""Profile-isolation regression tests for the Nous billing credential cache."""

from __future__ import annotations

import json
from pathlib import Path

import hermes_constants
import hermes_cli.auth as auth
from hermes_cli import nous_billing as billing


def _write_auth(home: Path, *, token: str, portal_base_url: str) -> None:
    (home / "auth.json").write_text(
        json.dumps({
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    "access_token": token,
                    "refresh_token": "refresh-token",
                    "client_id": "hermes-cli-vps",
                    "expires_at": "2999-01-01T00:00:00+00:00",
                    "portal_base_url": portal_base_url,
                }
            },
        }),
        encoding="utf-8",
    )


def _resolve_under_profile(home: Path) -> tuple[str, str]:
    reset_token = hermes_constants.set_hermes_home_override(str(home))
    try:
        return billing._resolve_token_and_base()
    finally:
        hermes_constants.reset_hermes_home_override(reset_token)


def test_token_cache_does_not_leak_across_multiplex_profile_contexts(
    monkeypatch, tmp_path
):
    """A profile must never receive a sibling profile's cached credentials."""
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    _write_auth(
        profile_a,
        token="token-for-profile-a",
        portal_base_url="https://profile-a.example.invalid",
    )
    _write_auth(
        profile_b,
        token="token-for-profile-b",
        portal_base_url="https://profile-b.example.invalid",
    )

    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    billing.invalidate_cached_token()

    # Bypass auth.py's separate startup memo so this test isolates the billing
    # cache while retaining real profile-scoped auth.json resolution.
    real_resolver = auth.resolve_nous_access_token
    resolved_profiles: list[str] = []

    def _resolve_from_profile_auth() -> str:
        resolved_profiles.append(hermes_constants.get_hermes_home().name)
        return real_resolver(insecure=True)

    monkeypatch.setattr(
        auth,
        "resolve_nous_access_token",
        _resolve_from_profile_auth,
    )

    result_a = _resolve_under_profile(profile_a)
    cached_result_a = _resolve_under_profile(profile_a)
    aliased_result_a = _resolve_under_profile(profile_a / ".." / "profile-a")
    result_b = _resolve_under_profile(profile_b)

    assert result_a == (
        "token-for-profile-a",
        "https://profile-a.example.invalid",
    )
    assert cached_result_a == result_a
    assert aliased_result_a == result_a
    assert result_b == (
        "token-for-profile-b",
        "https://profile-b.example.invalid",
    ), "profile B must not receive profile A's cached billing credentials"
    assert resolved_profiles == ["profile-a", "profile-b"]

    reset_token = hermes_constants.set_hermes_home_override(str(profile_a))
    try:
        billing.invalidate_cached_token()
    finally:
        hermes_constants.reset_hermes_home_override(reset_token)

    assert _resolve_under_profile(profile_b) == result_b
    assert _resolve_under_profile(profile_a) == result_a
    assert resolved_profiles == ["profile-a", "profile-b", "profile-a"]
