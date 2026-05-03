"""Tests for optional fixed dashboard session token (HERMES_DASHBOARD_SESSION_TOKEN)."""

import string

import pytest


@pytest.fixture(autouse=True)
def clear_dashboard_session_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)


def test_resolve_dashboard_session_token_random_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
    from hermes_cli.web_server import _resolve_dashboard_session_token

    token = _resolve_dashboard_session_token()
    assert len(token) >= 40  # secrets.token_urlsafe(32)
    allowed = set(string.ascii_letters + string.digits + "-_")
    assert set(token) <= allowed


@pytest.mark.parametrize(
    "bad_value",
    ["", "   ", "a" * 15],
)
def test_resolve_dashboard_session_token_fallback_when_invalid_length(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: str,
) -> None:
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", bad_value)
    from hermes_cli.web_server import _resolve_dashboard_session_token

    token = _resolve_dashboard_session_token()
    assert len(token) >= 40
    allowed = set(string.ascii_letters + string.digits + "-_")
    assert set(token) <= allowed


def test_resolve_dashboard_session_token_accepts_fixed_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "z" * 16
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", secret)
    from hermes_cli.web_server import _resolve_dashboard_session_token

    assert _resolve_dashboard_session_token() == secret


def test_resolve_dashboard_session_token_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "y" * 16
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"  {secret}  \n")
    from hermes_cli.web_server import _resolve_dashboard_session_token

    assert _resolve_dashboard_session_token() == secret


def test_resolve_dashboard_session_token_short_value_falls_back_even_if_placeholder_like(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values under 16 chars are rejected before placeholder checks."""
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "changeme")
    from hermes_cli.web_server import _resolve_dashboard_session_token

    token = _resolve_dashboard_session_token()
    assert len(token) >= 40
