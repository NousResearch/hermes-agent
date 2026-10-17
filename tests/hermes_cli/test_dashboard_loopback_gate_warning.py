"""A disengaged loopback gate must be loud when auth providers are configured (#139415 item 4).

``POST /auth/password-login`` is not behind the gate, so on a plain loopback bind with a
configured provider it answers ``{"ok": true}`` and sets a full cookie set — but only the gated
middleware ever attaches ``request.state.session``, so app clients (Hermes Go, Hermes-Relay)
then 401 on every session-backed route while the SPA keeps working via the injected token.
The dashboard used to say nothing about it; ``_configure_auth_gate`` now warns once.
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture(autouse=True)
def _hermetic_public_hosts(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    monkeypatch.setattr(web_server, "_dashboard_public_hosts", lambda: frozenset())
    clear_providers()
    yield
    clear_providers()


def _configure(host: str = "127.0.0.1") -> None:
    web_server._configure_auth_gate(host, False, None, None)


def test_loopback_bind_with_provider_warns_the_gate_is_disengaged(caplog):
    register_provider(StubAuthProvider())
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        _configure("127.0.0.1")
    assert web_server.app.state.auth_required is False
    assert "NOT engaged" in caplog.text
    # The warning names the configured provider and the way out.
    assert "stub" in caplog.text
    assert "--host 0.0.0.0" in caplog.text


def test_loopback_bind_without_provider_stays_silent(caplog):
    # No provider configured: an ungated loopback bind is the plain local mode, not a trap.
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        _configure("127.0.0.1")
    assert web_server.app.state.auth_required is False
    assert "NOT engaged" not in caplog.text


def test_gated_bind_reports_the_gate_instead(caplog):
    register_provider(StubAuthProvider())
    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        _configure("0.0.0.0")
    assert web_server.app.state.auth_required is True
    assert "NOT engaged" not in caplog.text
    assert "auth gate enabled" in caplog.text


def test_desktop_exempt_backend_does_not_get_the_loopback_warning(monkeypatch, caplog):
    # The Desktop-owned loopback backend authenticates with its per-spawn session token by
    # design (#96490) and logs its own notice; it must not also be called a trap.
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "stub-session-token")
    register_provider(StubAuthProvider())
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        _configure("127.0.0.1")
    assert web_server.app.state.auth_required is False
    assert "NOT engaged" not in caplog.text
