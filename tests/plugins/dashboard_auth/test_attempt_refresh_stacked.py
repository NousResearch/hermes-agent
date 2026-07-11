"""Regression test for the dashboard logout bug fixed in
``hermes_cli/dashboard_auth/middleware.py::_attempt_refresh``.

The bug: when multiple session providers are registered (e.g. ``nous``,
``basic``, ``self_hosted``) and a non-owning provider is iterated BEFORE the
owning one, that non-owning provider raises ``RefreshExpiredError`` on a
foreign RT it cannot recognise. The old loop aborted on the FIRST
``RefreshExpiredError``, so the owning provider never got a chance to refresh
its own live token — logging users out ~every 15 minutes.

The fix: a ``RefreshExpiredError`` from a NON-owning provider is treated as
"not my token, ask the next provider" (``continue``), and only short-circuits
to re-login when EVERY provider fails or the OWNING provider raises it.

This test registers two stacked providers, has the non-owning one raise
``RefreshExpiredError`` first, and asserts the owning provider still gets
called and its refresh succeeds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_cli.dashboard_auth import (
    RefreshExpiredError,
    Session,
)
from hermes_cli.dashboard_auth.base import DashboardAuthProvider
from hermes_cli.dashboard_auth.middleware import _attempt_refresh


def _make_session(provider: str = "owning") -> Session:
    """Build a minimal Session for refresh-success return values."""
    import time

    return Session(
        user_id="u1",
        email="",
        display_name=provider,
        org_id="",
        provider=provider,
        expires_at=int(time.time()) + 3600,
        access_token=f"at-{provider}",
        refresh_token=f"rt-{provider}",
    )


class _StubProvider(DashboardAuthProvider):
    """Minimal provider stub for the refresh-loop test.

    Only ``refresh_session`` is exercised here; the other protocol methods are
    no-ops since the middleware refresh path never calls them.
    """

    supports_session = True
    supports_token = False

    def __init__(self, name: str, *, raises: type[Exception] | None, returns=None):
        self.name = name
        self.display_name = name
        self._raises = raises
        self._returns = returns
        self.refresh_calls = 0

    # --- protocol surface (unused by _attempt_refresh) ----------------------

    def start_login(self, *, redirect_uri):  # pragma: no cover - unused
        raise NotImplementedError

    def complete_login(self, *, code, state, code_verifier, redirect_uri):  # pragma: no cover
        raise NotImplementedError

    def verify_session(self, *, access_token):  # pragma: no cover - unused
        return None

    def revoke_session(self, *, refresh_token):  # pragma: no cover - unused
        return None

    # --- the one method that matters ----------------------------------------

    def refresh_session(self, *, refresh_token):
        self.refresh_calls += 1
        if self._raises is not None:
            raise self._raises(f"{self.name} cannot rotate RT {refresh_token!r}")
        return self._returns


@pytest.fixture
def _stacked_providers(monkeypatch):
    """Register a non-owning provider that raises RefreshExpiredError FIRST,
    and an owning provider that succeeds SECOND, so the loop MUST continue
    past the first failure to reach the success."""
    non_owning = _StubProvider("basic", raises=RefreshExpiredError)
    owning = _StubProvider(
        "nous", raises=None, returns=_make_session("nous")
    )

    # list_session_providers() returns registration order; non-owning first
    # is exactly the misordering that triggered the logout bug.
    monkeypatch.setattr(
        "hermes_cli.dashboard_auth.middleware.list_session_providers",
        lambda: [non_owning, owning],
    )
    # Stub audit_log so the loop's REFRESH_FAILURE side-effect doesn't try to
    # JSON-serialize the fake request's IP — we're testing loop logic, not audit.
    monkeypatch.setattr("hermes_cli.dashboard_auth.middleware.audit_log", lambda *a, **kw: None)
    return non_owning, owning


def _fake_request():
    """A minimal request stand-in. audit_log is stubbed in _stacked_providers,
    so this only needs to exist for _attempt_refresh's signature."""
    return MagicMock()


def test_non_owning_refresh_expired_does_not_block_owning(_stacked_providers):
    """The core regression: a non-owning ``RefreshExpiredError`` must not
    abort the loop before the owning provider gets to refresh."""
    non_owning, owning = _stacked_providers
    request = _fake_request()

    result = _attempt_refresh(request, refresh_token="rt-nous", session_provider="nous")

    # The owning provider MUST have been called despite non_owning raising first.
    assert owning.refresh_calls == 1, "owning provider was never tried"
    # The refresh succeeded and reports the owning provider.
    assert result is not None
    new_session, provider_name = result
    assert provider_name == "nous"
    assert new_session.refresh_token == "rt-nous"


def test_every_provider_expired_forces_relogin(_stacked_providers, monkeypatch):
    """When NO provider can rotate the RT (each raises RefreshExpiredError
    for a token it does not own), the loop returns None → force re-login."""
    non_owning, _owning = _stacked_providers
    # Replace the owning provider with a second non-owner that also raises.
    second_non_owner = _StubProvider("self_hosted", raises=RefreshExpiredError)
    monkeypatch.setattr(
        "hermes_cli.dashboard_auth.middleware.list_session_providers",
        lambda: [non_owning, second_non_owner],
    )
    request = _fake_request()

    result = _attempt_refresh(request, refresh_token="rt-unknown", session_provider="nous")

    assert result is None, "expected None (force re-login) when no provider can refresh"
    assert non_owning.refresh_calls == 1
    assert second_non_owner.refresh_calls == 1


def test_owning_provider_refresh_expired_short_circuits(_stacked_providers, monkeypatch):
    """When the OWNING provider itself raises RefreshExpiredError, the session
    is genuinely dead — the loop MUST stop and force re-login (no point trying
    non-owning providers that can't rotate a foreign RT)."""
    owning_dead = _StubProvider("nous", raises=RefreshExpiredError)
    non_owner = _StubProvider("basic", raises=RefreshExpiredError)
    # Owning first this time.
    monkeypatch.setattr(
        "hermes_cli.dashboard_auth.middleware.list_session_providers",
        lambda: [owning_dead, non_owner],
    )
    monkeypatch.setattr("hermes_cli.dashboard_auth.middleware.audit_log", lambda *a, **kw: None)
    request = _fake_request()

    result = _attempt_refresh(request, refresh_token="rt-nous", session_provider="nous")

    assert result is None
    # Owning raised → fatal → non-owner must NOT be tried.
    assert owning_dead.refresh_calls == 1
    assert non_owner.refresh_calls == 0, "non-owning provider tried after owning fatal failure"
