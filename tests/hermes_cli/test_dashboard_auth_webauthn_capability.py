"""WebAuthn provider capability invariants (supports_webauthn protocol seam).

The ``webauthn`` provider is a credential provider without an OAuth redirect and
without a password form: capability is flagged exactly like ``supports_password``
was, the route layer dispatches the same way (a ``supports_webauthn`` provider
302s to ``/login`` like a password provider instead of running the OAuth dance),
and the optional hooks raise by default so a mis-flagged provider fails loudly.

These invariants need no ``fido2`` installation: they use a minimal in-test
provider that only carries the flags, mirroring
tests/hermes_cli/conftest_dashboard_auth.py's StubAuthProvider discipline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider, LoginStart, Session, clear_providers, register_provider)
from hermes_cli.dashboard_auth.base import assert_protocol_compliance
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


class WebauthnShapedProvider(StubAuthProvider):
    """StubAuthProvider (keeps start_login/complete_login working) flagged for webauthn."""

    name = "stubwebauthn"
    display_name = "Stub Passkeys"
    supports_webauthn = True


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_supports_webauthn_defaults_false_on_the_protocol():
    assert DashboardAuthProvider.supports_password is False
    assert DashboardAuthProvider.supports_token is False
    assert DashboardAuthProvider.supports_webauthn is False


def test_flag_false_means_no_dispatch_change_for_existing_providers():
    # The four in-tree provider shapes stay byte-compatible: a plain stub keeps
    # every legacy default.
    assert StubAuthProvider.supports_webauthn is False


def test_webauthn_hooks_raise_by_default():
    class _BareProvider(DashboardAuthProvider):
        name, display_name = "bare", "Bare (test)"

        def start_login(self, *, redirect_uri): ...
        def complete_login(self, **kwargs): ...
        def verify_session(self, *, access_token): ...
        def refresh_session(self, *, refresh_token): ...
        def revoke_session(self, *, refresh_token): ...

    provider = _BareProvider()
    with pytest.raises(NotImplementedError, match="does not support webauthn"):
        provider.webauthn_offered_for_request(None)
    with pytest.raises(NotImplementedError, match="does not support webauthn"):
        provider.webauthn_login_options(client_ip="203.0.113.9")
    with pytest.raises(NotImplementedError, match="does not support webauthn"):
        provider.complete_webauthn_login(handle="h", credential={})
    with pytest.raises(NotImplementedError, match="does not support webauthn"):
        provider.webauthn_register_options(client_ip="203.0.113.9")
    with pytest.raises(NotImplementedError, match="does not support webauthn"):
        provider.complete_webauthn_registration(handle="h", credential={}, label="k")


def test_assert_protocol_compliance_accepts_a_webauthn_flagged_class():
    assert_protocol_compliance(WebauthnShapedProvider)


# ---------------------------------------------------------------------------
# Dispatch branch: /auth/login for a webauthn provider -> /login (not OAuth)
# ---------------------------------------------------------------------------


@pytest.fixture()
def webauthn_only_registry():
    clear_providers()
    register_provider(WebauthnShapedProvider())
    yield
    clear_providers()


def test_auth_login_for_webauthn_provider_redirects_to_login_page(webauthn_only_registry):
    from hermes_cli import web_server

    app = web_server.app
    app.state.auth_required = True
    client = TestClient(app)
    resp = client.get("/auth/login?provider=stubwebauthn", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/login")


def test_auth_login_for_webauthn_provider_carries_next_param(webauthn_only_registry):
    from hermes_cli import web_server

    app = web_server.app
    app.state.auth_required = True
    client = TestClient(app)
    resp = client.get(
        "/auth/login?provider=stubwebauthn&next=%2Fdashboard", follow_redirects=False)
    assert resp.status_code == 302
    assert "next=%2Fdashboard" in resp.headers["location"]


def test_native_authorize_for_webauthn_provider_redirects_to_login_page(
    webauthn_only_registry):
    """The desktop's native authorize rides the same mediator: /login renders the
    ceremony surfaces, the broker handle keeps riding the PKCE cookie."""
    from hermes_cli import web_server
    from tests.hermes_cli.test_dashboard_auth_native_flow import _make_pkce

    _verifier, challenge = _make_pkce()
    app = web_server.app
    app.state.auth_required = True
    client = TestClient(app)
    resp = client.get(
        f"/auth/native/authorize?provider=stubwebauthn&code_challenge={challenge}"
        "&code_challenge_method=S256&redirect_uri=http%3A%2F%2F127.0.0.1%3A8930%2Fcb",
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/login")