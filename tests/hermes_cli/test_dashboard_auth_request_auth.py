"""Tests for trusted-request (``supports_request_auth``) dashboard auth.

Covers the protocol widening, the registry helper, the ``RemoteUserAuthProvider``
reference provider (the X-Remote-User / authenticated-reverse-proxy pattern —
honored only from a trusted peer, optional shared-secret header, username
sanitizing), and end-to-end through the real ``gated_auth_middleware``.
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    ProviderError,
    Session,
    clear_providers,
    list_request_auth_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.base import assert_protocol_compliance
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
from plugins.dashboard_auth.remote_user import (
    RemoteUserAuthProvider,
    _clean_username,
    _peer_is_trusted,
)


class _Hdr:
    """Minimal case-insensitive header map (modeled on starlette.Headers)."""

    def __init__(self, mapping):
        self._data = {k.lower(): str(v) for k, v in mapping.items()}

    def get(self, name: str, default: str = ""):
        return self._data.get(name.lower(), default)


def _req(peer: str, headers: dict):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=_Hdr(headers))


_TRUSTED = ["192.168.0.2", "10.0.0.0/8"]


def _prov(trusted=None, secret=""):
    return RemoteUserAuthProvider(
        trusted_proxies=trusted or ["192.168.0.2"],
        header="X-Remote-User",
        secret=secret,
        secret_header="X-Remote-User-Secret",
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_remote_user_is_protocol_compliant(self):
        assert assert_protocol_compliance(RemoteUserAuthProvider) is None

    def test_default_supports_request_auth_is_false(self):
        from hermes_cli.dashboard_auth.base import DashboardAuthProvider
        assert DashboardAuthProvider.supports_request_auth is False

    def test_compliance_rejects_flag_without_implementation(self):
        class Bad(DashboardAuthProvider):
            name = "bad"
            display_name = "Bad"
            supports_session = False
            supports_request_auth = True  # but does NOT override verify_request_auth

            def start_login(self, *, redirect_uri):
                raise NotImplementedError

            def complete_login(self, **kwargs):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

        with pytest.raises(TypeError, match="supports_request_auth"):
            assert_protocol_compliance(Bad)

    def test_list_request_auth_providers_filters(self):
        clear_providers()
        register_provider(_prov())
        register_provider(StubAuthProvider())  # not a request-auth provider
        names = {p.name for p in list_request_auth_providers()}
        assert names == {"remote-user"}
        clear_providers()


# ---------------------------------------------------------------------------
# Peer-allowlist / header trust (the security core)
# ---------------------------------------------------------------------------


class TestPeerTrust:
    def test_honored_from_trusted_peer(self):
        s = _prov().verify_request_auth(request=_req("192.168.0.2", {"X-Remote-User": "ryan"}))
        assert s is not None and s.user_id == "ryan" and s.provider == "remote-user"

    def test_honors_cidr_peer(self):
        s = _prov(trusted=["10.0.0.0/8"]).verify_request_auth(
            request=_req("10.1.2.3", {"X-Remote-User": "alice"})
        )
        assert s is not None and s.user_id == "alice"

    def test_denied_from_untrusted_peer_even_with_header(self):
        assert _prov().verify_request_auth(
            request=_req("203.0.113.5", {"X-Remote-User": "ryan"})
        ) is None

    def test_denied_with_no_peer(self):
        assert _prov().verify_request_auth(request=_req("", {"X-Remote-User": "x"})) is None

    def test_denied_without_header(self):
        assert _prov().verify_request_auth(request=_req("192.168.0.2", {})) is None

    def test_peer_is_trusted_helper(self):
        assert _peer_is_trusted("192.168.0.2", _TRUSTED)
        assert _peer_is_trusted("10.5.6.7", _TRUSTED)
        assert not _peer_is_trusted("8.8.8.8", _TRUSTED)
        assert not _peer_is_trusted("not-an-ip", _TRUSTED)
        assert not _peer_is_trusted("", _TRUSTED)

    def test_bad_trusted_entries_ignored_gracefully(self):
        assert _peer_is_trusted("10.5.6.7", ["bogus", "10.0.0.0/8"])


# ---------------------------------------------------------------------------
# Optional shared-secret header
# ---------------------------------------------------------------------------


class TestSecretHeader:
    def test_secret_header_required_when_configured(self):
        p = _prov(secret="s3cret-no-colon")
        # correct secret
        assert p.verify_request_auth(
            request=_req("192.168.0.2", {"X-Remote-User": "ryan", "X-Remote-User-Secret": "s3cret-no-colon"})
        ) is not None
        # wrong secret
        assert p.verify_request_auth(
            request=_req("192.168.0.2", {"X-Remote-User": "ryan", "X-Remote-User-Secret": "wrong"})
        ) is None
        # missing secret
        assert p.verify_request_auth(request=_req("192.168.0.2", {"X-Remote-User": "ryan"})) is None

    def test_non_ascii_secret_values_do_not_crash(self):
        # Regression: hmac.compare_digest(str, str) raises TypeError when
        # either side contains non-ASCII, and the presented header is
        # attacker-controlled — it must not surface as an unhandled 500.
        p = _prov(secret="sécret—¥")
        # matching non-ASCII secret -> accepted
        assert p.verify_request_auth(
            request=_req("192.168.0.2", {"X-Remote-User": "ryan", "X-Remote-User-Secret": "sécret—¥"})
        ) is not None
        # non-ASCII presented value with a wrong secret -> declined, no raise
        assert p.verify_request_auth(
            request=_req("192.168.0.2", {"X-Remote-User": "ryan", "X-Remote-User-Secret": "wrøng"})
        ) is None


# ---------------------------------------------------------------------------
# Username sanitizing
# ---------------------------------------------------------------------------


class TestUsernameSanitize:
    def test_clean_username(self):
        assert _clean_username("  ryan  ") == "ryan"

    def test_reject_blank(self):
        assert _clean_username("   ") == ""

    def test_reject_control_chars(self):
        assert _clean_username("ry\nan") == ""
        assert _clean_username("ry\x00an") == ""

    def test_reject_oversize(self):
        assert _clean_username("x" * 300) == ""


# ---------------------------------------------------------------------------
# End-to-end through the real middleware (direct drive, real peer IP)
# ---------------------------------------------------------------------------


def _make_request(**headers):
    """A genuine starlette.Request for a protected route from a trusted peer."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "http_version": "1.1",
        "asgi": {"version": "3.0"},
        "method": "GET",
        "scheme": "https",
        "path": "/protected",
        "raw_path": b"/protected",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("192.168.0.2", 12345),  # the trusted proxy peer
        "server": ("hermes.example.com", 443),
        "app": web_server.app,
        "state": {},
    }
    hdrs = [(b"host", b"hermes.example.com")]
    for k, v in headers.items():
        hdrs.append((k.lower().encode(), v.encode()))
    scope["headers"] = hdrs
    return Request(scope)


def _run_gate(req):
    """Run gated_auth_middleware against ``req`` with a capturing call_next."""
    import asyncio

    from starlette.responses import Response
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware

    called = []

    async def call_next(r):
        called.append(1)
        return Response("next", status_code=200)

    async def go():
        return await gated_auth_middleware(req, call_next), called

    resp, called = asyncio.run(go())
    return resp, called


class _BaseReqProvider(DashboardAuthProvider):
    """Minimal supports_request_auth provider (implements the abstract surface)."""

    supports_session = False
    supports_request_auth = True

    def start_login(self, *, redirect_uri):
        raise NotImplementedError

    def complete_login(self, **kwargs):
        raise NotImplementedError

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        raise NotImplementedError

    def revoke_session(self, *, refresh_token):
        return None


class _DeclineReqProvider(_BaseReqProvider):
    name = "decline-req"
    display_name = "Decline Request Auth"

    def verify_request_auth(self, *, request):
        return None


class _AcceptReqProvider(_BaseReqProvider):
    name = "accept-req"
    display_name = "Accept Request Auth"

    def verify_request_auth(self, *, request):
        return Session(
            user_id="accepted", email="", display_name="accepted", org_id="",
            provider=self.name, expires_at=int(time.time()) + 3600,
            access_token="", refresh_token="",
        )


class _BoomReqProvider(_BaseReqProvider):
    name = "boom-req"
    display_name = "Boom Request Auth"

    def verify_request_auth(self, *, request):
        raise ProviderError("backing store down")


@contextlib.contextmanager
def _gated_app(*providers):
    """Register providers + flip the gate on, restoring state afterwards."""
    clear_providers()
    for provider in providers:
        register_provider(provider)
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        yield
    finally:
        clear_providers()
        web_server.app.state.auth_required = prev


@pytest.fixture
def request_auth_gate():
    clear_providers()
    register_provider(_prov(trusted=["192.168.0.2"], secret="sec-no-colon-xxxxxxxx"))
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    yield
    clear_providers()
    web_server.app.state.auth_required = prev_required


class TestGateIntegration:
    def test_denied_without_header(self, request_auth_gate):
        resp, called = _run_gate(_make_request())
        assert called == []  # never reached the handler
        assert resp.status_code == 302
        assert resp.headers["location"].startswith("/login")

    def test_denied_from_untrusted_peer_with_header(self, request_auth_gate):
        from starlette.requests import Request

        req = _make_request(**{"X-Remote-User": "ryan"})
        req.scope["client"] = ("203.0.113.9", 12345)  # not the trusted proxy
        resp, called = _run_gate(req)
        assert called == []
        assert resp.status_code == 302

    def test_header_from_trusted_peer_authenticates(self, request_auth_gate):
        resp, called = _run_gate(
            _make_request(
                **{
                    "X-Remote-User": "ryan",
                    "X-Remote-User-Secret": "sec-no-colon-xxxxxxxx",
                }
            )
        )
        assert called == [1]  # passed through the gate
        assert resp.status_code == 200

    def test_wrong_secret_denied(self, request_auth_gate):
        resp, called = _run_gate(
            _make_request(**{"X-Remote-User": "ryan", "X-Remote-User-Secret": "nope"})
        )
        assert called == []
        assert resp.status_code == 302

    def test_vouch_success_emits_audit_event(self, request_auth_gate, monkeypatch):
        import hermes_cli.dashboard_auth.middleware as mw
        from hermes_cli.dashboard_auth.audit import AuditEvent

        calls = []
        monkeypatch.setattr(
            mw, "audit_log", lambda event, **k: calls.append((event, k))
        )

        resp, called = _run_gate(
            _make_request(
                **{"X-Remote-User": "ryan", "X-Remote-User-Secret": "sec-no-colon-xxxxxxxx"}
            )
        )
        assert resp.status_code == 200 and called == [1]
        assert len(calls) == 1
        event, fields = calls[0]
        assert event == AuditEvent.REQUEST_AUTH_SUCCESS
        assert fields["provider"] == "remote-user"
        assert fields["user_id"] == "ryan"
        assert fields["peer"] == "192.168.0.2"


class TestStackingAndFailClosed:
    def test_provider_outage_surfaces_503(self):
        with _gated_app(_BoomReqProvider()):
            resp, called = _run_gate(_make_request())
            assert resp.status_code == 503
            assert called == []  # never reached the handler

    def test_first_declining_provider_falls_through(self):
        with _gated_app(_DeclineReqProvider(), _AcceptReqProvider()):
            resp, called = _run_gate(_make_request(**{"X-Remote-User": "x"}))
            assert resp.status_code == 200
            assert called == [1]

    def test_decline_only_is_denied(self):
        with _gated_app(_DeclineReqProvider()):
            resp, called = _run_gate(_make_request(**{"X-Remote-User": "x"}))
            assert called == []
            assert resp.status_code == 302