"""Credential destination policy (andrexibiza P1-A): the plane bearer and the IdP client secret go
only to the configured https origin (plain http only to loopback) and never follow a redirect.

Every refusal is asserted BEFORE the credential leaves the process: a capturing transport (or a
real second listener for redirects) proves nothing was sent.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from plugins.config_backends.remote import client
from plugins.config_backends.remote import credentials as cred_mod
from plugins.config_backends.remote import destination
from plugins.config_backends.remote.credentials import PlaneCredentialError

from .stub_plane import CLIENT_ID, CLIENT_SECRET, INSTANCE, StubPlane, remote_env

PLANE_BEARER = "plane-bearer-secret"


class _Listener:
    """A loopback HTTP server on its own port (its own origin) recording every request; answers
    each with ``status`` and an optional ``Location``."""

    def __init__(self, status: int = 200, location: str | None = None) -> None:
        self.seen: list[dict] = []
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args) -> None:
                pass

            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                listener.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": self.rfile.read(n).decode() if n else ""})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            do_GET = do_POST = do_PATCH = _any

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def listeners():
    made: list[_Listener] = []

    def make(**kw) -> _Listener:
        made.append(_Listener(**kw))
        return made[-1]

    yield make
    for listener in made:
        listener.close()


@pytest.fixture
def nous_bearer(monkeypatch, tmp_path):
    """Cloud credential path with a known bearer; no IdP configured."""
    for k in ("GATEWAY_RELAY_IDP_TOKEN_URL", "GATEWAY_RELAY_IDP_CLIENT_ID", "GATEWAY_RELAY_IDP_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_CONFIG_INSTANCE_ID", INSTANCE)
    resolved: list[Path] = []

    def token(home):
        resolved.append(home)
        return PLANE_BEARER

    monkeypatch.setattr(client, "plane_token", token)
    return resolved


@pytest.fixture
def captured(monkeypatch):
    """Replace the credentialed opener: record what WOULD be sent, send nothing."""
    sent: list = []

    class _Resp:
        status, headers = 200, {}

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_(req, timeout):
        sent.append(req)
        return _Resp()

    monkeypatch.setattr(destination._OPENER, "open", open_)
    return sent


# --- config plane bearer ---------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://foreign.example:8443/plane",          # ambient non-loopback http
    "http://config-config.nousresearch.com",       # HTTPS -> HTTP downgrade of the real default
    "ftp://plane.example",
    "https://user:pw@plane.example",               # userinfo: ambiguous destination
    "https://plane.example/?next=https://evil.example",
    "https://",
])
def test_bearer_refused_before_send_for_forbidden_plane_url(monkeypatch, tmp_path, nous_bearer, captured, url):
    monkeypatch.setenv(client.URL_ENV, url)
    with pytest.raises(destination.CredentialDestinationError) as exc:
        client.request("GET", tmp_path, "default")
    assert exc.value.retryable is False
    assert captured == []          # nothing left the process
    assert nous_bearer == []       # the bearer was never even resolved


@pytest.mark.parametrize("url, expect", [
    ("", "https://config-config.nousresearch.com/v1/config/self?profile=default"),   # D19 default
    ("https://config.corp.internal:9443/plane/", "https://config.corp.internal:9443/plane/v1/config/self?profile=default"),
    ("http://localhost:8080", "http://localhost:8080/v1/config/self?profile=default"),
    ("http://127.0.0.1:9", "http://127.0.0.1:9/v1/config/self?profile=default"),
    ("http://[::1]:9", "http://[::1]:9/v1/config/self?profile=default"),
])
def test_bearer_sent_to_allowed_plane_origin(monkeypatch, tmp_path, nous_bearer, captured, url, expect):
    monkeypatch.setenv(client.URL_ENV, url)
    resp = client.request("GET", tmp_path, "default")
    assert resp.status == 200
    assert [r.full_url for r in captured] == [expect]
    assert captured[0].get_header("Authorization") == f"Bearer {PLANE_BEARER}"


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_plane_redirect_is_not_followed_with_the_bearer(monkeypatch, tmp_path, nous_bearer, listeners, code):
    """A real cross-origin redirect: the plane answers 3xx to another origin; the target never sees
    a request, let alone the bearer, and the refusal is not retried."""
    target = listeners()
    plane = listeners(status=code, location=f"{target.url}/v1/config/self?profile=default")
    monkeypatch.setenv(client.URL_ENV, plane.url)

    with pytest.raises(destination.CredentialDestinationError) as exc:
        client.request("GET", tmp_path, "default")

    assert exc.value.retryable is False
    assert len(plane.seen) == 1 and plane.seen[0]["auth"] == f"Bearer {PLANE_BEARER}"
    assert target.seen == []


def test_boot_fails_closed_on_forbidden_plane_url_without_retry(monkeypatch, tmp_path):
    """Through the real backend: a forbidden destination is a configuration error, so boot fails
    closed at once (no retry sleeps) and nothing is sent."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from plugins.config_backends import remote as remote_pkg
    from plugins.config_backends.remote import backend as backend_mod

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as plane:
        for k, v in remote_env(plane).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv(client.URL_ENV, plane.url.replace("127.0.0.1", "plane.example"))
        sleeps: list = []
        monkeypatch.setattr(backend_mod.time, "sleep", sleeps.append)
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        try:
            with pytest.raises(ConfigBackendUnavailable, match="https"):
                remote_pkg.get_remote_backend().boot(home)
        finally:
            remote_pkg._reset_for_tests()
            cred_mod._IDP_CACHE.clear()
        assert sleeps == []
        assert plane.requests == [] and plane.token_requests == 0


# --- IdP client secret ------------------------------------------------------------------------

def _idp_env(monkeypatch, token_url: str) -> None:
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", token_url)
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", CLIENT_SECRET)
    cred_mod._IDP_CACHE.clear()


@pytest.mark.parametrize("url", [
    "http://foreign.example:8443/token",
    "http://idp.corp.internal/token",               # downgrade: on-prem must still be https
    "https://user:pw@idp.example/token",
    "https://idp.example/token#frag",
])
def test_idp_secret_refused_before_send_for_forbidden_token_url(monkeypatch, captured, url):
    _idp_env(monkeypatch, url)
    with pytest.raises(destination.CredentialDestinationError) as exc:
        cred_mod._idp_token()
    assert exc.value.retryable is False
    assert captured == []


def test_idp_secret_sent_to_allowed_on_prem_https(monkeypatch, captured):
    _idp_env(monkeypatch, "https://idp.corp.internal/oauth2/token")
    with pytest.raises(PlaneCredentialError, match="no access_token"):  # the capture answers {}
        cred_mod._idp_token()
    assert [r.full_url for r in captured] == ["https://idp.corp.internal/oauth2/token"]
    assert CLIENT_SECRET in captured[0].data.decode()


@pytest.mark.parametrize("code", [302, 307, 308])
def test_idp_redirect_is_not_followed_with_the_secret(monkeypatch, listeners, code):
    target = listeners()
    idp = listeners(status=code, location=f"{target.url}/token")
    _idp_env(monkeypatch, f"{idp.url}/token")

    with pytest.raises(destination.CredentialDestinationError) as exc:
        cred_mod._idp_token()

    assert exc.value.retryable is False
    assert len(idp.seen) == 1 and CLIENT_SECRET in idp.seen[0]["body"]
    assert target.seen == []
