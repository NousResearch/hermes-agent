"""Browser-session to Android native-app PKCE handoff tests."""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
from fastapi.testclient import TestClient


from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth.routes import (
    _MOBILE_HANDOFF_RATE_MAX,
    _reset_mobile_handoff_rate_limit,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

ANDROID_REDIRECT_URI = "com.nousresearch.hermes.android://oauth/callback"
ANDROID_VERIFIER = "v" * 43
ANDROID_STATE = "android-state-7f4b3d"


def _code_challenge(verifier: str = ANDROID_VERIFIER) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _handoff_start_path(
    *,
    redirect_uri: str = ANDROID_REDIRECT_URI,
    code_challenge: str | None = None,
    code_challenge_method: str = "S256",
    state: str = ANDROID_STATE,
) -> str:
    return "/mobile-handoff/start?" + urlencode(
        {
            "redirect_uri": redirect_uri,
            "code_challenge": (
                _code_challenge()
                if code_challenge is None
                else code_challenge
            ),
            "code_challenge_method": code_challenge_method,
            "state": state,
        }
    )


@pytest.fixture
def gated_app(monkeypatch):
    clear_providers()
    register_provider(StubAuthProvider())
    native_flow._reset_for_tests()
    _reset_mobile_handoff_rate_limit()
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True

    client = TestClient(
        web_server.app,
        base_url="https://fly-app.fly.dev",
        follow_redirects=False,
    )
    yield client

    clear_providers()
    native_flow._reset_for_tests()
    _reset_mobile_handoff_rate_limit()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _logged_in(client: TestClient) -> None:
    login = client.get("/auth/login?provider=stub")
    assert login.status_code == 302
    callback = client.get(login.headers["location"])
    assert callback.status_code == 302
    assert callback.headers["location"] == "/"


def _walk_browser_handoff(client: TestClient, *, path: str | None = None) -> dict[str, str]:
    start_path = path or _handoff_start_path()

    auto_sso = client.get(start_path)
    assert auto_sso.status_code == 302
    auto_location = urlparse(auto_sso.headers["location"])
    assert auto_location.path == "/auth/login"
    auto_query = parse_qs(auto_location.query)
    assert auto_query["provider"] == ["stub"]
    assert auto_query["next"] == [start_path]

    upstream = client.get(auto_sso.headers["location"])
    assert upstream.status_code == 302
    callback = client.get(upstream.headers["location"])
    assert callback.status_code == 302
    callback_target = urlparse(callback.headers["location"])
    expected_target = urlparse(start_path)
    assert callback_target.path == expected_target.path
    assert parse_qs(callback_target.query) == parse_qs(expected_target.query)

    app_redirect = client.get(callback.headers["location"])
    assert app_redirect.status_code == 302
    parsed = urlparse(app_redirect.headers["location"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == ANDROID_REDIRECT_URI
    assert app_redirect.headers["cache-control"] == "no-store, no-cache, must-revalidate"

    values = parse_qs(parsed.query)
    return {key: item[0] for key, item in values.items()}


def test_unauthenticated_start_auto_sso_preserves_handoff_query(gated_app):
    path = _handoff_start_path()

    response = gated_app.get(path)

    assert response.status_code == 302
    location = urlparse(response.headers["location"])
    assert location.path == "/auth/login"
    query = parse_qs(location.query)
    assert query["provider"] == ["stub"]
    assert query["next"] == [path]
    # Only the challenge crosses the browser. The mobile verifier never does.
    assert ANDROID_VERIFIER not in response.headers["location"]


def test_authenticated_browser_mints_android_code_redeemable_for_bearer(gated_app):
    values = _walk_browser_handoff(gated_app)

    assert values["state"] == ANDROID_STATE
    assert values["base_url"] == "https://fly-app.fly.dev"
    assert len(values["code"]) >= 32

    mobile = TestClient(
        web_server.app,
        base_url="https://fly-app.fly.dev",
        follow_redirects=False,
    )
    token = mobile.post(
        "/auth/native/token",
        json={"code": values["code"], "code_verifier": ANDROID_VERIFIER},
    )
    assert token.status_code == 200
    body = token.json()
    assert body["token_type"] == "Bearer"
    assert body["access_token"]
    # Access-token-only: the browser's rotating, reuse-detected refresh token
    # is never shared with the app (a second, independent client rotating the
    # same token would break both). See test_mobile_handoff_never_shares...
    assert body["refresh_token"] == ""
    assert "set-cookie" not in {key.lower() for key in token.headers}

    me = mobile.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["user_id"] == "stub-user-1"

    replay = mobile.post(
        "/auth/native/token",
        json={"code": values["code"], "code_verifier": ANDROID_VERIFIER},
    )
    assert replay.status_code == 400


def test_logged_in_browser_handoff_skips_upstream_login(gated_app):
    _logged_in(gated_app)

    response = gated_app.get(_handoff_start_path())

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == ANDROID_REDIRECT_URI
    assert parse_qs(parsed.query)["state"] == [ANDROID_STATE]


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"redirect_uri": "https://evil.example/callback"}, "redirect URI"),
        ({"code_challenge": "short"}, "code challenge"),
        ({"code_challenge_method": "plain"}, "S256"),
        ({"state": ""}, "state"),
    ],
)
def test_mobile_handoff_rejects_invalid_native_parameters(
    gated_app,
    kwargs,
    detail,
):
    _logged_in(gated_app)

    response = gated_app.get(_handoff_start_path(**kwargs))

    assert response.status_code == 400
    assert detail.lower() in response.json()["detail"].lower()


def test_mobile_handoff_base_url_honors_forwarded_prefix(gated_app):
    _logged_in(gated_app)

    response = gated_app.get(
        _handoff_start_path(),
        headers={"X-Forwarded-Prefix": "/hermes"},
    )

    assert response.status_code == 302
    values = parse_qs(urlparse(response.headers["location"]).query)
    assert values["base_url"] == ["https://fly-app.fly.dev/hermes"]


def _browser_refresh_cookie(client: TestClient) -> str:
    """Return the browser session's ``hermes_session_rt`` value (any prefix)."""
    for cookie in client.cookies.jar:
        if cookie.name.endswith("hermes_session_rt"):
            return cookie.value or ""
    return ""


def test_mobile_handoff_never_shares_browser_refresh_token(gated_app):
    """The app's session is access-token-only.

    The browser holds a rotating, reuse-detected refresh token. Handing that
    same token to the Android app (a second, independent client) would let
    either side's rotation invalidate the other and trip Portal's reuse
    detection — potentially revoking the whole family. The handoff must never
    expose the browser RT in the deep link or the redeemed token response.
    """
    _logged_in(gated_app)
    browser_rt = _browser_refresh_cookie(gated_app)
    # Sanity: the browser session really does hold a (rotating) refresh token,
    # so the "must not leak it" assertions below are meaningful.
    assert browser_rt

    response = gated_app.get(_handoff_start_path())
    assert response.status_code == 302
    deep_link = response.headers["location"]
    assert browser_rt not in deep_link

    values = parse_qs(urlparse(deep_link).query)
    mobile = TestClient(
        web_server.app,
        base_url="https://fly-app.fly.dev",
        follow_redirects=False,
    )
    token = mobile.post(
        "/auth/native/token",
        json={"code": values["code"][0], "code_verifier": ANDROID_VERIFIER},
    )
    assert token.status_code == 200
    body = token.json()
    # Access-token-only: no refresh token minted for the app, and the browser's
    # rotating RT never appears anywhere in the native token response.
    assert body["access_token"]
    assert body["refresh_token"] == ""
    assert browser_rt not in token.text


def test_mobile_handoff_rate_limited_per_ip(gated_app):
    """One authenticated caller cannot exhaust the shared native-flow store.

    Each handoff pops its pending entry immediately, so the native-flow per-IP
    *pending* cap never engages; without the per-IP mint budget a single
    authenticated browser could keep the shared, capacity-bounded ``_issued``
    store full and 503 every other user's native login.
    """
    _logged_in(gated_app)

    for _ in range(_MOBILE_HANDOFF_RATE_MAX):
        ok = gated_app.get(_handoff_start_path())
        assert ok.status_code == 302

    blocked = gated_app.get(_handoff_start_path())
    assert blocked.status_code == 429
    assert "too many" in blocked.json()["detail"].lower()
