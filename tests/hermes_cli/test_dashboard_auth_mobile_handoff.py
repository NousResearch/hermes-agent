"""Dashboard-auth browser-to-Android session handoff tests."""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, quote, urlparse

import pytest

pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import routes as auth_routes
from hermes_cli.dashboard_auth.routes import (
    _reset_mobile_handoffs,
    _reset_password_rate_limit,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

ANDROID_REDIRECT_URI = "com.nousresearch.hermes.android://oauth/callback"
ANDROID_VERIFIER = "v" * 43


@pytest.fixture
def gated_app():
    clear_providers()
    register_provider(StubAuthProvider())
    _reset_mobile_handoffs()
    _reset_password_rate_limit()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    _reset_mobile_handoffs()
    _reset_password_rate_limit()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _code_challenge(verifier: str = ANDROID_VERIFIER) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _handoff_start_path(code_challenge: str | None = None) -> str:
    challenge = _code_challenge() if code_challenge is None else code_challenge
    return (
        f"/mobile-handoff/start?redirect_uri={quote(ANDROID_REDIRECT_URI, safe='')}"
        f"&code_challenge={quote(challenge, safe='')}"
    )


def _logged_in(client: TestClient) -> None:
    first = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert first.status_code == 302
    state = first.headers["location"].split("state=")[1]
    second = client.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert second.status_code == 302


def test_unauthenticated_handoff_start_enters_normal_login(gated_app):
    response = gated_app.get(_handoff_start_path(), follow_redirects=False)

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("/login?next=")
    assert "mobile-handoff%2Fstart" in location
    assert ANDROID_VERIFIER not in location


def test_unauthenticated_handoff_round_trips_through_login(gated_app):
    start = gated_app.get(_handoff_start_path(), follow_redirects=False)
    login_next = parse_qs(urlparse(start.headers["location"]).query)["next"][0]

    login = gated_app.get(
        f"/auth/login?provider=stub&next={quote(login_next, safe='')}",
        follow_redirects=False,
    )
    assert login.status_code == 302

    state = login.headers["location"].split("state=")[1]
    callback = gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert callback.status_code == 302
    callback_target = urlparse(callback.headers["location"])
    assert callback_target.path == "/mobile-handoff/start"
    assert parse_qs(callback_target.query) == parse_qs(urlparse(login_next).query)

    handoff = gated_app.get(callback.headers["location"], follow_redirects=False)
    assert handoff.status_code == 302

    params = parse_qs(urlparse(handoff.headers["location"]).query)
    mobile_client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    consume = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": params["code"][0], "verifier": ANDROID_VERIFIER},
    )

    assert consume.status_code == 200
    assert mobile_client.get("/api/auth/me").status_code == 200


def test_browser_session_mints_one_time_android_handoff_code(gated_app):
    _logged_in(gated_app)

    start = gated_app.get(_handoff_start_path(), follow_redirects=False)

    assert start.status_code == 302
    redirect = urlparse(start.headers["location"])
    assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == ANDROID_REDIRECT_URI

    params = parse_qs(redirect.query)
    code = params["code"][0]
    assert len(code) >= 32
    assert params["base_url"] == ["https://fly-app.fly.dev"]
    assert params["expires_in"] == ["90"]

    mobile_client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    consume = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": code, "verifier": ANDROID_VERIFIER},
    )
    assert consume.status_code == 200
    assert consume.json()["ok"] is True
    assert "hermes_session_at" in consume.headers.get("set-cookie", "")

    me = mobile_client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == "stub-user-1"

    reused = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": code, "verifier": ANDROID_VERIFIER},
    )
    assert reused.status_code == 400


def test_mobile_handoff_requires_verifier_to_consume_code(gated_app):
    _logged_in(gated_app)
    start = gated_app.get(_handoff_start_path(), follow_redirects=False)
    code = parse_qs(urlparse(start.headers["location"]).query)["code"][0]

    mobile_client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    wrong = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": code, "verifier": "wrong" * 10},
    )
    assert wrong.status_code == 400

    correct = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": code, "verifier": ANDROID_VERIFIER},
    )
    assert correct.status_code == 200


def test_mobile_handoff_rejects_expired_code(gated_app):
    _logged_in(gated_app)
    start = gated_app.get(_handoff_start_path(), follow_redirects=False)
    code = parse_qs(urlparse(start.headers["location"]).query)["code"][0]

    with auth_routes._mobile_handoffs_lock:
        _expires_at, payload = auth_routes._mobile_handoffs[code]
        auth_routes._mobile_handoffs[code] = (0.0, payload)

    mobile_client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    expired = mobile_client.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": code, "verifier": ANDROID_VERIFIER},
    )

    assert expired.status_code == 400


def test_mobile_handoff_rejects_non_app_redirect_uri(gated_app):
    _logged_in(gated_app)

    response = gated_app.get(
        "/mobile-handoff/start?redirect_uri=https%3A%2F%2Fevil.example%2Fcb",
        follow_redirects=False,
    )

    assert response.status_code == 400


def test_mobile_handoff_rejects_invalid_code_challenge(gated_app):
    _logged_in(gated_app)

    response = gated_app.get(_handoff_start_path("short"), follow_redirects=False)

    assert response.status_code == 400


def test_mobile_handoff_consume_requires_real_code(gated_app):
    response = gated_app.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": "not-a-real-code", "verifier": ANDROID_VERIFIER},
    )

    assert response.status_code == 400


def test_mobile_handoff_consume_requires_code_and_verifier(gated_app):
    response = gated_app.post(
        "/api/auth/mobile-handoff/consume",
        json={"code": "", "verifier": ""},
    )

    assert response.status_code == 400
