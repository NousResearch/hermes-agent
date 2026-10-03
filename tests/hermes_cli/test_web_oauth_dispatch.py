"""Regression tests for the OAuth dispatcher in hermes_cli.web_server.

Bug history (2026-05-09): the `_OAUTH_PROVIDER_CATALOG` had two entries
flagged ``flow: "pkce"`` — anthropic and minimax-oauth — and the
dispatcher ``start_oauth_login`` hardcoded ``_start_anthropic_pkce()``
for any pkce-flagged provider. So clicking "Login" next to MiniMax in
the dashboard's Keys tab silently launched the Anthropic/Claude OAuth
flow. The Anthropic dashboard flow was later removed entirely because
Hermes must not mint subscription OAuth tokens from an unattended HTTP
endpoint; only the approved external CLI path remains.

The fix:
  1. Catalog entry for minimax-oauth changed from ``flow: "pkce"`` to
     ``flow: "device_code"`` (the actual UX is verification URI + user
     code + background poll, with PKCE as a security extension).
  2. New MiniMax branch added to ``_start_device_code_flow``.
  3. Anthropic's catalog entry changed to ``flow: "external"`` and its
     dashboard start/submit routes now reject the removed flow. Any future
     provider added without an explicit flow gets a clean error instead of
     silently launching Anthropic OAuth.

These tests pin the corrected behavior.
"""
import base64
import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app
import hermes_cli.web_routers.oauth as _rt_oauth
import hermes_cli.web_server_oauth as _web_server_oauth

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _make_profile_home(tmp_path, monkeypatch, profile="coder"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_home = tmp_path / "profiles" / profile
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n")  # identity marker: a bare dir is not a profile
    return profile_home


def _fake_nous_device_data():
    return {
        "device_code": "device-code",
        "user_code": "NOUS-1234",
        "verification_uri": "https://portal.nousresearch.com/device",
        "verification_uri_complete": (
            "https://portal.nousresearch.com/device?user_code=NOUS-1234"
        ),
        "expires_in": 600,
        "interval": 5,
    }


def _oauth_test_jwt(*, label):
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    payload = {
        "sub": f"nas_user:{label}",
        "client_id": "hermes-cli",
        "account_tier": "free",
        "scope": "inference:invoke",
        "exp": int(time.time()) + 3600,
    }
    return f"{segment({'alg': 'RS256'})}.{segment(payload)}.sig"


def _seed_guest(home, *, portal_base_url=None):
    state = {
        "auth_method": "anonymous",
        "account_tier": "anonymous",
        "anon_token": f"anon-{home.name}",
        "client_id": "nas-anonymous",
    }
    if portal_base_url is not None:
        state["portal_base_url"] = portal_base_url
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "active_provider": "nous", "providers": {"nous": state}}),
        encoding="utf-8",
    )


def _wait_for_oauth_session(session_id, timeout=5):
    deadline = time.monotonic() + timeout
    session = {}
    while time.monotonic() < deadline:
        with _web_server_oauth._oauth_sessions_lock:
            session = dict(_web_server_oauth._oauth_sessions.get(session_id) or {})
        if session.get("status") != "pending":
            return session
        time.sleep(0.01)
    raise AssertionError(f"OAuth session did not finish: {session}")


def _invoke_scope_refusal():
    request = httpx.Request("POST", "https://portal.nousresearch.com/oauth/device/code")
    response = httpx.Response(
        400,
        json={
            "error": "invalid_scope",
            "error_description": "unsupported scope inference:invoke",
        },
        request=request,
    )
    return httpx.HTTPStatusError("invalid scope", request=request, response=response)


@pytest.fixture
def oauth_portal_pair():
    """Two real OAuth endpoints: the dashboard launch profile and a named profile."""
    hits = []
    servers = []
    threads = []
    release_status = threading.Event()
    release_status.set()

    def start(label):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                form = parse_qs(raw_body.decode("utf-8"))
                hits.append((label, self.path))
                if self.path == "/api/oauth/device/code":
                    payload = {
                        "device_code": f"{label}-device",
                        "user_code": f"{label.upper()}-NOUS",
                        "verification_uri": f"https://{label}.example/verify",
                        "verification_uri_complete": f"https://{label}.example/verify?code=1",
                        "expires_in": 600,
                        "interval": 5,
                    }
                elif self.path == "/oauth/code":
                    payload = {
                        "user_code": f"{label.upper()}-MINIMAX",
                        "verification_uri": f"https://{label}.example/verify",
                        "expired_in": 600,
                        "interval": 2000,
                        "state": form["state"][0],
                    }
                elif self.path == "/api/anonymous/promotion-intent":
                    body = json.loads(raw_body)
                    if body.get("device_code") != f"{label}-device":
                        self.send_response(400)
                        self.end_headers()
                        return
                    payload = {
                        "claim_code": f"{label.upper()}-CLAIM",
                        "claim_url": (
                            f"http://127.0.0.1:{getattr(self.server, 'server_port')}/claim/{label}"
                        ),
                        "expires_in": 600,
                        "interval": 0,
                    }
                elif self.path == "/api/anonymous/promotion-status":
                    if not release_status.wait(timeout=5):
                        self.send_response(503)
                        self.end_headers()
                        return
                    payload = {
                        "status": "completed",
                        "user_id": f"nas_user:{label}",
                        "account_email": f"{label}@example.test",
                    }
                elif self.path == "/api/oauth/token":
                    if form.get("device_code") != [f"{label}-device"]:
                        self.send_response(400)
                        self.end_headers()
                        return
                    payload = {
                        "access_token": _oauth_test_jwt(label=label),
                        "refresh_token": f"{label}-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "inference:invoke",
                        "inference_base_url": "https://inference-api.nousresearch.com/v1",
                    }
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A002
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        return f"http://127.0.0.1:{server.server_port}"

    try:
        yield {
            "launch": start("launch"),
            "profile": start("profile"),
            "hits": hits,
            "release_status": release_status,
        }
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def _configure_profile_portals(tmp_path, monkeypatch, portals):
    """Launch profile A and named profile B, each with conflicting HERMES/NOUS overrides."""
    from hermes_cli import anon_auth

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.setenv("HERMES_ANON_API_SECRET", "dashboard-test-secret")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", portals["launch"])
    monkeypatch.setenv("NOUS_PORTAL_BASE_URL", portals["profile"])
    monkeypatch.setattr(anon_auth, "DEFAULT_NOUS_PORTAL_URL", portals["launch"])

    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (profile_home / ".env").write_text(
        "\n".join((
            f"HERMES_PORTAL_BASE_URL={portals['profile']}",
            f"NOUS_PORTAL_BASE_URL={portals['launch']}",
            "",
        )),
        encoding="utf-8",
    )
    return profile_home


@pytest.mark.parametrize(
    ("provider_id", "override_name", "poller_name", "request_path"),
    [
        ("nous", "HERMES_PORTAL_BASE_URL", "_nous_plain_poller", "/api/oauth/device/code"),
        ("minimax-oauth", "MINIMAX_PORTAL_BASE_URL", "_minimax_poller", "/oauth/code"),
    ],
)
def test_named_profile_oauth_start_uses_its_own_endpoint(
    tmp_path, monkeypatch, oauth_portal_pair,
    provider_id, override_name, poller_name, request_path,
):
    """A->B->A dashboard starts must contact each profile's own OAuth endpoint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(override_name, oauth_portal_pair["launch"])
    if override_name == "HERMES_PORTAL_BASE_URL":
        monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (profile_home / ".env").write_text(
        f"{override_name}={oauth_portal_pair['profile']}\n", encoding="utf-8")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(
            f"hermes_cli.web_server_oauth.{poller_name}", return_value=None))
        if provider_id == "nous":
            stack.enter_context(patch("hermes_cli.anon_auth.guest_enabled", return_value=False))
        for profile in (None, "coder", None):
            suffix = f"?profile={profile}" if profile else ""
            response = client.post(
                f"/api/providers/oauth/{provider_id}/start{suffix}", headers=HEADERS)
            assert response.status_code == 200, response.text
            _web_server_oauth._oauth_sessions.pop(response.json()["session_id"], None)

    assert [label for label, path in oauth_portal_pair["hits"] if path == request_path] == [
        "launch", "profile", "launch"]


@pytest.mark.parametrize("stored_portal", [None, "stale"])
def test_named_profile_free_tier_promotion_uses_selected_endpoint_not_guest_state(
    tmp_path, monkeypatch, oauth_portal_pair, stored_portal,
):
    """A named-profile promotion ignores a missing/stale endpoint saved on its guest identity."""
    profile_home = _configure_profile_portals(tmp_path, monkeypatch, oauth_portal_pair)
    _seed_guest(
        profile_home,
        portal_base_url=(oauth_portal_pair["launch"] if stored_portal == "stale" else None),
    )

    response = client.post(
        "/api/providers/oauth/nous/start?profile=coder",
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    try:
        session = _wait_for_oauth_session(session_id)
        assert session["status"] == "approved", session
        assert response.json()["user_code"] == "PROFILE-CLAIM"
        for path in (
            "/api/oauth/device/code",
            "/api/anonymous/promotion-intent",
            "/api/anonymous/promotion-status",
            "/api/oauth/token",
        ):
            assert [label for label, hit_path in oauth_portal_pair["hits"] if hit_path == path] == [
                "profile"
            ]
        saved = json.loads((profile_home / "auth.json").read_text(encoding="utf-8"))
        assert saved["providers"]["nous"]["portal_base_url"] == oauth_portal_pair["profile"]
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)


def test_free_tier_background_flow_captures_scoped_endpoint_across_a_b_a(
    tmp_path, monkeypatch, oauth_portal_pair,
):
    """Device, intent, background status, and token calls stay on A->B->A's selected endpoints.

    Each profile deliberately stores the other profile's stale portal. Both also define the legacy
    ``NOUS_PORTAL_BASE_URL`` as the other endpoint, proving ``HERMES_PORTAL_BASE_URL`` precedence.
    The status calls are held until all three start routes return, so the token leg runs only after
    the ambient dashboard profile has changed twice.
    """
    profile_home = _configure_profile_portals(tmp_path, monkeypatch, oauth_portal_pair)
    _seed_guest(tmp_path, portal_base_url=oauth_portal_pair["profile"])
    _seed_guest(profile_home, portal_base_url=oauth_portal_pair["launch"])
    oauth_portal_pair["release_status"].clear()

    responses = []
    session_ids = []
    try:
        for profile in (None, "coder", None):
            suffix = f"?profile={profile}" if profile else ""
            response = client.post(
                f"/api/providers/oauth/nous/start{suffix}",
                headers=HEADERS,
            )
            assert response.status_code == 200, response.text
            responses.append(response.json())
            session_ids.append(response.json()["session_id"])

        deadline = time.monotonic() + 5
        status_calls = []
        while time.monotonic() < deadline:
            status_calls = [
                label for label, path in oauth_portal_pair["hits"]
                if path == "/api/anonymous/promotion-status"
            ]
            if len(status_calls) == 3:
                break
            time.sleep(0.01)
        assert len(status_calls) == 3
    finally:
        oauth_portal_pair["release_status"].set()

    try:
        sessions = [_wait_for_oauth_session(session_id) for session_id in session_ids]
        assert [session["status"] for session in sessions] == ["approved"] * 3
        assert [body["user_code"] for body in responses] == [
            "LAUNCH-CLAIM", "PROFILE-CLAIM", "LAUNCH-CLAIM"
        ]
        for path in ("/api/oauth/device/code", "/api/anonymous/promotion-intent"):
            assert [label for label, hit_path in oauth_portal_pair["hits"] if hit_path == path] == [
                "launch", "profile", "launch"
            ]
        for path in ("/api/anonymous/promotion-status", "/api/oauth/token"):
            labels = [label for label, hit_path in oauth_portal_pair["hits"] if hit_path == path]
            assert labels.count("launch") == 2
            assert labels.count("profile") == 1

        default_saved = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        profile_saved = json.loads((profile_home / "auth.json").read_text(encoding="utf-8"))
        assert default_saved["providers"]["nous"]["portal_base_url"] == oauth_portal_pair["launch"]
        assert profile_saved["providers"]["nous"]["portal_base_url"] == oauth_portal_pair["profile"]
    finally:
        for session_id in session_ids:
            _web_server_oauth._oauth_sessions.pop(session_id, None)


def test_minimax_login_does_not_launch_anthropic_flow():
    """Click 'Login' on MiniMax → MUST NOT return claude.ai auth_url."""
    fake_user_code_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://api.minimax.io/oauth/verify",
        # `expired_in` < 1e12 so the heuristic treats it as seconds.
        "expired_in": 600,
        "interval": 2000,
        "state": "stub-state",
    }
    with patch(
        "hermes_cli.auth._minimax_request_user_code",
        return_value=fake_user_code_resp,
    ), patch(
        "hermes_cli.auth._minimax_pkce_pair",
        return_value=("verifier-stub", "challenge-stub", "stub-state"),
    ), patch(
        "hermes_cli.web_server_oauth._minimax_poller",
        return_value=None,
    ):
        resp = client.post(
            "/api/providers/oauth/minimax-oauth/start",
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The bug used to return Anthropic's auth_url — make sure the response
    # references neither the auth_url field nor anything Claude-related.
    assert "auth_url" not in body
    assert "claude.ai" not in str(body).lower()

    # And the response IS the device-code shape pointing at MiniMax.
    assert body["flow"] == "device_code"
    assert "minimax" in body["verification_url"].lower()
    assert body["user_code"] == "ABCD-1234"
    assert body["expires_in"] == 600




def test_minimax_start_route_honors_poller_mock_on_owning_module(tmp_path, monkeypatch):
    """A monkeypatch on ``web_server_oauth._minimax_poller`` must intercept the
    background thread the /start route spawns.

    Regression: the router imported the poller functions at module level, so a
    test's patch on the owning module was a no-op — the REAL poller ran on the
    leaked daemon thread, hit the live MiniMax endpoint from CI, and its
    getaddrinfo call segfaulted a later test's collection (CI flake, run
    34323790818). The router must resolve pollers late, at spawn time.
    """
    import threading

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fake_user_code_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://api.minimax.io/oauth/verify",
        "expired_in": 600,
        "interval": 2000,
        "state": "stub-state",
    }
    mock_ran = threading.Event()
    real_network_hit = threading.Event()

    def fake_poller(session_id):
        mock_ran.set()

    def fail_poll_token(**kwargs):
        real_network_hit.set()
        raise AssertionError("real _minimax_poller body must not run under the mock")

    with patch(
        "hermes_cli.auth._minimax_request_user_code",
        return_value=fake_user_code_resp,
    ), patch(
        "hermes_cli.auth._minimax_pkce_pair",
        return_value=("verifier-stub", "challenge-stub", "stub-state"),
    ), patch(
        "hermes_cli.auth._minimax_poll_token",
        fail_poll_token,
    ), patch(
        "hermes_cli.web_server_oauth._minimax_poller",
        fake_poller,
    ):
        resp = client.post("/api/providers/oauth/minimax-oauth/start", headers=HEADERS)
        assert resp.status_code == 200, resp.text
        assert mock_ran.wait(timeout=5), "patched poller never ran — router bypassed the seam"
        assert not real_network_hit.is_set()
    _web_server_oauth._oauth_sessions.pop(resp.json()["session_id"], None)


def test_oauth_provider_status_uses_profile_query(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    profile_home = _make_profile_home(tmp_path, monkeypatch)
    observed_homes = []

    def fake_status():
        observed_homes.append(get_hermes_home())
        return {"logged_in": False, "source": None}

    fake_catalog = ({
        "id": "fake-oauth",
        "name": "Fake OAuth",
        "flow": "pkce",
        "cli_command": "hermes auth add fake-oauth",
        "docs_url": "https://example.com",
        "status_fn": fake_status,
    },)
    monkeypatch.setattr(_web_server_oauth, "_OAUTH_PROVIDER_CATALOG", fake_catalog)

    resp = client.get("/api/providers/oauth?profile=coder", headers=HEADERS)

    assert resp.status_code == 200, resp.text
    assert observed_homes == [profile_home]


def test_oauth_start_stores_profile_for_background_completion(tmp_path, monkeypatch):

    _make_profile_home(tmp_path, monkeypatch)
    fake_user_code_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://api.minimax.io/oauth/verify",
        "expired_in": 600,
        "interval": 2000,
        "state": "stub-state",
    }
    with patch(
        "hermes_cli.auth._minimax_request_user_code",
        return_value=fake_user_code_resp,
    ), patch(
        "hermes_cli.auth._minimax_pkce_pair",
        return_value=("verifier-stub", "challenge-stub", "stub-state"),
    ), patch(
        "hermes_cli.web_server_oauth._minimax_poller",
        return_value=None,
    ):
        resp = client.post(
            "/api/providers/oauth/minimax-oauth/start?profile=coder",
            headers=HEADERS,
        )

    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    try:
        assert _web_server_oauth._oauth_sessions[session_id]["profile"] == "coder"
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)


def test_oauth_session_cannot_be_polled_or_cancelled_from_another_profile(
    tmp_path, monkeypatch
):
    """A named-profile OAuth session must reject default-profile retargeting."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "profiles" / "worker").mkdir(parents=True)
    (tmp_path / "profiles" / "worker" / "config.yaml").write_text("{}\n")  # identity marker
    session_id, _session = _rt_oauth._new_oauth_session(
        "xai-oauth", "device_code", profile="worker"
    )
    try:
        poll_resp = client.get(
            f"/api/providers/oauth/xai-oauth/poll/{session_id}",
            headers=HEADERS,
        )
        assert poll_resp.status_code == 400, poll_resp.text
        assert "profile" in poll_resp.text.lower()

        cancel_resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}",
            headers=HEADERS,
        )
        assert cancel_resp.status_code == 400, cancel_resp.text
        assert "profile" in cancel_resp.text.lower()
        assert session_id in _web_server_oauth._oauth_sessions

        correct_poll = client.get(
            f"/api/providers/oauth/xai-oauth/poll/{session_id}?profile=worker",
            headers=HEADERS,
        )
        assert correct_poll.status_code == 200, correct_poll.text

        correct_cancel = client.delete(
            f"/api/providers/oauth/sessions/{session_id}?profile=worker",
            headers=HEADERS,
        )
        assert correct_cancel.status_code == 200, correct_cancel.text
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)






def test_codex_dashboard_worker_stops_polling_after_cancel(tmp_path, monkeypatch):
    """A real DELETE mid-poll must stop the worker before it exchanges/saves tokens.

    Regression for IA-01: cancelling only popped the session dict; the
    background worker kept polling/exchanging/saving regardless, and once
    the session was gone `_oauth_session_profile()` fell back to the
    caller's current profile scope instead of the one the login started
    in. The fix marks the dict `cancelled` before popping, and the worker
    checks that flag before every remaining step.

    Exercises the actual `DELETE /api/providers/oauth/sessions/{id}`
    endpoint (rather than mutating the session dict directly) so the
    endpoint/worker race and the real removal from `_oauth_sessions` are
    both under test.
    """
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 3,
                    "user_code": "CODEX-1234",
                })
            raise AssertionError(
                f"worker must stop before calling {url} once cancelled"
            )

    saved = []
    _make_profile_home(tmp_path, monkeypatch, profile="coder")
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(auth_mod, "_save_codex_tokens", lambda tokens: saved.append(tokens))

    sid, _ = _rt_oauth._new_oauth_session("openai-codex", "device_code", profile="coder")

    def fake_sleep(_interval):
        # Simulate a real concurrent DELETE /api/providers/oauth/sessions/{sid}
        # firing while the worker is asleep between polls.
        resp = client.delete(
            f"/api/providers/oauth/sessions/{sid}?profile=coder",
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    monkeypatch.setattr(ws.time, "sleep", fake_sleep)

    try:
        _rt_oauth._codex_full_login_worker(sid)

        assert saved == []
        assert sid not in _web_server_oauth._oauth_sessions
    finally:
        _web_server_oauth._oauth_sessions.pop(sid, None)


def test_codex_worker_final_save_is_atomic_with_cancel_delete(tmp_path, monkeypatch):
    """The final cancellation check and the token save must be one atomic
    section under `_oauth_sessions_lock`.

    Regression: checking `cancelled` and calling `_save_codex_tokens()` used
    to be two separate steps with no lock held across them, so a DELETE
    landing in that gap flipped the flag too late for the worker to see it
    and the tokens were saved anyway. This drives a real DELETE from another
    thread exactly while the worker holds the lock for its check+save, and
    asserts DELETE stays blocked for the whole critical section instead of
    slipping in between the check and the save.
    """
    import threading

    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 0,
                    "user_code": "CODEX-1234",
                })
            return _Resp(200, {
                "authorization_code": "auth-code",
                "code_verifier": "verifier",
            })

    class _TokenClient(_Client):
        def post(self, url, **kwargs):
            return _Resp(200, {"access_token": "at", "refresh_token": "rt"})

    clients = iter([_Client, _Client, _TokenClient])
    _make_profile_home(tmp_path, monkeypatch, profile="coder")
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: next(clients)(*a, **k))

    saved = []
    delete_threads = []
    delete_started = threading.Event()
    delete_finished = threading.Event()

    def fake_save(tokens):
        # We are inside the worker's critical section right now (holding
        # _oauth_sessions_lock). Fire a real DELETE from another thread and
        # prove it cannot complete until this section releases the lock.
        # Do NOT join the DELETE thread here: it is blocked on the very
        # lock this section holds, so joining here would deadlock.
        delete_thread = threading.Thread(target=_fire_delete, daemon=True)
        delete_threads.append(delete_thread)
        delete_thread.start()
        delete_started.wait(timeout=2)
        still_blocked = not delete_finished.wait(timeout=0.2)
        saved.append((tokens, still_blocked))

    def _fire_delete():
        delete_started.set()
        client.delete(
            f"/api/providers/oauth/sessions/{sid}?profile=coder",
            headers=HEADERS,
        )
        delete_finished.set()

    monkeypatch.setattr(auth_mod, "_save_codex_tokens", fake_save)
    monkeypatch.setattr(ws.time, "sleep", lambda *_a, **_k: None)

    # Lock order: every saver enters the profile scope (_SKILLS_PROFILE_LOCK) before
    # _oauth_sessions_lock; one saver taking them the other way round deadlocks with the rest.
    real_scope = _rt_oauth._profile_scope
    session_lock_held_at_scope = []

    @contextlib.contextmanager
    def recording_scope(profile):
        session_lock_held_at_scope.append(_web_server_oauth._oauth_sessions_lock.locked())
        with real_scope(profile) as scoped:
            yield scoped

    monkeypatch.setattr(_rt_oauth, "_profile_scope", recording_scope)

    sid, _ = _rt_oauth._new_oauth_session("openai-codex", "device_code", profile="coder")

    _rt_oauth._codex_full_login_worker(sid)
    assert session_lock_held_at_scope == [False], "profile scope entered while holding the session lock"

    # The lock is released now (worker returned), so the DELETE thread can
    # finally complete.
    delete_threads[0].join(timeout=2)

    assert len(saved) == 1
    tokens, delete_was_still_blocked_during_save = saved[0]
    assert tokens == {"access_token": "at", "refresh_token": "rt"}
    assert delete_was_still_blocked_during_save, (
        "DELETE must block until the worker's check+save critical section "
        "finishes, not slip in between the check and the save"
    )
    # DELETE arrived after the point of no return (save already committed),
    # so this is the legitimate too-late-to-cancel outcome: token saved,
    # session subsequently removed by the now-unblocked DELETE.
    assert sid not in _web_server_oauth._oauth_sessions




def test_nous_dashboard_poller_preserves_effective_scope_when_token_omits_scope(monkeypatch):
    from hermes_cli import auth as auth_mod

    session_id = "nous-effective-scope-test"
    _web_server_oauth._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "nous",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "hermes-cli",
        "device_code": "device-code",
        "interval": 5,
        "expires_at": time.time() + 600,
        "scope": auth_mod.DEFAULT_NOUS_SCOPE,
    }
    captured_state = {}

    def fake_refresh_nous_oauth_from_state(state, **kwargs):
        captured_state.update(state)
        return {**state, "agent_key": "jwt-agent-key"}

    monkeypatch.setattr(
        auth_mod,
        "_poll_for_token",
        lambda **kwargs: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "refresh_nous_oauth_from_state",
        fake_refresh_nous_oauth_from_state,
    )
    monkeypatch.setattr(auth_mod, "persist_nous_credentials", lambda state: None)

    try:
        _web_server_oauth._nous_plain_poller(session_id)
        assert captured_state["scope"] == auth_mod.DEFAULT_NOUS_SCOPE
        assert _web_server_oauth._oauth_sessions[session_id]["status"] == "approved"
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)






def test_anthropic_dashboard_oauth_is_removed_and_external():
    """Anthropic subscription OAuth is not minted by the dashboard anymore."""

    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}
    assert providers["anthropic"]["flow"] == "external"

    before_sessions = set(_web_server_oauth._oauth_sessions)
    start_resp = client.post(
        "/api/providers/oauth/anthropic/start",
        headers=HEADERS,
    )
    assert start_resp.status_code == 400, start_resp.text
    assert "claude.ai" not in start_resp.text

    submit_resp = client.post(
        "/api/providers/oauth/anthropic/submit",
        headers=HEADERS,
        json={"session_id": "unused", "code": "unused"},
    )
    assert submit_resp.status_code == 400, submit_resp.text
    assert set(_web_server_oauth._oauth_sessions) == before_sessions


def test_accounts_offers_every_oauth_provider_from_catalog():
    """PARITY CONTRACT: every accounts-tab provider in the unified catalog (the
    `hermes model` universe) must be offered by /api/providers/oauth. This keeps
    the desktop Accounts tab in lockstep with the CLI picker — no provider the
    CLI can sign into may be missing from the GUI.
    """
    from hermes_cli.provider_catalog import provider_catalog

    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    offered = {p["id"] for p in resp.json()["providers"]}
    for d in provider_catalog():
        if d.tab == "accounts":
            assert d.slug in offered, (
                f"{d.slug} is an accounts-tab provider in `hermes model` but is "
                f"missing from the desktop Accounts tab (/api/providers/oauth)"
            )






def test_external_oauth_disconnect_rejected_before_auth_mutation(monkeypatch):
    """DELETE must not pretend to remove credentials owned by another CLI."""
    from hermes_cli import auth as auth_mod

    def fail_clear_provider_auth(provider_id=None):
        raise AssertionError("external providers must not reach clear_provider_auth")

    monkeypatch.setattr(auth_mod, "clear_provider_auth", fail_clear_provider_auth)

    resp = client.delete("/api/providers/oauth/qwen-oauth", headers=HEADERS)
    assert resp.status_code == 400, resp.text


def test_env_sourced_oauth_status_is_not_disconnectable(monkeypatch):
    """An env/.env-backed Anthropic API key is removed from Keys, not OAuth Accounts."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}

    assert providers["anthropic"]["status"]["source"] == "env_var"
    assert providers["anthropic"]["disconnectable"] is False

    delete_resp = client.delete("/api/providers/oauth/anthropic", headers=HEADERS)
    assert delete_resp.status_code == 400, delete_resp.text


def test_xai_dashboard_poller_seeds_single_entry_and_clears_suppression(tmp_path, monkeypatch):
    """The dashboard device-code poller must leave exactly ONE pool entry — the
    singleton-seeded ``device_code`` source — and must NOT create a parallel
    ``manual:dashboard_*`` entry.

    Dedupe: a parallel dashboard entry would share the singleton's single-use
    refresh token, and two entries racing the same rotation ->
    ``refresh_token_reused`` (on main, the dashboard login inserted exactly
    such a duplicate alongside the singleton seed). The poller writes the
    singleton only; the seed is the single source of truth.

    Suppression: an interactive dashboard login must also clear any
    ``device_code`` suppression left by a prior ``hermes auth remove
    xai-oauth``.
    """
    from hermes_cli import auth as auth_mod
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    # Existing chat provider must not be overwritten by dashboard OAuth.
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "openrouter",
                "providers": {},
            }
        ),
        encoding="utf-8",
    )

    # Prior `hermes auth remove xai-oauth` left the source suppressed.
    auth_mod.suppress_credential_source("xai-oauth", "device_code")
    assert auth_mod.is_source_suppressed("xai-oauth", "device_code") is True

    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_discovery",
        lambda *a, **k: {"token_endpoint": "https://auth.x.ai/token"},
    )
    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_poll_device_token",
        lambda client, **kwargs: {
            "access_token": "xai-dashboard-access",
            "refresh_token": "rt-dashboard",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )

    session_id = "xai-dashboard-dedupe-test"
    _web_server_oauth._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "device_code": "device-code",
        "interval": 5,
        "expires_at": time.time() + 600,
    }
    try:
        _web_server_oauth._xai_device_poller(session_id)
        assert _web_server_oauth._oauth_sessions[session_id]["status"] == "approved"
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)

    # The interactive dashboard login cleared the suppression marker.
    assert auth_mod.is_source_suppressed("xai-oauth", "device_code") is False

    after = json.loads(auth_path.read_text(encoding="utf-8"))
    assert after["active_provider"] == "openrouter"
    assert after["providers"]["xai-oauth"]["tokens"]["access_token"] == "xai-dashboard-access"

    # The credential pool has exactly one entry, seeded from the
    # singleton as ``device_code`` — no parallel ``manual:dashboard_*``
    # duplicate sharing the single-use refresh token.
    entries = load_pool("xai-oauth").entries()
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    assert entries[0].refresh_token == "rt-dashboard"
    assert not any(
        getattr(e, "source", "").startswith("manual:dashboard") for e in entries
    )






def test_status_falls_through_to_generic_dispatcher_for_catalog_only_provider():
    """Accounts-tab providers with no hardcoded branch reflect REAL status.

    Providers appended to the Accounts tab from the unified provider_catalog()
    carry status_fn=None and may have no explicit branch in
    _resolve_provider_status. Before the fallthrough they rendered permanently
    logged-out; now they dispatch to hermes_cli.auth.get_auth_status (the
    canonical slug dispatcher) so membership AND status both auto-extend.
    """

    fake_status = {
        "logged_in": True,
        "provider": "some-future-oauth",
        "name": "Future OAuth Provider",
        "access_token": "sk-future-secret-token-xyz",
        "expires_at": "2026-12-01T00:00:00Z",
        "has_refresh_token": True,
    }
    with patch("hermes_cli.auth.get_auth_status", return_value=fake_status):
        out = _rt_oauth._resolve_provider_status("some-future-oauth", None)

    assert out["logged_in"] is True
    assert out["source"] == "some-future-oauth"
    assert out["source_label"] == "Future OAuth Provider"
    # Token is previewed, never returned whole.
    assert out["token_preview"] and "sk-future-secret-token-xyz" not in out["token_preview"]
    assert out["expires_at"] == "2026-12-01T00:00:00Z"
    assert out["has_refresh_token"] is True


def test_qwen_status_card_carries_the_getter_expiry_token_and_auth_file():
    """The Qwen card reads the keys get_qwen_auth_status actually returns."""
    from datetime import datetime

    expires_at_ms = 1_790_541_000_123
    qwen_status = {
        "logged_in": True, "auth_file": "/home/u/.qwen/oauth_creds.json", "source": "qwen-cli",
        "api_key": "qwen-access-token-secret-xyz", "expires_at_ms": expires_at_ms,
    }
    with patch("hermes_cli.auth.get_qwen_auth_status", return_value=qwen_status):
        out = _rt_oauth._resolve_provider_status("qwen-oauth", None)

    assert out["logged_in"] is True
    # Same wire shape as the other cards: an ISO string with an offset, same instant.
    assert isinstance(out["expires_at"], str)
    expires_at = datetime.fromisoformat(out["expires_at"])
    assert expires_at.tzinfo is not None
    assert round(expires_at.timestamp() * 1000) == expires_at_ms
    assert out["source_label"] == "/home/u/.qwen/oauth_creds.json"
    assert out["token_preview"] and "qwen-access-token-secret-xyz" not in out["token_preview"]






@pytest.mark.parametrize("provider", ["xai-oauth", "minimax-oauth"])
@pytest.mark.parametrize("event", ["cancel", "expire"])
def test_dashboard_poller_saves_only_into_the_profile_the_login_started_in(tmp_path, monkeypatch, provider, event):
    """A login started with ``?profile=coder`` saves into coder or nowhere. The pollers looked the
    profile up by session id AFTER the provider answered; a cancel (DELETE) or the 15-minute sweep
    had removed the registry entry by then, so the tokens landed in the dashboard's launch profile
    (the Codex path was fixed the same way as IA-01). A cancelled login saves nothing."""
    from hermes_cli import auth as auth_mod
    from hermes_constants import get_hermes_home

    coder_home = _make_profile_home(tmp_path, monkeypatch, profile="coder")
    saved_into = []
    sid, sess = _rt_oauth._new_oauth_session(provider, "device_code", profile="coder")

    def provider_answers(*_a, **_k):
        if event == "cancel":
            resp = client.delete(f"/api/providers/oauth/sessions/{sid}?profile=coder", headers=HEADERS)
            assert resp.status_code == 200, resp.text
        else:
            _web_server_oauth._oauth_sessions.pop(sid, None)
        return {"access_token": "at", "refresh_token": "rt", "id_token": "", "expires_in": 3600,
                "expired_in": 3600, "token_type": "Bearer"}

    record = lambda *_a, **_k: saved_into.append(str(get_hermes_home()))
    if provider == "xai-oauth":
        sess.update(device_code="dc", expires_at=time.time() + 600, interval=1)
        monkeypatch.setattr(auth_mod, "_xai_oauth_discovery", lambda *a, **k: {"token_endpoint": "https://x.test/t"})
        monkeypatch.setattr(auth_mod, "_xai_oauth_poll_device_token", provider_answers)
        monkeypatch.setattr(auth_mod, "_save_xai_oauth_tokens", record)
        monkeypatch.setattr(auth_mod, "mark_provider_active_if_unset", lambda *a, **k: None)
        monkeypatch.setattr(auth_mod, "unsuppress_credential_source", lambda *a, **k: None)
        poller = _web_server_oauth._xai_device_poller
    else:
        sess.update(portal_base_url="https://m.test", client_id="cid", user_code="uc", code_verifier="cv",
                    expired_in_raw=600, interval_ms=1000)
        monkeypatch.setattr(auth_mod, "_minimax_poll_token", provider_answers)
        monkeypatch.setattr(auth_mod, "_minimax_save_auth_state", record)
        poller = _web_server_oauth._minimax_poller
    try:
        poller(sid)
    finally:
        _web_server_oauth._oauth_sessions.pop(sid, None)

    assert saved_into == ([] if event == "cancel" else [str(coder_home)])
