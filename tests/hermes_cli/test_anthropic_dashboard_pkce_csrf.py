"""Regression tests for the Anthropic OAuth PKCE flow served by the web dashboard.

``hermes_cli.web_server._start_anthropic_pkce`` / ``_submit_anthropic_pkce``
implement a *second*, independent copy of the Anthropic PKCE login flow
(the first lives in ``agent.anthropic_adapter.run_hermes_oauth_login_pure``,
used by the CLI). The CLI copy was hardened by PR #10699 (issue #10693)
after PR #2647 silently reintroduced a vulnerable pattern that reused the
PKCE ``code_verifier`` as the OAuth ``state`` parameter — see
``tests/agent/test_anthropic_oauth_pkce.py``.

That hardening was never ported to the dashboard copy. This file proves the
dashboard flow currently:

1. Sends the PKCE ``code_verifier`` verbatim as the ``state`` query
   parameter on the authorization URL handed to the browser
   (``hermes_cli/web_server.py`` ``_start_anthropic_pkce``), leaking a value
   that must stay confidential (RFC 7636 §7.2) via browser history, Referer
   headers, and Anthropic's own access logs.
2. Never compares the ``state`` echoed back on the callback against the
   session-stored value in ``_submit_anthropic_pkce`` — the CSRF check
   required by RFC 6749 §10.12 is entirely absent on this path, unlike the
   CLI flow which aborts on ``received_state != oauth_state``.

Each test below asserts the *secure* behavior. Under the current code they
are expected to FAIL, which is the evidence that the bug is real.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

import pytest


def _import_web_server():
    pytest.importorskip("hermes_cli.web_server")
    from hermes_cli import web_server

    if not web_server._ANTHROPIC_OAUTH_AVAILABLE:
        pytest.skip("Anthropic OAuth adapter unavailable in this environment")
    return web_server


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture(autouse=True)
def _clean_oauth_sessions():
    """Every test gets a pristine in-memory session table."""
    web_server = _import_web_server()
    with web_server._oauth_sessions_lock:
        web_server._oauth_sessions.clear()
    yield
    with web_server._oauth_sessions_lock:
        web_server._oauth_sessions.clear()


def test_authorization_url_state_is_not_pkce_verifier():
    """The ``state`` query param must NOT equal the PKCE ``code_verifier``.

    Reusing the verifier as state leaks it via the authorization URL
    (browser history, Referer headers, auth-server access logs), defeating
    RFC 7636's confidentiality requirement for the verifier.
    """
    web_server = _import_web_server()

    session = web_server._start_anthropic_pkce()
    auth_url = session["auth_url"]
    qs = parse_qs(urlparse(auth_url).query)
    state_in_url = qs.get("state", [""])[0]

    with web_server._oauth_sessions_lock:
        stored_verifier = web_server._oauth_sessions[session["session_id"]]["verifier"]

    assert state_in_url != stored_verifier, (
        "Dashboard Anthropic PKCE flow sends the code_verifier as the "
        "OAuth 'state' parameter -- the exact pattern fixed for the CLI "
        "flow by PR #10699 (issue #10693), but never ported here. The "
        "verifier now leaks via the authorization URL."
    )


def test_verifier_never_appears_anywhere_in_auth_url():
    """Defense in depth: the verifier must not appear in ANY query value."""
    web_server = _import_web_server()

    session = web_server._start_anthropic_pkce()
    auth_url = session["auth_url"]

    with web_server._oauth_sessions_lock:
        stored_verifier = web_server._oauth_sessions[session["session_id"]]["verifier"]

    assert stored_verifier not in auth_url, (
        "PKCE code_verifier leaked verbatim into the authorization URL "
        "returned to the browser by _start_anthropic_pkce()."
    )


def test_state_is_cryptographically_independent_of_verifier():
    """``state`` should be an independently generated anti-CSRF token.

    A compliant implementation (mirroring
    ``agent.anthropic_adapter.run_hermes_oauth_login_pure``) generates
    ``state`` via a separate ``secrets.token_urlsafe(...)`` call rather than
    deriving it from the verifier/challenge at all.
    """
    web_server = _import_web_server()

    session_a = web_server._start_anthropic_pkce()
    session_b = web_server._start_anthropic_pkce()

    with web_server._oauth_sessions_lock:
        verifier_a = web_server._oauth_sessions[session_a["session_id"]]["verifier"]
        state_a = web_server._oauth_sessions[session_a["session_id"]]["state"]
        verifier_b = web_server._oauth_sessions[session_b["session_id"]]["verifier"]
        state_b = web_server._oauth_sessions[session_b["session_id"]]["state"]

    assert state_a != verifier_a, "session A: state == verifier (leak)"
    assert state_b != verifier_b, "session B: state == verifier (leak)"


def test_submit_pkce_rejects_state_mismatch(monkeypatch):
    """The token exchange must NOT proceed when the callback state does not
    match the state issued for this session (RFC 6749 Section 10.12 CSRF guard).

    This mirrors ``tests/agent/test_anthropic_oauth_pkce.py``'s state-mismatch
    coverage for the CLI flow, applied to the dashboard's HTTP-facing
    equivalent.
    """
    web_server = _import_web_server()

    session = web_server._start_anthropic_pkce()
    session_id = session["session_id"]

    captured_request: Dict[str, Any] = {}

    def fake_urlopen(req, *_a, **_kw):
        captured_request["url"] = req.full_url
        captured_request["data"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps({
            "access_token": "sk-ant-attacker-exchanged",
            "refresh_token": "sk-ant-refresh",
            "expires_in": 3600,
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    # Attacker completes their OWN authorization and gets the victim to
    # paste a code carrying a state that does not match what
    # _start_anthropic_pkce issued for the victim's session (classic OAuth
    # login-CSRF: the attacker's code gets bound to the victim's account
    # because the callback endpoint never checks state).
    result = web_server._submit_anthropic_pkce(
        session_id, "attacker-code#attacker-controlled-state"
    )

    assert not captured_request, (
        "Token exchange proceeded with a callback 'state' that did not "
        "match the session's issued state -- _submit_anthropic_pkce() "
        "performs no CSRF validation at all, unlike the CLI flow's "
        "'received_state != oauth_state' guard."
    )
    assert result.get("ok") is False


def test_submit_pkce_with_no_state_suffix_does_not_silently_succeed(monkeypatch):
    """Pasting a bare code (no ``#state`` suffix) must not bypass the CSRF
    check by silently falling back to the session's own state value.
    """
    web_server = _import_web_server()

    session = web_server._start_anthropic_pkce()
    session_id = session["session_id"]

    captured_request: Dict[str, Any] = {}

    def fake_urlopen(req, *_a, **_kw):
        captured_request["url"] = req.full_url
        captured_request["data"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps({
            "access_token": "sk-ant-should-not-be-issued",
            "refresh_token": "sk-ant-refresh",
            "expires_in": 3600,
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = web_server._submit_anthropic_pkce(session_id, "some-code-with-no-state")

    assert not captured_request, (
        "_submit_anthropic_pkce() fell back to sess['state'] when the "
        "callback carried no state suffix at all -- this makes the state "
        "check a no-op rather than a genuine CSRF guard."
    )
    assert result.get("ok") is False
