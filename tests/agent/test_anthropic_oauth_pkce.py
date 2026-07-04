"""Regression tests for the Anthropic OAuth PKCE flow.

Guards against re-introducing the bug where the PKCE ``code_verifier`` was
reused as the OAuth ``state`` parameter, leaking the verifier via the
authorization URL (browser history, Referer headers, auth-server logs) and
removing CSRF protection on the callback path.

History:
  - PR #1775 first fixed this on ``run_hermes_oauth_login()``.
  - PR #2647 (b17e5c10) added ``run_hermes_oauth_login_pure()`` and silently
    copy-pasted the pre-#1775 vulnerable pattern.
  - PR #3107 removed the old function, leaving only the regressed copy.
  - PR #10699 (issue #10693) fixed the regression on the surviving function.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


def _patch_oauth_flow(
    monkeypatch,
    *,
    callback_code: str,
    token_response: Dict[str, Any] | None = None,
    capture_token_request: Dict[str, Any] | None = None,
    capture_auth_url: Dict[str, str] | None = None,
) -> None:
    """Wire up monkeypatches that let ``run_hermes_oauth_login_pure()`` run
    end-to-end without touching a real browser, stdin, or HTTP endpoint.

    ``callback_code`` is the literal string the user would paste back into the
    terminal (``"<code>#<state>"`` format).
    ``capture_token_request`` and ``capture_auth_url`` are out-dict captures
    so the test can introspect what was sent to the auth URL and the token
    endpoint, respectively.
    """
    import urllib.request

    if token_response is None:
        token_response = {
            "access_token": "sk-ant-test-access",
            "refresh_token": "sk-ant-test-refresh",
            "expires_in": 3600,
        }

    def fake_open(url):
        if capture_auth_url is not None:
            capture_auth_url["url"] = url
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)
    # The flow now gates webbrowser.open() behind a graphical-browser check so
    # it never launches a console browser (w3m/lynx) inside the terminal. Tests
    # run headless, so force the GUI path to True — the URL capture relies on
    # webbrowser.open() being invoked.
    monkeypatch.setattr(
        "hermes_cli.auth._can_open_graphical_browser", lambda: True
    )
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: callback_code)

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, *_a, **_kw):
        if capture_token_request is not None:
            capture_token_request["url"] = req.full_url
            capture_token_request["data"] = json.loads(req.data.decode())
            capture_token_request["headers"] = dict(req.headers)
        return _FakeResponse(json.dumps(token_response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_authorization_url_state_is_not_pkce_verifier(monkeypatch, tmp_path):
    """The ``state`` parameter in the authorization URL must NOT equal the
    PKCE ``code_verifier``.

    Reusing the verifier as state leaks the verifier into browser history,
    Referer headers, and auth-server access logs — defeating RFC 7636.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured_url: Dict[str, str] = {}
    captured_token: Dict[str, Any] = {}
    _patch_oauth_flow(
        monkeypatch,
        # state echoed back unchanged so the CSRF guard passes
        callback_code="auth-code-from-anthropic#PLACEHOLDER",
        capture_auth_url=captured_url,
        capture_token_request=captured_token,
    )

    # Stub the callback parse: we need the state echoed back to match. To do
    # that without hardcoding the state value, override input() AFTER seeing
    # the auth URL.
    import builtins

    real_input_calls = {"count": 0}

    def fake_input(*_a, **_kw):
        real_input_calls["count"] += 1
        # First (and only) call is the "Authorization code:" prompt.
        url = captured_url.get("url", "")
        qs = parse_qs(urlparse(url).query)
        state = qs.get("state", [""])[0]
        return f"auth-code-from-anthropic#{state}"

    monkeypatch.setattr(builtins, "input", fake_input)

    from agent.anthropic_adapter import run_hermes_oauth_login_pure

    result = run_hermes_oauth_login_pure()
    assert result is not None, "OAuth flow should succeed with matching state"

    url = captured_url["url"]
    qs = parse_qs(urlparse(url).query)

    assert "state" in qs and qs["state"][0], "authorization URL must include state"
    assert "code_challenge" in qs, "authorization URL must include code_challenge"

    state_in_url = qs["state"][0]
    verifier_sent = captured_token["data"]["code_verifier"]

    # The whole point: state and verifier must be independent values.
    assert state_in_url != verifier_sent, (
        "PKCE code_verifier was reused as OAuth state — regression of #10693 / "
        "#1775. The verifier is supposed to be a secret known only to the "
        "client; placing it in the authorization URL leaks it via browser "
        "history, Referer headers, and auth-server logs."
    )

    # And the verifier MUST NOT appear anywhere in the URL.
    assert verifier_sent not in url, (
        "PKCE verifier leaked into authorization URL — regression of #10693"
    )


def test_login_token_exchange_uses_platform_claude_host(monkeypatch, tmp_path):
    """The login token exchange must hit ``platform.claude.com`` first.

    Anthropic migrated the OAuth token endpoint to ``platform.claude.com``;
    ``console.anthropic.com`` now 404s, so a hardcoded console host makes a
    fresh login impossible (issue #45250 / #49821). The refresh path already
    iterates the new host first — the login path must do the same.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured_token: Dict[str, Any] = {}
    captured_url: Dict[str, str] = {}
    _patch_oauth_flow(
        monkeypatch,
        callback_code="placeholder",
        capture_token_request=captured_token,
        capture_auth_url=captured_url,
    )

    import builtins

    def fake_input(*_a, **_kw):
        qs = parse_qs(urlparse(captured_url.get("url", "")).query)
        state = qs.get("state", [""])[0]
        return f"auth-code#{state}"

    monkeypatch.setattr(builtins, "input", fake_input)

    from agent.anthropic_adapter import run_hermes_oauth_login_pure

    result = run_hermes_oauth_login_pure()

    assert result is not None, "login should succeed against the live host"
    assert captured_token["url"] == "https://platform.claude.com/v1/oauth/token", (
        "login token exchange must target platform.claude.com first, not the "
        "dead console.anthropic.com host (regression of #45250 / #49821)"
    )


def test_login_token_exchange_falls_back_to_console_host(monkeypatch, tmp_path):
    """If ``platform.claude.com`` is unreachable, the login path must fall back
    to the legacy ``console.anthropic.com`` host — mirroring the refresh path's
    fallback list — rather than failing outright.
    """
    import urllib.request

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured_url: Dict[str, str] = {}
    _patch_oauth_flow(
        monkeypatch,
        callback_code="placeholder",
        capture_auth_url=captured_url,
    )

    attempts: list[str] = []

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, *_a, **_kw):
        attempts.append(req.full_url)
        if req.full_url.startswith("https://platform.claude.com"):
            raise RuntimeError("HTTP Error 404: Not Found")
        body = json.dumps(
            {
                "access_token": "sk-ant-test-access",
                "refresh_token": "sk-ant-test-refresh",
                "expires_in": 3600,
            }
        ).encode()
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    import builtins

    def fake_input(*_a, **_kw):
        qs = parse_qs(urlparse(captured_url.get("url", "")).query)
        state = qs.get("state", [""])[0]
        return f"auth-code#{state}"

    monkeypatch.setattr(builtins, "input", fake_input)

    from agent.anthropic_adapter import run_hermes_oauth_login_pure

    result = run_hermes_oauth_login_pure()

    assert result is not None, "login should succeed via the console fallback"
    assert attempts == [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ], "login must try platform.claude.com first, then fall back to console"


def test_callback_state_mismatch_aborts(monkeypatch, tmp_path, caplog):
    """If the state returned in the callback does not match the one we sent
    in the authorization URL, the flow must abort before exchanging the code.

    Without this check, an attacker who tricks the user into pasting a
    crafted ``<code>#<state>`` string can complete the token exchange — the
    CSRF protection that ``state`` is supposed to provide (RFC 6749 §10.12)
    would be absent.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured_token: Dict[str, Any] = {}
    _patch_oauth_flow(
        monkeypatch,
        callback_code="attacker-code#attacker-state-does-not-match",
        capture_token_request=captured_token,
    )

    from agent.anthropic_adapter import run_hermes_oauth_login_pure

    result = run_hermes_oauth_login_pure()

    assert result is None, "mismatched state must abort the flow"
    assert "url" not in captured_token, (
        "token exchange must NOT happen when state mismatches"
    )


def _http_error(url: str, code: int, retry_after: str | None = None):
    """Build a real ``urllib.error.HTTPError`` with optional Retry-After."""
    import io
    import urllib.error
    from email.message import Message

    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "error", headers, io.BytesIO(b""))


def _run_login_with_urlopen(monkeypatch, tmp_path, fake_urlopen):
    """Drive ``run_hermes_oauth_login_pure()`` end-to-end against the given
    ``urlopen`` stub, echoing the CSRF state back so the guard passes.
    Returns ``(result, sleeps)`` where ``sleeps`` records ``time.sleep`` calls.
    """
    import builtins
    import time
    import urllib.request

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured_url: Dict[str, str] = {}
    _patch_oauth_flow(
        monkeypatch,
        callback_code="placeholder",
        capture_auth_url=captured_url,
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    sleeps: list = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def fake_input(*_a, **_kw):
        qs = parse_qs(urlparse(captured_url.get("url", "")).query)
        state = qs.get("state", [""])[0]
        return f"auth-code#{state}"

    monkeypatch.setattr(builtins, "input", fake_input)

    from agent.anthropic_adapter import run_hermes_oauth_login_pure

    return run_hermes_oauth_login_pure(), sleeps


def test_login_token_exchange_retries_on_429(monkeypatch, tmp_path):
    """A transient HTTP 429 from the token endpoint must be retried (honoring
    ``Retry-After``) instead of failing the exchange outright.

    The authorization code the user pasted is single-use: if the exchange
    gives up on a transient rate limit, the code is burned and the user has
    to redo the whole browser round-trip. The Codex device-code login already
    retries 429s with capped backoff; the Anthropic login path must match.
    """
    attempts: list = []

    def fake_urlopen(req, *_a, **_kw):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            raise _http_error(req.full_url, 429, retry_after="3")
        body = json.dumps(
            {
                "access_token": "sk-ant-test-access",
                "refresh_token": "sk-ant-test-refresh",
                "expires_in": 3600,
            }
        ).encode()

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return body

        return _FakeResponse()

    result, sleeps = _run_login_with_urlopen(monkeypatch, tmp_path, fake_urlopen)

    assert result is not None, "a single transient 429 must not fail the login"
    assert attempts == [
        "https://platform.claude.com/v1/oauth/token",
        "https://platform.claude.com/v1/oauth/token",
    ], "the 429'd endpoint must be retried, not skipped to the fallback host"
    assert sleeps == [3], "backoff must honor the server's Retry-After header"


def test_login_429_exhausts_retries_then_falls_back(monkeypatch, tmp_path):
    """Persistent 429s at the primary host must exhaust the capped retry
    budget and then fall through to the legacy console host — preserving the
    endpoint-fallback semantics the host list was introduced for.
    """
    attempts: list = []

    def fake_urlopen(req, *_a, **_kw):
        attempts.append(req.full_url)
        if req.full_url.startswith("https://platform.claude.com"):
            raise _http_error(req.full_url, 429)
        body = json.dumps(
            {
                "access_token": "sk-ant-test-access",
                "refresh_token": "sk-ant-test-refresh",
                "expires_in": 3600,
            }
        ).encode()

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return body

        return _FakeResponse()

    result, sleeps = _run_login_with_urlopen(monkeypatch, tmp_path, fake_urlopen)

    assert result is not None, "login should still succeed via the console fallback"
    assert attempts == ["https://platform.claude.com/v1/oauth/token"] * 4 + [
        "https://console.anthropic.com/v1/oauth/token"
    ], "4 capped attempts at the primary host, then the fallback host"
    # Exponential backoff 2^1..2^3; no sleep after the final failed attempt.
    assert sleeps == [2, 4, 8]


def test_login_non_429_http_error_is_not_retried(monkeypatch, tmp_path):
    """Non-429 HTTP errors (e.g. the 404 from the dead console host) must NOT
    be retried — they fail fast to the next endpoint exactly as before.
    """
    attempts: list = []

    def fake_urlopen(req, *_a, **_kw):
        attempts.append(req.full_url)
        if req.full_url.startswith("https://platform.claude.com"):
            raise _http_error(req.full_url, 404)
        body = json.dumps(
            {
                "access_token": "sk-ant-test-access",
                "refresh_token": "sk-ant-test-refresh",
                "expires_in": 3600,
            }
        ).encode()

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return body

        return _FakeResponse()

    result, sleeps = _run_login_with_urlopen(monkeypatch, tmp_path, fake_urlopen)

    assert result is not None, "login should succeed via the console fallback"
    assert attempts == [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ], "a non-429 HTTP error must move straight to the next host, no retries"
    assert sleeps == [], "no backoff sleeps for non-429 errors"
