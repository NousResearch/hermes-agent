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

        def read(self, size: int = -1):
            if size is None or size < 0:
                return self._body
            return self._body[:size]

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

        def read(self, size: int = -1):
            if size is None or size < 0:
                return self._body
            return self._body[:size]

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
