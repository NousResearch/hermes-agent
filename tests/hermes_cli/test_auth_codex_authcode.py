"""Tests for the OpenAI Codex browser Authorization Code + PKCE OAuth flow.

Covers PKCE helpers, authorize-URL building, the loopback callback server,
token-exchange request shape, and error handling. Live OpenAI login is not
required here — the handshake with auth.openai.com is attested separately.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest

from hermes_cli import auth as auth_mod
from hermes_cli.auth import (
    AuthError,
    CODEX_OAUTH_AUTHORIZE_URL,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_SCOPE,
    CODEX_OAUTH_TOKEN_URL,
    _codex_authcode_build_authorize_url,
    _codex_authcode_free_port,
    _codex_authcode_make_callback_handler,
    _codex_authcode_wait_for_callback,
    _oauth_pkce_code_challenge,
    _oauth_pkce_code_verifier,
)


# --------------------------------------------------------------------------------------
# PKCE primitives
# --------------------------------------------------------------------------------------


def test_pkce_verifier_is_urlsafe_and_bounded():
    v = _oauth_pkce_code_verifier()
    assert isinstance(v, str) and v
    assert "+" not in v and "/" not in v and "=" not in v
    assert len(v) <= 128


def test_pkce_challenge_is_s256_of_verifier():
    import base64
    import hashlib

    v = _oauth_pkce_code_verifier()
    c = _oauth_pkce_code_challenge(v)
    expected = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).decode().rstrip("=")
    assert c == expected


# --------------------------------------------------------------------------------------
# Authorize URL
# --------------------------------------------------------------------------------------


def test_build_authorize_url_contains_required_oauth_params():
    url = _codex_authcode_build_authorize_url(
        client_id="cid-123",
        redirect_uri="http://127.0.0.1:8765/callback",
        scope="openai.chatgpt.experimental",
        state="state-xyz",
        code_challenge="challenge-abc",
    )
    assert url.startswith(CODEX_OAUTH_AUTHORIZE_URL + "?")
    for key in (
        "client_id=cid-123",
        "response_type=code",
        "redirect_uri=http%3A%2F%2F127.0.0.1%3A8765%2Fcallback",
        "scope=openai.chatgpt.experimental",
        "state=state-xyz",
        "code_challenge_method=S256",
        "code_challenge=challenge-abc",
    ):
        assert key in url, f"missing {key!r} in {url}"


def test_build_authorize_url_uses_loopback_redirect():
    url = _codex_authcode_build_authorize_url(
        client_id=CODEX_OAUTH_CLIENT_ID,
        redirect_uri="http://127.0.0.1:9999/callback",
        scope=CODEX_OAUTH_SCOPE,
        state="s",
        code_challenge="c",
    )
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A9999%2Fcallback" in url


# --------------------------------------------------------------------------------------
# Free loopback port
# --------------------------------------------------------------------------------------


def test_free_port_is_in_range_or_os_chosen():
    port = _codex_authcode_free_port()
    assert isinstance(port, int) and 1024 < port < 65536


# --------------------------------------------------------------------------------------
# Callback handler
# --------------------------------------------------------------------------------------


def _drive_callback(path: str) -> dict:
    handler_cls, result = _codex_authcode_make_callback_handler("/callback")
    handler = handler_cls.__new__(handler_cls)
    handler.path = path
    handler.wfile = BytesIO()
    handler.send_response = lambda *a, **k: None
    handler.send_header = lambda *a, **k: None
    handler.end_headers = lambda *a, **k: None
    handler.do_GET()
    return result


def test_callback_handler_captures_code_and_state():
    result = _drive_callback("/callback?code=AC123&state=st")
    assert result["code"] == "AC123"
    assert result["state"] == "st"
    assert result.get("error") is None


def test_callback_handler_captures_error():
    result = _drive_callback("/callback?error=access_denied&error_description=user%20cancelled")
    assert result["error"] == "access_denied"
    assert result["error_description"] == "user cancelled"


def test_callback_handler_404_on_wrong_path():
    result = _drive_callback("/other?code=AC123&state=st")
    assert result["code"] is None
    assert result["state"] is None


# --------------------------------------------------------------------------------------
# Wait-for-callback: success and timeout against a real in-process callback
# --------------------------------------------------------------------------------------


def test_wait_for_callback_receives_code():
    import threading
    import time
    import urllib.request

    port = _codex_authcode_free_port()
    path = "/callback"
    state = "mystate"

    def _hit_callback():
        time.sleep(0.4)
        url = f"http://127.0.0.1:{port}{path}?code=THE_CODE&state={state}"
        try:
            urllib.request.urlopen(url, timeout=2)
        except Exception:
            pass

    t = threading.Thread(target=_hit_callback, daemon=True)
    t.start()

    result = _codex_authcode_wait_for_callback(
        host="127.0.0.1",
        port=port,
        path=path,
        timeout_seconds=5.0,
    )
    assert result["code"] == "THE_CODE"
    assert result["state"] == state
    assert result.get("error") is None


def test_wait_for_callback_times_out():
    with pytest.raises(AuthError) as exc:
        _codex_authcode_wait_for_callback(
            host="127.0.0.1",
            port=_codex_authcode_free_port(),
            path="/callback",
            timeout_seconds=1.0,
        )
    assert exc.value.code == "authcode_timeout"


# --------------------------------------------------------------------------------------
# Token exchange shape
# --------------------------------------------------------------------------------------


def test_codex_authcode_login_exchanges_code_with_pkce_verifier(monkeypatch):
    captured = {}
    fixed = "fixed-state"

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"access_token": "AT", "refresh_token": "RT"}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data=None, headers=None):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            return _FakeResp()

    monkeypatch.setattr(auth_mod.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed))
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_wait_for_callback",
        lambda **kw: {"code": "AC", "state": fixed, "error": None, "error_description": None},
    )
    monkeypatch.setattr(auth_mod, "_codex_authcode_free_port", lambda: 8765)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod.httpx, "Client", _FakeClient)

    creds = auth_mod._codex_authcode_login(open_browser=True)
    assert creds["tokens"]["access_token"] == "AT"
    assert creds["tokens"]["refresh_token"] == "RT"
    assert creds["source"] == "authcode"
    assert creds["auth_mode"] == "chatgpt"
    assert captured["url"] == CODEX_OAUTH_TOKEN_URL
    assert captured["data"]["grant_type"] == "authorization_code"
    assert captured["data"]["code"] == "AC"
    assert captured["data"]["client_id"] == CODEX_OAUTH_CLIENT_ID
    assert "code_verifier" in captured["data"]
    assert captured["data"]["redirect_uri"].startswith("http://127.0.0.1:")
    assert captured["data"]["redirect_uri"].endswith("/callback")


def test_codex_authcode_login_raises_on_access_denied(monkeypatch):
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_wait_for_callback",
        lambda **kw: {
            "error": "access_denied",
            "error_description": "user cancelled",
            "code": None,
            "state": None,
        },
    )
    monkeypatch.setattr(auth_mod, "_codex_authcode_free_port", lambda: 8765)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *a, **k: None)
    with pytest.raises(AuthError) as exc:
        auth_mod._codex_authcode_login(open_browser=True)
    assert exc.value.code == "authcode_access_denied"


def test_codex_authcode_login_raises_on_state_mismatch(monkeypatch):
    fixed = "fixed-state"

    monkeypatch.setattr(auth_mod.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed))
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_wait_for_callback",
        lambda **kw: {"code": "AC", "state": "WRONG", "error": None, "error_description": None},
    )
    monkeypatch.setattr(auth_mod, "_codex_authcode_free_port", lambda: 8765)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *a, **k: None)
    with pytest.raises(AuthError) as exc:
        auth_mod._codex_authcode_login(open_browser=True)
    assert exc.value.code == "authcode_state_mismatch"


def test_codex_authcode_login_raises_on_no_code(monkeypatch):
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_wait_for_callback",
        lambda **kw: {"code": None, "state": None, "error": None, "error_description": None},
    )
    monkeypatch.setattr(auth_mod, "_codex_authcode_free_port", lambda: 8765)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *a, **k: None)
    with pytest.raises(AuthError) as exc:
        auth_mod._codex_authcode_login(open_browser=True)
    assert exc.value.code == "authcode_no_code"


def test_codex_authcode_login_raises_on_token_exchange_error(monkeypatch):
    fixed = "fixed-state"

    monkeypatch.setattr(auth_mod.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed))
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_wait_for_callback",
        lambda **kw: {"code": "AC", "state": fixed, "error": None, "error_description": None},
    )
    monkeypatch.setattr(auth_mod, "_codex_authcode_free_port", lambda: 8765)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *a, **k: None)

    class _FakeResp:
        status_code = 500

        def json(self):
            return {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data=None, headers=None):
            return _FakeResp()

    monkeypatch.setattr(auth_mod.httpx, "Client", _FakeClient)
    with pytest.raises(AuthError) as exc:
        auth_mod._codex_authcode_login(open_browser=True)
    assert exc.value.code == "token_exchange_error"


def test_login_openai_codex_uses_authcode_when_browser_available(monkeypatch):
    """``_login_openai_codex`` must read ``args`` (not delete it) and default
    to the authorization-code flow when a graphical browser is available."""
    called = {"authcode": 0, "device": 0}

    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", lambda: (_ for _ in ()).throw(AuthError("none")))
    monkeypatch.setattr(auth_mod, "_import_codex_cli_tokens", lambda: None)
    monkeypatch.setattr(auth_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    monkeypatch.setattr(
        auth_mod,
        "_codex_authcode_login",
        lambda **kw: (
            called.__setitem__("authcode", called["authcode"] + 1)
            or {
                "tokens": {"access_token": "AT", "refresh_token": "RT"},
                "base_url": "https://chatgpt.com/backend-api/codex",
                "last_refresh": "2026-08-28T00:00:00Z",
            }
        ),
    )
    monkeypatch.setattr(
        auth_mod,
        "_codex_device_code_login",
        lambda: called.__setitem__("device", called["device"] + 1) or (_ for _ in ()).throw(AssertionError("device")),
    )
    monkeypatch.setattr(auth_mod, "_save_codex_tokens", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "_update_config_for_provider", lambda *a, **k: "/tmp/config.yaml")

    args = SimpleNamespace(device_code=False, no_browser=False, timeout=None, scope=None)
    auth_mod._login_openai_codex(args, auth_mod.PROVIDER_REGISTRY["openai-codex"], force_new_login=True)
    assert called == {"authcode": 1, "device": 0}
