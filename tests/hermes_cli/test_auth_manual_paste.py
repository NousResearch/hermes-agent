"""Tests for the OAuth manual-paste fallback for browser-only remotes.

Regression coverage for [#26923](https://github.com/NousResearch/hermes-agent/issues/26923):
GCP Cloud Shell, GitHub Codespaces, AWS EC2 Instance Connect and
other browser-only remote consoles can't reach the
``http://127.0.0.1:56121/callback`` loopback listener bound on the
remote VM.  The previous SSH-tunnel hint was useless without a real
SSH client, leaving the user with no path forward.  This test file
locks in four things:

* ``_is_remote_session`` recognises the cloud-shell / Codespaces
  envvars (so the existing hint at least fires).
* ``_parse_pasted_callback`` accepts every form a user might paste
  (full URL, ``?code=...&state=...`` fragment, bare ``code=...``,
  bare opaque value) and returns the same shape the loopback HTTP
  handler does.
* ``_prompt_manual_callback_paste`` reads stdin and produces that
  same shape.
* ``_xai_oauth_loopback_login(manual_paste=True)`` skips the HTTP
  server entirely, validates ``state``, and goes straight to the
  token exchange — proving the paste path actually wires up.
"""

from __future__ import annotations

import builtins
import io
import contextlib

import pytest

from hermes_cli import auth as auth_mod


# ---------------------------------------------------------------------------
# _is_remote_session — broadened detection (#26923)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "envvar",
    [
        "SSH_CLIENT",
        "SSH_TTY",
        "CLOUD_SHELL",
        "CODESPACES",
        "CODESPACE_NAME",
        "GITPOD_WORKSPACE_ID",
        "REPL_ID",
        "STACKBLITZ",
    ],
)
def test_is_remote_session_detects_known_remote_envvar(monkeypatch, envvar):
    """Each documented remote-console env var must trip the check.

    The SSH ones preserve historical behaviour; the cloud-shell ones
    are what closes #26923.  Without these, the SSH hint never fires
    and the user has no signal that ``--manual-paste`` exists.
    """
    for name in (
        "SSH_CLIENT",
        "SSH_TTY",
        "CLOUD_SHELL",
        "CODESPACES",
        "CODESPACE_NAME",
        "GITPOD_WORKSPACE_ID",
        "REPL_ID",
        "STACKBLITZ",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(envvar, "1")
    assert auth_mod._is_remote_session() is True


def test_is_remote_session_false_when_no_remote_envvars(monkeypatch):
    for name in (
        "SSH_CLIENT",
        "SSH_TTY",
        "CLOUD_SHELL",
        "CODESPACES",
        "CODESPACE_NAME",
        "GITPOD_WORKSPACE_ID",
        "REPL_ID",
        "STACKBLITZ",
    ):
        monkeypatch.delenv(name, raising=False)
    assert auth_mod._is_remote_session() is False


# ---------------------------------------------------------------------------
# _parse_pasted_callback — accept every plausible paste form
# ---------------------------------------------------------------------------


def test_parse_full_callback_url():
    out = auth_mod._parse_pasted_callback(
        "http://127.0.0.1:56121/callback?code=abc123&state=deadbeef"
    )
    assert out == {
        "code": "abc123",
        "state": "deadbeef",
        "error": None,
        "error_description": None,
    }


def test_parse_callback_url_https_and_extra_params():
    out = auth_mod._parse_pasted_callback(
        "https://127.0.0.1:56121/callback?code=abc&state=xyz&scope=openid"
    )
    assert out["code"] == "abc"
    assert out["state"] == "xyz"


def test_parse_bare_query_string_with_leading_question_mark():
    out = auth_mod._parse_pasted_callback("?code=p1&state=s1")
    assert out["code"] == "p1"
    assert out["state"] == "s1"


def test_parse_bare_query_fragment_no_question_mark():
    out = auth_mod._parse_pasted_callback("code=p2&state=s2")
    assert out["code"] == "p2"
    assert out["state"] == "s2"


def test_parse_bare_opaque_code_value():
    """Some users only copy the ``code`` value itself."""
    out = auth_mod._parse_pasted_callback("ABCDEF-the-code-value")
    assert out["code"] == "ABCDEF-the-code-value"
    assert out["state"] is None


def test_parse_callback_with_error_field():
    out = auth_mod._parse_pasted_callback(
        "http://127.0.0.1:56121/callback?error=access_denied"
        "&error_description=user+rejected"
    )
    assert out["code"] is None
    assert out["error"] == "access_denied"
    assert out["error_description"] == "user rejected"


def test_parse_empty_input_returns_all_none():
    out = auth_mod._parse_pasted_callback("")
    assert out == {
        "code": None,
        "state": None,
        "error": None,
        "error_description": None,
    }


def test_parse_whitespace_only_returns_all_none():
    out = auth_mod._parse_pasted_callback("   \n\t  ")
    assert out["code"] is None


def test_parse_malformed_url_does_not_crash():
    out = auth_mod._parse_pasted_callback("http://[not a url")
    # Malformed URLs return all-None rather than raising — the caller
    # (state check) will reject the empty payload with a clear error.
    assert out["code"] is None


# ---------------------------------------------------------------------------
# _prompt_manual_callback_paste — stdin handling
# ---------------------------------------------------------------------------


def test_prompt_reads_stdin_and_parses(monkeypatch):
    monkeypatch.setattr(
        builtins, "input",
        lambda *_a, **_k: "http://127.0.0.1:56121/callback?code=abc&state=xyz",
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = auth_mod._prompt_manual_callback_paste(
            "http://127.0.0.1:56121/callback"
        )
    rendered = buf.getvalue()
    assert "Manual callback paste" in rendered
    assert "127.0.0.1:56121" in rendered
    assert out["code"] == "abc"
    assert out["state"] == "xyz"


def test_prompt_eof_returns_all_none(monkeypatch):
    def _raise_eof(*_a, **_k):
        raise EOFError()

    monkeypatch.setattr(builtins, "input", _raise_eof)
    with contextlib.redirect_stdout(io.StringIO()):
        out = auth_mod._prompt_manual_callback_paste(
            "http://127.0.0.1:56121/callback"
        )
    assert out["code"] is None


def test_prompt_keyboard_interrupt_returns_all_none(monkeypatch):
    def _raise_kbi(*_a, **_k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(builtins, "input", _raise_kbi)
    with contextlib.redirect_stdout(io.StringIO()):
        out = auth_mod._prompt_manual_callback_paste(
            "http://127.0.0.1:56121/callback"
        )
    assert out["code"] is None


# ---------------------------------------------------------------------------
# _xai_oauth_loopback_login(manual_paste=True) — full integration
# ---------------------------------------------------------------------------


class _StubTokenResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def test_xai_loopback_login_manual_paste_skips_http_server(monkeypatch):
    """``manual_paste=True`` must NOT bind a loopback HTTP server.

    Direct end-to-end regression for #26923: the whole point is that
    the listener is unreachable on browser-only remotes, so the paste
    path must avoid it entirely.  We assert this by replacing
    ``_xai_start_callback_server`` with a function that fails if
    invoked, then driving the full happy path with a stubbed prompt
    + stubbed token endpoint.
    """
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )

    def _server_must_not_be_called(*_a, **_k):
        raise AssertionError(
            "manual_paste=True must skip the loopback HTTP server "
            "(regression for #26923)"
        )

    monkeypatch.setattr(
        auth_mod, "_xai_start_callback_server", _server_must_not_be_called
    )

    captured_state: dict = {}

    def _fake_prompt(_redirect_uri):
        # Hermes generates state internally; we won't know it ahead of
        # time, so capture the state Hermes baked into the authorize
        # URL via a sneak peek on ``_xai_oauth_build_authorize_url``.
        return {
            "code": "fake-auth-code",
            "state": captured_state["value"],
            "error": None,
            "error_description": None,
        }

    monkeypatch.setattr(
        auth_mod, "_prompt_manual_callback_paste", _fake_prompt
    )

    original_build = auth_mod._xai_oauth_build_authorize_url

    def _capture_state(**kwargs):
        captured_state["value"] = kwargs["state"]
        return original_build(**kwargs)

    monkeypatch.setattr(
        auth_mod, "_xai_oauth_build_authorize_url", _capture_state
    )

    def _fake_token_post(*_a, **_k):
        return _StubTokenResponse(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": "",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )

    monkeypatch.setattr(auth_mod.httpx, "post", _fake_token_post)

    with contextlib.redirect_stdout(io.StringIO()):
        creds = auth_mod._xai_oauth_loopback_login(manual_paste=True)

    assert creds["tokens"]["access_token"] == "at"
    assert creds["tokens"]["refresh_token"] == "rt"
    assert "127.0.0.1:56121" in creds["redirect_uri"]


def test_xai_loopback_login_manual_paste_state_mismatch_raises(monkeypatch):
    """A pasted callback with the wrong state must still be rejected.

    The HTTP-server path uses the same state check; manual-paste
    must not be a CSRF bypass.
    """
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        auth_mod, "_prompt_manual_callback_paste",
        lambda _ru: {
            "code": "fake",
            "state": "WRONG-STATE",
            "error": None,
            "error_description": None,
        },
    )

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(auth_mod.AuthError) as exc:
            auth_mod._xai_oauth_loopback_login(manual_paste=True)
    assert exc.value.code == "xai_state_mismatch"


def test_xai_loopback_login_manual_paste_bare_code_succeeds(monkeypatch):
    """Bare-code paste (state=None) must complete login under manual_paste.

    xAI's consent page renders the authorization code in-page rather than
    redirecting through 127.0.0.1, so on remote/headless setups the only
    value the user can obtain is the opaque code with no ``state=``
    parameter. ``_parse_pasted_callback`` correctly returns
    ``state=None`` for that input. The login flow must accept this case
    (PKCE still protects the exchange); historically it raised
    ``xai_state_mismatch``. Regression for the bare-code branch of #26923.
    """
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        auth_mod, "_prompt_manual_callback_paste",
        lambda _ru: {
            "code": "bare-opaque-code",
            "state": None,
            "error": None,
            "error_description": None,
        },
    )

    def _fake_token_post(*_a, **_k):
        return _StubTokenResponse(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": "",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )

    monkeypatch.setattr(auth_mod.httpx, "post", _fake_token_post)

    with contextlib.redirect_stdout(io.StringIO()):
        creds = auth_mod._xai_oauth_loopback_login(manual_paste=True)

    assert creds["tokens"]["access_token"] == "at"
    assert creds["tokens"]["refresh_token"] == "rt"


def test_xai_loopback_login_loopback_path_rejects_missing_state(monkeypatch):
    """Loopback (manual_paste=False) must NOT accept ``state=None``.

    The bare-code relaxation only applies to the manual-paste path,
    where the user demonstrably has no way to supply ``state``. The
    HTTP-server path always sees ``state`` populated from the real
    callback query string, so missing state there means something is
    wrong (a malformed callback, an attacker-supplied request) and
    must still raise ``xai_state_mismatch``.
    """
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )

    class _StubServer:
        def shutdown(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr(
        auth_mod, "_xai_start_callback_server",
        lambda *_a, **_k: (
            _StubServer(),
            None,
            {"code": "fake", "state": None, "error": None,
             "error_description": None},
            "http://127.0.0.1:56121/callback",
        ),
    )
    monkeypatch.setattr(
        auth_mod, "_xai_wait_for_callback",
        lambda *_a, **_k: {
            "code": "fake",
            "state": None,
            "error": None,
            "error_description": None,
        },
    )
    monkeypatch.setattr(auth_mod, "_xai_validate_loopback_redirect_uri", lambda _u: None)
    monkeypatch.setattr(auth_mod, "_print_loopback_ssh_hint", lambda *_a, **_k: None)

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(auth_mod.AuthError) as exc:
            auth_mod._xai_oauth_loopback_login(manual_paste=False, open_browser=False)
    assert exc.value.code == "xai_state_mismatch"


def test_xai_loopback_login_manual_paste_missing_code_raises(monkeypatch):
    """Empty paste must surface as ``xai_code_missing``, not crash."""
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    captured: dict = {"state": None}
    original_build = auth_mod._xai_oauth_build_authorize_url

    def _capture(**kw):
        captured["state"] = kw["state"]
        return original_build(**kw)

    monkeypatch.setattr(auth_mod, "_xai_oauth_build_authorize_url", _capture)
    monkeypatch.setattr(
        auth_mod, "_prompt_manual_callback_paste",
        lambda _ru: {
            "code": None,
            "state": captured["state"],
            "error": None,
            "error_description": None,
        },
    )

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(auth_mod.AuthError) as exc:
            auth_mod._xai_oauth_loopback_login(manual_paste=True)
    assert exc.value.code == "xai_code_missing"


def test_xai_loopback_login_timeout_falls_back_to_manual_paste(monkeypatch):
    """Loopback timeout should offer the existing manual-paste path."""
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )

    class _StubServer:
        def shutdown(self):
            return None

        def server_close(self):
            return None

    class _StubThread:
        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        auth_mod,
        "_xai_start_callback_server",
        lambda: (
            _StubServer(),
            _StubThread(),
            {
                "code": None,
                "state": None,
                "error": None,
                "error_description": None,
            },
            "http://127.0.0.1:56121/callback",
        ),
    )

    captured: dict = {"state": None, "prompt_calls": 0}
    original_build = auth_mod._xai_oauth_build_authorize_url

    def _capture(**kwargs):
        captured["state"] = kwargs["state"]
        return original_build(**kwargs)

    monkeypatch.setattr(auth_mod, "_xai_oauth_build_authorize_url", _capture)

    def _raise_timeout(*_a, **_k):
        raise auth_mod.AuthError(
            "xAI authorization timed out waiting for the local callback.",
            provider="xai-oauth",
            code="xai_callback_timeout",
        )

    monkeypatch.setattr(auth_mod, "_xai_wait_for_callback", _raise_timeout)

    def _fake_prompt(_redirect_uri):
        captured["prompt_calls"] += 1
        return {
            "code": "manual-auth-code",
            "state": captured["state"],
            "error": None,
            "error_description": None,
        }

    monkeypatch.setattr(auth_mod, "_prompt_manual_callback_paste", _fake_prompt)
    monkeypatch.setattr(
        auth_mod.sys, "stdin", type("StubStdin", (), {"isatty": lambda self: True})()
    )
    monkeypatch.setattr(
        auth_mod.httpx,
        "post",
        lambda *_a, **_k: _StubTokenResponse(
            {
                "access_token": "at-timeout",
                "refresh_token": "rt-timeout",
                "id_token": "",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        ),
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        creds = auth_mod._xai_oauth_loopback_login(manual_paste=False)

    rendered = buf.getvalue()
    assert "xAI loopback callback timed out." in rendered
    assert "--manual-paste" in rendered
    assert captured["prompt_calls"] == 1
    assert creds["tokens"]["access_token"] == "at-timeout"
    assert creds["tokens"]["refresh_token"] == "rt-timeout"


def test_xai_loopback_login_timeout_noninteractive_reraises(monkeypatch):
    """Non-interactive stdin must keep the original timeout error."""
    monkeypatch.setattr(
        auth_mod, "_xai_oauth_discovery",
        lambda *_a, **_k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )

    class _StubServer:
        def shutdown(self):
            return None

        def server_close(self):
            return None

    class _StubThread:
        def join(self, timeout=None):
            return None

    monkeypatch.setattr(
        auth_mod,
        "_xai_start_callback_server",
        lambda: (
            _StubServer(),
            _StubThread(),
            {
                "code": None,
                "state": None,
                "error": None,
                "error_description": None,
            },
            "http://127.0.0.1:56121/callback",
        ),
    )

    monkeypatch.setattr(
        auth_mod,
        "_xai_wait_for_callback",
        lambda *_a, **_k: (_ for _ in ()).throw(
            auth_mod.AuthError(
                "xAI authorization timed out waiting for the local callback.",
                provider="xai-oauth",
                code="xai_callback_timeout",
            )
        ),
    )
    monkeypatch.setattr(
        auth_mod.sys, "stdin", type("StubStdin", (), {"isatty": lambda self: False})()
    )
    monkeypatch.setattr(
        auth_mod,
        "_prompt_manual_callback_paste",
        lambda *_a, **_k: pytest.fail("manual-paste fallback should not run"),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(auth_mod.AuthError) as exc:
            auth_mod._xai_oauth_loopback_login(manual_paste=False)
    assert exc.value.code == "xai_callback_timeout"


# ---------------------------------------------------------------------------
# _print_loopback_ssh_hint — now also mentions --manual-paste
# ---------------------------------------------------------------------------


def test_ssh_hint_mentions_manual_paste_for_non_ssh_remotes(monkeypatch):
    """Users on Cloud Shell / Codespaces have no real SSH client; the
    hint must point them at the new ``--manual-paste`` flag instead
    of leaving them stuck on the ``ssh -L`` recipe."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        auth_mod._print_loopback_ssh_hint(
            "http://127.0.0.1:56121/callback",
            docs_url=auth_mod.XAI_OAUTH_DOCS_URL,
        )
    rendered = buf.getvalue()
    assert "--manual-paste" in rendered
    assert "Cloud Shell" in rendered or "Codespaces" in rendered
