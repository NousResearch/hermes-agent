from __future__ import annotations

import socket
import threading
import urllib.request
from http.server import HTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli import auth as auth_mod


def test_store_provider_state_can_skip_active_provider() -> None:
    auth_store = {"active_provider": "nous", "providers": {}}

    auth_mod._store_provider_state(
        auth_store,
        "spotify",
        {"access_token": "abc"},
        set_active=False,
    )

    assert auth_store["active_provider"] == "nous"
    assert auth_store["providers"]["spotify"]["access_token"] == "abc"


def test_resolve_spotify_runtime_credentials_refreshes_without_changing_active_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        store["active_provider"] = "nous"
        auth_mod._store_provider_state(
            store,
            "spotify",
            {
                "client_id": "spotify-client",
                "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
                "api_base_url": auth_mod.DEFAULT_SPOTIFY_API_BASE_URL,
                "accounts_base_url": auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                "scope": auth_mod.DEFAULT_SPOTIFY_SCOPE,
                "access_token": "expired-token",
                "refresh_token": "refresh-token",
                "token_type": "Bearer",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    monkeypatch.setattr(
        auth_mod,
        "_refresh_spotify_oauth_state",
        lambda state, timeout_seconds=20.0: {
            **state,
            "access_token": "fresh-token",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )

    creds = auth_mod.resolve_spotify_runtime_credentials()

    assert creds["access_token"] == "fresh-token"
    persisted = auth_mod.get_provider_auth_state("spotify")
    assert persisted is not None
    assert persisted["access_token"] == "fresh-token"
    assert auth_mod.get_active_provider() == "nous"


def test_auth_spotify_status_command_reports_logged_in(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_mod,
        "get_auth_status",
        lambda provider=None: {
            "logged_in": True,
            "auth_type": "oauth_pkce",
            "client_id": "spotify-client",
            "redirect_uri": "http://127.0.0.1:43827/spotify/callback",
            "scope": "user-library-read",
        },
    )

    from hermes_cli.auth_commands import auth_status_command

    auth_status_command(SimpleNamespace(provider="spotify"))
    output = capsys.readouterr().out
    assert "spotify: logged in" in output
    assert "client_id: spotify-client" in output


def test_spotify_logout_does_not_reset_model_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  default: gemini-3-flash\n"
        "  provider: custom:local\n"
        "  base_url: http://localhost:11434/v1\n"
        "  api_key: ${LOCAL_API_KEY}\n",
        encoding="utf-8",
    )

    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        auth_mod._store_provider_state(
            store,
            "spotify",
            {
                "client_id": "spotify-client",
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            set_active=False,
        )
        auth_mod._save_auth_store(store)

    auth_mod.logout_command(SimpleNamespace(provider="spotify"))

    output = capsys.readouterr().out
    assert "Logged out of Spotify." in output
    assert "Model provider configuration was unchanged." in output
    assert auth_mod.get_provider_auth_state("spotify") is None
    assert config_path.read_text(encoding="utf-8") == (
        "model:\n"
        "  default: gemini-3-flash\n"
        "  provider: custom:local\n"
        "  base_url: http://localhost:11434/v1\n"
        "  api_key: ${LOCAL_API_KEY}\n"
    )


def test_spotify_interactive_setup_persists_client_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """The wizard writes HERMES_SPOTIFY_CLIENT_ID to .env and returns the value."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "wizard-client-123")
    # Prevent actually opening the browser during tests.
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    result = auth_mod._spotify_interactive_setup(
        redirect_uri_hint=auth_mod.DEFAULT_SPOTIFY_REDIRECT_URI,
    )
    assert result == "wizard-client-123"

    env_path = tmp_path / ".env"
    assert env_path.exists()
    env_text = env_path.read_text()
    assert "HERMES_SPOTIFY_CLIENT_ID=wizard-client-123" in env_text
    # Default redirect URI should NOT be persisted.
    assert "HERMES_SPOTIFY_REDIRECT_URI" not in env_text

    # Docs URL should appear in wizard output so users can find the guide.
    output = capsys.readouterr().out
    assert auth_mod.SPOTIFY_DOCS_URL in output


def test_spotify_interactive_setup_empty_aborts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input aborts cleanly instead of persisting an empty client_id."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(auth_mod, "webbrowser", SimpleNamespace(open=lambda *_a, **_k: False))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)

    with pytest.raises(SystemExit):
        auth_mod._spotify_interactive_setup(
            redirect_uri_hint=auth_mod.DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    env_path = tmp_path / ".env"
    if env_path.exists():
        assert "HERMES_SPOTIFY_CLIENT_ID" not in env_path.read_text()


# ── Callback handler tests ─────────────────────────────────────────────────


class _ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True


def _start_spotify_callback_server(path: str = "/callback") -> tuple[HTTPServer, threading.Thread, dict, str]:
    """Start a Spotify loopback callback server on an OS-assigned port."""
    handler_cls, result = auth_mod._make_spotify_callback_handler(path)
    server = _ReuseHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    redirect_uri = f"http://127.0.0.1:{port}{path}"
    return server, thread, result, redirect_uri


def _get_callback(redirect_uri: str, query: str = "") -> tuple[int, str]:
    """GET the loopback callback URL with an optional query string."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    target = redirect_uri + (("?" + query) if query else "")
    req = Request(target, method="GET")
    try:
        with urlopen(req, timeout=5.0) as resp:
            return resp.getcode(), resp.read().decode("utf-8", "replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def test_spotify_callback_handler_returns_400_when_callback_url_lacks_code_and_error():
    """Bare loopback URL (no code, no error) must not claim authorization received.

    Mirrors the xAI OAuth regression fix: when Spotify's auth backend fails to
    redirect and the user manually navigates to http://127.0.0.1:<port>/callback,
    the handler must return 400 "not received" rather than 200 "authorization
    received" — otherwise the browser shows a success page while the CLI wait
    loop still times out, leaving the user with a contradictory outcome.
    """
    server, thread, result, redirect_uri = _start_spotify_callback_server()
    try:
        status, body = _get_callback(redirect_uri)
        assert status == 400
        assert "not received" in body.lower()
        assert "hermes auth add spotify" in body
        # Wait loop must still see no code/error so it raises a real timeout,
        # rather than treating this empty hit as a successful callback.
        assert result["code"] is None
        assert result["error"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_spotify_callback_handler_accepts_callback_with_code():
    """A real OAuth redirect (code + state) still records both and shows success."""
    server, thread, result, redirect_uri = _start_spotify_callback_server()
    try:
        status, body = _get_callback(redirect_uri, query="code=abc&state=xyz")
        assert status == 200
        assert "Spotify authorization received" in body
        assert result["code"] == "abc"
        assert result["state"] == "xyz"
        assert result["error"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_spotify_callback_handler_records_error_callback():
    """A redirect carrying an `error` param must surface the failure page."""
    server, thread, result, redirect_uri = _start_spotify_callback_server()
    try:
        status, body = _get_callback(redirect_uri, query="error=access_denied")
        assert status == 200
        assert "Spotify authorization failed" in body
        assert result["error"] == "access_denied"
        assert result["code"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
