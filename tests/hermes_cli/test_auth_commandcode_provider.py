"""Tests for Command Code OAuth and CLI credential import (hermes_cli/auth_commandcode.py).

Covers:
- _commandcode_cli_auth_path
- _read_commandcode_cli_tokens
- _save_commandcode_cli_tokens
- validate_commandcode_api_key
- resolve_commandcode_runtime_credentials
- get_commandcode_auth_status
- _commandcode_oauth_login auto-import
- _OAUTH_CAPABLE_PROVIDERS registration
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_COMMANDCODE_BASE_URL,
    _commandcode_cli_auth_path,
    _read_commandcode_cli_tokens,
    _save_commandcode_cli_tokens,
    _commandcode_oauth_login,
    resolve_commandcode_runtime_credentials,
    get_commandcode_auth_status,
    validate_commandcode_api_key,
)
from hermes_cli.auth_commands import _OAUTH_CAPABLE_PROVIDERS, _OAUTH_ADD_SPECS


def _make_commandcode_tokens(
    api_key="test-commandcode-api-key",
    user_id="usr-12345",
    user_name="testuser",
    key_name="testkey",
    **extra,
):
    data = {
        "apiKey": api_key,
        "userId": user_id,
        "userName": user_name,
        "keyName": key_name,
        "authenticatedAt": "2026-09-07T00:00:00.000Z",
    }
    data.update(extra)
    return data


@pytest.fixture()
def commandcode_env(tmp_path, monkeypatch):
    """Redirect _commandcode_cli_auth_path to tmp_path/.commandcode/auth.json."""
    creds_path = tmp_path / ".commandcode" / "auth.json"
    monkeypatch.setattr(
        "hermes_cli.auth._commandcode_cli_auth_path", lambda: creds_path
    )
    monkeypatch.setattr(
        "hermes_cli.auth_commandcode._commandcode_cli_auth_path", lambda: creds_path
    )
    return tmp_path


def test_commandcode_cli_auth_path_returns_expected_location():
    path = _commandcode_cli_auth_path()
    assert path == Path.home() / ".commandcode" / "auth.json"


def test_read_commandcode_cli_tokens_success(commandcode_env):
    creds_path = commandcode_env / ".commandcode" / "auth.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = _make_commandcode_tokens()
    creds_path.write_text(json.dumps(tokens), encoding="utf-8")

    loaded = _read_commandcode_cli_tokens()
    assert loaded["apiKey"] == "test-commandcode-api-key"
    assert loaded["userId"] == "usr-12345"
    assert loaded["userName"] == "testuser"


def test_read_commandcode_cli_tokens_missing_file_raises_auth_error(commandcode_env):
    with pytest.raises(AuthError) as exc_info:
        _read_commandcode_cli_tokens()
    assert "Command Code CLI credentials not found" in str(exc_info.value)
    assert exc_info.value.code == "commandcode_auth_missing"


def test_read_commandcode_cli_tokens_missing_key_raises_auth_error(commandcode_env):
    creds_path = commandcode_env / ".commandcode" / "auth.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps({"userName": "user_without_key"}), encoding="utf-8")

    with pytest.raises(AuthError) as exc_info:
        _read_commandcode_cli_tokens()
    assert "missing apiKey" in str(exc_info.value)
    assert exc_info.value.code == "commandcode_auth_missing_key"


def test_save_commandcode_cli_tokens(commandcode_env):
    tokens = _make_commandcode_tokens()
    saved_path = _save_commandcode_cli_tokens(tokens)
    assert saved_path.exists()
    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert content["apiKey"] == "test-commandcode-api-key"
    assert content["userName"] == "testuser"


def test_validate_commandcode_api_key_success():
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "user": {"id": "usr-123", "userName": "rmlima"}
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = validate_commandcode_api_key("valid-key")
        assert res is not None
        assert res["userId"] == "usr-123"
        assert res["userName"] == "rmlima"


def test_validate_commandcode_api_key_failure():
    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        res = validate_commandcode_api_key("bad-key")
        assert res is None


def test_resolve_commandcode_runtime_credentials(commandcode_env):
    creds_path = commandcode_env / ".commandcode" / "auth.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = _make_commandcode_tokens()
    creds_path.write_text(json.dumps(tokens), encoding="utf-8")

    creds = resolve_commandcode_runtime_credentials()
    assert creds["provider"] == "commandcode"
    assert creds["api_key"] == "test-commandcode-api-key"
    assert creds["user_name"] == "testuser"
    assert creds["base_url"] == DEFAULT_COMMANDCODE_BASE_URL


def test_get_commandcode_auth_status_logged_in(commandcode_env):
    creds_path = commandcode_env / ".commandcode" / "auth.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = _make_commandcode_tokens()
    creds_path.write_text(json.dumps(tokens), encoding="utf-8")

    status = get_commandcode_auth_status()
    assert status["logged_in"] is True
    assert status["user_name"] == "testuser"


def test_commandcode_oauth_login_auto_import(commandcode_env):
    creds_path = commandcode_env / ".commandcode" / "auth.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = _make_commandcode_tokens()
    creds_path.write_text(json.dumps(tokens), encoding="utf-8")

    with patch("hermes_cli.auth_commandcode.validate_commandcode_api_key", return_value={"userId": "usr-12345", "userName": "testuser"}):
        result = _commandcode_oauth_login()
        assert result["api_key"] == "test-commandcode-api-key"
        assert result["userName"] == "testuser"
        assert result["source"] == "commandcode-cli"


def test_commandcode_in_oauth_capable_providers():
    assert "commandcode-oauth" in _OAUTH_CAPABLE_PROVIDERS
    assert "commandcode-oauth" in _OAUTH_ADD_SPECS
    spec = _OAUTH_ADD_SPECS["commandcode-oauth"]
    assert spec.token({"api_key": "my-key"}) == "my-key"


def test_callback_server_binds_before_returning(commandcode_env):
    """The returned listener is already bound and serving — the URL can trust its port."""
    import socket

    from hermes_cli import auth_commandcode as ac

    server, _result, used_port, thread = ac._commandcode_start_callback_server("state-x")
    try:
        assert used_port == ac.COMMANDCODE_CALLBACK_PORT or used_port > 0
        # A live listener must accept a connection immediately.
        probe = socket.create_connection(("127.0.0.1", used_port), timeout=2.0)
        probe.close()
    finally:
        ac._commandcode_stop_callback_server(server, thread)


def test_oauth_login_url_matches_listening_port_when_default_taken(commandcode_env, monkeypatch):
    """With 5959 occupied, bind-first guarantees the printed URL uses the ephemeral port.

    Regression for the reviewed failure mode: the listener used to fall back to an
    ephemeral port AFTER the URL (and browser open) committed to 5959, so the callback
    could never reach the flow.
    """
    import contextlib
    import io
    import json
    import re
    import socket
    import threading
    import urllib.request
    from types import SimpleNamespace

    from hermes_cli import auth_commandcode as ac

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", ac.COMMANDCODE_CALLBACK_PORT))
    blocker.listen(1)
    monkeypatch.setattr("hermes_cli.auth._can_open_graphical_browser", lambda: False)

    out = io.StringIO()
    outcome: dict = {}

    def _login():
        try:
            with contextlib.redirect_stdout(out):
                outcome["creds"] = ac._commandcode_oauth_login(
                    SimpleNamespace(no_browser=True, timeout=20.0), force_browser=True
                )
        except Exception as exc:  # pragma: no cover - surfaced via outcome
            outcome["error"] = exc

    t = threading.Thread(target=_login, daemon=True)
    t.start()

    url = None
    for _ in range(200):
        m = re.search(r"callback=http://127\.0\.0\.1:(\d+)/callback&state=(\S+)", out.getvalue())
        if m:
            url = m
            break
        threading.Event().wait(0.05)
    assert url is not None, f"auth URL never printed: {out.getvalue()!r}"
    port, state = int(url.group(1)), url.group(2)
    assert port != ac.COMMANDCODE_CALLBACK_PORT, "URL must reflect the actually-bound port"

    body = json.dumps({"state": state, "apiKey": "ck-wired", "userId": "u9", "userName": "raf"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/callback", data=body, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5).read()
    t.join(timeout=25)
    blocker.close()

    assert "error" not in outcome, outcome.get("error")
    assert outcome["creds"]["api_key"] == "ck-wired"
