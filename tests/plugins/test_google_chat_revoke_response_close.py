"""Google Chat revocation closes responses without depending on unused bodies."""

import io
import sys
import urllib.error
import urllib.request
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("http_error", [False, True])
def test_revoke_closes_response_and_removes_tokens(tmp_path, monkeypatch, capsys, http_error):
    from plugins.platforms.google_chat import oauth

    monkeypatch.setattr(oauth, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(oauth, "_ensure_deps", lambda: None)
    credentials = ModuleType("google.oauth2.credentials")
    credentials.Credentials = Mock()
    credentials.Credentials.from_authorized_user_file.return_value = SimpleNamespace(
        expired=False, refresh_token=None, token="test-token")
    transport = ModuleType("google.auth.transport.requests")
    transport.Request = Mock()
    monkeypatch.setitem(sys.modules, credentials.__name__, credentials)
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    token = oauth._token_path()
    pending = oauth._pending_auth_path()
    token.write_text("{}")
    pending.write_text("{}")
    response = io.BytesIO(b"unused")
    response.read = Mock(side_effect=OSError("body unavailable"))
    error = urllib.error.HTTPError("https://oauth.example/revoke", 400, "invalid", {}, response)
    request = Mock(side_effect=error) if http_error else Mock(return_value=response)
    monkeypatch.setattr(urllib.request, "urlopen", request)

    oauth.revoke()

    assert response.closed
    response.read.assert_not_called()
    assert not token.exists()
    assert not pending.exists()
    output = capsys.readouterr().out
    assert ("Remote revocation failed" in output) is http_error
    assert ("Token revoked with Google." in output) is not http_error
    req = request.call_args.args[0]
    assert req.get_method() == "POST"
    assert "token=test-token" in req.full_url
    assert request.call_args.kwargs["timeout"] == 15
