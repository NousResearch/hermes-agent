"""OAuth diagnostics close success and HTTP error responses on every read path."""

import importlib.util
import io
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def diagnose():
    path = Path(__file__).resolve().parents[2] / "optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py"
    spec = importlib.util.spec_from_file_location("diagnose_response_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("helper", ["_post", "_get_json"])
@pytest.mark.parametrize("read_error", [False, True])
def test_success_response_closed(diagnose, monkeypatch, helper, read_error):
    response = io.BytesIO(b'{"ok": true}')
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    if read_error:
        response.read = Mock(side_effect=OSError("read interrupted"))
    monkeypatch.setattr(diagnose.urllib.request, "urlopen", Mock(return_value=response))

    if read_error:
        with pytest.raises(OSError, match="read interrupted"):
            getattr(diagnose, helper)("https://oauth.example/test")
    elif helper == "_post":
        assert diagnose._post("https://oauth.example/test") == (
            200, response.headers, b'{"ok": true}')
    else:
        assert diagnose._get_json("https://oauth.example/test") == {"ok": True}
    assert response.closed


@pytest.mark.parametrize("helper", ["_post", "_get_json"])
def test_http_error_response_closed(diagnose, monkeypatch, helper):
    body = io.BytesIO(b'{"error": "invalid_token"}')
    error = urllib.error.HTTPError("https://oauth.example/test", 401, "invalid", {}, body)
    monkeypatch.setattr(diagnose.urllib.request, "urlopen", Mock(side_effect=error))

    if helper == "_post":
        assert diagnose._post("https://oauth.example/test") == (
            401, {}, b'{"error": "invalid_token"}')
    else:
        with pytest.raises(urllib.error.HTTPError) as raised:
            diagnose._get_json("https://oauth.example/test")
        assert raised.value is error
    assert body.closed


def test_invalid_json_closes_response(diagnose, monkeypatch):
    response = io.BytesIO(b"invalid json")
    monkeypatch.setattr(diagnose.urllib.request, "urlopen", Mock(return_value=response))

    with pytest.raises(ValueError):
        diagnose._get_json("https://oauth.example/test")
    assert response.closed


def test_post_error_body_read_failure_still_closes(diagnose, monkeypatch):
    body = io.BytesIO(b"unused")
    body.read = Mock(side_effect=OSError("read interrupted"))
    error = urllib.error.HTTPError("https://oauth.example/test", 401, "invalid", {}, body)
    monkeypatch.setattr(diagnose.urllib.request, "urlopen", Mock(side_effect=error))

    with pytest.raises(OSError, match="read interrupted"):
        diagnose._post("https://oauth.example/test")
    assert body.closed
