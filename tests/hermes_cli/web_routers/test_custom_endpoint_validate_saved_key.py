"""Tests for saved custom endpoint credential reuse and profile scoping on validate."""

from __future__ import annotations

import pytest
import hermes_yaml as yaml


@pytest.fixture
def _web_client(monkeypatch, _isolate_hermes_home):
    TestClient = pytest.importorskip("starlette.testclient").TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    class _BlockingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("async validation route used blocking httpx.Client")

    monkeypatch.setattr("httpx.Client", _BlockingClient)
    return client


def _install_fake_httpx(monkeypatch, *, expected_token: str):
    captured_calls: list[tuple[str, str, dict[str, str]]] = []

    class _Resp:
        def __init__(self, status=200, body=None, content_type="application/json"):
            self.status_code = status
            self.is_success = 200 <= status < 300
            self.headers = {"content-type": content_type}
            self._body = body if body is not None else {"data": [{"id": "gemini-3.8-flash"}]}

        def json(self):
            if isinstance(self._body, str):
                raise ValueError("not JSON")
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, *args, headers=None, **kwargs):
            captured_calls.append(("GET", url, dict(headers or {})))
            if url.endswith("/v1/models"):
                if (headers or {}).get("Authorization") == f"Bearer {expected_token}":
                    return _Resp(200, {"data": [{"id": "gemini-3.8-flash"}]})
                return _Resp(401, {"error": "Invalid token"})
            return _Resp(200, "<html>SPA</html>", content_type="text/html")

        async def post(self, url, *args, json=None, headers=None, **kwargs):
            captured_calls.append(("POST", url, dict(headers or {})))
            return _Resp(200, {"choices": []})

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    return captured_calls


def test_custom_endpoint_validate_reuses_saved_key_when_omitted(_web_client, monkeypatch):
    """Editing an existing endpoint and clicking Test without re-typing the API key
    must send the stored credential for the same origin, never drop auth."""
    from hermes_cli.config import get_config_path

    # 1. Endpoint saved via dashboard (key_env -> .env)
    _web_client.post(
        "/api/providers/custom-endpoints",
        json={
            "id": "myapi",
            "name": "MyAPI",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
            "api_key": "saved-endpoint-secret",
        },
    )

    captured_calls = _install_fake_httpx(monkeypatch, expected_token="saved-endpoint-secret")

    # Omitted api_key (what the Desktop Test button sends when the password input is blank)
    res = _web_client.post(
        "/api/providers/custom-endpoints/validate",
        json={
            "id": "myapi",
            "name": "MyAPI",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
        },
    ).json()
    assert res["ok"] is True
    assert res["models"] == ["gemini-3.8-flash"]
    assert captured_calls[0] == (
        "GET",
        "http://127.0.0.1:12168/v1/models",
        {"Accept": "application/json", "Authorization": "Bearer saved-endpoint-secret"},
    )
    assert captured_calls[1][0] == "POST"
    assert captured_calls[1][2]["Authorization"] == "Bearer saved-endpoint-secret"

    # Changing the origin (host/port/scheme) must NOT ship the old endpoint's saved secret.
    captured_calls.clear()
    res_other_origin = _web_client.post(
        "/api/providers/custom-endpoints/validate",
        json={
            "id": "myapi",
            "name": "MyAPI",
            "base_url": "http://other.example.com/v1",
            "model": "gemini-3.8-flash",
        },
    ).json()
    assert res_other_origin["ok"] is False
    assert "rejected the API key" in res_other_origin["message"]
    assert "Authorization" not in captured_calls[0][2]

    # Explicit api_key="" forces a keyless probe even when id is present.
    captured_calls.clear()
    res_explicit_blank = _web_client.post(
        "/api/providers/custom-endpoints/validate",
        json={
            "id": "myapi",
            "name": "MyAPI",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
            "api_key": "",
        },
    ).json()
    assert res_explicit_blank["ok"] is False
    assert "rejected the API key" in res_explicit_blank["message"]
    assert "Authorization" not in captured_calls[0][2]

    # Hand-written providers.<id> using ``api:`` + ``api_key:`` works identically.
    get_config_path().write_text(
        yaml.safe_dump({
            "providers": {
                "handwritten": {
                    "name": "Handwritten",
                    "api": "http://127.0.0.1:12168/v1",
                    "api_key": "saved-endpoint-secret",
                    "default_model": "gemini-3.8-flash",
                }
            }
        }),
        encoding="utf-8",
    )
    captured_calls.clear()
    res_handwritten = _web_client.post(
        "/api/providers/custom-endpoints/validate",
        json={
            "id": "handwritten",
            "name": "Handwritten",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
        },
    ).json()
    assert res_handwritten["ok"] is True
    assert captured_calls[0][2]["Authorization"] == "Bearer saved-endpoint-secret"

    # Unresolved ${VAR} template in api_key must not be shipped verbatim as a Bearer token.
    get_config_path().write_text(
        yaml.safe_dump({
            "providers": {
                "unresolved": {
                    "name": "Unresolved",
                    "base_url": "http://127.0.0.1:12168/v1",
                    "api_key": "${UNSET_CUSTOM_ENDPOINT_TEST_VAR}",
                    "default_model": "gemini-3.8-flash",
                }
            }
        }),
        encoding="utf-8",
    )
    captured_calls.clear()
    res_unresolved = _web_client.post(
        "/api/providers/custom-endpoints/validate",
        json={
            "id": "unresolved",
            "name": "Unresolved",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
        },
    ).json()
    assert res_unresolved["ok"] is False
    assert "Authorization" not in captured_calls[0][2]


def test_custom_endpoint_validate_respects_profile_scope(_web_client, monkeypatch):
    """?profile=<name> must resolve credentials from that profile's .env, never the default home."""
    from hermes_cli.profiles import create_profile

    create_profile("worker")

    _web_client.post(
        "/api/providers/custom-endpoints",
        json={
            "id": "shared",
            "name": "Shared",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
            "api_key": "default-profile-secret",
        },
    )
    _web_client.post(
        "/api/providers/custom-endpoints?profile=worker",
        json={
            "id": "shared",
            "name": "Shared",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
            "api_key": "worker-profile-secret",
        },
    )

    captured_calls = _install_fake_httpx(monkeypatch, expected_token="worker-profile-secret")
    res = _web_client.post(
        "/api/providers/custom-endpoints/validate?profile=worker",
        json={
            "id": "shared",
            "name": "Shared",
            "base_url": "http://127.0.0.1:12168/v1",
            "model": "gemini-3.8-flash",
        },
    ).json()
    assert res["ok"] is True
    assert captured_calls[0][2]["Authorization"] == "Bearer worker-profile-secret"
