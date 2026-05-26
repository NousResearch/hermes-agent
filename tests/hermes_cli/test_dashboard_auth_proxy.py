from __future__ import annotations

import base64

import pytest
from fastapi import Response
from fastapi.testclient import TestClient

from hermes_cli import dashboard_auth_proxy as proxy


def _basic(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode()
    return "Basic " + base64.b64encode(raw).decode()


@pytest.fixture(autouse=True)
def _proxy_env(monkeypatch, tmp_path):
    monkeypatch.delenv(proxy.USERNAME_ENV, raising=False)
    monkeypatch.delenv(proxy.PASSWORD_FILE_ENV, raising=False)
    monkeypatch.delenv(proxy.UPSTREAM_ENV, raising=False)
    password_file = tmp_path / "password"
    password_file.write_text("secret\n", encoding="utf-8")
    return password_file


def test_proxy_fails_closed_without_credentials():
    client = TestClient(proxy.app)

    response = client.get("/")

    assert response.status_code == 503
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_proxy_rejects_missing_or_bad_basic_auth(monkeypatch, _proxy_env):
    monkeypatch.setenv(proxy.USERNAME_ENV, "oscar")
    monkeypatch.setenv(proxy.PASSWORD_FILE_ENV, str(_proxy_env))
    client = TestClient(proxy.app)

    missing = client.get("/")
    bad = client.get("/", headers={"Authorization": _basic("oscar", "wrong")})

    assert missing.status_code == 401
    assert bad.status_code == 401
    assert missing.headers["WWW-Authenticate"].startswith("Basic")


def test_proxy_accepts_valid_basic_auth(monkeypatch, _proxy_env):
    monkeypatch.setenv(proxy.USERNAME_ENV, "oscar")
    monkeypatch.setenv(proxy.PASSWORD_FILE_ENV, str(_proxy_env))

    async def fake_proxy_to_upstream(request, path):
        return Response(f"proxied:{path}")

    monkeypatch.setattr(proxy, "_proxy_to_upstream", fake_proxy_to_upstream)
    client = TestClient(proxy.app)

    response = client.get("/kanban", headers={"Authorization": _basic("oscar", "secret")})

    assert response.status_code == 200
    assert response.text == "proxied:kanban"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_proxy_strips_proxy_auth_and_sets_upstream_host(monkeypatch):
    monkeypatch.setenv(proxy.UPSTREAM_ENV, "http://127.0.0.1:9119")

    headers = proxy._proxy_headers(
        {
            "Authorization": "Basic abc",
            "Host": "public.example",
            "X-Hermes-Session-Token": "session",
            "Connection": "keep-alive",
        },
    )

    assert "Authorization" not in headers
    assert headers["Host"] == "127.0.0.1:9119"
    assert headers["X-Hermes-Session-Token"] == "session"
    assert "Connection" not in headers
