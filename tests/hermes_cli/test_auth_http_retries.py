"""Regression tests for retry-enabled auth HTTP clients."""

from hermes_cli.auth import (
    refresh_codex_oauth_pure,
    _xai_oauth_discovery,
)


class _StubHTTPResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.calls.append(("post", args, kwargs))
        return self._response

    def request(self, *args, **kwargs):
        self.calls.append(("request", args, kwargs))
        return self._response


def _patch_retry_client(monkeypatch, response):
    holder = {}

    class _FakeTransport:
        def __init__(self, *, retries, verify=True):
            holder["transport_retries"] = retries
            holder["transport_verify"] = verify

    def _client_factory(*args, **kwargs):
        holder["client_kwargs"] = kwargs
        holder["client"] = _StubHTTPClient(response)
        return holder["client"]

    monkeypatch.setattr("hermes_cli.auth.httpx.HTTPTransport", _FakeTransport)
    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _client_factory)
    return holder


def test_refresh_codex_oauth_pure_uses_retry_transport(monkeypatch):
    holder = _patch_retry_client(
        monkeypatch,
        _StubHTTPResponse(
            200,
            {
                "access_token": "access-new",
                "refresh_token": "refresh-new",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        ),
    )

    result = refresh_codex_oauth_pure("access-old", "refresh-old")

    assert result["access_token"] == "access-new"
    assert holder["transport_retries"] == 3
    assert holder["transport_verify"] is True
    assert "transport" in holder["client_kwargs"]


def test_xai_oauth_discovery_uses_retry_transport(monkeypatch):
    holder = _patch_retry_client(
        monkeypatch,
        _StubHTTPResponse(
            200,
            {
                "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
                "token_endpoint": "https://auth.x.ai/oauth2/token",
            },
        ),
    )

    payload = _xai_oauth_discovery()

    assert payload["token_endpoint"] == "https://auth.x.ai/oauth2/token"
    assert holder["transport_retries"] == 3
    assert holder["transport_verify"] is True
    assert holder["client"].calls[0][0] == "request"
