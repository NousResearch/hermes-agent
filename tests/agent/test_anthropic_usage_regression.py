"""Offline Anthropic usage regressions; no real credentials or network."""
import httpx
import pytest
from agent import account_usage


@pytest.fixture
def usage_http(monkeypatch):
    calls = []
    payload = {"five_hour": {"utilization": 20}}
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oat01-ambient")

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    real_client = httpx.Client
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    return payload, calls


def test_usage_rate_limit_is_visible_not_silently_missing(monkeypatch):
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oat01-fixture")
    real_client = httpx.Client
    def handle(request):
        return httpx.Response(429, json={"error": {"message": "private response not for display"}})
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    snapshot = account_usage.fetch_account_usage("anthropic")
    assert snapshot is not None
    assert not snapshot.available
    text = "\n".join(account_usage.render_account_usage_lines(snapshot))
    assert "429" in text
    assert "private response" not in text


def test_usage_uses_active_oauth_not_different_ambient_account(usage_http):
    _, calls = usage_http
    snapshot = account_usage.fetch_account_usage("anthropic", api_key="sk-ant-oat01-active")
    assert snapshot is not None and snapshot.available
    assert calls[0].headers["authorization"] == "Bearer sk-ant-oat01-active"


@pytest.mark.parametrize("token", ["", "sk-ant-api03-fixture"])
def test_explicit_non_oauth_never_falls_back_to_ambient(usage_http, token):
    _, calls = usage_http
    snapshot = account_usage.fetch_account_usage("anthropic", api_key=token)
    assert snapshot is None or not snapshot.available
    assert calls == []


def test_proxy_credential_is_not_forwarded_to_anthropic(usage_http):
    _, calls = usage_http
    snapshot = account_usage.fetch_account_usage(
        "anthropic", base_url="https://proxy.example/v1", api_key="eyJproxy-credential",
    )
    assert snapshot is not None and not snapshot.available
    assert calls == []
