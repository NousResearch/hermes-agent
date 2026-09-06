"""Offline transport contract; no real credentials or provider requests."""
import json

import httpx
import pytest

from agent import account_usage as usage


@pytest.mark.parametrize("outcome, expected", [
    (401, "auth"), (403, "auth"), (429, "rate_limit"), (503, "network"),
    (404, "unsupported"), ("timeout", "network"), ("invalid", "unsupported"),
])
def test_typed_fetch_sanitizes_transport_failures(monkeypatch, outcome, expected):
    def respond(request):
        if outcome == "timeout":
            raise httpx.ReadTimeout("fixture-secret in URL and header", request=request)
        if outcome == "invalid":
            return httpx.Response(200, text="fixture-secret")
        return httpx.Response(outcome, json={"error": "fixture-secret"})

    client = httpx.Client
    monkeypatch.setattr(usage.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    result = usage.fetch_codex_quota(api_key="fixture-secret")
    assert result.error == expected
    assert "fixture-secret" not in json.dumps(result.to_dict())
    assert usage.codex_quota_snapshot(result) is None


def test_selected_pool_token_never_borrows_singleton_account(monkeypatch):
    monkeypatch.setattr(usage, "resolve_codex_runtime_credentials", lambda **kw: {
        "api_key": "selected-token", "source": "credential_pool", "base_url": ""})
    monkeypatch.setattr(usage, "_read_codex_tokens", lambda: {
        "tokens": {"access_token": "different-token", "account_id": "wrong-account"}})
    token, _, account = usage._resolve_codex_usage_credentials(None, None)
    assert token == "selected-token"
    assert account is None


def test_resolver_errors_never_retry_another_account(monkeypatch):
    calls = []
    def resolve(*args):
        calls.append(True)
        raise httpx.ConnectError("fixture-secret")
    monkeypatch.setattr(usage, "_resolve_codex_usage_credentials", resolve)
    assert usage.fetch_codex_quota().error == "network"
    assert len(calls) == 1
    monkeypatch.setattr(usage, "_resolve_codex_usage_credentials", lambda *a: (_ for _ in ()).throw(
        usage.AuthError("fixture-secret", code="codex_rate_limited")))
    assert usage.fetch_codex_quota().error == "rate_limit"


def test_absent_credit_flag_is_unknown_not_false():
    result = usage.parse_codex_quota({}, fetched_at=1800000000)
    assert result.credits_unlimited is None
