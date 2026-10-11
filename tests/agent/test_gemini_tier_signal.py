"""Tier inference through real HTTP serialization, without external requests."""
import httpx
import pytest

from agent import gemini_native_adapter as gemini


@pytest.mark.parametrize(
    "status,headers,body,expected",
    [
        (200, {}, '{"candidates":[]}', "unknown"),
        (204, {}, "", "unknown"),
        (200, {"x-ratelimit-limit-requests-per-day": "bad"}, "{}", "unknown"),
        (200, {"X-RateLimit-Limit-Requests-Per-Day": "100"}, "{}", "free"),
        (200, {"x-ratelimit-limit-requests-per-day": "2000"}, "{}", "paid"),
        (429, {}, "generate_content_free_tier_requests", "free"),
        (429, {}, "rate limited", "paid"),
        (401, {}, "unauthorized", "unknown"),
    ],
)
def test_probe_tier_response_contract(monkeypatch, status, headers, body, expected):
    requests = []
    real_client = httpx.Client

    def respond(request):
        requests.append(request)
        return httpx.Response(status, headers=headers, text=body)

    def client(**kwargs):
        assert kwargs["timeout"] == 3.0
        return real_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(gemini.httpx, "Client", client)
    assert gemini.probe_gemini_tier(
        "fixture-key", "https://proxy.example.test/custom", model="caller-model", timeout=3.0
    ) == expected
    assert len(requests) == 1
    assert requests[0].url.path == "/custom/v1beta/models/caller-model:generateContent"
    assert requests[0].url.params["key"] == "fixture-key"


def test_probe_missing_key_and_transport_error_remain_nonblocking(monkeypatch):
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise httpx.ConnectError("fixture offline")

    monkeypatch.setattr(gemini.httpx, "Client", unavailable)
    assert gemini.probe_gemini_tier("") == "unknown"
    assert calls == []
    assert gemini.probe_gemini_tier("fixture-key") == "unknown"
    assert len(calls) == 1
