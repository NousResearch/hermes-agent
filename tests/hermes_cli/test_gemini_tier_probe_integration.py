"""The setup gate consumes the real Gemini probe without paid-tier guesswork."""
from types import SimpleNamespace

import httpx
import pytest

from hermes_cli.model_setup_flows import _gemini_tier_ok


@pytest.mark.parametrize("headers,allowed,label", [
    ({}, True, "could not verify"),
    ({"x-ratelimit-limit-requests-per-day": "malformed"}, True, "could not verify"),
    ({"x-ratelimit-limit-requests-per-day": "2000"}, True, "paid"),
    ({"x-ratelimit-limit-requests-per-day": "100"}, False, "free"),
])
def test_real_probe_to_setup_gate(monkeypatch, capsys, headers, allowed, label):
    real_client = httpx.Client
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers=headers, json={"candidates": []})

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(respond)
    ))
    monkeypatch.setenv("GEMINI_BASE_URL", "https://fixture.example.test/selected")
    assert _gemini_tier_ok(
        "fixture-key", SimpleNamespace(inference_base_url="https://unused.example.test"), "GEMINI_BASE_URL"
    ) is allowed
    text = capsys.readouterr().out
    assert label in text
    if label != "paid":
        assert "Tier check: paid" not in text
    assert len(requests) == 1
    assert requests[0].url.host == "fixture.example.test"
    assert requests[0].url.path.startswith("/selected/v1beta/models/")
