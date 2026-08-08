"""Tests for detect_local_server_type endpoint-probe ordering.

The probe now checks ``/v1/models`` first and returns
"openai-compatible" on a 200, short-circuiting the Ollama / LM Studio
/ llama.cpp / vLLM probes. This avoids spurious 404 health-check noise
against standard OpenAI-compatible servers. See
agent/model_metadata.py (detect_local_server_type).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.model_metadata import detect_local_server_type


def _resp(status_code, *, json_body=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.json = MagicMock(return_value=json_body if json_body is not None else {})
    return r


def _client_routing(route):
    """Build a fake httpx.Client whose .get(url) dispatches via ``route``.

    ``route`` maps a URL suffix → response (or raises if the suffix isn't
    found, mimicking a connection error to that endpoint).
    """
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)

    def _get(url, *a, **k):
        for suffix, resp in route.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        # Unmatched endpoint → simulate connection refused.
        raise ConnectionError(f"no route for {url}")

    client.get = MagicMock(side_effect=_get)
    return client


def test_v1_models_200_returns_openai_compatible():
    """A 200 on /v1/models short-circuits to 'openai-compatible'."""
    client = _client_routing({"/v1/models": _resp(200, json_body={"data": []})})
    with patch("httpx.Client", return_value=client):
        result = detect_local_server_type("http://localhost:8000/v1", api_key="sk-x")
    assert result == "openai-compatible"


def test_v1_models_checked_before_other_probes():
    """When /v1/models answers 200, the Ollama/llama.cpp/vLLM endpoints are
    never probed — that's the noise-reduction the fix delivers."""
    client = _client_routing({"/v1/models": _resp(200, json_body={"data": []})})
    with patch("httpx.Client", return_value=client):
        detect_local_server_type("http://localhost:8000/v1")

    probed = [call.args[0] for call in client.get.call_args_list]
    assert any(u.endswith("/v1/models") for u in probed)
    # None of the downstream probes should have fired.
    for noisy in ("/api/tags", "/v1/props", "/props", "/version", "/api/v1/models"):
        assert not any(u.endswith(noisy) for u in probed), f"unexpected probe: {noisy}"


def test_falls_through_to_ollama_when_v1_models_unavailable():
    """If /v1/models isn't OpenAI-compatible (connection error), the probe
    still detects Ollama via /api/tags."""
    client = _client_routing({
        "/v1/models": ConnectionError("refused"),
        "/api/v1/models": _resp(404),
        "/api/tags": _resp(200, json_body={"models": [{"name": "llama3"}]}),
    })
    with patch("httpx.Client", return_value=client):
        result = detect_local_server_type("http://localhost:11434")
    assert result == "ollama"


def test_v1_models_non_200_does_not_claim_openai_compatible():
    """A non-200 on /v1/models must not be mistaken for openai-compatible;
    detection continues and (here) finds nothing."""
    client = _client_routing({
        "/v1/models": _resp(404),
        "/api/v1/models": _resp(404),
        "/api/tags": _resp(404),
        "/v1/props": _resp(404),
        "/props": _resp(404),
        "/version": _resp(404),
    })
    with patch("httpx.Client", return_value=client):
        result = detect_local_server_type("http://localhost:9999")
    assert result is None
