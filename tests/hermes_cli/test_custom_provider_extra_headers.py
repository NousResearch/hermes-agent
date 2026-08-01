"""Tests for per-provider ``extra_headers`` in providers / custom_providers config.

PR #3526 salvage — user-configurable extra HTTP headers on LLM API calls
(reverse proxies, gateways, custom auth such as Cloudflare Access tokens).
"""

import json

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    apply_custom_provider_extra_headers_to_client_kwargs,
    get_custom_provider_extra_headers,
    get_custom_provider_session_affinity,
    normalize_extra_headers,
)
from hermes_cli import models as models_mod




def test_normalize_entry_keeps_extra_headers():
    normalized = _normalize_custom_provider_entry(
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"X-Custom-Auth": "tok", "X-Client-Name": "hermes"},
        }
    )
    assert normalized is not None
    assert normalized["extra_headers"] == {
        "X-Custom-Auth": "tok",
        "X-Client-Name": "hermes",
    }


def test_session_affinity_requires_explicit_matching_provider_opt_in():
    config = {
        "privacy": {
            "allow_third_party_identifiers": True,
        },
    }
    providers = [
        {
            "name": "opted-in-proxy",
            "base_url": "https://proxy.example.com/anthropic/",
            "session_affinity": True,
        },
        {
            "name": "default-proxy",
            "base_url": "https://other.example.com/anthropic",
        },
    ]

    assert get_custom_provider_session_affinity(
        "https://proxy.example.com/anthropic",
        custom_providers=providers,
        config=config,
    ) is True
    assert get_custom_provider_session_affinity(
        "https://other.example.com/anthropic",
        custom_providers=providers,
        config=config,
    ) is False


def test_session_affinity_requires_generic_privacy_opt_in():
    assert get_custom_provider_session_affinity(
        "https://proxy.example.com/anthropic",
        custom_providers=[
            {
                "name": "proxy",
                "base_url": "https://proxy.example.com/anthropic",
                "session_affinity": True,
            },
        ],
        config={"privacy": {"allow_third_party_identifiers": False}},
    ) is False


def test_normalize_entry_keeps_session_affinity_opt_in():
    normalized = _normalize_custom_provider_entry(
        {
            "name": "my-proxy",
            "base_url": "https://proxy.example.com/anthropic",
            "session_affinity": True,
        }
    )

    assert normalized is not None
    assert normalized["session_affinity"] is True












def test_fetch_api_models_sends_extra_headers_to_models_probe(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "proxy-model"}]}).encode()

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {
            key.lower(): value
            for key, value in request.header_items()
        }
        return FakeResponse()

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", fake_urlopen)

    models = models_mod.fetch_api_models(
        "proxy-key",
        "https://llm.internal.example.com/v1",
        headers={
            "sleeve-harness": "hermes",
            "sleeve-base-url": "http://localhost:8081/v1",
        },
    )

    assert models == ["proxy-model"]
    assert captured["url"] == "https://llm.internal.example.com/v1/models"
    assert captured["headers"]["authorization"] == "Bearer proxy-key"
    assert captured["headers"]["sleeve-harness"] == "hermes"
    assert captured["headers"]["sleeve-base-url"] == "http://localhost:8081/v1"
