"""Tests for per-provider ``extra_headers`` in providers / custom_providers config.

PR #3526 salvage — user-configurable extra HTTP headers on LLM API calls
(reverse proxies, gateways, custom auth such as Cloudflare Access tokens).
"""

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    apply_custom_provider_extra_headers_to_client_kwargs,
    get_custom_provider_extra_headers,
    normalize_extra_headers,
)
from hermes_cli import models as models_mod


def test_normalize_extra_headers_stringifies_and_drops_none():
    assert normalize_extra_headers({"X-Int": 7, "X-Str": "v", "X-None": None}) == {
        "X-Int": "7",
        "X-Str": "v",
    }


def test_normalize_extra_headers_rejects_non_dict_and_empty():
    for bad in (None, "x", 42, ["a"], {}):
        assert normalize_extra_headers(bad) == {}


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


def test_normalize_entry_drops_invalid_extra_headers():
    for bad in ("not-a-dict", {}, 42, ["a"]):
        normalized = _normalize_custom_provider_entry(
            {
                "name": "my-proxy",
                "base_url": "https://llm.internal.example.com/v1",
                "extra_headers": bad,
            }
        )
        assert normalized is not None
        assert "extra_headers" not in normalized


def test_normalize_entry_stringifies_values_and_skips_none():
    normalized = _normalize_custom_provider_entry(
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"X-Int": 7, "X-None": None},
        }
    )
    assert normalized is not None
    assert normalized["extra_headers"] == {"X-Int": "7"}


def test_get_custom_provider_extra_headers_matches_base_url():
    providers = [
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"CF-Access-Client-Id": "xxxx.access"},
        }
    ]
    # trailing-slash and case insensitive match, mirroring the TLS helper
    headers = get_custom_provider_extra_headers(
        "https://LLM.internal.example.com/v1/",
        custom_providers=providers,
    )
    assert headers == {"CF-Access-Client-Id": "xxxx.access"}


def test_get_custom_provider_extra_headers_no_match_returns_empty():
    providers = [
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"X-Secret": "s"},
        }
    ]
    assert get_custom_provider_extra_headers(
        "https://other.example.com/v1", custom_providers=providers,
    ) == {}
    # prefix look-alike host must not match (no substring bypass)
    assert get_custom_provider_extra_headers(
        "https://llm.internal.example.com.attacker.test/v1",
        custom_providers=providers,
    ) == {}


def test_apply_extra_headers_merges_onto_existing_defaults():
    client_kwargs = {
        "api_key": "x",
        "base_url": "https://llm.internal.example.com/v1",
        "default_headers": {"User-Agent": "curl/8.7.1", "X-Keep": "1"},
    }
    providers = [
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"User-Agent": "override", "X-New": "2"},
        }
    ]
    apply_custom_provider_extra_headers_to_client_kwargs(
        client_kwargs,
        "https://llm.internal.example.com/v1",
        custom_providers=providers,
    )
    assert client_kwargs["default_headers"] == {
        "User-Agent": "override",  # provider-specific value wins
        "X-Keep": "1",             # untouched defaults preserved
        "X-New": "2",
    }


def test_apply_extra_headers_noop_without_match():
    client_kwargs = {"api_key": "x", "base_url": "https://other.example.com/v1"}
    providers = [
        {
            "name": "my-proxy",
            "base_url": "https://llm.internal.example.com/v1",
            "extra_headers": {"X-Secret": "s"},
        }
    ]
    apply_custom_provider_extra_headers_to_client_kwargs(
        client_kwargs,
        "https://other.example.com/v1",
        custom_providers=providers,
    )
    assert "default_headers" not in client_kwargs


def test_fetch_api_models_sends_extra_headers_to_models_probe(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "proxy-model"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = {
                key.lower(): value for key, value in (headers or {}).items()
            }
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

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
