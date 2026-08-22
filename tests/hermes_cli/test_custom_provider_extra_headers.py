"""Tests for per-provider ``extra_headers`` in providers / custom_providers config.

PR #3526 salvage — user-configurable extra HTTP headers on LLM API calls
(reverse proxies, gateways, custom auth such as Cloudflare Access tokens).
"""

import json

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
    """Match by normalized base_url returns the entry's extra_headers."""
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
    """No matching base_url yields empty dict."""
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


def test_get_custom_provider_extra_headers_preserves_extra_path_segment():
    """Extra path segment after normalisation is still a mismatch."""
    providers = [
        {
            "base_url": "https://llm.internal.example.com/v1//",
            "extra_headers": {"Authorization": "secret"},
        }
    ]
    assert get_custom_provider_extra_headers(
        "https://llm.internal.example.com/v1",
        custom_providers=providers,
    ) == {}


def test_get_custom_provider_extra_headers_skips_alias_without_headers():
    """Bug #74465: an earlier entry matching the same URL but without
    extra_headers must not shadow a later entry that DOES have headers."""
    providers = [
        {
            "name": "direct",
            "base_url": "http://127.0.0.1:8787/v1",
            # no extra_headers
        },
        {
            "name": "aether-router",
            "base_url": "http://127.0.0.1:8787/v1",
            "extra_headers": {"X-Aether-Route": "my-route"},
        },
    ]
    headers = get_custom_provider_extra_headers(
        "http://127.0.0.1:8787/v1",
        custom_providers=providers,
    )
    assert headers == {"X-Aether-Route": "my-route"}


def test_get_custom_provider_extra_headers_skips_empty_header_dict_alias():
    """An earlier entry with an explicit but empty extra_headers dict must
    not shadow a later entry that carries real headers."""
    providers = [
        {
            "name": "bare-alias",
            "base_url": "http://127.0.0.1:8787/v1",
            "extra_headers": {},
        },
        {
            "name": "proxied",
            "base_url": "http://127.0.0.1:8787/v1",
            "extra_headers": {"Authorization": "Bearer tok"},
        },
    ]
    headers = get_custom_provider_extra_headers(
        "http://127.0.0.1:8787/v1",
        custom_providers=providers,
    )
    assert headers == {"Authorization": "Bearer tok"}


def test_apply_extra_headers_merges_onto_existing_defaults():
    """apply_custom_provider_extra_headers_to_client_kwargs merges headers,
    with provider-specific values winning over existing defaults."""
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
    """No matching base_url -> no default_headers key added."""
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


# ---------------------------------------------------------------------------
# Model-aware header selection for a base_url shared by several named
# custom_providers[] entries (mirrors hermes-webui#7176 / PR #7177's
# provider-selection fix — see that PR's review for the header-drop defect
# this closes: a shared endpoint's resolved entry B could have its
# extra_headers silently replaced by sibling entry A's headers because
# get_custom_provider_extra_headers() only matched by base_url).
# ---------------------------------------------------------------------------


SHARED_URL = "https://gateway.example/v1"

_SHARED_ENTRIES = [
    {
        "name": "Gateway OpenAI Chat",
        "base_url": SHARED_URL,
        "extra_headers": {"CF-Access-Client-Id": "openai-chat-id"},
        "models": {"gpt-5": {}, "gpt-5-mini": {}},
    },
    {
        "name": "Gateway Claude",
        "base_url": SHARED_URL,
        "extra_headers": {"CF-Access-Client-Id": "claude-id"},
        "models": {"claude-sonnet-5@default": {}, "claude-opus-5@default": {}},
    },
]


def test_get_custom_provider_extra_headers_prefers_owning_entry_with_model_id():
    """When two entries share one base_url, model_id must select the
    entry's headers that actually owns the requested model, not just the
    first declared entry."""
    headers = get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=_SHARED_ENTRIES, model_id="claude-sonnet-5@default"
    )
    assert headers == {"CF-Access-Client-Id": "claude-id"}

    headers_openai = get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=_SHARED_ENTRIES, model_id="gpt-5"
    )
    assert headers_openai == {"CF-Access-Client-Id": "openai-chat-id"}


def test_get_custom_provider_extra_headers_falls_back_without_model_id():
    """Existing callers that don't pass model_id keep the prior
    first-declared-entry behavior — no behavior change for them."""
    headers = get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=_SHARED_ENTRIES,
    )
    assert headers == {"CF-Access-Client-Id": "openai-chat-id"}


def test_get_custom_provider_extra_headers_unclaimed_model_falls_back_to_first():
    """A model_id that no shared entry claims falls back to the first
    base_url match, preserving prior behavior rather than failing closed."""
    headers = get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=_SHARED_ENTRIES, model_id="totally-unknown-model",
    )
    assert headers == {"CF-Access-Client-Id": "openai-chat-id"}


def test_get_custom_provider_extra_headers_owning_entry_without_headers_returns_empty():
    """If the entry that owns the requested model declares NO extra_headers
    of its own, its sibling's headers must NOT leak in — the absence of
    headers on the owning entry is itself meaningful and must not silently
    fall back to a differently-scoped entry's headers."""
    entries = [
        {
            "name": "Gateway OpenAI Chat",
            "base_url": SHARED_URL,
            "extra_headers": {"CF-Access-Client-Id": "openai-chat-id"},
            "models": {"gpt-5": {}},
        },
        {
            "name": "Gateway Claude",
            "base_url": SHARED_URL,
            # No extra_headers on the Claude entry.
            "models": {"claude-sonnet-5@default": {}},
        },
    ]
    headers = get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=entries, model_id="claude-sonnet-5@default",
    )
    assert headers == {}


def test_get_custom_provider_extra_headers_singular_model_field_owns_headers():
    """Ownership via the singular ``model`` field (not just ``models``
    allowlist) also selects the correct entry's headers."""
    entries = [
        {
            "name": "A",
            "base_url": SHARED_URL,
            "model": "model-a",
            "extra_headers": {"X-Which": "a"},
        },
        {
            "name": "B",
            "base_url": SHARED_URL,
            "model": "model-b",
            "extra_headers": {"X-Which": "b"},
        },
    ]
    assert get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=entries, model_id="model-b",
    ) == {"X-Which": "b"}
    assert get_custom_provider_extra_headers(
        SHARED_URL, custom_providers=entries, model_id="model-a",
    ) == {"X-Which": "a"}


def test_apply_extra_headers_model_id_selects_owning_entry_headers():
    """End-to-end: apply_custom_provider_extra_headers_to_client_kwargs with
    model_id merges the OWNING entry's headers, not the first declared."""
    client_kwargs = {"api_key": "x", "base_url": SHARED_URL}
    apply_custom_provider_extra_headers_to_client_kwargs(
        client_kwargs, SHARED_URL, custom_providers=_SHARED_ENTRIES,
        model_id="claude-sonnet-5@default",
    )
    assert client_kwargs["default_headers"] == {"CF-Access-Client-Id": "claude-id"}


def test_apply_extra_headers_without_model_id_keeps_prior_behavior():
    """apply_custom_provider_extra_headers_to_client_kwargs without model_id
    is unchanged: first declared entry sharing the base_url wins."""
    client_kwargs = {"api_key": "x", "base_url": SHARED_URL}
    apply_custom_provider_extra_headers_to_client_kwargs(
        client_kwargs, SHARED_URL, custom_providers=_SHARED_ENTRIES,
    )
    assert client_kwargs["default_headers"] == {"CF-Access-Client-Id": "openai-chat-id"}
