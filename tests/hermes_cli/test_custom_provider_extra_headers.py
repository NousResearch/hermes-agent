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


def test_lowercase_auth_and_user_agent_replace_sdk_defaults_on_the_wire():
    """A configured header differing only in case from an SDK-owned one goes out ONCE, with the
    configured value (the SDK merges plain dicts: ``authorization`` used to ride beside its own
    ``Authorization``)."""
    import http.server
    import threading

    import openai

    seen = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append([(k.lower(), v) for k, v in self.headers.items() if k.lower() in ("authorization", "user-agent")])
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "probe"}}')

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    client_kwargs = {"api_key": "sk-default", "base_url": base_url, "default_headers": {"User-Agent": "hermes"}}
    apply_custom_provider_extra_headers_to_client_kwargs(client_kwargs, base_url, custom_providers=[{
        "name": "gw", "base_url": base_url,
        "extra_headers": {"authorization": "Bearer gateway", "user-agent": "CorpWAF/1.0"},
    }])
    try:
        openai.OpenAI(max_retries=0, **client_kwargs).chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}])
    except openai.BadRequestError:
        pass
    finally:
        server.shutdown()
    assert sorted(seen[0]) == [("authorization", "Bearer gateway"), ("user-agent", "CorpWAF/1.0")]


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
