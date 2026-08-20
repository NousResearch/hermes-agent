"""Contract tests for the MrScraper web plugin and native tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.web.mrscraper import tools as mt
from plugins.mrscraper_client import MrScraperAPIError, MrScraperClient


class CaptureClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def reset(cls):
        cls.calls = []

    def primary_get(self, path, *, params=None):
        self.calls.append(("GET", path, params, None))
        return {"ok": True}

    def primary_post(self, path, body):
        self.calls.append(("POST", path, None, body))
        return {"ok": True}

    def serp_search(self, body, *, html=False):
        self.calls.append(("SERP", html, None, body))
        return "<html>ok</html>" if html else {"results": []}


@pytest.fixture(autouse=True)
def _capture_client(monkeypatch):
    CaptureClient.reset()
    monkeypatch.setattr(mt, "MrScraperClient", CaptureClient)


def test_bundled_prompts_are_exact_n8n_asset() -> None:
    bundled = Path(mt.__file__).with_name("structured_data_prompts.json")
    assert hashlib.sha256(bundled.read_bytes()).hexdigest() == (
        "3d9c15e8ebe7ad8cb04281251311200c1d3413452f14f252dc9ed3a8aae8533a"
    )
    assert list(mt.STRUCTURED_DATA_PROMPTS) == mt.STRUCTURED_CATEGORIES


def test_all_fourteen_web_tools_have_typed_schemas() -> None:
    assert len(mt.MRSCRAPER_TOOLS) == 14
    names = [name for name, _schema, _handler in mt.MRSCRAPER_TOOLS]
    assert len(names) == len(set(names))
    for name, schema, handler in mt.MRSCRAPER_TOOLS:
        assert schema["name"] == name
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["additionalProperties"] is False
        assert callable(handler)


def test_crawl_maps_defaults_and_omits_blank_patterns() -> None:
    mt.crawl_website_urls({"url": "https://example.com", "include_patterns": ""})
    assert CaptureClient.calls == [
        (
            "POST",
            "/api/v1/scrapers-ai",
            None,
            {
                "graph": "map",
                "url": "https://example.com",
                "maxDepth": 2,
                "maxPages": 50,
                "limit": 50,
            },
        )
    ]


def test_prompt_schema_is_compact_and_appended_once() -> None:
    mt.extract_page_by_prompt({
        "url": "https://example.com/p",
        "prompt": "Extract product",
        "output_schema": {"name": "string", "price": "number"},
        "mode": "Cheap",
        "proxy_country": "ID",
    })
    body = CaptureClient.calls[0][3]
    assert body["graph"] == "general"
    assert body["mode"] == "Cheap"
    assert body["proxyCountry"] == "ID"
    assert body["message"] == (
        "Extract product\n\nReturn the output as JSON matching this schema:\n"
        '{"name":"string","price":"number"}'
    )
    assert body["message"].count("Return the output") == 1


def test_listing_maps_max_pages_and_item_schema_label() -> None:
    mt.extract_listings({
        "url": "https://example.com/list",
        "output_schema": {"title": "string"},
        "max_pages": 3,
    })
    body = CaptureClient.calls[0][3]
    assert body["maxPages"] == 3
    assert body["message"] == (
        'Return each item as JSON matching this schema:\n{"title":"string"}'
    )


def test_listing_invalid_minimum_raises() -> None:
    with pytest.raises(mt.MrScraperError, match="max_pages must be at least 1"):
        mt.extract_listings({"url": "https://example.com/list", "max_pages": 0})


def test_structured_data_uses_exact_selected_prompt() -> None:
    mt.extract_structured_data({
        "url": "https://example.com/product",
        "category": "product",
    })
    body = CaptureClient.calls[0][3]
    assert body["message"] == mt.STRUCTURED_DATA_PROMPTS["product"]
    assert "category" not in body


def test_serp_html_preserves_text_and_boolean_mapping() -> None:
    result = mt.search_google_serp({
        "query": "Hermes",
        "format": "html",
        "render_js": False,
    })
    assert result == "<html>ok</html>"
    assert CaptureClient.calls[0][1] is True
    assert CaptureClient.calls[0][3]["renderJs"] is False


@pytest.mark.parametrize("field", ["region", "language"])
def test_serp_rejects_non_two_letter_codes(field: str) -> None:
    with pytest.raises(mt.MrScraperError, match=f"{field} must be a two-letter code"):
        mt.search_google_serp({"query": "Hermes", field: "usa"})


def test_results_query_and_result_id_encoding() -> None:
    mt.get_results({"scraper_id": "abc", "page": 2, "sort_order": "ASC"})
    assert CaptureClient.calls[0][2] == {
        "filters[scraperId]": "abc",
        "page": 2,
        "pageSize": 10,
        "sort": "createdAt",
        "sortOrder": "ASC",
    }
    mt.get_result_detail({"result_id": "a/b ?"})
    assert CaptureClient.calls[1][1] == "/api/v1/results/a%2Fb%20%3F"


def test_latest_results_uses_fixed_sort() -> None:
    mt.get_latest_results({"scraper_id": "abc", "count": 4})
    assert CaptureClient.calls[0][2] == {
        "filters[scraperId]": "abc",
        "page": 1,
        "pageSize": 4,
        "sort": "createdAt",
        "sortOrder": "DESC",
    }


def test_latest_results_invalid_count_raises() -> None:
    with pytest.raises(mt.MrScraperError, match="count must be at least 1"):
        mt.get_latest_results({"scraper_id": "abc", "count": 0})


def test_ai_map_run_preserves_zero_and_false_defaults() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com",
        "agent_type": "map",
        "max_depth": 0,
        "limit": 1,
    })
    body = CaptureClient.calls[0][3]
    assert body["maxDepth"] == 0
    assert body["maxPages"] == 50
    assert body["limit"] == 1
    assert "agent_type" not in body


def test_ai_general_maps_boolean_defaults_and_omits_blank_selector() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com",
        "wait_for_selector": "",
    })
    assert CaptureClient.calls[0][3] == {
        "scraperId": "s1",
        "url": "https://example.com",
        "maxRetry": 3,
        "bypassProxy": False,
        "html": False,
        "markdown": False,
        "renderJavascript": False,
        "returnCookies": False,
        "screenshot": False,
        "useHomePage": False,
    }


def test_ai_listing_maps_conditional_defaults() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com/list",
        "agent_type": "listing",
    })
    body = CaptureClient.calls[0][3]
    assert body["maxPages"] == 5
    assert body["timeout"] == 300
    assert body["stream"] is False


def test_ai_general_rejects_listing_only_parameter() -> None:
    with pytest.raises(mt.MrScraperError, match="max_pages"):
        mt.run_existing_scraper({
            "scraper_type": "ai",
            "scraper_id": "s1",
            "url": "https://example.com",
            "agent_type": "general",
            "max_pages": 2,
        })


def test_manual_run_preserves_empty_collections_and_stringifies_screenshot() -> None:
    mt.run_existing_scraper({
        "scraper_type": "manual",
        "scraper_id": "m1",
        "url": "https://example.com",
        "cookies": [],
        "paginator": {},
        "token_cap": 0,
        "screenshot": False,
    })
    method, path, _params, body = CaptureClient.calls[0]
    assert (method, path) == ("POST", "/api/v1/scrapers-manual-rerun")
    assert body["cookies"] == []
    assert body["paginator"] == {}
    assert body["tokenCap"] == 0
    assert body["screenshot"] == "false"
    assert body["bypassProxy"] is True
    assert body["homePageTimeout"] == 10
    assert body["timeout"] == 600


def test_manual_run_rejects_ai_parameters() -> None:
    with pytest.raises(mt.MrScraperError, match="agent_type"):
        mt.run_existing_scraper({
            "scraper_type": "manual",
            "scraper_id": "m1",
            "url": "https://example.com",
            "agent_type": "general",
        })


def test_manual_run_rejects_non_object_cookie() -> None:
    with pytest.raises(mt.MrScraperError, match="cookies must be an array of objects"):
        mt.run_existing_scraper({
            "scraper_type": "manual",
            "scraper_id": "m1",
            "url": "https://example.com",
            "cookies": ["not-an-object"],
        })


@pytest.mark.parametrize(
    "scraper_type,expected_path",
    [
        ("ai", "/api/v1/scrapers-ai-rerun/bulk"),
        ("manual", "/api/v1/scrapers-manual-rerun/bulk"),
    ],
)
def test_batch_endpoint_and_text_normalization(scraper_type, expected_path) -> None:
    mt.run_existing_scraper_batch({
        "scraper_type": scraper_type,
        "scraper_id": "s1",
        "urls": "https://a.example,\nhttps://b.example\n",
    })
    assert CaptureClient.calls[0][1] == expected_path
    assert CaptureClient.calls[0][3]["urls"] == [
        "https://a.example",
        "https://b.example",
    ]


def test_batch_rejects_non_string_array_entries() -> None:
    with pytest.raises(mt.MrScraperError, match="urls must be an array of strings"):
        mt.run_existing_scraper_batch({
            "scraper_type": "ai",
            "scraper_id": "s1",
            "urls": [123],
        })


def test_client_primary_and_serp_auth_headers(monkeypatch) -> None:
    response = SimpleNamespace(
        ok=True,
        status_code=200,
        text='{"ok":true}',
        headers={"Content-Type": "application/json"},
        json=lambda: {"ok": True},
    )
    request = MagicMock(return_value=response)
    monkeypatch.setattr("plugins.mrscraper_client.requests.request", request)
    client = MrScraperClient(token="not-a-real-secret")

    client.primary_get("/api/v1/subscription-accounts")
    primary_call = request.call_args
    assert primary_call.args[:2] == (
        "GET",
        "https://api.app.mrscraper.com/api/v1/subscription-accounts",
    )
    assert primary_call.kwargs["headers"]["x-api-token"] == "not-a-real-secret"
    client.serp_search({"query": "x"})
    serp_call = request.call_args
    assert serp_call.args[:2] == (
        "POST",
        "https://sync.scraper.mrscraper.com/api/google/serp/v2/sync",
    )
    assert serp_call.kwargs["headers"]["Authorization"] == ("Bearer not-a-real-secret")


def test_client_error_redacts_token_and_truncates_body(monkeypatch) -> None:
    secret = "runtime" + "-secret-value"
    response = SimpleNamespace(
        ok=False,
        status_code=401,
        text=f"bad token {secret} " + ("x" * 1000),
        headers={"Content-Type": "application/json"},
    )
    monkeypatch.setattr(
        "plugins.mrscraper_client.requests.request", MagicMock(return_value=response)
    )
    with pytest.raises(MrScraperAPIError) as raised:
        MrScraperClient(token=secret).primary_get("/api/v1/results")
    assert secret not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)
    assert "truncated" in str(raised.value)


def test_plugin_registers_provider_and_fourteen_tools() -> None:
    import plugins.web.mrscraper as plugin

    ctx = SimpleNamespace(
        providers=[],
        tools=[],
        register_web_search_provider=lambda provider: ctx.providers.append(provider),
        register_tool=lambda **kwargs: ctx.tools.append(kwargs),
    )
    plugin.register(ctx)
    assert [provider.name for provider in ctx.providers] == ["mrscraper"]
    assert len(ctx.tools) == 14


def test_web_provider_normalizes_serp(monkeypatch) -> None:
    from plugins.web.mrscraper.provider import MrScraperWebSearchProvider

    monkeypatch.setattr(
        "plugins.web.mrscraper.provider.search_google_serp",
        lambda _args: {
            "organic_results": [
                {"title": "One", "link": "https://one", "snippet": "First"}
            ]
        },
    )
    result = MrScraperWebSearchProvider().search("query", limit=1)
    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "One",
                    "url": "https://one",
                    "description": "First",
                    "position": 1,
                }
            ]
        },
    }


def test_web_provider_extract_uses_rendered_markdown(monkeypatch) -> None:
    from plugins.web.mrscraper.provider import MrScraperWebSearchProvider

    monkeypatch.setattr("plugins.web.mrscraper.provider.is_safe_url", lambda _url: True)
    monkeypatch.setattr(
        "plugins.web.mrscraper.provider.check_website_access", lambda _url: None
    )
    fetch = MagicMock(
        return_value={
            "data": {
                "markdown": "# Example",
                "metadata": {"title": "Example", "lang": "en"},
            }
        }
    )
    monkeypatch.setattr("plugins.web.mrscraper.provider.fetch_rendered_html", fetch)

    result = MrScraperWebSearchProvider().extract(["https://example.com"])

    fetch.assert_called_once_with({
        "url": "https://example.com",
        "html": False,
        "markdown": True,
    })
    assert result == [
        {
            "url": "https://example.com",
            "title": "Example",
            "content": "# Example",
            "raw_content": "# Example",
            "metadata": {"title": "Example", "lang": "en"},
        }
    ]


def test_web_provider_rejects_unsafe_url_without_fetch(monkeypatch) -> None:
    from plugins.web.mrscraper.provider import MrScraperWebSearchProvider

    monkeypatch.setattr(
        "plugins.web.mrscraper.provider.is_safe_url", lambda _url: False
    )
    fetch = MagicMock()
    monkeypatch.setattr("plugins.web.mrscraper.provider.fetch_rendered_html", fetch)

    result = MrScraperWebSearchProvider().extract(["http://127.0.0.1/private"])

    assert result == [
        {"url": "http://127.0.0.1/private", "title": "", "error": "Unsafe URL"}
    ]
    fetch.assert_not_called()
