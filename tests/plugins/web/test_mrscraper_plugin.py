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


def test_existing_scraper_schemas_explain_mode_selection() -> None:
    schemas = {name: schema for name, schema, _handler in mt.MRSCRAPER_TOOLS}
    single = schemas["mrscraper_run_existing_scraper"]
    properties = single["parameters"]["properties"]
    assert "scraper_type" in single["description"]
    assert "agent_type" in single["description"]
    assert properties["scraper_type"]["default"] == "ai"
    assert properties["agent_type"]["default"] == "general"
    assert "scraper_type" not in single["parameters"]["required"]
    assert "General" in properties["render_javascript"]["description"]
    assert "Listing" in properties["max_pages"]["description"]
    assert "Map" in properties["include_patterns"]["description"]
    assert "Manual" in properties["home_page"]["description"]

    batch = schemas["mrscraper_run_existing_scraper_batch"]
    assert set(batch["parameters"]["properties"]) == {
        "scraper_type",
        "scraper_id",
        "urls",
    }
    assert "single-run" in batch["description"]


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


def test_ai_map_run_uses_defaults_and_omits_blank_patterns() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com",
        "agent_type": "map",
        "include_patterns": "",
        "exclude_patterns": "  ",
    })
    body = CaptureClient.calls[0][3]
    assert body == {
        "scraperId": "s1",
        "url": "https://example.com",
        "maxRetry": 3,
        "maxDepth": 2,
        "maxPages": 50,
        "limit": 50,
    }


def test_ai_map_run_serializes_user_patterns() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com",
        "agent_type": "map",
        "include_patterns": "^/products/|^/offers/",
        "exclude_patterns": "^/account/",
    })
    body = CaptureClient.calls[0][3]
    assert body["includePatterns"] == "^/products/|^/offers/"
    assert body["excludePatterns"] == "^/account/"


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
    }


def test_ai_listing_maps_conditional_defaults() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com/list",
        "agent_type": "listing",
    })
    body = CaptureClient.calls[0][3]
    assert body["maxPages"] == 1
    assert body["timeout"] == 300
    assert body["stream"] is False


def test_ai_listing_accepts_max_pages_override() -> None:
    mt.run_existing_scraper({
        "scraper_type": "ai",
        "scraper_id": "s1",
        "url": "https://example.com/list",
        "agent_type": "listing",
        "max_pages": 7,
    })
    assert CaptureClient.calls[0][3]["maxPages"] == 7


def test_ai_listing_serializes_enabled_home_page_only() -> None:
    mt.run_existing_scraper({
        "scraper_id": "s1",
        "url": "https://example.com/list",
        "agent_type": "listing",
        "use_home_page": True,
    })
    assert CaptureClient.calls[0][3]["useHomePage"] is True


def test_ai_general_ignores_listing_map_and_manual_fields() -> None:
    mt.run_existing_scraper({
        "scraper_id": "s1",
        "url": "https://example.com",
        "max_pages": 2,
        "timeout": 5,
        "stream": True,
        "max_depth": 9,
        "limit": 2,
        "include_patterns": "include",
        "exclude_patterns": "exclude",
        "cookie_jar": "jar",
        "cookies": [{"name": "session"}],
        "home_page": True,
        "home_page_timeout": 20,
        "paginator": {"next": ".next"},
        "proxy": "proxy",
        "record": True,
        "return_cookie": True,
        "token_cap": 10,
    })
    body = CaptureClient.calls[0][3]
    assert "maxPages" not in body
    assert "timeout" not in body
    assert "stream" not in body
    assert "maxDepth" not in body
    assert "limit" not in body
    assert "includePatterns" not in body
    assert "excludePatterns" not in body
    assert "cookieJar" not in body
    assert "cookies" not in body
    assert "homePage" not in body
    assert "homePageTimeout" not in body
    assert "paginator" not in body
    assert "proxy" not in body
    assert "record" not in body
    assert "returnCookie" not in body
    assert "tokenCap" not in body


def test_ai_listing_ignores_map_and_manual_fields() -> None:
    mt.run_existing_scraper({
        "scraper_id": "s1",
        "url": "https://example.com/list",
        "agent_type": "listing",
        "max_depth": 9,
        "limit": 2,
        "include_patterns": "include",
        "exclude_patterns": "exclude",
        "home_page": True,
        "return_cookie": True,
        "token_cap": 10,
    })
    body = CaptureClient.calls[0][3]
    assert "maxDepth" not in body
    assert "limit" not in body
    assert "includePatterns" not in body
    assert "excludePatterns" not in body
    assert "homePage" not in body
    assert "returnCookie" not in body
    assert "tokenCap" not in body


def test_ai_map_ignores_general_listing_home_page_and_manual_fields() -> None:
    mt.run_existing_scraper({
        "scraper_id": "s1",
        "url": "https://example.com/map",
        "agent_type": "map",
        "bypass_proxy": True,
        "html": True,
        "markdown": True,
        "render_javascript": True,
        "return_cookies": True,
        "screenshot": True,
        "use_home_page": True,
        "wait_for_selector": ".ready",
        "timeout": 5,
        "stream": True,
        "home_page": True,
        "return_cookie": True,
    })
    body = CaptureClient.calls[0][3]
    for field in (
        "bypassProxy",
        "html",
        "markdown",
        "renderJavascript",
        "returnCookies",
        "screenshot",
        "useHomePage",
        "waitForSelector",
        "timeout",
        "stream",
        "homePage",
        "returnCookie",
    ):
        assert field not in body


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
    assert "homePage" not in body
    assert body["homePageTimeout"] == 10
    assert body["timeout"] == 600


def test_manual_run_serializes_enabled_home_page_only() -> None:
    mt.run_existing_scraper({
        "scraper_type": "manual",
        "scraper_id": "m1",
        "url": "https://example.com",
        "home_page": True,
    })
    assert CaptureClient.calls[0][3]["homePage"] is True


def test_manual_run_ignores_all_ai_only_parameters() -> None:
    mt.run_existing_scraper({
        "scraper_type": "manual",
        "scraper_id": "m1",
        "url": "https://example.com",
        "agent_type": "listing",
        "render_javascript": True,
        "return_cookies": True,
        "use_home_page": True,
        "wait_for_selector": ".ready",
        "max_pages": 5,
        "max_depth": 3,
        "limit": 4,
        "include_patterns": "include",
        "exclude_patterns": "exclude",
    })
    body = CaptureClient.calls[0][3]
    for field in (
        "agentType",
        "renderJavascript",
        "returnCookies",
        "useHomePage",
        "waitForSelector",
        "maxPages",
        "maxDepth",
        "limit",
        "includePatterns",
        "excludePatterns",
    ):
        assert field not in body


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
    assert CaptureClient.calls[0] == (
        "POST",
        expected_path,
        None,
        {
            "scraperId": "s1",
            "urls": ["https://a.example", "https://b.example"],
        },
    )


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
