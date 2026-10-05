"""Ordered credential failover contracts; HTTP fixtures are not live billing verification."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from plugins.web.firecrawl import provider as fc


@pytest.fixture
def configured(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("web:\n  backend: firecrawl\n  cache_enabled: false\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    monkeypatch.setenv("FIRECRAWL_API_KEYS", '["first-secret", "second-secret", "third-secret"]')
    monkeypatch.setattr(fc, "keyless_search", lambda *a: pytest.fail("keyless routing"))
    monkeypatch.setattr(fc, "keyless_extract", lambda *a: pytest.fail("keyless routing"))
    monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: False)
    return tmp_path


@pytest.mark.parametrize("failure", [402, 401, 403, 429, 500, "timeout", "page", "text402", "requests402"])
@pytest.mark.parametrize("operation", ["search", "extract"])
def test_ordered_failover_is_billing_only_and_sticky(configured, monkeypatch, caplog, failure, operation):
    calls = []

    class Client:
        def __init__(self, api_key, **kwargs):
            self.key = api_key

        def search(self, **kwargs):
            calls.append(self.key)
            if self.key == "first-secret":
                if failure == "timeout":
                    raise TimeoutError("first-secret timed out")
                if failure == "text402":
                    raise RuntimeError("page says 402 insufficient credits first-secret")
                if failure == "page":
                    return {"web": [{"url": "https://example.com", "description": "402 Payment Required"}]}
                if failure == "requests402":
                    import requests
                    response = requests.Response()
                    response.status_code = 402
                    raise requests.HTTPError("first-secret", response=response)
                response = httpx.Response(failure, request=httpx.Request("POST", "https://api.firecrawl.dev/v2/search"))
                raise httpx.HTTPStatusError("first-secret second-secret third-secret insufficient credits", request=response.request, response=response)
            return {"web": [{"url": "https://example.com", "title": "OK"}]}

        def scrape(self, **kwargs):
            self.search()
            return {"markdown": "content", "metadata": {"sourceURL": "https://example.com"}}

    monkeypatch.setattr(fc, "Firecrawl", Client)
    p = fc.FirecrawlWebSearchProvider()
    if operation == "search":
        result = p.search("test")
    else:
        page = asyncio.run(p.extract(["https://example.com"]))[0]
        result = {"success": not bool(page.get("error")), "page": page}
    assert result["success"] is (failure in (402, "requests402", "page"))
    assert calls == (["first-secret", "second-secret"] if failure in (402, "requests402") else ["first-secret"])
    if failure in (402, "requests402"):
        with ThreadPoolExecutor(max_workers=6) as executor:
            assert all(r["success"] for r in executor.map(lambda _: p.search("parallel"), range(12)))
        assert calls.count("first-secret") == 1
        assert asyncio.run(p.extract(["https://example.com"]))[0]["content"] == "content"
        assert set(calls[1:]) == {"second-secret"}
        monkeypatch.setenv("FIRECRAWL_API_KEYS", '["third-secret", "second-secret"]')
        assert p.search("new config")["success"]
        assert calls[-1] == "third-secret"
    for key in ("first-secret", "second-secret", "third-secret"):
        assert key not in json.dumps(result) + caplog.text


@pytest.mark.parametrize("raw", ['not-json-secret', '"scalar-secret"', '{"key":"object-secret"}', '["valid-secret", 42]', '["valid-secret", ""]'])
def test_invalid_plural_fails_closed_without_secret_echo(configured, monkeypatch, raw):
    monkeypatch.setenv("FIRECRAWL_API_KEYS", raw)
    with pytest.raises(ValueError, match="FIRECRAWL_API_KEYS") as exc:
        fc._get_firecrawl_client()
    assert raw not in str(exc.value)
    from tools.web_tools_rescue import _rescue_eligible
    assert not _rescue_eligible(fc.FirecrawlWebSearchProvider())
