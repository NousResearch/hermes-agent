import json

import httpx
import pytest

from tools.web_tools import (
    check_web_crawl_api_key,
    check_web_extract_api_key,
    check_web_search_api_key,
    web_crawl_tool,
    web_extract_tool,
    web_search_tool,
)

pytestmark = pytest.mark.integration

_SEARXNG_BASE_URL = "http://localhost:8888"
_SEARCH_ONLY_ENV_KEYS = (
    "EXA_API_KEY",
    "PARALLEL_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "TAVILY_API_KEY",
)


def _searxng_available() -> bool:
    try:
        response = httpx.get(f"{_SEARXNG_BASE_URL}/config", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


@pytest.fixture
def searxng_search_only_home(tmp_path, monkeypatch):
    for key in _SEARCH_ONLY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "web:\n"
        "  backend: searxng\n"
        "  search_backend: searxng\n"
        "  searxng:\n"
        f"    base_url: {_SEARXNG_BASE_URL}\n"
        "    timeout: 20\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.skipif(not _searxng_available(), reason="Local SearXNG is not available on http://localhost:8888")
def test_live_searxng_search_backend_supports_search_only(searxng_search_only_home):
    assert check_web_search_api_key() is True
    assert check_web_extract_api_key() is False
    assert check_web_crawl_api_key() is False

    result = json.loads(web_search_tool("OpenAI news", limit=3))

    assert result["success"] is True
    assert len(result["data"]["web"]) >= 1
    assert result["data"]["web"][0]["url"]


@pytest.mark.asyncio
@pytest.mark.skipif(not _searxng_available(), reason="Local SearXNG is not available on http://localhost:8888")
async def test_live_searxng_search_only_config_returns_extract_error(searxng_search_only_home):
    result = json.loads(await web_extract_tool(["https://example.com"], use_llm_processing=False))

    assert result["success"] is False
    assert "SearXNG supports search only" in result["error"]


@pytest.mark.asyncio
@pytest.mark.skipif(not _searxng_available(), reason="Local SearXNG is not available on http://localhost:8888")
async def test_live_searxng_search_only_config_returns_crawl_error(searxng_search_only_home):
    result = json.loads(await web_crawl_tool("https://example.com", use_llm_processing=False))

    assert result["success"] is False
    assert "SearXNG supports search only" in result["error"]
