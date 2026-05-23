from __future__ import annotations

from unittest.mock import MagicMock, patch

from plugins.web.scrapecreators.provider import ScrapeCreatorsWebSearchProvider


def test_scrapecreators_availability_requires_key(monkeypatch):
    provider = ScrapeCreatorsWebSearchProvider()

    monkeypatch.delenv("SCRAPECREATORS_API_KEY", raising=False)
    assert provider.is_available() is False

    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "key")
    assert provider.is_available() is True


def test_scrapecreators_search_normalizes_google_results(monkeypatch):
    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "key")
    response = MagicMock()
    response.json.return_value = {
        "results": [
            {
                "title": "OpenAI",
                "url": "https://openai.com/",
                "description": "AI research",
                "position": 3,
            }
        ]
    }
    response.raise_for_status.return_value = None

    with patch("plugins.web.scrapecreators.provider.httpx.get", return_value=response) as get:
        result = ScrapeCreatorsWebSearchProvider().search("OpenAI", limit=1)

    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "OpenAI",
                    "url": "https://openai.com/",
                    "description": "AI research",
                    "position": 3,
                }
            ]
        },
    }
    assert get.call_args.kwargs["params"] == {"query": "OpenAI"}
    assert get.call_args.kwargs["headers"]["x-api-key"] == "key"


def test_scrapecreators_search_accepts_nested_data_shape(monkeypatch):
    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "key")
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "organic_results": [
                {
                    "title": "Docs",
                    "link": "https://example.com/docs",
                    "snippet": "Example documentation",
                    "rank": 2,
                }
            ]
        }
    }
    response.raise_for_status.return_value = None

    with patch("plugins.web.scrapecreators.provider.httpx.get", return_value=response):
        result = ScrapeCreatorsWebSearchProvider().search("docs", limit=5)

    assert result["success"] is True
    assert result["data"]["web"] == [
        {
            "title": "Docs",
            "url": "https://example.com/docs",
            "description": "Example documentation",
            "position": 2,
        }
    ]
