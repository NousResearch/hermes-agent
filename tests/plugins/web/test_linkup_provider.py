"""Linkup web provider: request shape and Hermes result normalization."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _linkup_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "lnk-test")
    monkeypatch.delenv("LINKUP_SEARCH_DEPTH", raising=False)
    monkeypatch.delenv("LINKUP_BASE_URL", raising=False)


def _response(payload: dict, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.text = ""
    return response


def test_search_posts_search_results_and_maps_hits() -> None:
    from plugins.web.linkup.provider import LinkupWebSearchProvider

    captured: dict = {}

    def _post(url, json, timeout, headers):
        captured.update(url=url, json=json, timeout=timeout, headers=headers)
        return _response({"results": [
            {"name": "Example", "url": "https://example.com", "content": "A snippet"},
        ]})

    with patch("plugins.web.linkup.provider.httpx.post", side_effect=_post):
        result = LinkupWebSearchProvider().search("who makes Hermes", limit=5)

    assert captured["url"] == "https://api.linkup.so/v1/search"
    assert captured["json"] == {
        "q": "who makes Hermes", "depth": "fast", "outputType": "searchResults", "maxResults": 5,
    }
    assert captured["headers"]["Authorization"] == "Bearer lnk-test"
    assert result == {
        "success": True,
        "data": {"web": [{
            "title": "Example", "url": "https://example.com", "description": "A snippet", "position": 1,
        }]},
    }


def test_search_honors_depth_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from plugins.web.linkup.provider import LinkupWebSearchProvider

    monkeypatch.setenv("LINKUP_SEARCH_DEPTH", "standard")
    captured: dict = {}

    def _post(url, json, timeout, headers):
        captured["depth"] = json["depth"]
        return _response({"results": []})

    with patch("plugins.web.linkup.provider.httpx.post", side_effect=_post):
        result = LinkupWebSearchProvider().search("q", limit=1)

    assert captured["depth"] == "standard"
    assert result["success"] is True
    assert result["data"]["web"] == []


def test_search_without_key_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from plugins.web.linkup.provider import LinkupWebSearchProvider

    monkeypatch.delenv("LINKUP_API_KEY", raising=False)
    with patch("plugins.web.linkup.provider.httpx.post") as post:
        result = LinkupWebSearchProvider().search("q")
    post.assert_not_called()
    assert result["success"] is False
    assert "LINKUP_API_KEY" in result["error"]


def test_extract_fetches_each_url_as_markdown() -> None:
    from plugins.web.linkup.provider import LinkupWebSearchProvider

    calls: list = []

    def _post_with_text(url, json, timeout, headers):
        calls.append(json["url"])
        if json["url"].endswith("/bad"):
            failed = _response({}, status=500)
            failed.text = "upstream down"
            return failed
        return _response({"markdown": "page body"})

    with patch("plugins.web.linkup.provider.httpx.post", side_effect=_post_with_text):
        docs = LinkupWebSearchProvider().extract(["https://example.com/ok", "https://example.com/bad"])

    assert calls == ["https://example.com/ok", "https://example.com/bad"]
    assert docs[0]["url"] == "https://example.com/ok"
    assert docs[0]["content"] == "page body"
    assert docs[1]["url"] == "https://example.com/bad"
    assert "upstream down" in docs[1]["error"]
