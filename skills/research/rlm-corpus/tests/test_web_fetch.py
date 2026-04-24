"""Unit tests for the web-fetch backends.

All HTTP is mocked via ``unittest.mock.patch`` — no network is hit.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from web_fetch import (
    CloudflareFetcher,
    FetchedPage,
    JinaFetcher,
    make_fetcher,
)


# ---------------------------------------------------------------------------
# Cloudflare /markdown
# ---------------------------------------------------------------------------


def _mock_response(json_body, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = str(json_body)
    return resp


def test_cloudflare_requires_credentials(monkeypatch):
    for var in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "RLM_CF_ACCOUNT_ID",
        "RLM_CF_API_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareFetcher()


def test_cloudflare_fetch_markdown_unwraps_string_result():
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok")
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            {"success": True, "result": "# Hello\n\nbody text"}
        )
        page = fetcher.fetch_markdown("https://example.com")

    assert isinstance(page, FetchedPage)
    assert page.url == "https://example.com"
    assert "# Hello" in page.markdown

    args, kwargs = mock_post.call_args
    assert args[0].endswith("/accounts/acct/browser-rendering/markdown")
    assert kwargs["headers"]["Authorization"] == "Bearer tok"
    assert kwargs["json"] == {"url": "https://example.com"}


def test_cloudflare_fetch_markdown_unwraps_dict_result():
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok")
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            {"success": True, "result": {"markdown": "hello", "title": "My Page"}}
        )
        page = fetcher.fetch_markdown("https://example.com")
    assert page.markdown == "hello"
    assert page.title == "My Page"


def test_cloudflare_surfaces_api_errors():
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok")
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            {"success": False, "errors": [{"code": 7003, "message": "nope"}]}
        )
        with pytest.raises(RuntimeError, match="(?i)cloudflare api error"):
            fetcher.fetch_markdown("https://example.com")


# ---------------------------------------------------------------------------
# Cloudflare /crawl
# ---------------------------------------------------------------------------


def test_cloudflare_crawl_submits_polls_and_collects():
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok", poll_interval=0)

    submit = _mock_response({"success": True, "result": {"id": "job-123"}})
    running = _mock_response({
        "success": True,
        "result": {"id": "job-123", "status": "running", "records": [
            {"url": "https://site/a", "title": "A", "markdown": "# A"},
        ]},
    })
    completed = _mock_response({
        "success": True,
        "result": {"id": "job-123", "status": "completed", "records": [
            {"url": "https://site/b", "title": "B", "markdown": "# B"},
        ]},
    })

    with patch("requests.post", return_value=submit) as mp, \
         patch("requests.get", side_effect=[running, completed]) as mg:
        pages = fetcher.crawl(
            "https://site/",
            max_depth=1,
            limit=10,
            include_patterns=["/docs/*"],
            exclude_patterns=["*.pdf"],
        )

    assert [p.url for p in pages] == ["https://site/a", "https://site/b"]
    assert pages[0].markdown == "# A"

    _, submit_kwargs = mp.call_args
    body = submit_kwargs["json"]
    assert body["url"] == "https://site/"
    assert body["maxDepth"] == 1
    assert body["limit"] == 10
    assert body["formats"] == ["markdown"]
    assert body["render"] is True
    assert body["includePatterns"] == ["/docs/*"]
    assert body["excludePatterns"] == ["*.pdf"]
    assert mg.call_count == 2


def test_cloudflare_crawl_handles_cursor_pagination():
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok", poll_interval=0)
    submit = _mock_response({"success": True, "result": {"id": "job-x"}})

    page_1 = _mock_response({
        "success": True,
        "result": {
            "status": "running",
            "records": [{"url": "u1", "markdown": "m1"}],
            "cursor": "CURSOR-A",
        },
    })
    page_2 = _mock_response({
        "success": True,
        "result": {
            "status": "running",
            "records": [{"url": "u2", "markdown": "m2"}],
            "cursor": "CURSOR-B",
        },
    })
    page_3 = _mock_response({
        "success": True,
        "result": {
            "status": "completed",
            "records": [{"url": "u3", "markdown": "m3"}],
        },
    })

    with patch("requests.post", return_value=submit), \
         patch("requests.get", side_effect=[page_1, page_2, page_3]) as mg:
        pages = fetcher.crawl("https://site/")

    assert [p.url for p in pages] == ["u1", "u2", "u3"]
    # Verify the cursor is threaded through subsequent polls.
    cursors = [c.kwargs.get("params", {}).get("cursor") for c in mg.call_args_list]
    assert cursors == [None, "CURSOR-A", "CURSOR-B"]


def test_cloudflare_crawl_respects_deadline(monkeypatch):
    fetcher = CloudflareFetcher(account_id="acct", api_token="tok", poll_interval=0)
    submit = _mock_response({"success": True, "result": {"id": "job-t"}})
    stuck = _mock_response({
        "success": True,
        "result": {"status": "running", "records": []},
    })
    # Never-terminating stream
    with patch("requests.post", return_value=submit), \
         patch("requests.get", return_value=stuck), \
         pytest.raises(TimeoutError):
        fetcher.crawl("https://site/", timeout=0)


# ---------------------------------------------------------------------------
# Jina
# ---------------------------------------------------------------------------


def test_jina_fetch_returns_body_text():
    fetcher = JinaFetcher()
    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.text = "# Jina markdown"
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        page = fetcher.fetch_markdown("https://example.com/article")

    assert page.markdown == "# Jina markdown"
    url_called = mock_get.call_args.args[0]
    assert url_called.startswith("https://r.jina.ai/")


def test_jina_does_not_support_crawl():
    fetcher = JinaFetcher()
    with pytest.raises(NotImplementedError):
        fetcher.crawl("https://example.com/")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_fetcher_selects_jina_without_credentials(monkeypatch):
    monkeypatch.setenv("RLM_WEB_FETCHER", "jina")
    fetcher = make_fetcher()
    assert fetcher.name == "jina"


def test_make_fetcher_selects_cloudflare_with_credentials(monkeypatch):
    monkeypatch.setenv("RLM_WEB_FETCHER", "cloudflare")
    monkeypatch.setenv("RLM_CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("RLM_CF_API_TOKEN", "tok")
    fetcher = make_fetcher()
    assert fetcher.name == "cloudflare"


def test_make_fetcher_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_fetcher("unknown-backend")
