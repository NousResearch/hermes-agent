"""Tests for news_verify._web_search layering.

- Tavily primary path parses results (title/snippet/url).
- Falls through to the DuckDuckGo scrape when Tavily fails.
- Returns [] (never fabricates) when everything fails.
"""
import requests

import blog.news_verify as nv


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_tavily_primary(monkeypatch):
    monkeypatch.setattr(nv, "_tavily_key", lambda: "test-key")

    def fake_post(self, url, **kw):
        assert "tavily" in url
        return _Resp(200, {"results": [
            {"title": "Idempotency-Key", "content": "Stripe header docs",
             "url": "https://stripe.com/docs/api/idempotent_requests"},
            {"title": "Backoff", "content": "AWS retry guidance",
             "url": "https://aws.amazon.com/blogs/architecture/"},
        ]})

    monkeypatch.setattr(requests.Session, "post", fake_post)
    hits = nv._web_search("Stripe Idempotency-Key")
    assert [h["title"] for h in hits] == ["Idempotency-Key", "Backoff"]
    assert hits[0]["url"].startswith("https://stripe.com")


def test_falls_back_to_ddg(monkeypatch):
    monkeypatch.setattr(nv, "_tavily_key", lambda: "test-key")

    def fail_tavily(self, url, **kw):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests.Session, "post", fail_tavily)

    ddg_html = ('<tr class="result"><td><a class="result-link" '
                'href="https://example.com/x">Example</a></td>'
                '<td class="result-snippet">Some snippet</td></tr>')
    monkeypatch.setattr(requests, "post",
                        lambda url, **kw: _Resp(200, text=ddg_html))
    hits = nv._web_search("anything")
    assert hits and hits[0]["url"] == "https://example.com/x"


def test_all_failures_return_empty(monkeypatch):
    monkeypatch.setattr(nv, "_tavily_key", lambda: "")

    def fail(self, *a, **kw):
        raise requests.RequestException("down")

    monkeypatch.setattr(requests, "post", fail)
    assert nv._web_search("anything") == []
