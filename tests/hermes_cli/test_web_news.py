"""Tests for the dashboard Command Center news feed (``GET /api/news``)."""
import pytest

from hermes_cli.web_routers import news


RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Older &amp; wiser</title><link>https://example.com/a</link>
<description>&lt;p&gt;Hello <![CDATA[<b>world</b>]]>&lt;/p&gt;</description>
<pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>No link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>a</title>
<entry><title>Newest</title><link rel="alternate" href="https://example.org/n"/>
<updated>2026-09-20T12:00:00Z</updated><summary>Fresh</summary></entry>
</feed>"""


def test_parse_rss_strips_html_and_drops_unsafe_links():
    items = news.parse_feed(RSS, "Src")
    assert [i["title"] for i in items] == ["Older & wiser"]
    assert items[0]["summary"] == "Hello world"
    assert items[0]["source"] == "Src"
    assert items[0]["published"] is not None


def test_parse_atom():
    items = news.parse_feed(ATOM, "A")
    assert items[0]["link"] == "https://example.org/n"
    assert items[0]["summary"] == "Fresh"


def test_parse_rejects_dtd():
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel/></rss>'
    with pytest.raises(ValueError):
        news.parse_feed(bomb, "x")


def test_resolve_feeds_defaults_and_custom():
    assert news.resolve_news_feeds({}) == news.DEFAULT_NEWS_FEEDS
    feeds = news.resolve_news_feeds({"dashboard": {"news_feeds": [
        "https://www.example.com/rss",
        {"name": "Mine", "url": "https://x.dev/feed"},
        "file:///etc/passwd",
        42,
    ]}})
    assert feeds == [
        {"name": "example.com", "url": "https://www.example.com/rss"},
        {"name": "Mine", "url": "https://x.dev/feed"},
    ]


@pytest.fixture
def client(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_endpoint_merges_sorts_and_caches(client, monkeypatch):
    monkeypatch.setattr(news, "_cache", {})
    monkeypatch.setattr(news, "load_config", lambda: {"dashboard": {"news_feeds": [
        {"name": "R", "url": "https://r.test/rss"},
        {"name": "A", "url": "https://a.test/atom"},
        {"name": "Down", "url": "https://down.test/rss"},
    ]}})
    calls = []

    async def fake_fetch(_client, feed):
        calls.append(feed["url"])
        if "down" in feed["url"]:
            return [], {**feed, "ok": False, "count": 0, "error": "boom"}
        payload = RSS if feed["url"].endswith("rss") else ATOM
        items = news.parse_feed(payload, feed["name"])
        return items, {**feed, "ok": True, "count": len(items), "error": None}

    monkeypatch.setattr(news, "_fetch_feed", fake_fetch)

    resp = client.get("/api/news")
    assert resp.status_code == 200
    data = resp.json()
    assert [i["title"] for i in data["items"]] == ["Newest", "Older & wiser"]
    assert [f["ok"] for f in data["feeds"]] == [True, True, False]
    assert len(calls) == 3

    client.get("/api/news", params={"limit": 1})
    assert len(calls) == 3, "second call should be served from cache"
    client.get("/api/news", params={"refresh": "true"})
    assert len(calls) == 6


def test_endpoint_requires_token():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app

    assert TestClient(app).get("/api/news").status_code == 401


def _items(source, n, start=0):
    return [{"source": source, "link": f"https://{source}/{i}", "title": str(i), "published": 1000 - start - i}
            for i in range(n)]


def test_a_chatty_source_cannot_crowd_out_the_others_but_a_quiet_day_still_fills_the_page():
    loud, quiet = _items("loud", 30), _items("quiet", 5, start=100)
    merged = sorted(loud + quiet, key=lambda i: -i["published"])

    page = news.cap_per_source(merged, 20)

    assert sum(1 for i in page if i["source"] == "quiet") == 5
    assert sum(1 for i in page if i["source"] == "loud") == 15 and len(page) == 20
    # one source alone still returns a full page
    assert len(news.cap_per_source(loud, 20)) == 20
    # and the page stays newest-first
    assert [i["published"] for i in page] == sorted((i["published"] for i in page), reverse=True)


@pytest.mark.asyncio
async def test_same_host_feeds_never_overlap_while_other_hosts_run_in_parallel(monkeypatch):
    import asyncio
    monkeypatch.setattr(news, "_SAME_HOST_GAP_S", 0.05)
    running, overlap, started = {}, [], []

    async def fake_fetch(client, feed):
        host = feed["url"].split("/")[2]
        running[host] = running.get(host, 0) + 1
        if running[host] > 1:
            overlap.append(host)
        started.append(host)
        await asyncio.sleep(0.05)
        running[host] -= 1
        return [], {"name": feed["name"], "url": feed["url"], "ok": True, "count": 0, "error": None}

    monkeypatch.setattr(news, "_fetch_feed", fake_fetch)
    feeds = [{"name": n, "url": u} for n, u in (("a", "https://r.com/a"), ("b", "https://r.com/b"),
                                                 ("c", "https://other.com/c"))]

    data = await news.collect_news(feeds)

    assert overlap == [] and started[:2] == ["r.com", "other.com"]
    assert [f["name"] for f in data["feeds"]] == ["a", "b", "c"]
