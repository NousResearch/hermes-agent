"""Tests for optional-skills/research/rss-feeds/scripts/feed.py — parsing and discovery contracts."""

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "research" / "rss-feeds" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import feed  # noqa: E402

RSS = b"""<?xml version="1.0"?><rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>Blog &amp; Notes</title>
<item><title>Older</title><link>https://ex.com/a</link><pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate>
  <dc:creator>Ann</dc:creator><description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description></item>
<item><title>Newer</title><link>https://ex.com/b</link><pubDate>Thu, 04 Sep 2026 08:30:00 +0200</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Atom Site</title>
<entry><title>Entry</title><link rel="self" href="https://ex.com/self"/><link rel="alternate" href="https://ex.com/post"/>
<updated>2026-09-03T12:00:00Z</updated><author><name>Bob</name></author><content type="html">&lt;p&gt;Body&lt;/p&gt;</content></entry>
</feed>"""

JSONFEED = b'{"version":"https://jsonfeed.org/version/1.1","title":"JF","items":[{"id":"1","url":"https://ex.com/j","title":"J1","date_published":"2026-09-02T00:00:00Z","authors":[{"name":"Cy"}],"content_text":"txt"}]}'

PAGE = b"""<html><head><title>x</title>
<link rel="alternate" type="application/atom+xml" href="/atom/everything/">
<link rel="stylesheet" href="/s.css"></head><body></body></html>"""


def test_all_three_formats_normalise_to_the_same_entry_shape_and_utc_dates():
    """RSS (RFC 822), Atom (ISO), JSON Feed (ISO) parse to identical keys with UTC-normalised dates,
    and Atom picks the rel=alternate link over rel=self."""
    rss, atom, jf = feed.parse_feed(RSS), feed.parse_feed(ATOM), feed.parse_feed(JSONFEED, "application/feed+json")
    keys = {"title", "link", "published", "author", "summary"}
    for f in (rss, atom, jf):
        assert f["entries"] and all(set(e) == keys for e in f["entries"])
    assert rss["title"] == "Blog & Notes"
    assert rss["entries"][0]["summary"] == "Hello world"  # HTML stripped, entities decoded
    assert rss["entries"][1]["published"] == "2026-09-04T06:30:00+00:00"  # +0200 → UTC
    assert atom["entries"][0]["link"] == "https://ex.com/post"
    assert jf["entries"][0]["author"] == "Cy"
    newest = feed.filter_entries(rss["entries"], limit=1, since="2026-09-02")
    assert [e["title"] for e in newest] == ["Newer"]


def test_page_url_discovers_advertised_feed_then_reads_it():
    """A non-feed page falls through to <link rel=alternate> discovery (resolved against the page URL)
    and `read` returns the parsed feed tagged with where it was discovered from."""
    responses = {
        "https://ex.com/blog/": (PAGE, "text/html"),
        "https://ex.com/atom/everything/": (ATOM, "application/atom+xml"),
    }
    with mock.patch.object(feed, "fetch", side_effect=lambda u: responses[u]):
        assert feed.discover("https://ex.com/blog/") == ["https://ex.com/atom/everything/"]
        result = feed.read("https://ex.com/blog/")
    assert result["url"] == "https://ex.com/atom/everything/"
    assert result["discovered_from"] == "https://ex.com/blog/"
    assert result["entries"][0]["title"] == "Entry"
    # no advertised feed → well-known paths are proposed, never an empty list
    with mock.patch.object(feed, "fetch", return_value=(b"<html><body>plain</body></html>", "text/html")):
        candidates = feed.discover("https://plain.example/")
    assert candidates and all(c.startswith("https://plain.example/") for c in candidates)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
