from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "reddit-access"
    / "scripts"
    / "reddit_rss.py"
)
spec = importlib.util.spec_from_file_location("reddit_rss", SCRIPT)
assert spec and spec.loader
reddit_rss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reddit_rss)


ATOM = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Eight inch phone</title>
    <link href="https://www.reddit.com/r/phones/comments/abc/example/" />
    <author><name>/u/curi</name></author>
    <published>2026-07-11T00:00:00Z</published>
    <content type="html">A &lt;b&gt;useful&lt;/b&gt; device&lt;br/&gt;with space.</content>
  </entry>
</feed>'''

EXPECTED_RECORD = {
    "title": "Eight inch phone",
    "url": "https://www.reddit.com/r/phones/comments/abc/example/",
    "author": "curi",
    "published": "2026-07-11T00:00:00Z",
    "text": "A useful device with space.",
    "subreddit": "r/phones",
    "source": "https://www.reddit.com/r/phones/.rss",
}


def test_feed_url_for_subreddit_and_query():
    assert reddit_rss.feed_url("r/phones", None) == "https://www.reddit.com/r/phones/.rss"
    assert reddit_rss.feed_url(None, "8 inch phone") == "https://www.reddit.com/search.rss?q=8+inch+phone"


def test_feed_url_requires_exactly_one_source():
    for subreddit, query in [(None, None), ("phones", "phone")]:
        with pytest.raises(ValueError):
            reddit_rss.feed_url(subreddit, query)


def test_parse_atom_normalizes_read_only_record():
    result = reddit_rss.parse_feed(ATOM, "https://www.reddit.com/r/phones/.rss")
    assert result == [EXPECTED_RECORD]


def test_fetch_sends_request_and_parses_response():
    response = MagicMock()
    response.status = 200
    response.read.return_value = ATOM
    response.__enter__.return_value = response

    with patch.object(reddit_rss.urllib.request, "urlopen", return_value=response) as urlopen:
        result = reddit_rss.fetch(
            "https://www.reddit.com/r/phones/.rss",
            timeout=7.5,
            user_agent="test-agent",
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://www.reddit.com/r/phones/.rss"
    assert request.get_header("User-agent") == "test-agent"
    assert request.get_header("Accept") == "application/atom+xml"
    assert urlopen.call_args.kwargs == {"timeout": 7.5}
    assert result == [EXPECTED_RECORD]


def test_fetch_rejects_non_200_response():
    response = MagicMock()
    response.status = 503
    response.__enter__.return_value = response

    with patch.object(reddit_rss.urllib.request, "urlopen", return_value=response):
        with pytest.raises(RuntimeError, match=r"Reddit RSS returned HTTP 503"):
            reddit_rss.fetch(
                "https://www.reddit.com/r/phones/.rss",
                timeout=15.0,
                user_agent="test-agent",
            )
