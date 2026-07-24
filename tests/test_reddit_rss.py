from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "optional-skills" / "research" / "reddit-access" / "scripts" / "reddit_rss.py"
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


def test_feed_url_for_subreddit_and_query():
    assert reddit_rss.feed_url("r/phones", None) == "https://www.reddit.com/r/phones/.rss"
    assert reddit_rss.feed_url(None, "8 inch phone") == "https://www.reddit.com/search.rss?q=8+inch+phone"


def test_feed_url_requires_exactly_one_source():
    for subreddit, query in [(None, None), ("phones", "phone")]:
        try:
            reddit_rss.feed_url(subreddit, query)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_parse_atom_normalizes_read_only_record():
    result = reddit_rss.parse_feed(ATOM, "https://www.reddit.com/r/phones/.rss")
    assert result == [
        {
            "title": "Eight inch phone",
            "url": "https://www.reddit.com/r/phones/comments/abc/example/",
            "author": "curi",
            "published": "2026-07-11T00:00:00Z",
            "text": "A useful device with space.",
            "subreddit": "r/phones",
            "source": "https://www.reddit.com/r/phones/.rss",
        }
    ]
