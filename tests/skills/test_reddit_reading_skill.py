"""Tests for optional-skills/social-media/reddit-reading/scripts/reddit.py — backend selection and throttle handling."""

import io
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "social-media" / "reddit-reading" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import reddit  # noqa: E402

THREAD_ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><author><name>/u/op</name></author><title>Post title</title>
  <link href="https://www.reddit.com/r/test/comments/abc123/post_title/"/><updated>2026-09-01T00:00:00+00:00</updated>
  <content type="html">&lt;div&gt;body text&lt;/div&gt; submitted by /u/op [link] [comments]</content></entry>
<entry><author><name>/u/c1</name></author><title>/u/c1 on Post title</title>
  <link href="https://www.reddit.com/r/test/comments/abc123/post_title/k1/"/><updated>2026-09-01T01:00:00+00:00</updated>
  <content type="html">&lt;p&gt;first comment&lt;/p&gt;</content></entry>
</feed>"""


def _http_error(code, headers):
    return urllib.error.HTTPError("https://www.reddit.com/x", code, "msg", headers, io.BytesIO(b""))


def test_anonymous_thread_uses_atom_feed_and_waits_out_a_429_exactly_once(monkeypatch):
    """Without OAuth credentials the .rss endpoint is used; a 429 sleeps for x-ratelimit-reset and retries once,
    and the parsed thread separates the post from its comments with feed noise stripped."""
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    assert reddit.oauth_credentials() is None

    calls = []
    sleeps = []

    class Resp(io.BytesIO):
        headers = {"x-ratelimit-remaining": "0.0"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise _http_error(429, {"x-ratelimit-reset": "7"})
        return Resp(THREAD_ATOM)

    with mock.patch.object(reddit.urllib.request, "urlopen", fake_urlopen), \
         mock.patch.object(reddit.time, "sleep", sleeps.append):
        post = reddit.atom_thread("test", "abc123", limit=10)

    assert calls[0].startswith("https://www.reddit.com/r/test/comments/abc123/.rss") and len(calls) == 2
    assert sleeps == [8]  # reset + 1s margin, one retry only
    assert post["title"] == "Post title" and post["author"] == "op"
    assert post["body"] == "body text"  # "submitted by … [link] [comments]" footer stripped
    assert [c["author"] for c in post["comments"]] == ["c1"]

    # a second 429 after the retry propagates instead of looping
    with mock.patch.object(reddit.urllib.request, "urlopen", side_effect=_http_error(429, {})), \
         mock.patch.object(reddit.time, "sleep", lambda s: None), pytest.raises(urllib.error.HTTPError):
        reddit._get("https://www.reddit.com/r/test/.rss")


def test_oauth_credentials_route_to_oauth_host_and_flatten_nested_comments(monkeypatch):
    """With REDDIT_CLIENT_ID/SECRET the script talks to oauth.reddit.com with a bearer token and
    returns nested comments flattened with depth and scores — data the anonymous path cannot provide."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "sec")
    assert reddit.oauth_credentials() == ("cid", "sec")

    listing = [
        {"data": {"children": [{"kind": "t3", "data": {"title": "T", "author": "op", "subreddit": "test", "score": 42,
                                                      "num_comments": 2, "created_utc": 1.0, "permalink": "/r/test/comments/abc123/t/",
                                                      "is_self": True, "url": "https://www.reddit.com/r/test/comments/abc123/t/", "selftext": "s"}}]}},
        {"data": {"children": [{"kind": "t1", "data": {"author": "a", "score": 5, "body": "top", "permalink": "/p/1",
                                                      "replies": {"data": {"children": [{"kind": "t1", "data": {"author": "b", "score": 1, "body": "reply", "replies": ""}}]}}}},
                               {"kind": "more", "data": {}}]}},
    ]
    seen = {}

    def fake_api(path, token, **params):
        seen["path"], seen["token"] = path, token
        return listing

    with mock.patch.object(reddit, "_api", fake_api):
        post = reddit.api_thread("tok", "test", "abc123", limit=10)

    assert seen == {"path": "/r/test/comments/abc123", "token": "tok"}
    assert post["score"] == 42 and post["url"] == "https://www.reddit.com/r/test/comments/abc123/t/"
    assert [(c["author"], c["depth"], c["score"]) for c in post["comments"]] == [("a", 0, 5), ("b", 1, 1)]

    # the bearer header actually reaches the OAuth host
    captured = {}

    def fake_get(url, headers=None, retry_on_429=True):
        captured["url"], captured["headers"] = url, headers
        return b'{"data": {"children": []}}', {}

    with mock.patch.object(reddit, "_get", fake_get):
        reddit._api("/r/test/hot", "tok", limit=1)
    assert captured["url"].startswith("https://oauth.reddit.com/r/test/hot?")
    assert captured["headers"]["Authorization"] == "Bearer tok"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
