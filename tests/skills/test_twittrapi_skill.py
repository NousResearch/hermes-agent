"""Tests for optional-skills/social-media/twittrapi/scripts/twittrapi.py — request shape and cursor paging."""

import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "social-media" / "twittrapi" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import twittrapi


class _Resp(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.status = 200
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_request_sends_key_header_simple_format_and_encoded_query(monkeypatch, capsys):
    """The key travels only in X-API-KEY (never the URL), format=simple is always added, and
    X search operators are URL-encoded as one query value."""
    monkeypatch.setenv("TWITTRAPI_API_KEY", "test-only-key")
    seen = []

    def fake_urlopen(req, timeout):
        seen.append(req)
        return _Resp({"data": [{"id": "1"}], "nextCursor": "c1", "hasNextPage": True},
                     {"X-Remaining-Balance": "4.2"})

    with mock.patch.object(twittrapi.urllib.request, "urlopen", fake_urlopen):
        rc = twittrapi.main(["/search", "searchItem=from:XDevelopers lang:en", "category=Latest"])

    assert rc == 0 and len(seen) == 1
    req = seen[0]
    assert req.get_method() == "GET"
    assert req.full_url.startswith("https://api.twittrapi.com/search?")
    assert "format=simple" in req.full_url
    assert "searchItem=from%3AXDevelopers+lang%3Aen" in req.full_url
    assert "test-only-key" not in req.full_url
    assert req.get_header("X-api-key") == "test-only-key"
    assert not req.get_header("User-agent", "").startswith("Python-urllib")
    out, err = capsys.readouterr()
    assert json.loads(out)["nextCursor"] == "c1"
    assert "X-Remaining-Balance: 4.2" in err


def test_pages_follow_next_cursor_and_stop_on_repeat(monkeypatch, capsys):
    """--pages passes nextCursor as cursor, merges data, and stops early when the cursor repeats."""
    monkeypatch.setenv("TWITTRAPI_API_KEY", "test-only-key")
    pages = [
        {"data": [{"id": "1"}], "nextCursor": "c1", "hasNextPage": True},
        {"data": [{"id": "2"}], "nextCursor": "c1", "hasNextPage": True},
    ]
    urls = []

    def fake_urlopen(req, timeout):
        urls.append(req.full_url)
        return _Resp(pages[len(urls) - 1])

    with mock.patch.object(twittrapi.urllib.request, "urlopen", fake_urlopen):
        rc = twittrapi.main(["/user/-1/followers", "username=XDevelopers", "--pages", "5"])

    assert rc == 0 and len(urls) == 2
    assert "cursor" not in urls[0] and "cursor=c1" in urls[1]
    result = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in result["data"]] == ["1", "2"]
    assert result["hasNextPage"] is False


def test_http_error_returns_1_and_missing_key_returns_2(monkeypatch, capsys):
    monkeypatch.setenv("TWITTRAPI_API_KEY", "test-only-key")
    err = urllib.error.HTTPError("https://api.twittrapi.com/x", 402, "Payment Required", {},
                                 io.BytesIO(b'{"error": "no balance"}'))
    with mock.patch.object(twittrapi.urllib.request, "urlopen", side_effect=err):
        assert twittrapi.main(["/user/by/username/XDevelopers"]) == 1
    out, stderr = capsys.readouterr()
    assert "no balance" in out and "HTTP 402" in stderr

    monkeypatch.delenv("TWITTRAPI_API_KEY")
    with mock.patch.object(twittrapi.urllib.request, "urlopen") as urlopen:
        assert twittrapi.main(["/balance"]) == 2
    urlopen.assert_not_called()
    assert "TWITTRAPI_API_KEY" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
