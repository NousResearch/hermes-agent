"""Tests for the artlist skill connector (auth + search payload construction).

Behaviour tests against the shipped script with mocked `requests` — no live
network, no browser. The download path (Playwright) is exercised manually.
"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "artlist"
    / "scripts"
    / "artlist.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("artlist_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_song_url_format():
    mod = _load()
    assert (
        mod.song_url("a-beam-of-light", "6001611")
        == "https://artlist.io/royalty-free-music/song/a-beam-of-light/6001611"
    )


def test_get_access_token_exchanges_session_cookie(tmp_path):
    mod = _load()
    (tmp_path / "state.json").write_text(
        json.dumps({"cookies": [{"name": "session-token", "value": "s3cr3t", "domain": ".artlist.io"}]}),
        encoding="utf-8",
    )
    with patch.object(mod.requests, "get") as get:
        get.return_value = Mock(raise_for_status=lambda: None, json=lambda: {"accessToken": "tok"})
        assert mod.get_access_token(str(tmp_path)) == "tok"
        assert get.call_args.args[0] == "https://artlist.io/api/auth/session"
        assert "session-token=s3cr3t" in get.call_args.kwargs["headers"]["Cookie"]


def test_search_builds_payload_and_uses_bearer(tmp_path):
    mod = _load()
    (tmp_path / "state.json").write_text(
        json.dumps({"cookies": [{"name": "session-token", "value": "s3cr3t", "domain": ".artlist.io"}]}),
        encoding="utf-8",
    )
    with patch.object(mod.requests, "get") as get, patch.object(mod.requests, "post") as post:
        get.return_value = Mock(raise_for_status=lambda: None, json=lambda: {"accessToken": "tok"})
        post.return_value = Mock(
            raise_for_status=lambda: None, json=lambda: {"data": {"songList": {"songs": []}}}
        )
        mod.search(str(tmp_path), term="elegant", category_ids=[62])
        assert post.call_args.args[0] == "https://search-api.artlist.io/v2/graphql"
        body = post.call_args.kwargs["json"]
        assert body["variables"]["searchTerm"] == "elegant"
        assert body["variables"]["categoryIds"] == [62]
        assert body["variables"]["songSortType"] == "NEWEST"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"
