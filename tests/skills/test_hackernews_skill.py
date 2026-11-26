from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research"
    / "hackernews"
    / "scripts"
    / "hn.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("hackernews_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_strip_html_decodes_entities_and_paragraphs():
    mod = load_module()

    assert mod._strip_html("Hello &amp; <b>HN</b><p>Next<br>line") == "Hello & HN\n\nNext\nline"


def test_normalize_item_adds_hn_url_and_plain_text():
    mod = load_module()

    row = mod._normalize_item(
        {
            "id": 8863,
            "type": "story",
            "by": "pg",
            "time": 1175714200,
            "title": "My YC app: Dropbox",
            "url": "https://www.getdropbox.com/u/2/screencast.html",
            "score": 111,
            "descendants": 71,
            "text": "<p>Hello &amp; welcome</p>",
        }
    )

    assert row["id"] == 8863
    assert row["hn_url"] == "https://news.ycombinator.com/item?id=8863"
    assert row["time_iso"] == "2007-04-04T19:16:40Z"
    assert row["text"] == "Hello & welcome"


def test_list_items_fetches_ids_and_items():
    mod = load_module()

    def fake_firebase(path):
        if path == "topstories":
            return [1, 2, 3]
        if path == "item/1":
            return {"id": 1, "type": "story", "title": "One", "by": "alice"}
        if path == "item/2":
            return {"id": 2, "type": "story", "title": "Two", "by": "bob", "dead": True}
        if path == "item/3":
            return {"id": 3, "type": "story", "title": "Three", "by": "carol"}
        raise AssertionError(path)

    with patch.object(mod, "_firebase", side_effect=fake_firebase):
        rows = mod.list_items("top", 2)

    assert len(rows) == 2
    assert [row["title"] for row in rows] == ["One", "Three"]
    assert all(row["id"] != 2 for row in rows)


def test_get_item_with_comments_fetches_top_level_comments():
    mod = load_module()

    def fake_firebase(path):
        payloads = {
            "item/10": {"id": 10, "type": "story", "title": "Story", "kids": [11, 12, 13]},
            "item/11": {"id": 11, "type": "comment", "by": "a", "text": "<p>first"},
            "item/12": {"id": 12, "type": "comment", "deleted": True},
            "item/13": {"id": 13, "type": "comment", "by": "c", "text": "third"},
        }
        return payloads[path]

    with patch.object(mod, "_firebase", side_effect=fake_firebase):
        item = mod.get_item(10, include_comments=True, comment_limit=3)

    assert item["kids"] == [11, 12, 13]
    assert item["comments_fetched"] == 2
    assert [comment["id"] for comment in item["comments_detail"]] == [11, 13]


def test_get_user_normalizes_public_profile():
    mod = load_module()

    with patch.object(
        mod,
        "_firebase",
        return_value={
            "id": "pg",
            "created": 1160418111,
            "karma": 155040,
            "about": "<b>Lisp</b>",
            "submitted": [1, 2, 3],
        },
    ):
        user = mod.get_user("pg")

    assert user["id"] == "pg"
    assert user["karma"] == 155040
    assert user["about"] == "Lisp"
    assert user["submitted_count"] == 3
    assert user["profile_url"] == "https://news.ycombinator.com/user?id=pg"


def test_search_uses_algolia_and_normalizes_hits():
    mod = load_module()

    payload = {
        "hits": [
            {
                "objectID": "123",
                "title": "Ask HN: Example",
                "author": "alice",
                "points": 42,
                "num_comments": 7,
                "created_at_i": 1700000000,
                "_tags": ["story", "ask_hn"],
            }
        ]
    }

    with patch.object(mod, "_fetch_json", return_value=payload) as fetch_json:
        rows = mod.search("example", 5, by_date=True, tags="ask_hn")

    called_url = fetch_json.call_args[0][0]
    assert "/search_by_date?" in called_url
    assert "query=example" in called_url
    assert "tags=ask_hn" in called_url
    assert rows[0]["id"] == 123
    assert rows[0]["hn_url"] == "https://news.ycombinator.com/item?id=123"
    assert rows[0]["comments"] == 7


def test_main_prints_json_for_top_command(capsys):
    mod = load_module()

    with patch.object(mod, "list_items", return_value=[{"id": 1, "title": "One"}]):
        exit_code = mod.main(["top", "-n", "1"])

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert json.loads(stdout) == [{"id": 1, "title": "One"}]


def test_main_prints_json_error_to_stderr(capsys):
    mod = load_module()

    with patch.object(mod, "get_user", side_effect=RuntimeError("HN user 'missing' was not found")):
        exit_code = mod.main(["user", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.err) == {"error": "HN user 'missing' was not found"}
