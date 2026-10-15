"""Tests for optional-skills/research/olostep: skill contract and CLI helper (no network)."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "research" / "olostep"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import olostep_cli

SKILL_TEXT = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def test_author_credits_the_human_first():
    author = re.search(r"^author: (.*)$", SKILL_TEXT, re.MULTILINE).group(1)
    assert author.startswith("Umer K. (umerkay)")
    assert "ZeeshanAdilButt" in author


def test_sections_follow_the_modern_order():
    assert SKILL_TEXT.count("# Olostep Skill") == 1
    positions = [SKILL_TEXT.index(s) for s in SECTIONS]
    assert positions == sorted(positions)


def test_every_scrape_call_uses_url_to_scrape():
    calls = re.findall(r"scrapes\.create\(([^)]*)\)", SKILL_TEXT)
    assert calls
    for args in calls:
        assert "url_to_scrape=" in args
        assert not re.search(r"(?<![_\w])url=", args)


def test_answers_are_read_from_json_content_and_sources():
    assert "answer.json_content" in SKILL_TEXT
    assert "answer.sources" in SKILL_TEXT
    assert "print(result.answer)" not in SKILL_TEXT


def test_script_is_referenced_by_skill_relative_path():
    assert "SKILL_DIR/scripts/olostep_cli.py" in SKILL_TEXT
    assert (SKILL_DIR / "scripts" / "olostep_cli.py").is_file()


class FakeClient:
    def __init__(self):
        self.calls = {}
        outer = self

        class Searches:
            def create(self, query, **kw):
                outer.calls["search"] = (query, kw)
                links = [
                    {"url": "https://a.test", "title": "A", "description": "da", "markdown_content": "x"},
                    {"url": "https://b.test", "title": None, "description": None},
                ]
                return SimpleNamespace(links=links)

        class Scrapes:
            def create(self, **kw):
                outer.calls["scrape"] = kw
                # the real ScrapeResult only sets attributes that came back
                return SimpleNamespace(markdown_content="# Hi")

        class Answers:
            def create(self, **kw):
                outer.calls["answer"] = kw
                return SimpleNamespace(json_content='{"book_title": "T"}', sources=["https://s.test"])

        self.searches, self.scrapes, self.answers = Searches(), Scrapes(), Answers()


def test_search_returns_only_title_url_description():
    client = FakeClient()
    out = olostep_cli.search(client, "q", limit=3, include_domains=["a.test"], exclude_domains=[])
    assert out[0] == {"title": "A", "url": "https://a.test", "description": "da"}
    assert out[1]["title"] is None
    assert client.calls["search"] == (
        "q",
        {"limit": 3, "include_domains": ["a.test"], "exclude_domains": None},
    )


def test_scrape_passes_url_to_scrape_and_formats_list():
    client = FakeClient()
    assert olostep_cli.scrape(client, "https://example.com") == "# Hi"
    assert client.calls["scrape"] == {"url_to_scrape": "https://example.com", "formats": ["markdown"]}


def test_scrape_missing_format_raises_instead_of_attribute_error():
    with pytest.raises(RuntimeError, match="html"):
        olostep_cli.scrape(FakeClient(), "https://example.com", fmt="html")


def test_answer_parses_json_content_and_returns_sources():
    client = FakeClient()
    out = olostep_cli.answer(client, "task", {"book_title": ""})
    assert out == {"answer": {"book_title": "T"}, "sources": ["https://s.test"]}
    assert client.calls["answer"] == {"task": "task", "json_format": {"book_title": ""}}


def test_answer_without_json_format_omits_the_argument():
    client = FakeClient()
    olostep_cli.answer(client, "task")
    assert client.calls["answer"] == {"task": "task"}


def test_missing_api_key_exits_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("OLOSTEP_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        olostep_cli.get_client()
    assert "OLOSTEP_API_KEY" in str(exc.value)


def test_main_reports_request_errors_without_a_traceback(monkeypatch):
    client = FakeClient()

    def boom(**kw):
        raise ValueError("bad request")

    client.scrapes.create = boom
    monkeypatch.setattr(olostep_cli, "get_client", lambda: client)
    with pytest.raises(SystemExit) as exc:
        olostep_cli.main(["scrape", "https://example.com"])
    assert "bad request" in str(exc.value)


def test_main_prints_search_json(monkeypatch, capsys):
    monkeypatch.setattr(olostep_cli, "get_client", FakeClient)
    assert olostep_cli.main(["search", "q", "--limit", "2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["url"] == "https://a.test"
