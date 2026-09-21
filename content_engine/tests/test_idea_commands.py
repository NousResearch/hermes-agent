"""Tests for idea-approval Discord commands (approve-idea / reject-idea)."""
import sys
from pathlib import Path

_engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_engine_root))

import pytest
from blog import idea_backlog as ib
from blog import blog_approval as ba


PURPOSE_FIELDS = {
    "post_thesis": "The durable thesis this post will argue.",
    "concrete_takeaway": "A reader can apply the decision framework next week.",
    "evidence_anchor": "arXiv:2401.01234, section 3",
    "gap_claim": "Existing coverage explains the tool but not the operating trade-off.",
    "stream_format_rationale": "AI essay because the mechanism needs long-view analysis.",
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "STORE_PATH", tmp_path / "idea-backlog.jsonl")
    monkeypatch.setattr(ib, "BLOG_TOPICS_DIR", tmp_path / "blog_topics")
    # blog_approval's handle_discord_command imports idea_backlog fresh each call;
    # redirect its store paths via the module attribute swap.
    yield


def test_parse_idea_commands():
    assert ba.parse_discord_command("!approve-idea idea-abc") == {
        "command": "approve-idea", "slug": "idea-abc", "args": "",
    }
    assert ba.parse_discord_command("!reject-idea idea-abc because reasons") == {
        "command": "reject-idea", "slug": "idea-abc", "args": "because reasons",
    }
    # legacy blog commands still parse
    assert ba.parse_discord_command("!approve my-slug")["command"] == "approve"


def test_handle_approve_idea(monkeypatch, tmp_path):
    # monkeypatch idea_backlog store used inside blog_approval's lazy import
    monkeypatch.setattr(ib, "STORE_PATH", tmp_path / "idea-backlog.jsonl")
    monkeypatch.setattr(ib, "BLOG_TOPICS_DIR", tmp_path / "blog_topics")
    r = ib.add("Idea X", "concept", ["blog"], "ai", "arxiv:5", **PURPOSE_FIELDS)
    res = ba.handle_discord_command(f"!approve-idea {r['id']}")
    assert res["handled"] is True
    assert res["action"] == "idea_approved"
    # queued to stream backlog
    assert (tmp_path / "blog_topics" / "ai.jsonl").exists()


def test_handle_reject_idea(monkeypatch, tmp_path):
    monkeypatch.setattr(ib, "STORE_PATH", tmp_path / "idea-backlog.jsonl")
    monkeypatch.setattr(ib, "BLOG_TOPICS_DIR", tmp_path / "blog_topics")
    r = ib.add("Idea Y", "concept", [], "pm", "arxiv:6", **PURPOSE_FIELDS)
    res = ba.handle_discord_command(f"!reject-idea {r['id']}")
    assert res["handled"] is True
    assert res["action"] == "idea_rejected"


def test_handle_idea_not_found():
    res = ba.handle_discord_command("!approve-idea idea-nope")
    assert res["handled"] is True
    assert res["action"] == "not_found"
