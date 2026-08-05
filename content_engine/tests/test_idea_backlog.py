"""Tests for the durable idea-backlog module (research → blog review)."""
import sys
from pathlib import Path

# Ensure content_engine is importable.
_engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_engine_root))

import pytest
from blog import idea_backlog as ib


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    """Point the store at a temp path for each test."""
    monkeypatch.setattr(ib, "STORE_PATH", tmp_path / "idea-backlog.jsonl")
    monkeypatch.setattr(ib, "BACKLOG_DIR", tmp_path / "backlog")
    yield


def test_add_and_dedupe():
    r1 = ib.add("Title A", "concept A", ["blog"], "ai", "arxiv:1")
    assert r1["status"] == "added"
    # same (source,title) fingerprint → duplicate, no new record
    r2 = ib.add("Title A", "concept A", ["blog"], "ai", "arxiv:1")
    assert r2["status"] == "duplicate"
    assert r2["id"] == r1["id"]
    # different title → new record
    r3 = ib.add("Title B", "concept B", [], "pm", "arxiv:2")
    assert r3["status"] == "added"
    assert len(ib.list_pending()) == 2


def test_add_defaults_and_unknown_stream():
    r = ib.add("T", "c", [], "not-a-stream", "")
    assert r["status"] == "added"
    recs = ib._read()
    assert recs[0]["stream"] == "ai"  # falls back to ai
    assert recs[0]["source"] == ""


def test_approve_queues_to_stream_backlog():
    r = ib.add("T", "c", ["blog"], "ai", "arxiv:9")
    assert r["status"] == "added"
    res = ib.approve(r["id"])
    assert res["status"] == "approved"
    q = ib.BACKLOG_DIR / "ai.jsonl"
    assert q.exists()
    lines = [l for l in q.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert r["id"] in lines[0]
    # status flipped
    recs = ib._read()
    assert recs[0]["status"] == "approved"
    assert recs[0]["decided_at"] is not None
    # no longer pending
    assert r["id"] not in [x["id"] for x in ib.list_pending()]


def test_reject_flips_status():
    r = ib.add("T", "c", [], "ai", "")
    ib.reject(r["id"])
    recs = ib._read()
    assert recs[0]["status"] == "rejected"
    assert recs[0]["decided_at"] is not None


def test_approve_unknown_id():
    res = ib.approve("idea-does-not-exist")
    assert res["status"] == "not_found"


def test_idea_cards_shape():
    ib.add("Idea 1", "concept one", ["blog", "LinkedIn"], "ai", "arxiv:1")
    ib.add("Idea 2", "concept two", [], "pm", "arxiv:2")
    cards = ib.idea_cards()
    assert len(cards) == 2
    for c in cards:
        assert c["group"] == "IDEAS"
        assert c["id"]
        assert "!approve-idea" in c["pane"]
        assert "!reject-idea" in c["pane"]
        assert "<h1>" in c["pane"]
