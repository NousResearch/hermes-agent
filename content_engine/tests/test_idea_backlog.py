"""Tests for the durable idea-backlog module (research → blog review)."""
import sys
from pathlib import Path

# Ensure content_engine is importable.
_engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_engine_root))

import pytest
from blog import idea_backlog as ib


PURPOSE_FIELDS = {
    "post_thesis": "The durable thesis this post will argue.",
    "concrete_takeaway": "A reader can apply the decision framework next week.",
    "evidence_anchor": "arXiv:2401.01234, section 3",
    "gap_claim": "Existing coverage explains the tool but not the operating trade-off.",
    "stream_format_rationale": "AI essay because the mechanism needs long-view analysis.",
}

def _add(title="Title A", concept="concept A", angles=None, stream="ai", source="arxiv:1", **overrides):
    fields = PURPOSE_FIELDS | overrides
    return ib.add(title, concept, angles or ["blog"], stream, source, **fields)


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    """Point the store at a temp path for each test."""
    monkeypatch.setattr(ib, "STORE_PATH", tmp_path / "idea-backlog.jsonl")
    monkeypatch.setattr(ib, "BLOG_TOPICS_DIR", tmp_path / "blog_topics")
    yield


def test_add_and_dedupe():
    r1 = _add()
    assert r1["status"] == "added"
    # same (source,title) fingerprint → duplicate, no new record
    r2 = _add()
    assert r2["status"] == "duplicate"
    assert r2["id"] == r1["id"]
    # different title → new record
    r3 = _add("Title B", "concept B", [], "pm", "arxiv:2")
    assert r3["status"] == "added"
    assert len(ib.list_pending()) == 2


def test_add_defaults_and_unknown_stream():
    r = _add("T", "c", [], "not-a-stream", "")
    assert r["status"] == "added"
    recs = ib._read()
    assert recs[0]["stream"] == "ai"  # falls back to ai
    assert recs[0]["source"] == ""


def test_add_fails_closed_when_any_required_purpose_field_is_blank():
    for field in PURPOSE_FIELDS:
        result = _add(**{field: "  "})
        assert result == {"status": "invalid", "missing_fields": [field]}
    assert ib._read() == []


def test_add_persists_required_purpose_fields():
    result = _add()
    assert result["status"] == "added"
    record = ib._read()[0]
    for field, value in PURPOSE_FIELDS.items():
        assert record[field] == value


def test_cli_add_requires_purpose_led_fields(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "idea_backlog.py", "add", "--title", "T", "--concept", "c",
    ])
    with pytest.raises(SystemExit) as exc_info:
        ib._cli()
    assert exc_info.value.code == 2


def test_approve_queues_to_router_stream_queue_with_intake_fields_and_priority():
    r = _add("T", "c", ["blog"], "ai", "arxiv:9")
    assert r["status"] == "added"
    res = ib.approve(r["id"])
    assert res["status"] == "approved"
    q = ib.BLOG_TOPICS_DIR / "ai.jsonl"
    assert q.exists()
    assert not (ib.BLOG_TOPICS_DIR / "backlog" / "ai.jsonl").exists()
    lines = [l for l in q.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = __import__("json").loads(lines[0])
    assert entry["topic_id"] == r["id"]
    assert entry["priority"] == 9
    for field, value in PURPOSE_FIELDS.items():
        assert entry[field] == value
    # status flipped
    recs = ib._read()
    assert recs[0]["status"] == "approved"
    assert recs[0]["decided_at"] is not None
    # no longer pending
    assert r["id"] not in [x["id"] for x in ib.list_pending()]


def test_reject_flips_status():
    r = _add("T", "c", [], "ai", "")
    ib.reject(r["id"])
    recs = ib._read()
    assert recs[0]["status"] == "rejected"
    assert recs[0]["decided_at"] is not None


def test_approve_unknown_id():
    res = ib.approve("idea-does-not-exist")
    assert res["status"] == "not_found"


def test_idea_cards_shape():
    ib.add("Idea 1", "concept one", ["blog", "LinkedIn"], "ai", "arxiv:1", **PURPOSE_FIELDS)
    ib.add("Idea 2", "concept two", [], "pm", "arxiv:2", **PURPOSE_FIELDS)
    cards = ib.idea_cards()
    assert len(cards) == 2
    for c in cards:
        assert c["group"] == "IDEAS"
        assert c["id"]
        assert "!approve-idea" in c["pane"]
        assert "!reject-idea" in c["pane"]
        assert "<h1>" in c["pane"]
