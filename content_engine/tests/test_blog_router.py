"""Tests for blog.blog_router — per-stream topic selection + dedup.

The router reuses database.get_recently_used_topics / log_topic_usage with
brand=f"blog_{stream}" (mirrors article_pipeline's cross-run dedup). It gathers
candidate topics from the stream's sources, drops recently-used ids, and picks
the highest-priority. record() writes back AFTER a successful publish.
"""
import pytest

import blog.blog_router as br
from blog.blog_streams import STREAMS


@pytest.fixture(autouse=True)
def _isolate_reservations(tmp_path, monkeypatch):
    """Point the reservation file at a per-test tmp path.

    reserve() writes to a real persistent JSONL; without isolation, a stale
    reservation (180min TTL) from one test blocks topic ids in another and
    across runs. Isolating the path keeps the suite deterministic.
    """
    monkeypatch.setattr(br, "TOPIC_RESERVATIONS_PATH", tmp_path / "reservations.jsonl")
    monkeypatch.setattr(br, "FAILED_GENERATOR_PATH", tmp_path / "failed_generator.jsonl")


def _fake_topic(tid, priority=5, summary="a topic", tags=None, source=None):
    t = {
        "topic_id": tid,
        "title_hint": summary,
        "tags": tags or [],
        "source_override": source,  # None = use stream config source
        "signals": [{"signal_id": tid, "summary": summary, "priority": priority}],
        "priority": priority,
    }
    return t


def test_track_failed_generator_increments_attempts():
    """Repeated failures for the same topic_id accumulate, not duplicate rows."""
    assert br.track_failed_generator("t1", "ai", "gate fail") == 1
    assert br.track_failed_generator("t1", "ai", "gate fail again") == 2
    assert br.track_failed_generator("t1", "ai", "gate fail third") == 3
    entries = br._read_failed_generator_entries()
    assert len(entries) == 1
    assert entries[0]["attempts"] == 3
    assert entries[0]["last_error"] == "gate fail third"


def test_choose_excludes_quarantined_topic_after_threshold(monkeypatch):
    """A topic that has failed generation >= GENERATOR_FAILURE_THRESHOLD times
    is excluded from choose() so it stops being re-burned every cron cycle."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates",
                         lambda stream: [_fake_topic("bad-topic", 9), _fake_topic("good-topic", 5)])
    for _ in range(br.GENERATOR_FAILURE_THRESHOLD):
        br.track_failed_generator("bad-topic", "ai", "quality gate rejected")
    result = br.choose("ai")
    assert result is not None
    assert result["topic_id"] == "good-topic"


def test_choose_still_returns_topic_below_threshold(monkeypatch):
    """A topic with fewer failures than the threshold is still eligible (retries as designed)."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates", lambda stream: [_fake_topic("t1", 7)])
    br.track_failed_generator("t1", "ai", "quality gate rejected")
    result = br.choose("ai")
    assert result is not None
    assert result["topic_id"] == "t1"


def test_record_clears_failed_generator_tracking(monkeypatch):
    """A topic that eventually succeeds has its failure history wiped."""
    br.track_failed_generator("t1", "ai", "quality gate rejected")
    assert br.get_quarantined_generator_topics(threshold=1)
    monkeypatch.setattr(br.db, "log_topic_usage", lambda **kw: None)
    br.record("ai", "t1", "A Title")
    assert br._read_failed_generator_entries() == []


def test_choose_returns_topic_dict_or_none(monkeypatch):
    """choose(stream) returns {topic_id, title_hint, tags, source, signals} or None."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates", lambda stream: [_fake_topic("t1", 7)])
    out = br.choose("builder")
    assert out is not None
    assert out["topic_id"] == "t1"
    assert "title_hint" in out
    assert "tags" in out
    assert "source" in out
    assert "signals" in out


def test_choose_returns_none_when_no_candidates(monkeypatch):
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates", lambda stream: [])
    assert br.choose("ai") is None


def test_choose_excludes_recently_used(monkeypatch):
    """Topics within the recency window are excluded."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: ["used1"])
    monkeypatch.setattr(br, "_gather_candidates",
                       lambda stream: [_fake_topic("used1", 9), _fake_topic("fresh1", 5)])
    out = br.choose("pm")
    assert out is not None
    assert out["topic_id"] == "fresh1"


def test_choose_picks_highest_priority(monkeypatch):
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    cands = [_fake_topic("low", 3), _fake_topic("high", 9), _fake_topic("mid", 6)]
    monkeypatch.setattr(br, "_gather_candidates", lambda stream: cands)
    out = br.choose("ai")
    assert out["topic_id"] == "high"


def test_choose_uses_stream_source_from_config(monkeypatch):
    """The returned source matches the stream's configured source."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates", lambda stream: [_fake_topic("t1")])
    out = br.choose("ai")
    assert out["source"] == STREAMS["ai"]["source"]  # research-paper


def test_choose_uses_stream_base_tags(monkeypatch):
    """The returned tags include the stream's base_tags."""
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_candidates",
                       lambda stream: [_fake_topic("t1", tags=["extra-tag"])])
    out = br.choose("ai")
    assert "ai" in out["tags"]  # base tag from config
    assert "extra-tag" in out["tags"]  # topic tag merged


def test_record_writes_to_db_with_blog_brand(monkeypatch):
    """record() calls database.log_topic_usage with brand=f'blog_{stream}'."""
    calls = []
    monkeypatch.setattr(br.db, "log_topic_usage",
                        lambda topic_id, brand, topic_text, platform="", **kw: calls.append(
                            (topic_id, brand, topic_text, platform)))
    br.record("builder", "tid1", "title text")
    # quality_score=None is passed as 5th arg.
    assert ("tid1", "blog_builder", "title text", "blog") in calls
    # The call tuple format is (topic_id, brand, topic_text, platform, quality_score)
    # Check that it has the right first 4 even when quality_score is passed.
    match = [c for c in calls if c[0] == "tid1" and c[1] == "blog_builder"]
    assert match, "Expected tid1 call not found"


def test_record_is_defensive_on_db_error(monkeypatch):
    """record() swallows DB exceptions so a failed publish doesn't crash."""
    monkeypatch.setattr(br.db, "log_topic_usage",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    # Must not raise.
    br.record("ai", "tid1", "title")


def test_recent_used_is_defensive(monkeypatch):
    """_recent_used returns [] on DB error (degrades to in-run-only dedup)."""
    monkeypatch.setattr(br.db, "get_recently_used_topics",
                        lambda brand, days: (_ for _ in ()).throw(RuntimeError("db down")))
    assert br._recent_used("ai") == []


def test_gather_candidates_builder_uses_activity_collector(monkeypatch):
    """Builder pulls from frameworks + its own backlog queue + activity signals."""
    monkeypatch.setattr(br.ac, "collect_all", lambda: {
        "signals": [{"signal_id": "gh1", "summary": "push", "priority": 8,
                     "pillar": "agent_build_notes"}],
    })
    queued = [_fake_topic("queued-builder", priority=9, summary="approved builder idea")]
    monkeypatch.setattr(br, "_read_manual_queue", lambda stream: queued if stream == "builder" else [])
    cands = br._gather_candidates("builder")
    ids = {c["topic_id"] for c in cands}
    # Framework seeds present.
    assert any(i.startswith("fw-") for i in ids), "Framework seeds should be present"
    # Builder's own backlog queue (builder.jsonl) is now wired in. Do not rely
    # on a topic-id prefix: curated builder backlog items can come from research
    # synthesis, repo activity, or manual queue sources.
    builder_queue_ids = {c["topic_id"] for c in queued}
    assert ids & builder_queue_ids, "Builder queue should be present"
    # The activity signal is still present.
    assert "gh1" in ids, "Activity signal should still be present"


def test_gather_candidates_unknown_stream_returns_empty(monkeypatch):
    """An unknown stream name returns [] (no crash)."""
    assert br._gather_candidates("nonexistent") == []


def test_manual_queue_skips_placeholder_stubs(tmp_path, monkeypatch):
    """Placeholder stubs ('New Concept', empty) from external writers are dropped."""
    qf = tmp_path / "ai.jsonl"
    qf.write_text(
        '{"topic_id": "real-1", "title_hint": "A real topic", "priority": 7}\n'
        '{"topic_id": "manual-1", "title_hint": "New Concept", "priority": 10}\n'
        '{"topic_id": "manual-2", "title_hint": "", "priority": 10}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    out = br._read_manual_queue("ai")
    ids = {c["topic_id"] for c in out}
    assert ids == {"real-1"}, f"placeholders should be skipped, got {ids}"


def test_manual_queue_preserves_approved_idea_editorial_brief(tmp_path, monkeypatch):
    """Purpose-led fields survive queue ingestion into the generation plan."""
    qf = tmp_path / "ai.jsonl"
    qf.write_text(
        '{"topic_id":"idea-1","title_hint":"A real topic","priority":9,'
        '"post_thesis":"The thesis","concrete_takeaway":"Do this",'
        '"evidence_anchor":"PR #42","gap_claim":"Different angle",'
        '"stream_format_rationale":"AI essay"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_framework_candidates", lambda: [])
    out = br.choose("ai")
    assert out["editorial_brief"] == {
        "post_thesis": "The thesis",
        "concrete_takeaway": "Do this",
        "evidence_anchor": "PR #42",
        "gap_claim": "Different angle",
        "stream_format_rationale": "AI essay",
    }


# -- Approach A research-roundup router provenance ----------------------------

def test_research_queue_keeps_stream_source_provenance(tmp_path, monkeypatch):
    """Research queue entries without an explicit source_override fall through
    to the stream's configured source (curated-roundup), not manual_queue."""
    qf = tmp_path / "research.jsonl"
    qf.write_text(
        '{"topic_id": "r1", "title_hint": "Agent memory roundup", "priority": 7}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    out = br._read_manual_queue("research")
    assert len(out) == 1
    # source_override is None → choose() will pick STREAMS["research"]["source"].
    assert out[0]["source_override"] is None


def test_research_choose_returns_curated_roundup_source(tmp_path, monkeypatch):
    """choose('research') returns the stream's curated-roundup provenance."""
    qf = tmp_path / "research.jsonl"
    qf.write_text(
        '{"topic_id": "r1", "title_hint": "Agent memory roundup", "priority": 7}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    monkeypatch.setattr(br, "_recent_used", lambda stream: [])
    monkeypatch.setattr(br, "_gather_framework_candidates", lambda: [])
    out = br.choose("research")
    assert out is not None
    assert out["source"] == STREAMS["research"]["source"]  # curated-roundup


def test_research_queue_respects_explicit_source_override(tmp_path, monkeypatch):
    """An explicit source_override on a research queue entry still wins."""
    qf = tmp_path / "research.jsonl"
    qf.write_text(
        '{"topic_id": "r1", "title_hint": "Agent memory roundup", "priority": 7, '
        '"source_override": "manual_queue"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    out = br._read_manual_queue("research")
    assert out[0]["source_override"] == "manual_queue"


def test_ai_pm_queue_still_default_to_manual_queue(tmp_path, monkeypatch):
    """AI/PM queue entries still default to manual_queue (regression guard)."""
    qf = tmp_path / "ai.jsonl"
    qf.write_text(
        '{"topic_id": "a1", "title_hint": "A real topic", "priority": 7}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(br, "_manual_queue_path", lambda stream: qf)
    out = br._read_manual_queue("ai")
    assert out[0]["source_override"] == "manual_queue"