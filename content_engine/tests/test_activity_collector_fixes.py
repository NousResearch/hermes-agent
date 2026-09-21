"""Tests for activity_collector fixes.

Covers:
- Research digest HTML parsing (using a real saved fixture)
- Architecture signal rotation
- Section classification (news vs tools vs signal)
- Signal output format for downstream consumers
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_collector import (
    _extract_research_digest,
    collect_research_digest,
    collect_architecture_insights,
    _ensure_state,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "research-brief-fixture.html"


# ── Research digest: raw extraction ──


def test_extract_research_digest_reads_file():
    """_extract_research_digest should read the latest digest file and return items."""
    assert FIXTURE_PATH.exists(), f"Fixture missing: {FIXTURE_PATH}"
    # Monkey-patch RUNBOOKS_DIR to point at fixtures so _extract uses our file
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        items = _extract_research_digest()
    finally:
        ac.RUNBOOKS_DIR = original_dir
    assert len(items) > 0, "Should extract items from fixture"
    assert all("title" in i for i in items), "Each item must have a title"
    assert all("section" in i for i in items), "Each item must have a section"
    assert all("url" in i for i in items), "Each item must have a URL"


def test_extract_research_digest_clean_titles():
    """Titles should be clean text, not raw HTML."""
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        items = _extract_research_digest()
    finally:
        ac.RUNBOOKS_DIR = original_dir
    for item in items:
        title = item["title"]
        assert "<a" not in title, f"Title still contains <a> tag: {title[:40]}"
        assert "</a>" not in title, f"Title still contains </a>: {title[:40]}"
        assert "&amp;" not in title, f"Title still has encoded &amp;: {title[:40]}"
        assert len(title) > 5, f"Title too short: {title}"


def test_extract_research_digest_has_urls():
    """Each item should have a source URL extracted from <a href>."""
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        items = _extract_research_digest()
    finally:
        ac.RUNBOOKS_DIR = original_dir
    for item in items:
        assert item["url"], f"Missing URL for: {item['title'][:40]}"
        assert item["url"].startswith("http"), f"URL should be absolute: {item['url']}"


def test_extract_research_digest_has_summaries():
    """Each item should have a summary (may be empty if format differs)."""
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        items = _extract_research_digest()
    finally:
        ac.RUNBOOKS_DIR = original_dir
    assert any(
        item.get("summary") for item in items
    ), "At least some items should have summaries"


def test_extract_research_digest_section_classification():
    """Items should be classified into sections."""
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        items = _extract_research_digest()
    finally:
        ac.RUNBOOKS_DIR = original_dir
    sections = {i["section"] for i in items}
    assert "news" in sections, "Should have news items"
    # At least one non-news section (tools or signal)
    non_news = sections - {"news"}
    assert non_news, (
        f"Expected at least one non-news section, got only: {sections}"
    )


# ── Research digest: collect signals ──


def test_collect_research_digest_returns_signals():
    """collect_research_digest should return properly typed signals."""
    import activity_collector as ac
    # Use a fresh state so no previous tests have pre-marked these as used
    fresh_state = {"used_signals": [], "cycle_signals": {}}
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        signals = collect_research_digest(fresh_state)
    finally:
        ac.RUNBOOKS_DIR = original_dir
    assert isinstance(signals, list)
    for s in signals:
        assert "signal_type" in s
        assert s["signal_type"] in ("research_tool", "research_signal")
        assert "signal_id" in s
        assert s["signal_id"].startswith("research:")
        assert "variables" in s
        assert "title" in s["variables"]
        assert "summary" in s["variables"]


def test_collect_research_digest_skips_news():
    """News items should not appear as signals (they're skipped intentionally)."""
    import activity_collector as ac
    original_dir = ac.RUNBOOKS_DIR
    try:
        ac.RUNBOOKS_DIR = FIXTURE_DIR
        state = _ensure_state()
        signals = collect_research_digest(state)
    finally:
        ac.RUNBOOKS_DIR = original_dir
    for s in signals:
        # News items are never returned — only tools or signal sections
        assert s["signal_type"] in ("research_tool", "research_signal")


# ── Architecture rotation ──


def test_architecture_rotation_cycles_through_topics():
    """Repeated calls should cycle through different topics."""
    fresh_state = {"used_signals": [], "cycle_signals": {"arch_rotation_idx": 0}}
    seen = set()
    for i in range(8):
        signals = collect_architecture_insights(fresh_state)
        assert len(signals) == 1, "Should return exactly one signal"
        topic = signals[0]["variables"]["topic"]
        seen.add(topic)
    assert len(seen) >= 4, (
        f"Should see at least 4 different topics across 8 calls, got {len(seen)}"
    )


def test_architecture_rotation_wraps_around():
    """Rotation should wrap back to the first topic after exhausting the list."""
    # Use a fresh state dict, not _ensure_state(), to avoid cross-test pollution
    fresh_state = {"used_signals": [], "cycle_signals": {"arch_rotation_idx": 0}}

    first_signals = collect_architecture_insights(fresh_state)
    first_topic = first_signals[0]["variables"]["topic"]

    # Call 5 more times to advance through the remaining 5 topics
    for _ in range(5):
        collect_architecture_insights(fresh_state)

    # 7th call wraps back to index 0
    seventh_signals = collect_architecture_insights(fresh_state)
    seventh_topic = seventh_signals[0]["variables"]["topic"]
    assert seventh_topic == first_topic, (
        f"After wrapping, expected {first_topic}, got {seventh_topic}"
    )


def test_architecture_rotation_unique_signal_ids():
    """Each rotation call should produce a unique signal_id."""
    fresh_state = {"used_signals": [], "cycle_signals": {"arch_rotation_idx": 0}}
    ids = set()
    for i in range(6):
        s = collect_architecture_insights(fresh_state)
        sid = s[0]["signal_id"]
        assert sid not in ids, f"Duplicate signal_id: {sid}"
        ids.add(sid)
    assert len(ids) == 6, "Should have 6 unique signal IDs for 6 topics"


def test_architecture_signals_have_required_fields():
    """Architecture signals should have all fields downstream expects."""
    fresh_state = {"used_signals": [], "cycle_signals": {"arch_rotation_idx": 0}}
    signals = collect_architecture_insights(fresh_state)
    s = signals[0]
    assert s["signal_type"] == "architecture"
    assert s["priority"] >= 5
    assert "topic_label" in s["variables"]
    assert "description" in s["variables"]
    assert "topic" in s["variables"]
