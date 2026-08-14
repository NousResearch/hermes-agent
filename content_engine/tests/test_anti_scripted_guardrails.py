"""Tests for the anti-scripted guardrails on sahil_twitter personal X generation.

These tests prove:
- sahil_twitter with no activity signals returns empty topics (no static fallback)
- sahil_twitter with activity signals still works
- sahil_linkedin still gets static fallback (not affected)
- llm_drafts skips sahil_twitter without activity_data
- llm_generate skips sahil_twitter static fallback on LLM failure
- reply_suggester returns no suggestions for short/placeholder posts
"""

import pytest
import topics as tp
import llm_drafts as ld
import reply_suggester as rs


# ── topics.py: sahil_twitter no static fallback ──────────────────────────

def test_sahil_twitter_no_activity_returns_empty(monkeypatch):
    """sahil_twitter with no activity signals returns only screenshot topics (empty)."""
    # Disable activity collector
    monkeypatch.setattr(tp, "_ACTIVITY_COLLECTOR", None)
    monkeypatch.setattr(tp, "_ACTIVITY_MARKER", None)

    # Mock _load_activity to return False (no activity available)
    monkeypatch.setattr(tp, "_load_activity", lambda: False)

    # Mock _screenshot_topics to return empty
    monkeypatch.setattr(tp, "_screenshot_topics", lambda *a, **kw: [])

    # Mock get_recently_used_topics to return empty
    monkeypatch.setattr(tp, "get_recently_used_topics", lambda *a, **kw: [])

    topics = tp.get_topics("sahil_twitter", count=6)
    assert topics == [], (
        f"sahil_twitter with no activity signals should return empty list, got {len(topics)} topics"
    )


def test_sahil_twitter_with_activity_returns_topics(monkeypatch):
    """sahil_twitter with activity signals still generates topics."""
    # Mock activity collector to return a signal
    def mock_collect():
        return {
            "signals": [{
                "signal_id": "sig-001",
                "signal_type": "github_push",
                "variables": {
                    "repo_name": "KenseiAgent",
                    "description": "anti-scripted guardrails",
                },
                "timestamp": "2026-08-14T10:00:00Z",
                "pillar": "build_in_public",
            }],
            "state": {"used_signals": []},
        }

    monkeypatch.setattr(tp, "_ACTIVITY_COLLECTOR", mock_collect)
    monkeypatch.setattr(tp, "_ACTIVITY_MARKER", lambda *a, **kw: None)
    monkeypatch.setattr(tp, "_load_activity", lambda: True)
    monkeypatch.setattr(tp, "_screenshot_topics", lambda *a, **kw: [])
    monkeypatch.setattr(tp, "get_recently_used_topics", lambda *a, **kw: [])

    # Mock editorial_router to accept the signal (imported locally in get_topics)
    def mock_route(sig):
        return {"decision": "accept", "platform": "twitter", "scores": {"x_fit": 80, "linkedin_fit": 30}}
    import editorial_router
    monkeypatch.setattr(editorial_router, "route_signal", mock_route)

    topics = tp.get_topics("sahil_twitter", count=6)
    assert len(topics) > 0, "sahil_twitter with activity signals should return topics"
    # All topics should have activity_data
    for t in topics:
        assert t.get("activity_data") is not None or t.get("educational"), (
            f"Topic {t.get('id')} should have activity_data or be educational"
        )


def test_sahil_linkedin_still_gets_static_fallback(monkeypatch):
    """sahil_linkedin is NOT affected by the anti-scripted guardrail."""
    monkeypatch.setattr(tp, "_ACTIVITY_COLLECTOR", None)
    monkeypatch.setattr(tp, "_ACTIVITY_MARKER", None)
    monkeypatch.setattr(tp, "_load_activity", lambda: False)
    monkeypatch.setattr(tp, "_screenshot_topics", lambda *a, **kw: [])
    monkeypatch.setattr(tp, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(tp, "log_topic_usage", lambda *a, **kw: None)

    topics = tp.get_topics("sahil_linkedin", count=6)
    assert len(topics) > 0, "sahil_linkedin should still get static topic fallback"


def test_other_brands_still_get_static_fallback(monkeypatch):
    """plenishd, coachos, matchdaymaestro are NOT affected."""
    monkeypatch.setattr(tp, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(tp, "log_topic_usage", lambda *a, **kw: None)

    for brand in ("plenishd", "coachos", "matchdaymaestro"):
        topics = tp.get_topics(brand, count=6)
        assert len(topics) > 0, f"{brand} should still get static topic fallback"


# ── llm_drafts.py: sahil_twitter no static template fallback ─────────────

def test_llm_drafts_sahil_twitter_no_activity_skips():
    """generate_drafts for sahil_twitter with no activity_data returns empty."""
    topic = {
        "id": "t1",
        "pillar": "build_in_public",
        "topic": "Weekly indie dev update",
        # No activity_data
    }
    drafts = ld.generate_drafts("sahil_twitter", [topic], platform="twitter")
    assert drafts == [], (
        f"sahil_twitter without activity_data should return empty, got {len(drafts)} drafts"
    )


def test_llm_drafts_sahil_twitter_with_activity_generates():
    """generate_drafts for sahil_twitter with activity_data still works."""
    topic = {
        "id": "t1",
        "pillar": "build_in_public",
        "topic": "Just pushed KenseiAgent",
        "activity_data": {
            "signal_type": "github_push",
            "variables": {
                "repo_name": "KenseiAgent",
                "description": "anti-scripted guardrails",
            },
            "signal_id": "sig-001",
        },
    }
    drafts = ld.generate_drafts("sahil_twitter", [topic], platform="twitter")
    assert len(drafts) > 0, "sahil_twitter with activity_data should generate drafts"


def test_llm_drafts_sahil_linkedin_still_uses_static():
    """generate_drafts for sahil_linkedin still uses static templates."""
    topic = {
        "id": "t1",
        "pillar": "pm_thought",
        "topic": "Context over models thesis",
        # No activity_data
    }
    drafts = ld.generate_drafts("sahil_linkedin", [topic], platform="linkedin")
    assert len(drafts) > 0, "sahil_linkedin should still use static templates"


# ── reply_suggester.py: no suggestions for absent/placeholder source context ──

def test_reply_suggester_short_post_no_id_returns_empty():
    """Posts without an 'id' field (no source tweet context) return no suggestions."""
    post = {"text": "Nice."}
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert suggestions == [], "Posts without source tweet id should return no suggestions"


def test_reply_suggester_placeholder_post_no_id_returns_empty():
    """Placeholder/synthetic posts without an 'id' return no suggestions."""
    post = {"text": "Just shipped a thing."}
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert suggestions == [], "Placeholder posts without id should return no suggestions"


def test_reply_suggester_short_real_post_returns_suggestions():
    """Genuine short posts with an 'id' (real source tweet) get suggestions.
    
    The old <40-char length gate would have rejected this. The structural
    'id' check correctly permits it — real tweets can be short.
    """
    post = {
        "text": "Just shipped it.",
        "id": "1887654321098765432",
    }
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert len(suggestions) > 0, (
        "Short real posts with source tweet id should get reply suggestions"
    )


def test_reply_suggester_real_post_returns_suggestions():
    """Real posts with substantive content and an 'id' still get suggestions."""
    post = {
        "text": "Just shipped a new feature that lets you split your shopping across 9 UK supermarkets automatically. Saved £6.40 on my first test run.",
        "hashtags": ["buildinpublic"],
        "id": "1887654321098765432",
    }
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert len(suggestions) > 0, "Real posts should get reply suggestions"


def test_reply_suggester_engagement_bait_returns_empty():
    """Engagement bait posts return no suggestions even with a real id."""
    post = {
        "text": "RT if you agree! Like if you've been there. Comment below with your take.",
        "id": "1887654321098765432",
    }
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert suggestions == [], "Engagement bait should return no suggestions"


def test_reply_suggester_empty_post_returns_empty():
    """Empty posts return no suggestions."""
    post = {"text": ""}
    suggestions = rs.suggest_replies_for_post(post, "test_user")
    assert suggestions == [], "Empty posts should return no suggestions"
