"""Tests for slopless.py — verification cases + edge cases."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from slopless import audit_slop, batch_lint


# ─── Verification checklist from SKILL.md ──────────────────────────────────

def test_boilerplate_mantra_detects_short_lines():
    """Boilerplate mantras should trigger slop >= 4."""
    r = audit_slop("Shipping apps. Breaking things. Fixing them. Repeat.", context="twitter")
    assert r["slop_score"] >= 4, f"expected >= 4, got {r['slop_score']}"


def test_specific_tweet_passes():
    """Specific tweet should score < 3."""
    r = audit_slop("Just shipped v2.3 of Plenishd. 47% faster pantry scan on Claude 3.5.", context="twitter")
    assert r["slop_score"] < 3, f"expected < 3, got {r['slop_score']}"


def test_boilerplate_list_non_empty():
    from slopless import BOILERPLATE_MANTRAS
    assert len(BOILERPLATE_MANTRAS) > 0


# ─── Pattern-specific tests ─────────────────────────────────────────────────

def test_empty_text():
    r = audit_slop("")
    assert r["slop_score"] == 0
    assert r["passed"] is True


def test_template_itis_detected():
    r = audit_slop(
        "In the fast-paced world of AI, every AI tool now has a 'smart summary' feature. "
        "So, what are you waiting for?"
    )
    assert r["breakdown"].get("template_itis", 0) >= 2


def test_zero_specificity():
    r = audit_slop(
        "This innovative solution provides a seamless, powerful, and intuitive "
        "experience for users navigating the landscape of modern technology."
    )
    assert r["slop_score"] >= 3, f"expected >= 3, got {r}"


def test_very_specific_passes():
    r = audit_slop(
        "Migrated our Express API to Convex last week. "
        "PostgreSQL latency dropped from 340ms to 45ms on p95. "
        "Deployment took 2 hours total."
    )
    assert r["passed"] is True
    assert r["breakdown"].get("generic_filler", 0) == 0


def test_hashtag_bloat_twitter():
    r = audit_slop(
        "Just shipped #AI #Startup #Tech #Innovation #Hustle", context="twitter"
    )
    assert r["breakdown"].get("hashtags", 0) == 1


def test_ai_isms_detected():
    r = audit_slop(
        "Let's dive into the landscape of AI. "
        "It's important to note that in today's digital age, "
        "this is a testament to the rich tapestry of innovation."
    )
    assert r["breakdown"].get("ai_isms", 0) >= 2


def test_linkedin_ai_isms_doubled():
    r = audit_slop(
        "Let's dive into the landscape of innovation. "
        "It's important to note that at the end of the day, "
        "this is a testament to the rich tapestry of success.",
        context="linkedin"
    )
    # LinkedIn doubles the AI-ism weight → higher than general
    r_gen = audit_slop(
        "Let's dive into the landscape of innovation. "
        "It's important to note that at the end of the day, "
        "this is a testament to the rich tapestry of success.",
        context="general"
    )
    assert r["slop_score"] >= r_gen["slop_score"]


def test_blog_relaxes_over_polished():
    stanza = "Built it.\n\nBroke it.\n\nFixed it.\n\nShipped it."
    r_blog = audit_slop(stanza, context="blog")
    assert r_blog["breakdown"].get("over_polished", 0) == 0


def test_adverb_bloat():
    r = audit_slop(
        "She quickly, efficiently, seamlessly, brilliantly, perfectly, "
        "and comprehensively completed the task."
    )
    assert r["breakdown"].get("adverb_bloat", 0) == 1


def test_score_capped_at_10():
    worst_case = (
        "Shipping apps. Breaking things. Tap to play.\n"
        "In the fast-paced world of AI, let's dive in.\n"
        "It's important to note that in today's digital age, "
        "this is a testament to the rich tapestry.\n"
        "At the end of the day, when it comes to innovation, "
        "a game-changer that unlocks the full potential.\n"
        "#AI #Startup #Tech #Innovation #Hustle #Growth\n"
        "Quickly efficiently seamlessly brilliantly."
    )
    r = audit_slop(worst_case)
    assert r["slop_score"] <= 10


def test_context_unknown_uses_general():
    r = audit_slop("delve into the rich tapestry", context="unknown_context")
    assert r["slop_score"] == audit_slop("delve into the rich tapestry", context="general")["slop_score"]


def test_batch_lint():
    drafts = [
        {"id": "clean", "body": "Shipped v2.3 of Plenishd. 47% scan speed improvement.", "context": "twitter"},
        {"id": "sloppy", "body": "In today's digital age, let's dive into the rich tapestry of innovation.", "context": "general"},
    ]
    results = batch_lint(drafts)
    assert len(results) == 2
    assert results[0]["id"] == "clean"
    assert results[1]["id"] == "sloppy"
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
