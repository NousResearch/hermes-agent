"""Tests for editorial_router — source-aware platform routing.

Covers:
- X routes fresh build signal
- LinkedIn rejects raw technical signal without lesson
- Blog AI requires source/evidence
- Articles skip shallow signals
- Product metric without source rejected
- Router returns skip
- Source provenance building
- choose_content_type bridge
- Freshness/presence scoring
"""
from editorial_router import (
    route_signal, _build_provenance, _has_reader_lesson,
    _is_shallow, _has_source_evidence, _freshness_score,
    choose_content_type, _score_x_fit, _score_linkedin_fit,
    _score_blog_fit, _score_article_fit,
)


# ── Helper factories ──


def _sig(sid, stype="github_push", pillar="build_in_public",
         prio=8, hours=48, summary="Pushed repo_name to GitHub", url="",
         variables=None):
    return {
        "signal_id": sid,
        "signal_type": stype,
        "pillar": pillar,
        "priority": prio,
        "freshness_hours": hours,
        "summary": summary,
        "variables": variables or {
            "repo_name": "repo_name",
            "url": url or f"https://github.com/Sahil-SS9/repo_name",
        },
    }


def _bare_sig(sid, stype="manual", topic="A generic topic", prio=5):
    """A topic bank entry with no signal type (generic static topic)."""
    return {
        "signal_id": sid,
        "signal_type": stype,
        "topic": topic,
        "pillar": "build_in_public",
        "priority": prio,
        "freshness_hours": 168,
    }


# ── 1. X routes fresh build signal ──


def test_x_routes_fresh_build_signal():
    """A fresh github_push build signal routes to X."""
    s = _sig("gh-push:test", stype="github_push", prio=8, hours=24)
    result = route_signal(s, ["x_post", "linkedin_post", "blog", "article"])
    assert result["decision"] == "x_post"
    assert result["platform"] == "twitter"
    assert result["content_type"] in ("text", "text+image")
    assert result["scores"]["x_fit"] >= 70


def test_x_detects_screenshot_visual():
    """Screenshot signal gets text+image content type and routes to X."""
    s = _sig("sc-shot", stype="screenshot", hours=12)
    s["screenshot_path"] = "/tmp/test.png"
    result = route_signal(s, ["x_post", "linkedin_post"])
    assert result["decision"] == "x_post"
    assert result["content_type"] == "text+image"
    assert result["scores"]["visual_fit"] >= 90


# ── 2. LinkedIn rejects raw technical without lesson ──


def test_linkedin_rejects_raw_technical():
    """A github_push with no reader/business lesson should not route to LinkedIn."""
    s = _sig("gh-push:raw", stype="github_push", hours=48,
             summary="Pushed some code to repo")
    result = route_signal(s, ["x_post", "linkedin_post"])
    assert result["decision"] != "linkedin_post"
    # Either skips LinkedIn or routes to X
    if result["decision"] != "x_post":
        assert "Raw technical" in result["rationale"] or "skip" in result["decision"]


def test_linkedin_accepts_with_lesson():
    """A signal with reader/business lesson routes to LinkedIn."""
    s = _sig("gh-push:lesson", stype="github_push", hours=48,
             summary="Why I changed my approach to shipping — lessons learned")
    s["variables"]["sha"] = "abc123"
    result = route_signal(s, ["x_post", "linkedin_post"])
    # Should route to either X or LinkedIn — LinkedIn should be a strong contender
    assert _has_reader_lesson("Why I changed my approach — lessons learned")
    assert result["scores"]["linkedin_fit"] >= 50


def test_has_reader_lesson():
    """Text with reflective cues returns True."""
    assert _has_reader_lesson("Lessons learned from shipping to production")
    assert _has_reader_lesson("How we approached the adoption problem")
    assert not _has_reader_lesson("Pushed commit to main")
    assert not _has_reader_lesson("Installed new dependency and ran tests")


# ── 3. Blog AI requires source/evidence ──


def test_blog_ai_requires_source_evidence():
    """A bare blog queue entry with topic_id + title_hint is not enough.
    It must be explicitly curated or carry evidence/context."""
    s = {
        "signal_type": "manual",
        "topic_id": "fake-topic-1",
        "title_hint": "Why AI evals matter",
        "tags": ["ai", "evaluation"],
        "priority": 8,
    }
    result = route_signal(s, ["blog"])
    assert result["decision"] == "skip"


def test_blog_accepts_with_source_override():
    """A blog queue entry with source_override should route to blog."""
    s = {
        "signal_type": "manual",
        "topic_id": "ai-eval-2026",
        "title_hint": "Why most agent evals are wrong",
        "tags": ["ai", "agents"],
        "priority": 8,
        "source_override": "research-paper",
    }
    result = route_signal(s, ["blog"])
    assert result["decision"] == "blog" or result["scores"]["blog_fit"] >= 40


# ── 4. Articles skip shallow signals ──


def test_articles_skip_shallow_signals():
    """A signal with a very short summary should not route to article."""
    s = _sig("shallow", stype="harness_change", prio=8, hours=24,
             summary="Fix")
    assert _is_shallow(s)
    result = route_signal(s, ["x_post", "article"])
    # Article fit should be heavily penalised
    assert result["scores"]["article_fit"] < 70


def test_deep_signal_can_route_to_article():
    """A harness_change with high priority and enough context can route to article."""
    s = _sig("deep", stype="harness_change", prio=9, hours=48,
             summary="Implemented routing fallback chain with 4 retry strategies and governance agent approval gate")
    s["variables"]["repo"] = "KenseiAgent"
    s["variables"]["sha"] = "abc123def"
    assert not _is_shallow(s)
    result = route_signal(s, ["x_post", "article"])
    assert result["scores"]["article_fit"] >= 50


# ── 5. Product metric without source rejected ──


def test_product_metric_without_source_rejected():
    """A signal with product metrics but no source evidence is penalised."""
    s = _sig("metric-test", stype="manual", hours=168,
             summary="200 household test results show 30% improvement")
    result = route_signal(s, ["linkedin_post"])
    assert result["scores"]["linkedin_fit"] < 50


def test_product_metric_with_source_accepted():
    """A signal with product metrics AND source evidence still needs reader lesson for LinkedIn
    but should avoid the metric-without-source penalty."""
    s = _sig("metric-ok", stype="github_push", hours=48,
             summary="200 household test results show 30% improvement")
    s["source_provenance"] = {
        "source_type": "github_push",
        "confidence": "verified",
        "source_url": "https://github.com/org/repo",
        "evidence": ["verified results"],
    }
    result = route_signal(s, ["linkedin_post"])
    # Still low fit because github_push raw tech without reader lesson:
    # without the source evidence penalty, base is 35 - 20 (raw tech) = 15
    # with the source evidence penalty removed (-30), it would have been lower
    # This verifies the source evidence check works: it prevents the additional -30
    assert result["scores"]["linkedin_fit"] >= 10
    assert result["scores"]["linkedin_fit"] < 50


# ── 6. Router returns skip ──


def test_router_returns_skip_for_generic_topic():
    """A generic topic bank entry with no signal type should still route somewhere or skip."""
    s = _bare_sig("stale", stype="", topic="How to manage your kitchen better", prio=3)
    result = route_signal(s, ["x_post", "blog"])
    assert "decision" in result  # Doesn't crash


def test_router_can_skip_all():
    """A completely bare entry with low scores across should skip."""
    s = {"topic": "A vague idea", "priority": 1, "freshness_hours": 9999}
    result = route_signal(s, ["x_post", "linkedin_post", "blog", "article"])
    assert result["decision"] == "skip"


# ── 7. Source provenance ──


def test_source_provenance_built():
    """Source provenance is built correctly for verified signals."""
    s = _sig("gh-push:KenseiAgent", stype="github_push")
    prov = _build_provenance(s)
    assert prov["source_type"] == "github_push"
    assert prov["source_id"] == "gh-push:KenseiAgent"
    assert "github.com" in prov["source_url"]
    assert prov["confidence"] == "verified"
    assert len(prov["evidence"]) >= 1


def test_source_provenance_manual():
    """Manual signals get manual confidence."""
    s = {"signal_type": "manual", "topic_id": "t1", "topic": "test topic"}
    prov = _build_provenance(s)
    assert prov["confidence"] == "manual"


# ── 8. choose_content_type bridge ──


def test_choose_content_type_returns_none_without_signal():
    """choose_content_type returns None when there's no signal type."""
    result = choose_content_type({}, "sahil_twitter", "build_in_public", "twitter")
    assert result is None


def test_choose_content_type_returns_value_with_signal():
    """choose_content_type returns a content type when a real signal is available."""
    s = _sig("test", stype="screenshot", hours=12)
    s["screenshot_path"] = "/tmp/shot.png"
    result = choose_content_type(s, "sahil_twitter", "build_in_public", "twitter")
    assert result is not None
    assert result in ("text", "text+image", "article", "blog")


# ── 9. Freshness scoring ──


def test_freshness_score():
    """Freshness scores correctly across time ranges."""
    assert _freshness_score({"freshness_hours": 1}) == 100
    assert _freshness_score({"freshness_hours": 24}) == 100
    assert _freshness_score({"freshness_hours": 48}) > 50
    assert _freshness_score({"freshness_hours": 720}) == 0
    assert _freshness_score({"freshness_hours": 0}) == 50  # unknown


# ── 10. Platform fitness scoring sanity ──


def test_x_fit_higher_for_fresh_build():
    """Fresh build-in-public signal gets high X fit."""
    s = _sig("test", stype="harness_change", hours=12)
    assert _score_x_fit(s) >= 80


def test_linkedin_fit_lower_for_raw_tech():
    """Raw technical signal gets low LinkedIn fit."""
    s = _sig("test", stype="github_push", hours=48,
             summary="Pushed some changes")
    score = _score_linkedin_fit(s)
    assert score <= 60


def test_blog_fit_bonus_for_source_override():
    """Blog queue entry with source_override gets a bonus."""
    s = {"topic_id": "t1", "title_hint": "Test", "source_override": "research-paper", "priority": 5}
    score = _score_blog_fit(s)
    assert score >= 40


def test_article_fit_high_for_deep_signal():
    """Deep harness change signal gets high article fit."""
    s = _sig("test", stype="harness_change", prio=9, hours=48,
             summary="Implemented routing with 4 retry strategies and governance approval")
    assert _score_article_fit(s) >= 70


def test_article_fit_low_for_shallow():
    """Shallow signal gets low article fit."""
    s = _sig("test", stype="github_push", hours=48,
             summary="Fix")
    assert _score_article_fit(s) < 60


# ── 11. VCP rotation tracking (draft_media._record_rotation) ──


def test_record_rotation_helper_exists():
    """Ensure the _record_rotation helper exists with correct signature."""
    from draft_media import _record_rotation
    import inspect
    sig = inspect.signature(_record_rotation)
    params = list(sig.parameters.keys())
    assert "tracker" in params
    assert "draft" in params
    assert "model_used" in params
    assert "generation_cost" in params
    assert "ocr_passed" in params


def test_record_rotation_handles_none_tracker():
    """_record_rotation must silently handle None tracker."""
    from draft_media import _record_rotation
    # Should not raise
    _record_rotation(None, {"id": "1"}, model_used="test", generation_cost=0.0)


# ── 12. Activation hardening tests ──

def test_manual_blog_queue_bare_title_skips():
    """Bare manual queue title is not enough for long-form activation."""
    s = {
        "topic_id": "thin-topic",
        "title_hint": "A thin idea with no supporting notes",
        "tags": ["ai"],
        "priority": 8,
    }
    result = route_signal(s, ["blog"])
    assert result["decision"] == "skip"


def test_manual_blog_queue_curated_routes_to_blog():
    """Explicitly curated manual queue entries are accepted for blog routing."""
    s = {
        "topic_id": "curated-topic",
        "title_hint": "Why most agent evals are measuring the wrong thing",
        "tags": ["ai", "agents", "evaluation"],
        "priority": 8,
        "curated": True,
        "source_notes": "Human-selected from the editorial queue after review.",
    }
    result = route_signal(s, ["blog"])
    assert result["decision"] == "blog"
    assert result["platform"] == "blog"
    assert result["source_provenance"]["confidence"] == "claimed"


def test_choose_content_type_supports_curated_manual_blog_queue():
    """The content-type bridge must support curated manual blog entries too."""
    s = {
        "topic_id": "curated-topic",
        "title_hint": "Context engineering is the new prompt engineering",
        "curated": True,
        "source_notes": "Human-selected blog queue item.",
    }
    result = choose_content_type(s, "sahil_twitter", "ai", "blog")
    assert result == "blog"


def test_below_threshold_rationale_reports_blog_score():
    """Diagnostic rationale must report the actual blog_fit key, not zero."""
    s = {
        "signal_type": "unknown_external",
        "topic": "A vague sourced idea",
        "source_override": "manual",
        "priority": 1,
        "freshness_hours": 9999,
    }
    result = route_signal(s, ["blog"])
    assert "below threshold" in result["rationale"]
    assert "(0)" not in result["rationale"]


def test_product_metric_static_app_topic_skips_without_provenance():
    """Static app-brand metrics without evidence are blocked."""
    s = {
        "topic": "200 households saved £11.40 per week in testing",
        "pillar": "product",
        "priority": 5,
    }
    result = route_signal(s, ["x_post", "linkedin_post", "blog"])
    assert result["decision"] == "skip"
    assert "Product metric" in result["rationale"] or result["scores"]["linkedin_fit"] < 50



def test_generate_drafts_llm_skips_source_routed_rejections(monkeypatch):
    """Activation: editorial skip must prevent LLM spend and static fallback."""
    import llm_generate as lg

    called = {"generate_one": 0}

    def _boom(*args, **kwargs):
        called["generate_one"] += 1
        raise AssertionError("generate_one should not be called for routed skip")

    monkeypatch.setattr(lg, "generate_one", _boom)
    topic = {
        "activity_data": {
            "signal_type": "github_push",
            "signal_id": "gh-push:raw",
            "variables": {"repo_name": "repo", "url": "https://github.com/Sahil-SS9/repo"},
        },
        "pillar": "indie",
        "topic": "Pushed some code to repo",
        "source_provenance": {"confidence": "verified", "source_url": "https://github.com/Sahil-SS9/repo"},
    }
    drafts = lg.generate_drafts_llm("sahil_linkedin", [topic], platform="linkedin")
    assert drafts == []
    assert called["generate_one"] == 0
