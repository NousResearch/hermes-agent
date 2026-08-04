"""Tests for the article_pipeline orchestrator — end-to-end flow.

The pipeline:
  1. router.choose()  -> plan or None
  2. generator.write()  -> draft (or None on LLM dead)
  3. gates.check()  -> GateResult; if fail, retry once with feedback
  4. illustrator.illustrate()  -> illustrated body
  5. assembler.bundle()  -> ArticleBundle on disk
  6. database.insert_draft(content_type='article')  -> DB row
  7. discord_digest.post_article()  -> preview message

All external deps are stubbed; the tests assert the orchestration order
and the no-tech-debt wiring.
"""
import discord_digest as dd
from pathlib import Path
import article_pipeline as ap


def _stub_bundle(d, tmp_path):
    """Return a dd.ArticleBundle-shaped value for tests that stub assemble."""
    return dd.ArticleBundle(
        dir=tmp_path / "bundle", article_md=d["body_md"],
        article_md_path=tmp_path / "bundle" / "article.md",
        image_paths=[], title=d.get("title", "t"), lede=d.get("body_md", "")[:200],
        mode=d.get("mode", "deep_dive"), pillar=d.get("pillar", "h"),
    )


def test_pipeline_runs_router_then_generator_then_gates(monkeypatch, tmp_path):
    """Order: router -> generator -> gates. Short-circuit on router None."""
    calls = []

    def fake_router(state):
        calls.append("router")
        return None
    monkeypatch.setattr(ap, "router_choose", fake_router)
    monkeypatch.setattr(ap, "deliver_preview", lambda *a, **kw: calls.append("deliver"))
    out = ap.run(out_root=tmp_path, deliver=False)
    assert out["status"] in ("skipped_router", "skipped_disabled", "ok")
    assert "router" in calls
    # No draft generated.
    assert "deliver" not in calls


def test_pipeline_skips_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", False)
    calls = []
    monkeypatch.setattr(ap, "router_choose", lambda s: calls.append("r") or None)
    out = ap.run(out_root=tmp_path, deliver=False)
    assert out["status"] == "skipped_disabled"
    assert "r" not in calls


def test_pipeline_skips_when_router_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "router_choose", lambda s: None)
    out = ap.run(out_root=tmp_path, deliver=False)
    assert out["status"] == "skipped_router"


def test_pipeline_happy_path_inserts_draft_with_article_content_type(monkeypatch, tmp_path):
    """The orchestrator persists a drafts row with content_type='article' on success."""
    plan = {"mode": "deep_dive", "signals": [{"signal_id": "x", "signal_type": "harness_change",
            "priority": 8, "summary": "x", "repo": "KenseiAgent", "sha": "abc",
            "variables": {"summary": "x"}}], "pillar": "harness_tuning", "title_hint": "How I tuned routing"}
    draft = {"title": "t", "body_md": "long enough body", "mode": "deep_dive",
             "pillar": "h", "slug": "t", "signals": plan["signals"],
             "context": "ctx", "kb_snippets": []}
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "generate_draft", lambda plan, brand: draft)
    monkeypatch.setattr(ap, "gate_draft", lambda d: ("ok", []))
    monkeypatch.setattr(ap, "illustrate", lambda d, out_dir, **kw: d["body_md"])
    monkeypatch.setattr(ap, "assemble", lambda ill, d, out_root, dry_run: _stub_bundle(d, tmp_path))
    inserted = []
    monkeypatch.setattr(ap, "persist_article_draft",
                        lambda **kw: inserted.append(kw) or "draft-id-1")
    monkeypatch.setattr(ap, "deliver_preview", lambda b: None)
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda *a, **kw: None)
    out = ap.run(out_root=tmp_path, deliver=True)
    assert out["status"] == "ok"
    assert inserted, "expected persist_article_draft to be called"


def test_pipeline_secret_scan_blocks_persist_and_send(monkeypatch, tmp_path):
    """A planted sk- token is redacted before any DB write or Discord post."""
    plan = {"mode": "deep_dive", "signals": [{"signal_id": "x", "signal_type": "harness_change",
            "priority": 8, "summary": "x", "repo": "KenseiAgent", "sha": "abc",
            "variables": {"summary": "x"}}], "pillar": "harness_tuning", "title_hint": "x"}
    draft = {"title": "t", "body_md": "body", "mode": "deep_dive", "pillar": "h",
             "slug": "t", "signals": [], "context": "context with sk-abc...7890 secret",
             "kb_snippets": []}
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "generate_draft", lambda plan, brand: draft)
    sent = []
    monkeypatch.setattr(ap, "gate_draft", lambda d: ("ok", []))
    monkeypatch.setattr(ap, "illustrate", lambda d, out_dir, **kw: d["body_md"])
    monkeypatch.setattr(ap, "assemble",
                        lambda ill, d, out_root, dry_run: _stub_bundle(
                            {**d, "body_md": "body with sk-abc...7890 token"}, tmp_path))
    monkeypatch.setattr(ap, "persist_article_draft", lambda **kw: "draft-id")
    monkeypatch.setattr(ap, "deliver_preview", lambda b: sent.append(b))
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda *a, **kw: None)
    ap.run(out_root=tmp_path, deliver=True)
    # The body sent to Discord must have the redaction, not the raw token.
    assert sent, "expected the pipeline to send a bundle"
    sent_body = sent[0].article_md
    assert "sk-abc...7890" not in sent_body
    assert "***REDACTED***" in sent_body


def test_persist_article_draft_registers_separate_longform_approval(monkeypatch, tmp_path):
    """Article persistence writes both the legacy draft and durable approval row."""
    drafts = []
    approvals = []
    monkeypatch.setattr(ap.db, "insert_draft", lambda **kwargs: drafts.append(kwargs))
    monkeypatch.setattr(ap.db, "register_article_approval", lambda **kwargs: approvals.append(kwargs))
    bundle = tmp_path / "bundle"

    article_id = ap.persist_article_draft(
        id="article-1", brand="sahil_linkedin", platform="linkedin", pillar="h",
        title="T", body_md="body", bundle_path=bundle, slop_issues="",
    )

    assert article_id == "article-1"
    assert drafts[0]["content_type"] == "article"
    assert approvals == [{
        "article_id": "article-1", "bundle_path": bundle,
        "brand": "sahil_linkedin", "platform": "linkedin",
    }]


# ── Cross-run topic dedup ──

def _dedup_plan(sid="x"):
    sig = {"signal_id": sid, "signal_type": "harness_change", "priority": 8,
           "summary": f"summary {sid}", "repo": "KenseiAgent", "sha": "abc",
           "variables": {"summary": sid}}
    return {"mode": "deep_dive", "signals": [sig], "pillar": "h", "title_hint": "t"}


def test_recent_article_topics_seed_router_exclusion(monkeypatch, tmp_path):
    """run() seeds state['used'] from the persistent log so the router cannot
    re-pick a signal used as an article inside the recency window."""
    seen_state = {}

    def fake_router(state):
        seen_state["used"] = list(state.get("used", []))
        return None  # skip is fine; we only care about the seeded exclusion

    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: ["already-used"])
    monkeypatch.setattr(ap, "router_choose", fake_router)
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    ap.run(out_root=tmp_path, deliver=False)
    assert seen_state["used"] == ["already-used"]


def test_recent_article_topics_query_uses_article_brand(monkeypatch, tmp_path):
    """The recency read is keyed by the dedicated 'article' brand, not a platform."""
    captured = {}

    def fake_recent(brand, days=30):
        captured["brand"] = brand
        captured["days"] = days
        return []

    monkeypatch.setattr(ap.db, "get_recently_used_topics", fake_recent)
    monkeypatch.setattr(ap, "router_choose", lambda s: None)
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    ap.run(out_root=tmp_path, deliver=False)
    assert captured["brand"] == ap._ARTICLE_TOPIC_BRAND
    assert captured["brand"] == "article"


def test_topic_recorded_after_successful_run(monkeypatch, tmp_path):
    """A signal that becomes a published article is written back to the log."""
    plan = _dedup_plan("x")
    logged = []
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage",
                        lambda **kw: logged.append(kw))
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "_run_for_brand",
                        lambda p, brand, out_root, deliver: {"status": "ok", "brand": brand})
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    ap.run(out_root=tmp_path, deliver=False)
    assert logged, "expected the topic to be logged after success"
    assert logged[0]["topic_id"] == "x"
    assert logged[0]["brand"] == "article"


def test_topic_not_recorded_when_run_fails(monkeypatch, tmp_path):
    """A failed generation must not burn the topic; it should remain available."""
    plan = _dedup_plan("x")
    logged = []
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: logged.append(kw))
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "_run_for_brand",
                        lambda p, brand, out_root, deliver: {"status": "skipped_gate", "brand": brand})
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    ap.run(out_root=tmp_path, deliver=False)
    assert not logged, "a failed run must not record the topic as used"


def test_run_all_records_topic_once_when_any_brand_succeeds(monkeypatch, tmp_path):
    """run_all logs the topic once if at least one platform published."""
    plan = _dedup_plan("x")
    logged = []
    statuses = {"sahil_twitter": "ok", "sahil_linkedin": "skipped_gate"}
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: logged.append(kw))
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "_run_for_brand",
                        lambda p, brand, out_root, deliver: {"status": statuses[brand], "brand": brand})
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    out = ap.run_all(out_root=tmp_path, deliver=False)
    assert out["status"] == "ok"
    # One log call per signal in the plan, regardless of brand count.
    assert len(logged) == 1
    assert logged[0]["topic_id"] == "x"


def test_run_all_no_record_when_all_brands_fail(monkeypatch, tmp_path):
    plan = _dedup_plan("x")
    logged = []
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: logged.append(kw))
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "_run_for_brand",
                        lambda p, brand, out_root, deliver: {"status": "skipped_llm_dead", "brand": brand})
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    out = ap.run_all(out_root=tmp_path, deliver=False)
    assert out["status"] == "skipped_all"
    assert not logged
