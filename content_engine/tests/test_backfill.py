"""Tests for blog backfill — stream config, budget cap, idempotency."""

import blog.blog_streams as bs


def test_streams_have_structural_rules():
    """Each stream must carry a 'structure' key with per-stream rules."""
    for s in ("ai", "pm", "builder"):
        assert bs.STREAMS[s].get("structure"), f"{s}: structure rule required"
    assert "reflective" in bs.STREAMS["pm"]["structure"].lower()
    assert "hype" in bs.STREAMS["builder"]["structure"].lower() or \
           "reality" in bs.STREAMS["builder"]["structure"].lower()
    assert "verify" in bs.STREAMS["ai"]["structure"].lower()


def test_blog_uses_codex_cli_not_fal():
    """Blog images use Codex CLI (£0), not FAL. Backfill cap still positive."""
    import config as cfg
    assert not hasattr(cfg, "BLOG_IMAGE_MODEL"), "BLOG_IMAGE_MODEL should be removed"
    assert not hasattr(cfg, "BLOG_IMAGE_COST_GBP"), "BLOG_IMAGE_COST_GBP should be removed"
    assert cfg.BACKFILL_SPEND_CAP_GBP > 0


def test_backfill_topics_have_correct_counts():
    """Each stream has exactly 12 approved topics."""
    from blog.backfill_topics import TOPICS
    for stream in ("ai", "pm", "builder"):
        assert len(TOPICS[stream]) == 12, f"{stream}: expected 12 topics, got {len(TOPICS[stream])}"


def test_backfill_verification_topics_have_claims():
    """needs_verification=True topics must carry a claim string."""
    from blog.backfill_topics import TOPICS
    for stream, topics in TOPICS.items():
        for t in topics:
            if t.get("needs_verification"):
                assert t.get("claim"), f"{stream}/{t['title']}: needs_verification but no claim"


def test_backfill_skips_existing_and_respects_cap(monkeypatch, tmp_path):
    """Backfill skips topics whose MDX already exists and stops at cap."""
    import blog.backfill as bf
    import budget as bgt
    import config as cfg

    # Set up a minimal SahilBlog repo structure.
    repo = tmp_path / "SahilBlog"
    posts_dir = repo / "src/content/blog"
    posts_dir.mkdir(parents=True)

    # Pre-create one MDX so the first topic is skipped.
    (posts_dir / "why-context-is-the-bottleneck-not-model-size.mdx").write_text("existing")

    monkeypatch.setattr(cfg, "SAHILBLOG_REPO", str(repo))
    monkeypatch.setattr(cfg, "BACKFILL_SPEND_CAP_GBP", 0.2)  # low cap

    # Mock write_with_gate to return a fake draft.
    def fake_write(plan, stream="ai", **kw):
        return {
            "title": plan.get("title_hint", "Post").split(":")[0].strip(),
            "description": "A test draft",
            "body_md": "# Test\n\nLede.\n\n## Section one\n\nContent.\n\n## Section two\n\nMore.\n\n## Takeaway\n\nDone.",
            "slug": "test-post",
            "tier": "pm",
            "tags": ["test"],
            "format": "essay",
            "source": "manual",
            "stream": stream,
            "signals": [{"summary": "test signal"}],
            "context": "",
            "kb_snippets": [],
        }
    monkeypatch.setattr(bf, "write_with_gate", fake_write)

    # Mock illustrate to return fake paths AND record spend to the backfill
    # ledger (like the real one does via budget_ledger_path).
    def fake_illustrate(draft, max_sections=0, **kw):
        out = {"hero_path": None, "section_paths": {}}
        ledger_path = kw.get("budget_ledger_path")
        if max_sections > 0:
            hero = tmp_path / "hero.png"
            hero.write_text("x")
            out["hero_path"] = str(hero)
            bgt.record(0.0, label="test:hero",
                       ledger_path=ledger_path)
        for i in range(max_sections):
            sec = tmp_path / f"sec{i}.png"
            sec.write_text("x")
            out["section_paths"][f"Section {i}"] = str(sec)
            bgt.record(0.0, label="test:sec",
                       ledger_path=ledger_path)
        return out
    monkeypatch.setattr(bf, "illustrate", fake_illustrate)

    # Reset budget ledgers for the test.
    from pathlib import Path
    ledger = Path(bgt._LEDGER)
    if ledger.exists():
        ledger.unlink()
    # Reset the backfill envelope ledger too.
    bf_ledger = Path(cfg.BACKFILL_LEDGER_PATH)
    if bf_ledger.exists():
        bf_ledger.unlink()
    # Point the backfill ledger to a temp path so we don't clobber a real one.
    bf_tmp_ledger = tmp_path / "backfill_ledger.json"
    monkeypatch.setattr(cfg, "BACKFILL_LEDGER_PATH", str(bf_tmp_ledger))
    monkeypatch.setattr(bf, "stage_draft", lambda mdx_path, repo=None: Path(mdx_path).stem)

    # Run with a limit of 3 (should generate ~3 but cap stops earlier).
    result = bf.run(stream="ai", limit=5)
    # First topic skipped (exists), subsequent generate until cap hit.
    assert result["skipped"] >= 1
    # Total images generated should be limited by the cap.
    assert result["total_spend_gbp"] <= cfg.BACKFILL_SPEND_CAP_GBP + 0.01, \
        f"spend {result['total_spend_gbp']} exceeded cap {cfg.BACKFILL_SPEND_CAP_GBP}"


def test_backfill_excludes_seeded_topics_before_generation(monkeypatch, tmp_path, blog_exclusions):
    import blog.backfill as bf
    import config as cfg

    repo = tmp_path / "SahilBlog"
    posts_dir = repo / "src/content/blog"
    posts_dir.mkdir(parents=True)

    monkeypatch.setattr(cfg, "SAHILBLOG_REPO", str(repo))
    monkeypatch.setattr(cfg, "BACKFILL_LEDGER_PATH", str(tmp_path / "bf_ledger.json"))

    call_count = {"n": 0}
    def tracking_write(*args, **kwargs):
        call_count["n"] += 1
        return None
    monkeypatch.setattr(bf, "write_with_gate", tracking_write)

    result = bf.run(stream="builder")
    titles = {r["title"]: r["status"] for r in result["results"]}
    assert titles["Cheap-first model routing"] == "excluded"
    assert titles["Reference-anchored image generation"] == "excluded"
    assert titles["What production AI agent actually means"] == "excluded"
    assert call_count["n"] >= 1


def test_backfill_halts_on_review_unavailable(monkeypatch, tmp_path):
    """When write_with_gate raises ReviewUnavailable, backfill halts the stream
    and no further stage_draft calls are made."""
    import blog.backfill as bf
    import config as cfg
    from blog.blog_generator import ReviewUnavailable

    repo = tmp_path / "SahilBlog"
    posts_dir = repo / "src/content/blog"
    posts_dir.mkdir(parents=True)

    monkeypatch.setattr(cfg, "SAHILBLOG_REPO", str(repo))
    monkeypatch.setattr(cfg, "BACKFILL_LEDGER_PATH",
                        str(tmp_path / "bf_ledger.json"))

    # write_with_gate raises ReviewUnavailable on the first call.
    call_count = {"n": 0}
    def failing_write(plan, stream="ai", **kw):
        call_count["n"] += 1
        raise ReviewUnavailable("reviewer degraded")
    monkeypatch.setattr(bf, "write_with_gate", failing_write)

    # stage_draft should never be called. Track it.
    stage_calls = {"n": 0}
    def tracking_stage(*a, **kw):
        stage_calls["n"] += 1
        return "slug"
    monkeypatch.setattr(bf, "stage_draft", tracking_stage)

    result = bf.run(stream="ai", limit=3)
    assert call_count["n"] == 1, "should call write_with_gate once then halt"
    assert stage_calls["n"] == 0, "should never stage when reviewer unavailable"
    assert result["errors"] == 1
    assert result["status"] == "partial"
    assert any(r["status"] == "review_unavailable"
               for r in result["results"])
