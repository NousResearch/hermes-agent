"""Tests for the blocking image/brand wiring remediation.

Final review blocker: candidate image work was mis-wired. SahilBlog path is
blog/blog_pipeline.py -> blog/blog_illustrator.py -> blog/art_director.py.
Candidate changes in article_illustrator.py + prompt_engine.py affect X
Articles only. SahilBlog does not import prompt_engine.

These tests prove the three post-remediation invariants:

  (R1) SahilBlog uses the real blog art path with varied assigned layouts
       (one visual family per article + distinct explicit hero/section
       layouts via art_director.compose_prompt). No X-article pipeline /
       prompt_engine import is needed for SahilBlog image variety.

  (R2) article_pipeline._run_for_brand threads the actual requested brand
       into the illustration draft, and article_illustrator does NOT
       default X articles to "sahilblog". sahil_twitter and sahil_linkedin
       remain distinct (different BRAND_STYLE_MAP entries + different
       thread-through to the illustration draft).

  (R3) The "sahilblog" BRAND_STYLE_MAP entry is removed from prompt_engine
       and the default fallback no longer rebrands unknown brands to
       sahilblog. The remaining image-variety changes (MJ registry
       treatment + position preset rotation) are still consumed by the X
       article path and continue to work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import article_generator as gen_mod
import article_illustrator as ai
import article_pipeline as ap
from prompt_engine import BRAND_STYLE_MAP, _DEFAULT, build_image_prompt


# ── Shared stubs ─────────────────────────────────────────────────────────


def _stub_plan():
    sig = {
        "signal_id": "harness:KenseiAgent:abc",
        "signal_type": "harness_change", "priority": 8,
        "summary": "tune model routing", "repo": "KenseiAgent", "sha": "abc",
        "variables": {"summary": "tune model routing"},
    }
    return {
        "mode": "deep_dive", "signals": [sig], "pillar": "harness_tuning",
        "title_hint": "How I tuned routing",
    }


def _stub_draft():
    return {
        "title": "How I tuned routing", "body_md": "long enough body",
        "mode": "deep_dive", "pillar": "harness_tuning", "slug": "t",
        "signals": [_stub_plan()["signals"][0]], "context": "ctx",
        "kb_snippets": [],
    }


# ── R1: SahilBlog uses the real blog art path with varied assigned layouts


def _blog_draft(tmp_path: Path):
    return {
        "title": "Token-Maxing at the Edge",
        "description": "A counterintuitive claim about edges and inference.",
        "body_md": (
            "# Token-Maxing at the Edge\n\nA counterintuitive claim.\n\n"
            "## The mechanism\n\nThe numbers tell a story.\n\n"
            "## Worked example\n\nHere is the code.\n\n"
            "## What I'd try next\n\nThe takeaway.\n"
        ),
        "stream": "ai",
    }


@pytest.fixture
def blog_isolated(monkeypatch, tmp_path):
    """Stub the blog path: deterministic brief + no Codex subprocess + isolated
    P11 reference pack so tests never touch the LLM or the provider."""
    import blog.blog_illustrator as bi
    import config

    monkeypatch.setattr(bi, "ROTATION_STATE_PATH", tmp_path / "skill_rotation.json")

    root = tmp_path / "refs"
    root.mkdir()
    rows, core_rows = [], []
    for reference_id, role in (
        ("layout-fixture", "layout"),
        ("style-fixture", "style"),
        ("composition-fixture", "composition"),
    ):
        content = reference_id.encode("utf-8")
        rel = f"{reference_id}.png"
        (root / rel).write_bytes(content)
        row = {
            "record_schema_version": "2", "reference_id": reference_id,
            "path": rel, "sha256": hashlib.sha256(content).hexdigest(),
            "provenance_class": "sahil_curated",
            "ownership_or_usage_basis": "test fixture",
            "usage_classification": "review-required",
            "allowed_roles": [role], "parent_reference_id": None,
        }
        rows.append(row)
        core_rows.append({**row, "core_role": role, "core_tag": "fixture",
                          "curation_status": "visually-reviewed-core-candidate-test",
                          "blocked_roles": ["generation", "publication"],
                          "visual_rationale": "isolated test reference"})
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (root / "core-pack.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in core_rows), encoding="utf-8")
    monkeypatch.setattr(config, "IMAGERY_ANCHORS_DIR", str(root))
    return bi


def test_blog_path_does_not_import_prompt_engine_or_article_illustrator():
    """SahilBlog variety must flow live through blog/blog_illustrator +
    blog/art_director, NOT through the X-article pipeline."""
    import inspect
    import blog.blog_illustrator as bi

    src = inspect.getsource(bi)
    import_lines = [
        l.strip() for l in src.splitlines()
        if l.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    assert "prompt_engine" not in joined, (
        "SahilBlog must not import the X-article prompt pipeline")
    assert "article_illustrator" not in joined, (
        "SahilBlog must not import the X-article illustrator")


def test_blog_path_uses_compose_prompt_with_distinct_assigned_layouts_per_asset(
    blog_isolated, monkeypatch, tmp_path,
):
    """bi.illustrate must call art_director.compose_prompt with a distinct
    assigned_layout per asset (hero vs each section) while sharing one
    visual family across the whole article."""
    bi = blog_isolated

    # Brief that mimics what art_director._validate produces from a real
    # concept plan: one style, one palette, one motif, but a per-scene
    # composition for each asset key.
    def fake_brief(draft, headings, recent_styles=None,
                   recent_concept_fingerprints=None, llm=None):
        return {
            "style": "technical-diorama",
            "selection_seed": "abc123def456789",
            "palette": "stone grey, brass, warm amber",
            "motif": "a recurring archway",
            "art_direction": "vast, awe-of-scale, one warm focal light.",
            "layout": "architectural cross-section",
            "layout_variants": ["control hall", "vault map"],
            "asset_layouts": {
                "hero": "wide proscenium collision",
                "section-01": "side-on process cutaway",
                "section-02": "miniature factory overview",
            },
            "text_policy": "labels",
            "hero_prompt": "hero for Token-Maxing at the Edge",
            "section_prompts": {h: f"section image for {h}" for h in headings},
        }
    monkeypatch.setattr(bi, "build_art_brief", fake_brief)

    captured: list[str] = []
    def fake_generate(prompt, out_path, **kw):
        captured.append(prompt)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("png", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)

    bi.illustrate(_blog_draft(tmp_path), out_dir=tmp_path, max_sections=2)

    assert len(captured) == 3, "expected hero + 2 section prompts"
    hero, s1, s2 = captured

    # ONE visual family per article: every prompt names the same style.
    assert all("Technical Diorama" in p for p in captured)
    assert all("stone grey, brass, warm amber" in p for p in captured)
    assert all("a recurring archway" in p for p in captured)

    # DISTINCT explicit hero/section layouts: each assigned layout appears
    # ONLY in its own asset's prompt, never in the others.
    assert "wide proscenium collision" in hero
    assert "wide proscenium collision" not in s1
    assert "wide proscenium collision" not in s2

    assert "side-on process cutaway" in s1
    assert "side-on process cutaway" not in hero
    assert "side-on process cutaway" not in s2

    assert "miniature factory overview" in s2
    assert "miniature factory overview" not in hero
    assert "miniature factory overview" not in s1


# ── R2: article_pipeline threads the requested brand into the illustration


def test_run_for_brand_threads_sahil_twitter_brand_to_illustration(monkeypatch, tmp_path):
    """For an X Twitter article, the illustration draft must carry
    brand='sahil_twitter' (NOT 'sahilblog')."""
    plan = _stub_plan()
    draft = _stub_draft()
    captured: list[dict] = []

    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "generate_draft", lambda plan, brand: draft)
    monkeypatch.setattr(ap, "gate_draft", lambda d: ("ok", []))
    monkeypatch.setattr(ap, "illustrate", lambda d, out_dir, **kw: (
        captured.append(d) or d["body_md"]))
    monkeypatch.setattr(ap, "assemble",
                        lambda ill, d, out_root, dry_run: type(
                            "B", (), {"dir": tmp_path / "b",
                                      "article_md": d["body_md"],
                                      "article_md_path": tmp_path / "b" / "a.md",
                                      "image_paths": [], "title": d.get("title", "t"),
                                      "lede": "", "mode": "deep_dive",
                                      "pillar": "h"})())
    monkeypatch.setattr(ap, "persist_article_draft", lambda **kw: "draft-id")
    monkeypatch.setattr(ap, "deliver_preview", lambda b: None)
    monkeypatch.setattr(ap, "_record_article_topics", lambda p: None)
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: None)

    ap._run_for_brand(plan, "sahil_twitter", tmp_path, deliver=False)

    assert captured, "illustrate() was not invoked"
    assert captured[0].get("brand") == "sahil_twitter", (
        f"X Twitter article got brand={captured[0].get('brand')!r}; "
        "article_pipeline did not thread the requested brand into the "
        "illustration draft, so the X path silently fell back to the "
        "default (which was previously 'sahilblog').")
    assert captured[0].get("platform") == "twitter", (
        f"X Twitter article got platform={captured[0].get('platform')!r}; "
        "article_pipeline did not thread the platform.")


def test_run_for_brand_threads_sahil_linkedin_brand_and_platform(monkeypatch, tmp_path):
    """For an X LinkedIn article, the illustration draft must carry
    brand='sahil_linkedin' and platform='linkedin' — distinct from Twitter."""
    plan = _stub_plan()
    draft = _stub_draft()
    captured: list[dict] = []

    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "generate_draft", lambda plan, brand: draft)
    monkeypatch.setattr(ap, "gate_draft", lambda d: ("ok", []))
    monkeypatch.setattr(ap, "illustrate", lambda d, out_dir, **kw: (
        captured.append(d) or d["body_md"]))
    monkeypatch.setattr(ap, "assemble",
                        lambda ill, d, out_root, dry_run: type(
                            "B", (), {"dir": tmp_path / "b",
                                      "article_md": d["body_md"],
                                      "article_md_path": tmp_path / "b" / "a.md",
                                      "image_paths": [], "title": d.get("title", "t"),
                                      "lede": "", "mode": "deep_dive",
                                      "pillar": "h"})())
    monkeypatch.setattr(ap, "persist_article_draft", lambda **kw: "draft-id")
    monkeypatch.setattr(ap, "deliver_preview", lambda b: None)
    monkeypatch.setattr(ap, "_record_article_topics", lambda p: None)
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: None)

    ap._run_for_brand(plan, "sahil_linkedin", tmp_path, deliver=False)

    assert captured, "illustrate() was not invoked"
    assert captured[0].get("brand") == "sahil_linkedin", (
        f"X LinkedIn article got brand={captured[0].get('brand')!r}; "
        "article_pipeline did not thread the requested brand into the "
        "illustration draft.")
    assert captured[0].get("platform") == "linkedin", (
        f"X LinkedIn article got platform={captured[0].get('platform')!r}; "
        "article_pipeline did not thread the platform, so LinkedIn would "
        "lose its distinct (landscape) aspect.")


def test_run_for_brand_threads_brand_on_retry_path_too(monkeypatch, tmp_path):
    """When the gate fails the first draft and the pipeline rebuilds the
    article from retry feedback, the rebuild must STILL carry brand and
    platform into the illustration draft."""
    plan = _stub_plan()
    first_draft = _stub_draft()
    # Mark the first draft as gate-fail.
    gate_calls = {"n": 0}
    def gate(d):
        gate_calls["n"] += 1
        return ("ok", []) if gate_calls["n"] > 1 else ("fail", ["too short"])
    captured: list[dict] = []
    monkeypatch.setattr(ap, "ARTICLE_ENABLED", True)
    monkeypatch.setattr(ap, "router_choose", lambda s: plan)
    monkeypatch.setattr(ap, "generate_draft", lambda plan, brand: first_draft)
    monkeypatch.setattr(ap, "gate_draft", gate)
    monkeypatch.setattr(ap, "illustrate", lambda d, out_dir, **kw: (
        captured.append(d) or d["body_md"]))
    monkeypatch.setattr(ap, "assemble",
                        lambda ill, d, out_root, dry_run: type(
                            "B", (), {"dir": tmp_path / "b",
                                      "article_md": d["body_md"],
                                      "article_md_path": tmp_path / "b" / "a.md",
                                      "image_paths": [], "title": d.get("title", "t"),
                                      "lede": "", "mode": "deep_dive",
                                      "pillar": "h"})())
    monkeypatch.setattr(ap, "persist_article_draft", lambda **kw: "draft-id")
    monkeypatch.setattr(ap, "deliver_preview", lambda b: None)
    monkeypatch.setattr(ap, "_record_article_topics", lambda p: None)
    monkeypatch.setattr(ap.db, "get_recently_used_topics", lambda *a, **kw: [])
    monkeypatch.setattr(ap.db, "log_topic_usage", lambda **kw: None)

    # Force the retry path to return a body so we go past the gate retry.
    monkeypatch.setattr(gen_mod, "_call_llm_first",
                        lambda sys, user: "# Rebuilt\n\nlong enough body")
    monkeypatch.setattr(gen_mod, "_extract_title",
                        lambda body: "Rebuilt")

    ap._run_for_brand(plan, "sahil_linkedin", tmp_path, deliver=False)

    assert captured, "illustrate() was not invoked"
    assert captured[0].get("brand") == "sahil_linkedin"
    assert captured[0].get("platform") == "linkedin"


def test_article_illustrator_does_not_default_to_sahilblog_for_x_articles(
    monkeypatch, tmp_path,
):
    """article_illustrator.illustrate must never inject brand='sahilblog'
    for an X article when the draft has no brand key — SahilBlog is NOT
    an X-article caller, so the X path must default to its own brand."""
    from pathlib import Path
    import article_illustrator as ai

    out = tmp_path / "out"
    # Capture the brand seen by each per-image draft.
    seen_brands: list[str] = []
    seen_platforms: list[str] = []
    def fake_generate(draft, **kwargs):
        seen_brands.append(draft.get("brand"))
        seen_platforms.append(draft.get("platform"))
        return "/tmp/sentinel.png"
    monkeypatch.setattr(ai, "generate_post_image", fake_generate)
    monkeypatch.setattr(ai, "verify_text", lambda path, expected: (True, []))
    monkeypatch.setattr(ai, "budget_can_spend", lambda cost: True)

    draft = _stub_draft()  # NO 'brand' key on the draft.
    ai.illustrate(draft, out_dir=out, density="hero-only", max_images=2)

    assert seen_brands, "expected at least one image-draft call"
    for brand in seen_brands:
        assert brand != "sahilblog", (
            f"X-article image draft was rebranded to {brand!r}; "
            "article_illustrator must NOT default X articles to "
            "'sahilblog' (SahilBlog is not a caller of this module).")
    # When the pipeline does not thread a brand, the safe X default is
    # 'sahil_twitter'.
    for brand in seen_brands:
        assert brand == "sahil_twitter", (
            f"X article default brand should be 'sahil_twitter', "
            f"got {brand!r}")


def test_article_illustrator_threads_real_brand_into_every_image_draft(
    monkeypatch, tmp_path,
):
    """When the pipeline threads brand='sahil_linkedin', every per-image
    draft must carry it (not silently collapse to twitter or sahilblog)."""
    import article_illustrator as ai

    out = tmp_path / "out"
    seen_brands: list[str] = []
    def fake_generate(draft, **kwargs):
        seen_brands.append(draft.get("brand"))
        return "/tmp/sentinel.png"
    monkeypatch.setattr(ai, "generate_post_image", fake_generate)
    monkeypatch.setattr(ai, "verify_text", lambda path, expected: (True, []))
    monkeypatch.setattr(ai, "budget_can_spend", lambda cost: True)

    draft = {**_stub_draft(), "brand": "sahil_linkedin", "platform": "linkedin"}
    ai.illustrate(draft, out_dir=out, density="per-section", max_images=6)

    assert seen_brands, "expected at least one image-draft call"
    for brand in seen_brands:
        assert brand == "sahil_linkedin", (
            f"per-image draft brand should track the threaded brand; "
            f"got {brand!r}")


def test_sahil_twitter_and_sahil_linkedin_image_prompts_remain_distinct():
    """The prompt_engine path: brand-spec for twitter vs linkedin must
    remain distinct (different scene preset, different data preset,
    different brand ref). prompt_engine must NEVER collapse an X-article
    request onto the removed 'sahilblog' entry."""
    draft_t = {"brand": "sahil_twitter", "platform": "twitter",
               "title": "X title", "body_text": "body",
               "pillar": "harness_tuning"}
    draft_l = {"brand": "sahil_linkedin", "platform": "linkedin",
               "title": "X title", "body_text": "body",
               "pillar": "harness_tuning"}

    spec_t = BRAND_STYLE_MAP["sahil_twitter"]
    spec_l = BRAND_STYLE_MAP["sahil_linkedin"]
    assert spec_t["scene"] != spec_l["scene"], (
        "twitter and linkedin scene presets collapsed")
    assert spec_t["ref"] != spec_l["ref"], (
        "twitter and linkedin brand refs collapsed")

    prompt_t, _a_t, _m_t, _e_t = build_image_prompt(draft_t)
    prompt_l, _a_l, _m_l, _e_l = build_image_prompt(draft_l)
    assert spec_t["ref"] in prompt_t
    assert spec_l["ref"] in prompt_l
    assert spec_l["ref"] not in prompt_t
    assert spec_t["ref"] not in prompt_l

    # Aspect ratio must reflect the threaded platform.
    _, aspect_t, _, _ = build_image_prompt(draft_t)
    _, aspect_l, _, _ = build_image_prompt(draft_l)
    assert aspect_t == "square", f"twitter should be square, got {aspect_t!r}"
    assert aspect_l == "landscape", f"linkedin should be landscape, got {aspect_l!r}"


# ── R3: prompt_engine must not redefine 'sahilblog' and must not default
# unknown brands to the 'sahilblog' entry.


def test_prompt_engine_brand_map_does_not_contain_sahilblog():
    """The sahilblog entry in prompt_engine.BRAND_STYLE_MAP is dead code:
    SahilBlog is generated by blog/blog_illustrator + blog/art_director
    (Codex CLI), which never imports prompt_engine. Keeping it both
    misleads readers into believing prompt_engine is the SahilBlog path
    AND — via _DEFAULT — silently rebrands any draft without a brand."""
    assert "sahilblog" not in BRAND_STYLE_MAP, (
        "prompt_engine.BRAND_STYLE_MAP still carries a 'sahilblog' "
        "entry; the real SahilBlog path is blog/art_director and never "
        "reads prompt_engine.")


def test_prompt_engine_default_brand_is_sahil_twitter_not_sahilblog():
    """An X-article draft with brand='unknown' must fall back to the
    sahil_twitter spec (the canonical X default), not silently
    rebrand to sahilblog."""
    spec_unknown = BRAND_STYLE_MAP.get("unknown", _DEFAULT)
    spec_twitter = BRAND_STYLE_MAP["sahil_twitter"]
    assert spec_unknown is spec_twitter, (
        "prompt_engine _DEFAULT now silently picks the (removed) "
        "sahilblog spec for unknown brands; X articles without a "
        "brand get rebrand pollution.")


def test_prompt_engine_mj_style_treatment_still_flows_for_x_articles():
    """The MJ-style treatment variation is a broadly-valid X-article
    variety feature that IS correctly consumed by build_image_prompt.
    Removing the mis-wired SahilBlog bits must NOT delete it."""
    import prompt_engine as pe
    draft = {"brand": "sahil_twitter", "platform": "twitter",
             "title": "X title", "body_text": "body",
             "pillar": "harness_tuning", "slug": "x-title", "id": "x1"}
    prompt, *_ = build_image_prompt(draft)
    # Either registry cache is populated and adds a TREATMENT VARIATION
    # line, or the registry file is missing and the call degrades cleanly.
    if pe._MJ_REGISTRY_CACHE is not None:
        assert "TREATMENT VARIATION" in prompt, (
            "MJ treatment variation is wired in _prompt_parts but no "
            "longer appears in the built prompt")
    else:
        # Registry unavailable — ensure the call still returns a valid
        # prompt (degrades to no MJ line, not a crash).
        assert "TYPE:" in prompt