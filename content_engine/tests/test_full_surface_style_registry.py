"""Tests for full-surface extended style-menu wiring (blog / X / LinkedIn / workflows).

Covers every modified caller:
  (a) style_registry.pick_variation — deterministic, varied, neutral
  (b) image_jobs.prepare_image_request — registry_seed propagation + manifest
  (c) draft_media._resolve_extended_variation + _generate_native_codex blend pass
  (d) blog_illustrator extended_traits injection (blog path)
  (e) art_director.compose_prompt merges extended neutral traits
  (f) image_backends.ImageBackendRouter.plan records registry provenance
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CE = _REPO / "content_engine"
for p in (str(_CE), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── (a) pick_variation ──────────────────────────────────────────────────────

def test_pick_variation_deterministic():
    from style_registry import pick_variation
    a = pick_variation(42)
    b = pick_variation(42)
    assert a == b
    assert a["style_slug"]


def test_pick_variation_varies_across_seeds():
    from style_registry import pick_variation
    seen = {pick_variation(s)["style_slug"] for s in range(20)}
    assert len(seen) > 1  # cross-post variation


def test_pick_variation_never_returns_protected_or_sref():
    from style_registry import pick_variation, resolve_fragment
    for s in range(50):
        v = pick_variation(s)
        frag = resolve_fragment(style_slug=v["style_slug"], blend_slugs=v["blend_slugs"], seed=s)
        assert frag
        assert "sref" not in frag.lower()


# ── (b) prepare_image_request seed propagation ─────────────────────────────

def test_prepare_image_request_records_seed():
    from image_jobs import prepare_image_request
    r = prepare_image_request(
        prompt="x", style="mythic-tech-codex",
        blend=["steampunk", "synthwave"], registry_seed=12345,
    )
    assert r.registry_seed == 12345
    assert r.registry_slugs == ("steampunk", "synthwave")
    assert r.registry_traits


def test_prepare_image_request_seed_default_none():
    from image_jobs import prepare_image_request
    r = prepare_image_request(prompt="x", style="mythic-tech-codex")
    assert r.registry_seed is None
    assert r.registry_traits is None


def test_prepare_image_request_same_seed_deterministic():
    from image_jobs import prepare_image_request
    a = prepare_image_request(prompt="x", style="mythic-tech-codex",
                              blend=["steampunk", "synthwave"], registry_seed=7)
    b = prepare_image_request(prompt="x", style="mythic-tech-codex",
                              blend=["steampunk", "synthwave"], registry_seed=7)
    assert a.registry_traits == b.registry_traits


# ── (c) draft_media variation + native path ────────────────────────────────

def _draft(**kw):
    d = {"id": "d1", "title": "Test post", "brand": "sahil_twitter", "platform": "twitter",
         "visual_description": "A scene", "body_md": "# T\n\nBody"}
    d.update(kw)
    return d


def test_draft_media_variation_seed_deterministic():
    import draft_media
    s1 = draft_media._variation_seed(_draft())
    s2 = draft_media._variation_seed(_draft())
    assert s1 == s2
    assert draft_media._variation_seed(_draft(id="other")) != s1


def test_draft_media_user_pinned_style_skips_variation():
    import draft_media
    # User-pinned style (draft["style"]) → extended menu disabled.
    d = _draft(style="saga-noir")
    assert draft_media._resolve_extended_variation(d) is None


def test_draft_media_editorial_studio_still_gets_variation():
    import draft_media
    # Auto-selected editorial studio (not user-pinned) → extended menu applies
    # additively on top of the studio.
    d = _draft(_editorial={"studio": "dark-cyberpunk-hud"})
    v = draft_media._resolve_extended_variation(d)
    assert v is not None
    assert v["style_slug"]


def test_draft_media_no_explicit_style_gets_variation():
    import draft_media
    d = _draft()  # no _editorial.studio
    v = draft_media._resolve_extended_variation(d)
    assert v is not None
    assert v["style_slug"]


def test_draft_media_native_path_passes_blend(monkeypatch):
    import draft_media
    captured = {}

    class FakeCompleted:
        output_path = Path("/tmp/out.png")

    def fake_prepare(**kw):
        captured.update(kw)
        from image_jobs import PreparedImageRequest
        return PreparedImageRequest(
            prompt=kw["prompt"], style_id="mythic-tech-codex", backend="codex",
            references=(), registry_traits="traits",
            registry_slugs=tuple(kw.get("blend") or ()),
            registry_seed=kw.get("registry_seed"),
        )

    def fake_stage(request, **kw):
        return object()

    def fake_execute(request, staged, **kw):
        return FakeCompleted()

    import image_jobs
    import image_job_service
    monkeypatch.setattr(image_jobs, "prepare_image_request", fake_prepare)
    monkeypatch.setattr(image_job_service, "stage_and_plan_image_job", fake_stage)
    monkeypatch.setattr(image_job_service, "execute_staged_image_job", fake_execute)
    # Force native path (brand in NATIVE_CODEX_SOCIAL_BRANDS) with no studio.
    d = _draft()
    d["_editorial"] = {}  # type: ignore[assignment]
    out = draft_media._generate_native_codex(d, output_dir="/tmp/gi-test")
    assert out == "/tmp/out.png"
    # blend may be empty or a pair; if pair, seed must be recorded
    if captured.get("blend"):
        assert captured["registry_seed"] is not None


# ── (d) blog_illustrator extended_traits ───────────────────────────────────

def test_blog_illustrator_injects_extended_traits():
    from blog.blog_illustrator import _resolve_extended_traits_for_brief

    brief = {"style": "editorial", "layout": "hero", "selection_seed": 77}
    out = _resolve_extended_traits_for_brief(brief)
    assert out is not None
    assert out["blend_seed"] == 77
    assert out["fragment"]
    assert "sref" not in out["fragment"].lower()


def test_blog_illustrator_preserves_explicit_extended_traits():
    from blog.blog_illustrator import _resolve_extended_traits_for_brief

    brief = {"style": "editorial", "extended_traits": {"fragment": "explicit"}}
    out = _resolve_extended_traits_for_brief(brief)
    assert out == {"fragment": "explicit"}


def test_art_director_compose_prompt_merges_extended_traits():
    from blog.art_director import compose_prompt
    brief = {
        "style": "editorial",
        "extended_traits": {"fragment": "linocut print, high contrast, neutral traits"},
    }
    out = compose_prompt("concept", brief)
    assert "Additional neutral style traits: linocut print, high contrast, neutral traits." in out


def test_art_director_compose_prompt_without_extended_unchanged():
    from blog.art_director import compose_prompt
    brief = {"style": "editorial"}
    out = compose_prompt("concept", brief)
    assert "Additional neutral style traits" not in out


# ── (f) manifest provenance ─────────────────────────────────────────────────

def test_router_plan_records_registry_provenance(tmp_path):
    from image_backends import ImageBackendRouter
    from image_jobs import prepare_image_request

    root = tmp_path / "staging"
    root.mkdir()
    job = root / "job1"
    job.mkdir()
    request = prepare_image_request(
        prompt="p", style="mythic-tech-codex",
        blend=["steampunk", "synthwave"], registry_seed=42,
    )
    router = ImageBackendRouter()
    plan = router.plan(request, [], job_dir=job, staging_root=root)
    import json
    manifest = json.loads(plan.manifest_path.read_text())
    assert manifest["registry_slugs"] == ["steampunk", "synthwave"]
    assert manifest["registry_seed"] == 42
    assert manifest["registry_traits"]
    assert "sref" not in json.dumps(manifest).lower()
    assert manifest["preview_only"] is True
