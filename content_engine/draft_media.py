"""Quality-first, budget-capped image generation for content drafts.

Degrading FAL chain, then a free fallback:
  1. krea_medium  (~£0.02/img, aesthetic detail)   ── primary
  2. z_image      (~£0.008/img, fast + detailed)
  3. flux_klein   (~£0.0048/img, fast)
  4. Pollinations (free, no key) via image_generator

Each tier is tried until one returns an image, so a rate-limit or outage on the
primary degrades quality rather than failing. Budget target: image + video
under £5-10/month (video defaults to free local ffmpeg). Set CONTENT_IMAGE_MODEL
to override the primary without a code change.

Gemini/nano-banana were dropped: the available Google keys have no working
quota (verified 2026-05-30). Output is a local PNG so the asset can be attached
to Discord for review and later published through Postiz.
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# Degrading quality/cost chain. The primary is overridable via env for tuning.
_PRIMARY = os.getenv("CONTENT_IMAGE_MODEL", "krea_medium").strip()
FAL_CHAIN: List[str] = [_PRIMARY] + [m for m in ("z_image", "flux_klein") if m != _PRIMARY]

# On-brand art direction so even an un-enriched prompt looks intentional.
# Brand palettes are the canonical ones from the project brief.
BRAND_STYLE = {
    "plenishd": "warm dark background #2C2A28, Plenishd yellow #FBBF24 accent, modern, appetising, clean sans-serif",
    "coachos": "editorial near-monochrome, restrained green accent, premium sports design, clean grid, high contrast",
    "matchdaymaestro": "bold matchday energy, stadium-night palette, punchy football graphic, high contrast",
    "sahil_twitter": "dark terminal aesthetic, monospace code overlay, build-in-public, minimal",
    "sahil_linkedin": "professional editorial graphic, clean typography, grey tones with a single accent, restrained",
}


def build_prompt(brand: str, body_text: str, visual_description: str = "") -> str:
    """Derive an on-brand image prompt. Prefers an explicit art-directed prompt."""
    style = BRAND_STYLE.get((brand or "").lower(), "modern, clean, high-contrast social media graphic")
    core = (visual_description or body_text or "social media visual").strip()
    return (
        f"Social media graphic for {brand}. {core[:240]}. "
        f"Style: {style}. No watermark, no gibberish text, professional, eye-catching."
    )


# Default illustrated base. krea-class beats z-image turbo decisively; the prompt
# engine (Track A) supplies the art direction. Overridable via env.
from model_config import BASE_MODEL  # noqa: E402


# Platform -> generation aspect. The base image must be born at the final
# aspect; cropping a square into 9:16 throws away ~60% of the composition.
_PLATFORM_ASPECT = {
    "instagram": "portrait_4_5",
    "tiktok": "portrait_9_16",
    "linkedin": "landscape",
}


def aspect_for(platform: str) -> str:
    return _PLATFORM_ASPECT.get((platform or "").lower(), "square")


def generate_draft_image(
    prompt: str,
    brand: str = "",
    platform: str = "",
    draft_id: str = "",
    model: Optional[str] = None,
    negative_prompt: str = "",
    aspect: Optional[str] = None,
) -> Optional[str]:
    """Generate a raw (textless) base image down the degrading chain.

    Returns a local PNG path or None. The caller passes a finished prompt
    (use ``build_scene_prompt``). ``model`` overrides the primary for one call.
    """
    primary = model or BASE_MODEL
    chain = [primary] + [m for m in FAL_CHAIN if m != primary]
    aspect = aspect or aspect_for(platform)

    # 1. FAL tiers, degrading on failure/rate-limit.
    for tier in chain:
        try:
            import fal_client

            path = fal_client.generate_image(
                prompt, model=tier, aspect=aspect, negative_prompt=negative_prompt
            )
            if path and os.path.exists(path):
                print(f"[draft_media] base generated via {tier}: {path}")
                return path
        except Exception as exc:  # noqa: BLE001 (provider failures must fall through)
            print(f"[draft_media] FAL {tier} failed: {exc}")

    # 2. Free fallback: Pollinations.
    try:
        from image_generator import _pollinations_generate

        img_bytes = _pollinations_generate(prompt)
        out_dir = OUTPUT_ROOT / (brand or "misc") / (platform or "x")
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"{draft_id or uuid.uuid4().hex[:8]}_base.png"
        fpath.write_bytes(img_bytes)
        print(f"[draft_media] pollinations (free) saved: {fpath} ({len(img_bytes)} bytes)")
        return str(fpath)
    except Exception as exc:  # noqa: BLE001
        # The pollinations error embeds the full prompt as a URL-encoded query
        # string (thousands of chars); truncate so it doesn't flood the log/digest.
        print(f"[draft_media] pollinations failed: {str(exc)[:160]}")

    return None


def _headline_from(draft: dict) -> str:
    """Pick a short, punchy headline from the draft for the image overlay."""
    title = (draft.get("title") or "").strip()
    if 8 <= len(title) <= 70:
        return title
    body = (draft.get("body_text") or "").strip()
    # first sentence / line, trimmed
    for sep in ("\n", ". ", "! ", "? "):
        if sep in body:
            body = body.split(sep)[0]
            break
    body = body.lstrip("#@ ").strip()
    return (body[:68].rsplit(" ", 1)[0] if len(body) > 68 else body) or (title or "")


def _verify(path, expected):
    """OCR gate. Two cases:

    - expected is non-empty: every expected string must be present in the
      OCR output (fuzzy match). Returns (ok, missing).
    - expected is empty (SCENE/HERO): the image must be textless. If OCR
      detects significant text, returns (False, ["<unexpected text>"]).

    In both cases, ``ok=False`` triggers a regen with a different model.
    """
    from text_integrity import verify_text, has_significant_text
    if not expected:
        # Textless type: reject if the model added text
        if has_significant_text(path):
            return False, ["<unexpected text in textless image>"]
        return True, []
    return verify_text(path, expected)


def _can_spend(cost):
    import budget
    return budget.can_spend(cost)


def _record_rotation(tracker, draft, model_used="", generation_cost=0.0, ocr_passed=1):
    """Record an image generation in the rotation tracker.

    Shared helper so both the legacy and transplant paths call the same
    recording logic. Silent (never raises) — rotation tracking is advisory.
    """
    if tracker is None:
        return
    try:
        editorial = draft.get("_editorial", {})
        tracker.record(
            article_id=draft.get("id", ""),
            brand=draft.get("brand", ""),
            target_surface=draft.get("platform", "social"),
            studio=editorial.get("studio", ""),
            intent=editorial.get("intent", ""),
            narrative=editorial.get("narrative", ""),
            layout=editorial.get("layout", ""),
            composition=editorial.get("composition", ""),
            palette=editorial.get("palette", ""),
            baoyu_type=editorial.get("baoyu_type", ""),
            baoyu_style=editorial.get("baoyu_style", ""),
            model_used=model_used,
            generation_cost=generation_cost,
            ocr_passed=ocr_passed,
        )
    except Exception as exc:  # noqa: BLE001 — rotation tracking is never a blocker
        print(f"[draft_media] rotation tracker record failed: {exc}")


def _native_aspect(aspect: str) -> str:
    """Map a prompt-engine aspect to the native seam's three-way aspect."""
    if aspect == "landscape":
        return "landscape"
    if str(aspect).startswith("portrait"):
        return "portrait"
    return "square"


def _variation_seed(draft: dict) -> int:
    """Deterministic per-post seed for the extended style-menu variation.

    Same post content → same seed → same style/blend (article-coherent).
    Different posts → different seed → deliberate cross-article variation.
    """
    import hashlib
    identity = str(draft.get("id") or draft.get("title") or draft.get("prompt") or "post")
    return int(hashlib.sha256(f"midlib-var:{identity}".encode()).hexdigest()[:8], 16)


def _resolve_extended_variation(draft: dict) -> Optional[dict]:
    """Return {style_slug, blend_slugs} from the extended menu, or None when the
    draft pins an explicit user style (existing styles stay unchanged).

    A user-pinned style (``draft["style"]``) wins and disables the extended
    menu. The editorial decision engine's auto-selected studio
    (``draft["_editorial"]["studio"]``) is NOT user-pinned: the extended menu
    applies additively on top of it (base style + neutral registry traits),
    which is the intended additive model for blog/X/LinkedIn auto-routed jobs.
    """
    if draft.get("style"):
        # User-pinned explicit style — extended menu never overrides it.
        return None
    try:
        from style_registry import pick_variation
        seed = _variation_seed(draft)
        return pick_variation(seed)
    except Exception as exc:  # noqa: BLE001 — registry is never a blocker
        print(f"[draft_media] extended variation unavailable: {exc}")
        return None


def _generate_native_codex(draft: dict, output_dir=None) -> Optional[str]:
    """Generate one image via the explicit native Codex seam (approval-only).

    Uses the existing prompt engine for the art-directed prompt and the
    editorial decision engine's studio for the style. Fail-closed: raises on
    any seam error so the caller returns None rather than falling back to FAL.
    """
    from image_jobs import prepare_image_request
    from image_job_service import execute_staged_image_job, stage_and_plan_image_job
    from prompt_engine import build_image_prompt

    prompt, aspect, _model_key, _expected = build_image_prompt(draft)
    studio = (draft.get("_editorial") or {}).get("studio") or "chromatic-institute"

    staging_root = Path(output_dir or OUTPUT_ROOT) / "image_staging"
    job_id = f"social-{uuid.uuid4().hex[:12]}"

    # Extended style menu: optional deterministic blend when the draft has no
    # explicit studio. Explicit styles keep working unchanged.
    variation = _resolve_extended_variation(draft)
    blend_slugs = (variation or {}).get("blend_slugs") or []
    registry_style = None
    if variation and not blend_slugs:
        registry_style = variation.get("style_slug")
    request = prepare_image_request(
        prompt=prompt,
        style=studio,
        backend="codex",
        blend=blend_slugs,
        registry_style=registry_style,
        registry_seed=_variation_seed(draft) if (blend_slugs or registry_style) else None,
    )
    staged = stage_and_plan_image_job(request, staging_root=staging_root, job_id=job_id)
    completed = execute_staged_image_job(
        request,
        staged,
        staging_root=staging_root,
        job_id=job_id,
        aspect_ratio=_native_aspect(aspect),
    )
    return str(completed.output_path)


def generate_post_image(draft, model=None, output_dir=None, scene_prompt=None):
    """Build a baoyu-method prompt, generate via Seedream, OCR-verify the baked-in
    text, regenerate (reseed -> nano-banana) on failure within budget. Returns a
    publish-ready PNG path with text already in the artwork (no overlay).

    Enriches the draft with editorial metadata from the decision engine and
    records successful generations in the rotation tracker.
    """
    import os

    # ── Editorial decision engine ──
    tracker = None
    try:
        from image_router import decide
        from rotation_tracker import RotationTracker

        tracker = RotationTracker()
        history = tracker.recent_history(draft.get("brand", ""), 10)
        editorial = decide(draft, recent_history=history)
        draft["_editorial"] = editorial
    except Exception as _exc:  # noqa: BLE001 — never let the decision engine break generation
        print(f"[draft_media] decision engine error: {_exc}; using defaults")
        draft["_editorial"] = {}

    # Personal brands (X/LinkedIn) use the explicit native Codex seam
    # (image_jobs + image_job_service), which is approval-only and fail-closed.
    # There is no FAL fallback: on any seam error we return None rather than
    # silently degrading to the legacy FAL chain.
    try:
        from config import NATIVE_CODEX_SOCIAL_BRANDS
        if draft.get("brand") in NATIVE_CODEX_SOCIAL_BRANDS:
            _p = _generate_native_codex(draft, output_dir=output_dir)
            if _p and os.path.exists(_p):
                _record_rotation(tracker, draft,
                                 model_used="openai-codex",
                                 generation_cost=0.0,
                                 ocr_passed=1)
                return _p
    except Exception as _exc:  # noqa: BLE001 (fail-closed: no FAL fallback)
        print(f"[draft_media] native codex path error: {_exc}; no FAL fallback")
        return None

    import budget
    from fal_client import MODEL_COST_GBP
    from prompt_engine import build_image_prompt

    draft_id = draft.get("id") or os.urandom(4).hex()
    prompt, aspect, model_key, expected = build_image_prompt(draft)
    chain = [model or model_key, "seedream45", "nano_banana"]
    seen, attempts = set(), []
    for m in chain:
        if m in seen:
            continue
        seen.add(m)
        cost = MODEL_COST_GBP.get(m, 0.032)
        if not _can_spend(cost):
            print(f"[draft_media] budget cap; skip {m}")
            continue
        path = generate_draft_image(
            prompt, brand=draft.get("brand", ""),
            platform=draft.get("platform", ""), draft_id=draft_id,
            model=m, negative_prompt="", aspect=aspect,
        )
        if not path or not os.path.exists(path):
            continue
        budget.record(cost, label=f"{m}:{draft_id}")
        ok, missing = _verify(path, expected)
        attempts.append((m, ok, missing))
        if ok:
            try:
                from database import update_draft_image_prompt
                update_draft_image_prompt(draft_id, prompt)
            except Exception as exc:
                print(f"[draft_media] prompt persist failed: {exc}")

            # ── Record in rotation tracker (shared helper) ──
            _record_rotation(tracker, draft,
                             model_used=m,
                             generation_cost=cost,
                             ocr_passed=1)

            print(f"[draft_media] image OK via {m}: {path}")
            return path
        print(f"[draft_media] {m} text mismatch {missing}; escalating")
    print(f"[draft_media] all attempts failed text gate: {attempts}; returning last path")
    return path if 'path' in locals() and path and os.path.exists(path) else None


def generate_draft_video(
    image_path: str, caption: str = "", brand: str = "", draft_id: str = "",
    hero: bool = False,
) -> Optional[str]:
    """Motion video for a draft. Degrading chain, best first:

      1. AI image-to-video (FAL seedance_fast, ~£0.09/5s) — budget-gated, only
         when ``hero`` is set or CONTENT_AI_VIDEO=1. Real motion from the
         branded image; this is the gold-standard tier.
      2. HyperFrames (£0) — GSAP-animated full-bleed overlay on the image.
      3. ffmpeg Ken Burns (£0) — full-bleed slow zoom, always succeeds.

    A silent static frame is no longer an acceptable output: every tier here
    produces motion, and the ffmpeg tier mixes in a music bed when one exists
    under output/music/.
    """
    if not image_path or not os.path.exists(image_path):
        print("[draft_media] video needs a base image; none available.")
        return None

    out_dir = OUTPUT_ROOT / (brand or "misc") / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / f"{draft_id or uuid.uuid4().hex[:8]}.mp4")

    # 1. Budget-gated AI motion (seedance image-to-video).
    if hero or os.getenv("CONTENT_AI_VIDEO", "").strip() == "1":
        try:
            import budget
            from ai_video_generator import MODEL_COST, generate_video as ai_generate

            model = os.getenv("CONTENT_AI_VIDEO_MODEL", "seedance_fast").strip()
            cost_gbp = round(5 * MODEL_COST.get(model, 0.05) * 0.8, 3)  # USD->GBP approx
            if budget.can_spend(cost_gbp):
                motion = (caption or "subtle cinematic motion").strip()[:200]
                path = ai_generate(
                    brand=brand, operation="image2vid",
                    prompt=f"Subtle cinematic camera motion, gentle parallax, atmosphere. {motion}",
                    image_path=image_path, model=model, duration=5,
                    aspect_ratio="9:16", filename=f"{draft_id or uuid.uuid4().hex[:8]}.mp4",
                )
                if path and os.path.exists(path):
                    budget.record(cost_gbp, label=f"video:{model}:{draft_id}")
                    print(f"[draft_media] AI video saved: {path}")
                    return path
            else:
                print(f"[draft_media] budget cap; skipping AI video ({model})")
        except Exception as exc:  # noqa: BLE001
            print(f"[draft_media] AI video failed: {exc}")

    # 2. HyperFrames animated overlay (free).
    try:
        from hyperframes_video import generate_screenshot_overlay_video

        path = generate_screenshot_overlay_video(
            brand, overlay_text=_headline_from({"title": caption, "body_text": caption}),
            screenshot_path=image_path, draft_id=draft_id,
        )
        if path and os.path.exists(path):
            print(f"[draft_media] hyperframes video saved: {path}")
            return path
    except Exception as exc:  # noqa: BLE001
        print(f"[draft_media] hyperframes failed: {exc}")

    # 3. ffmpeg Ken Burns fallback (free, always works).
    try:
        from video_generator import create_image_slideshow_video

        path = create_image_slideshow_video(
            image_paths=[image_path],
            captions=None,
            output_path=out,
            duration_per_image=6.0,
        )
        if path and os.path.exists(path):
            print(f"[draft_media] video saved: {path}")
            return path
    except Exception as exc:  # noqa: BLE001
        print(f"[draft_media] video generation failed: {exc}")
    return None
