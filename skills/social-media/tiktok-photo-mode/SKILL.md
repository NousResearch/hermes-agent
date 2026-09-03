---
name: tiktok-photo-mode
description: >-
  The TikTok Photo Mode format-execution skill. Use when someone wants a "TikTok photo carousel,"
  "photo mode," "TikTok slideshow," "swipeable TikTok post," or to publish a multi-image TikTok post
  via WoopSocial (postType PHOTO). Builds a TikTok-native carousel that wins on completion rate —
  hook slide 1, progressive reveal with cliffhangers, reward + one CTA — AND sets the exact
  WoopSocial PHOTO publish config (privacy, comments, music, truthful disclosure flags). Uses the
  SWIPE framework. Reads brand-profile + goals-and-kpis first. The agent writes the slide plan +
  overlay copy + caption + config; the IMAGES are made by a human/image skill, not WoopSocial;
  WoopSocial publishes the native PHOTO post; metrics never fabricated and photo mode is never
  promised as a reach hack. Distinct from tiktok-script (VIDEO) and carousel-writer (cross-platform
  copy).
version: 1.0.0
---

# tiktok-photo-mode

A **format-execution skill** — produce a TikTok-native Photo Mode carousel (craft for completion) **and**
the exact WoopSocial PHOTO publish config. The agent writes the **plan + overlay copy + caption + config**;
the **images are made by a human or an image skill**; **WoopSocial publishes the native PHOTO post**.

## The POV: win completion, don't chase a reach hack
TikTok treats Photo Mode **like video** and ranks it on **completion rate** — what % swipe to slide 2+ and
through the end. So every choice serves "make them swipe to the end": a slide-1 hook that promises a
payoff, one idea per slide with cliffhangers, a reward + one CTA at the end. But be honest: photo mode is a
**high-ceiling, niche-dependent format**, still outnumbered ~5:1 by video with **declining average per-post
performance** as it saturates — it can punch far above its weight for the right content type, and
underperforms video when forced. **Never promise virality.** Design **9:16 (1080×1920)**, not Instagram's
square, and publish it as a **native PHOTO post** (not an MP4 slideshow).

## Read these first
1. **brand-profile** — niche + voice (does a swipeable format fit?).
2. **goals-and-kpis** — the goal (saves / comments / follows / reach).

## The framework: SWIPE
(Depth: `references/the-swipe-framework.md`.)
- **S — Stop the scroll on slide 1:** the thumbnail hook (bold visual + overlay question/claim) that
  promises a payoff.
- **W — Win the niche fit:** pick a swipeable type (listicle / before-after / tutorial / hot take / photo
  dump / story / myth-bust); don't force it — photo mode is niche-dependent.
- **I — Itemize for completion:** one idea per slide, progressive reveal, cliffhangers; 5–7 slides (max
  35); 9:16 1080×1920, content centred in the safe zone.
- **P — Pay off + prompt:** reward on the final slide + one CTA; question in the caption; don't beg for likes.
- **E — Equip the publish config:** `postType=PHOTO`, `privacyLevel` (from platform-inputs), `allowComment`,
  `autoAddMusic` (applies to PHOTO), truthful `isYourBrand`/`isBrandedContent`, `DIRECT_POST` vs
  `MEDIA_UPLOAD`; 1–2 hashtags + overlay keywords (TikTok SEO); **validate before publish.**

## The reality (verify-quarterly)
The honest performance tension (high-ceiling vs video-dominance + declining averages); completion rate as
the primary signal; 9:16 1080×1920 + safe zone; 5–7 slides; hook / progressive reveal / cliffhanger /
reward; sound (autoAddMusic) + text overlays as TikTok SEO; 1–2 hashtags; native-vs-MP4; winning content
types + engagement tactics: `references/photo-mode-2026-reality.md`. Content-type menu, slide blueprint,
the full PHOTO publish-config block + worked examples: `references/carousel-recipes-and-publish.md`.

## Honest scope (never violate)
- **The agent writes the slide plan + overlay copy + caption + publish config.** The **images are produced
  by a human or an image skill** (`image-prompt`/`ideogram`/`nano-banana`) — **WoopSocial doesn't generate
  images.**
- **WoopSocial publishes the native PHOTO post** (`postType=PHOTO`, raw-bytes upload) and **auto-discloses
  AI on TikTok.** **No update endpoint** → edit = delete + recreate (only while `NOT_STARTED`).
- **Disclosure flags set truthfully** (`isBrandedContent` when paid/gifted; `isYourBrand` for own-brand
  promo) — never falsely "organic." **Never promise virality / a "5×"; never fabricate metrics** (native
  analytics only). (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**tiktok-photo-mode (this)** = the TikTok-native PHOTO carousel (completion craft + the WoopSocial PHOTO
publish config) · **tiktok-script** = the TikTok VIDEO script (duet/stitch/cover live there) ·
**carousel-writer** = cross-platform carousel copy craft · **reels-script**/**youtube-shorts** = video ·
**caption-writer**/**hook-writer** = the caption + overlay lines this skill arranges.

## Where this connects
Reads first: **brand-profile**, **goals-and-kpis**. Copy: **hook-writer**, **caption-writer**. Images
(external to WoopSocial): **image-prompt**/**ideogram**/**nano-banana**, **carousel-writer** (adapt down to
9:16). Publish: **scheduling-and-queue → WoopSocial** (native PHOTO), **platform-specs-and-validation**
(validate + postType/field rules), **content-calendar**. Reuse/measure: **content-recycling** (→ IG/
Pinterest), **analytics-and-reporting** (completion/saves), **experimentation-and-ab-testing** (A/B the first slide).
Images stay external.

## Definition of done
A swipeable content type matched to the niche and goal; a slide-1 hook that promises a payoff; a 5–7 slide
progressive reveal with cliffhangers, one idea per slide, designed 9:16 1080×1920 with content in the safe
zone; a final-slide reward + one CTA and a caption question (not a like-beg); 1–2 hashtags + overlay
keywords for TikTok SEO; the complete WoopSocial PHOTO publish config (postType PHOTO, privacyLevel,
allowComment, autoAddMusic, truthful isYourBrand/isBrandedContent, postMode) validated before publish;
images produced externally, WoopSocial publishing the native PHOTO post with AI auto-disclosure, edits via
delete+recreate, nothing fabricated and no promised virality; correctly distinguished from tiktok-script
and carousel-writer.
