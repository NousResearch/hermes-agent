---
name: instagram-reels-publishing
description: >-
  The Instagram Reels publish-mechanics skill -- distinct from reels-script (the script), instagram-seo (the
  keyword caption), and platform-specs-and-validation (cross-platform validation). Use when someone has a
  finished Reel and needs to publish it correctly: specs, cover, safe zone, publish settings. It gets the
  Reel OUT the way Instagram displays it: 9:16 spec-clean video, the safe zone that survives the feed and
  profile-grid crops, a custom grid-safe cover (chosen before publish, can't edit after), the keyword
  caption, and audio. Uses the FRAME framework. Reads brand-profile + the Reel asset first. The agent
  specs the publish settings, cover brief, and specs/safe-zone checklist, and advises audio/tags.
  WoopSocial publishes postType REEL + video + caption + cover on a schedule; trending audio,
  Share-to-Feed, tags, and Reel Insights stay native/in-app; the video/cover come from other tools.
  Remove other-platform watermarks; metrics never fabricated.
version: 1.0.0
---

# instagram-reels-publishing

The **Reel publish-mechanics skill** — get a finished Reel **out** the way Instagram displays it (specs, safe
zone, cover, settings). The agent **specs the publish settings + the cover brief + the specs/safe-zone
checklist + advises audio/tags**; **WoopSocial publishes postType REEL + video + caption + cover**; **trending
audio, Share-to-Feed, tags, quality + Insights are native/in-app**, and the video/cover/burned-in captions
come from other tools.

## The POV: the Reel is made elsewhere — this gets it OUT spec-clean
Great Reels get squashed/cropped/blurred at the **publish step** — a sizing/settings problem, not a content
problem. Design for **how Instagram actually displays a Reel**: full **9:16** in the Reels tab, a **4:5** crop
in the feed, and the **3:4 (1080×1440)** profile-grid crop (the 2025 rectangular-grid change) — so keep critical content **centered**
and in the **safe zone.** The **cover** is chosen before publish and **can't be edited after.** WoopSocial
publishes **postType REEL + video + caption + cover**; the trending audio, Share-to-Feed, tags, and quality
settings are native/in-app, and the video/cover/captions come from other tools.

## Read these first
1. **brand-profile** — cover/caption style.
2. the **finished Reel asset** (+ topic).

## The framework: FRAME
(Depth: `references/the-frame-framework.md`.)
- **F — Frame it 9:16, spec-clean:** 1080×1920, MP4/MOV H.264 30/60fps ≥1080p ≤4GB (~15–20 Mbps); off 0.5625
  → crops/letterboxes; **remove other-platform watermarks.**
- **R — Reserve the safe zone:** top ~250px + bottom ~350px are UI; keep **hook text/face/CTA** in the central
  safe zone (survives the **4:5 feed + 3:4 grid crops**).
- **A — Art the cover for the grid:** choose before publishing; a **custom upload beats a video frame**; key
  elements + bold ≥60px text in the **3:4-grid-safe center**; **can't edit after upload** (→ `ideogram`/
  `nano-banana`).
- **M — Match caption + audio:** publish the **keyword caption** (← `instagram-seo`); audio beats silent +
  trending audio is pushed (native); the first-3-sec hook (← `reels-script`); burned-in captions in the safe
  zone (← `captions-and-clipping`).
- **E — Emit on schedule + mind what's native:** WoopSocial publishes **postType REEL + video + caption +
  cover**, scheduled + spec-validated (→ `platform-specs-and-validation`); **native/in-app:** trending audio,
  Share-to-Feed, collab/product tags, media-quality, Reel Insights.

## The reality (verify-quarterly)
9:16 1080×1920 canonical (off 0.5625 crops/letterboxes); MP4/MOV H.264 30/60fps ≥1080p ≤4GB ~15–20 Mbps;
length ~90s / 3 min in-app / ~15 min third-party; the 3 crop contexts (Reels tab 9:16, feed 4:5, **profile
grid 3:4 1080×1440** [2025 rectangular-grid change], link 1.91:1); safe zones (top ~250 / bottom ~350 UI;
central ~1320 safe; March 2026 unified FB+IG safe zone); cover (chosen before publish, frame or custom upload,
**can't edit after**, renders 9:16 feed + 3:4 grid, custom > frame, bold ≥60px center); audio (with-audio >
silent, trending pushed [native]); hook first 3s; burn captions (safe zone); remove watermarks; blurry fixes
(15–20 Mbps VBR, ~30-min processing, Rec.709, desaturate reds); media-quality native: `references/reels-
publishing-2026-reality.md`. The spec sheet + pre-publish checklist, the safe-zone map, the cover brief, the
publishable-vs-native table, the publish block + worked examples: `references/specs-cover-and-publish.md`.

## Honest scope (never violate)
- **The agent** specs the publish settings, writes the **cover brief**, runs the **specs/safe-zone checklist**,
  and **advises** trending-audio/Share-to-Feed/tags.
- **WoopSocial publishes** **postType: REEL + the 9:16 video + the caption (`content.text`) + the cover
  (REEL-only)** on a **schedule**, validated via `platform-specs-and-validation`.
- **WoopSocial does NOT:** select **trending audio** (IG's licensed library — not via API; native), toggle
  **Share-to-Feed** (native), add **collaborator/product tags** (native), trim the **video**, set
  **media-quality** (native), or read **Reel Insights** (native). It can't **render the video** (video tools/
  creator), **design the cover** (image tools), or **burn in captions** (`captions-and-clipping`).
- **9:16 1080×1920** or it crops/blurs; **content in the safe zone** (survives the 4:5 feed + 3:4 grid crops);
  **custom cover — can't edit after upload**; **remove other-platform watermarks**; **never fabricate reach.**
  (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**instagram-reels-publishing (this)** = the publish MECHANICS (specs/safe-zone/cover/settings) · **reels-
script** = the script · **instagram-seo** = the keyword caption (pairs — published here) · **platform-specs-
and-validation** = the cross-platform validation (this uses it) · **captions-and-clipping** = the burn-in ·
**tiktok-video-publishing**/**youtube-publishing-and-metadata** = the sibling publish skills · **thumbnail-
design** = YouTube thumbnails (the cover is the IG analogue) · **instagram-growth** = the strategy.

## Where this connects
Reads first: **brand-profile** + the **finished Reel asset.** Pulls from: **reels-script** (the hook/script),
**instagram-seo** (the keyword caption), the **video tools** (`veo-3`/`luma`/`kling`/`heygen` via
**ai-video**) for the clip, **ideogram**/**nano-banana** for the **cover**, **captions-and-clipping** for the
burned-in captions. Validates with: **platform-specs-and-validation** (9:16, ≤4GB, MP4/MOV). Publishes via:
**scheduling-and-queue → WoopSocial** (postType REEL + video + caption + cover). Measures with: native **Reel
Insights** + **analytics-and-reporting**. Trending audio, Share-to-Feed, tags, quality, Insights + asset
creation stay native/other-tools.

## Definition of done
A finished Reel published spec-clean: 9:16 at 1080×1920 (MP4/MOV H.264, ≤4GB, no other-platform watermark),
with the hook/face/CTA centered in the safe zone so it survives the 4:5 feed + 3:4 grid crops; a custom
grid-safe cover briefed to an image tool (bold ≥60px text in the 3:4 center, chosen before publish since it
can't be edited after); the keyword caption (from instagram-seo) and audio in place, with the hook routed to
reels-script and the burn-in to captions-and-clipping; postType REEL + video + caption + cover published via
WoopSocial on a schedule and validated via platform-specs-and-validation, while trending audio, Share-to-Feed,
collaborator/product tags, media-quality, and Insights stay native/in-app; nothing fabricated; correctly
distinguished from reels-script, instagram-seo, and platform-specs-and-validation.
