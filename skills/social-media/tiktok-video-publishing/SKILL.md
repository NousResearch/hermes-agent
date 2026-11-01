---
name: tiktok-video-publishing
description: >-
  The TikTok VIDEO publish-config skill. Use when someone wants to "publish/schedule a TikTok video,"
  "configure a TikTok upload," set the caption/cover/privacy/duet-stitch/disclosure, or post a TikTok video
  via WoopSocial (postType VIDEO). Configures the publish for reach, discovery, and compliant disclosure
  AND writes the search-optimized caption: the caption is search (lead-load the primary keyword), duet/
  stitch ON is free distribution, the cover frame wins the profile grid + search, and the disclosure flags
  (isBrandedContent/isYourBrand) are set truthfully. Uses the REACH framework. Reads brand-profile +
  goals-and-kpis + the script/video first. The VIDEO FILE is filmed by a human or made by the video skills
  (not WoopSocial); WoopSocial publishes via raw-bytes upload and auto-discloses AI on TikTok; trending
  sound / pinned comment / replies are native or human; no update endpoint (edit = delete+recreate);
  metrics never fabricated. Distinct from tiktok-script (the VIDEO SCRIPT) and tiktok-photo-mode (PHOTO).
version: 1.0.0
---

# tiktok-video-publishing

A **format-execution skill** — configure the TikTok VIDEO publish for reach + discovery + compliant
disclosure **and** write the search-optimized caption. The agent **configures the publish + writes the
caption**; the **video FILE is an input**; **WoopSocial publishes via `postType=VIDEO`** and auto-discloses
AI on TikTok.

## The POV: the publish config is the most controllable reach layer
The script gets filmed once, but the **publish config is set at upload and is the most controllable reach/
SEO layer.** Four levers: the **caption is search** (TikTok is a search engine; only the first ~100–150
chars show — lead with the primary keyword, not a teaser); **duet/stitch ON is free distribution** (others
remixing your video grows its engagement cluster); the **cover frame wins the profile grid + search**; and
**disclosure** (the `isBrandedContent`/`isYourBrand` flags) keeps you compliant. The **video file is an
input** — this skill makes it **land correctly and get found.**

## Read these first
1. **brand-profile** — voice/niche (and the search phrasing the audience uses).
2. **goals-and-kpis** — the goal (reach / saves / shop conversions).
3. the **script/video** (often from **tiktok-script**) — the topic to optimize the caption around.

## The framework: REACH
(Depth: `references/the-reach-framework.md`.)
- **R — Rank the caption:** lead-load the primary keyword (first ~100 chars) → value-prop → secondary
  keywords + niche hashtags; long-tail; mirror in on-screen text.
- **E — Engage-open settings:** `allowComment`/`allowDuet`/`allowStitch` **ON** for reach (restrict only
  for genuine brand-safety).
- **A — Attract with the cover:** a legible title-card `cover` image (the video's cover/thumbnail) that
  wins the profile grid + search.
- **C — Confirm disclosure:** `isBrandedContent`/`isYourBrand` **truthfully**, #ad up front; AI
  auto-disclosed on TikTok but **prohibited AI uses stay prohibited**; branded content can't be private.
- **H — Handle mode + specs + validate:** `privacyLevel` from platform-inputs; `DIRECT_POST` vs
  `MEDIA_UPLOAD` (trending sound); 1080×1920 9:16, ≤1GB, cap from `maxVideoPostDurationSec`; **validate**;
  no update → delete+recreate.

## The reality (verify-quarterly)
Caption-as-search (first ~100 chars, long-tail, Google surfacing, hashtag lift, pinned-comment = native);
duet/stitch/comments as reach levers; cover wins profile grid + search; privacy + disclosure (branded vs
your-brand, AI labels/prohibited uses, branded-can't-be-private); specs + length + DIRECT_POST vs
MEDIA_UPLOAD (trending sound): `references/video-publishing-2026-reality.md`. Caption-SEO formula,
reach-settings defaults, the full VIDEO publish-config block + worked examples:
`references/publish-config-and-recipes.md`.

## Honest scope (never violate)
- **The agent configures the publish + writes the caption.** The **video FILE is filmed by a human or made
  by the video skills** (`veo-3`/`luma`/`kling` + edited) — **WoopSocial doesn't film/edit/generate video.**
- **WoopSocial publishes via `postType=VIDEO`** (raw-bytes upload) and **auto-discloses AI on TikTok.**
  **No update** → edit = delete + recreate (only while `NOT_STARTED`).
- **Native/human:** **pinned comment + replies** (no comment surface) and **licensed trending sound** (use
  `MEDIA_UPLOAD`). **Disclosure truthful; prohibited AI uses stay prohibited; branded content can't be
  private.** **Never promise the FYP/virality; never fabricate metrics** (native analytics only). (Scope,
  distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**tiktok-video-publishing (this)** = the VIDEO publish-config + search caption (duet/stitch/cover apply;
autoAddMusic ignored) · **tiktok-script** = writes the VIDEO SCRIPT (the content) · **tiktok-photo-mode** =
the PHOTO carousel (autoAddMusic applies; duet/stitch/cover don't) · **reels-script**/**youtube-shorts** =
other video formats · **caption-writer**/**hook-writer** = generic copy.

## Where this connects
Reads first: **brand-profile**, **goals-and-kpis**, the **script/video** (**tiktok-script**). Content
inputs (external to WoopSocial): **tiktok-script**, the **video** cluster (**veo-3**/**luma**/**kling**) +
editing, **image-prompt**/**ideogram**/**nano-banana** (cover frame), **hook-writer**/**caption-writer**.
Publish: **scheduling-and-queue → WoopSocial** (`postType=VIDEO`), **platform-specs-and-validation**
(validate + field rules), **content-calendar**. Engage/reuse/measure: **community-management** (pinned
comment + replies), **content-recycling** (→ Reels/Shorts), **analytics-and-reporting**, **experimentation-and-ab-testing**
(A/B caption/cover). Filming/editing, trending sound, and comments stay external/native.

## Definition of done
A search-optimized caption (primary keyword lead-loaded, value-prop, secondary keywords + niche hashtags,
mirrored in on-screen text); reach settings open (comment/duet/stitch ON unless brand-safety); a legible
title-card cover frame; truthful disclosure flags with #ad up front (and AI auto-disclosed, prohibited AI
uses excluded, branded never private); the full WoopSocial VIDEO config (postType VIDEO, privacyLevel from
platform-inputs, postMode chosen, specs right) validated before publish; the video file produced
externally, WoopSocial publishing postType VIDEO with AI auto-disclosure, edits via delete+recreate,
trending sound/pinned comment/replies left native, nothing fabricated and no promised FYP; correctly
distinguished from tiktok-script and tiktok-photo-mode.
