# Scope, distinctions + connections

tiktok-photo-mode is a **format-execution skill** — it produces a TikTok-native Photo Mode carousel
(craft for completion) **and** the exact WoopSocial PHOTO publish config. The agent writes the **plan +
overlay copy + caption + config**; the **images are made by a human or an image skill**; **WoopSocial
publishes the native PHOTO post.**

## Honest scope (never violate)
- **The agent writes the slide-by-slide plan, text-overlay copy, caption, and publish config.** The
  **images themselves are produced by a human or an image skill** (`image-prompt`/`ideogram`/`nano-banana`) —
  **WoopSocial does not generate images.**
- **WoopSocial publishes the native PHOTO post** (`postType=PHOTO`, raw-bytes upload, server-side MIME
  detection) and **auto-discloses AI-generated content on TikTok.** **No update endpoint** → edit = delete
  + recreate (only while delivery is `NOT_STARTED`).
- **Disclosure flags set truthfully** — `isBrandedContent` when a third party paid/gifted, `isYourBrand`
  when promoting your own brand. Never set them falsely to look organic.
- **Never promise virality / a "5×".** Photo mode is **high-ceiling but niche-dependent**, outnumbered ~5:1
  by video with declining average per-post performance. **Never fabricate metrics** — read native analytics.
- **Verify-quarterly** the format limits + algorithm weighting (`references/photo-mode-2026-reality.md`).

## Distinct from its siblings
- **tiktok-photo-mode (this)** — the **TikTok-native PHOTO carousel** (completion craft + the WoopSocial
  PHOTO publish config: privacyLevel, autoAddMusic, disclosure flags, postMode).
- **tiktok-script** — the **TikTok VIDEO script** (`postType=VIDEO`; duet/stitch/cover apply there, not here).
- **carousel-writer** — **cross-platform carousel copy craft** (IG/LinkedIn-leaning); this skill is the
  TikTok-native execution + publish layer.
- **reels-script** / **youtube-shorts** — **video** formats.
- **caption-writer** / **hook-writer** — supply the **caption + overlay lines** this skill arranges.

## Where this connects
- **Reads first:** `brand-profile` (niche/voice), `goals-and-kpis` (saves/comments/follows/reach).
- **Copy:** `hook-writer` (slide-1 hook), `caption-writer` (caption + CTA), `brand-profile` (voice).
- **Images (external to WoopSocial):** `image-prompt`/`ideogram`/`nano-banana` (the slides), `carousel-writer`
  (if adapting a cross-platform carousel down to 9:16).
- **Publish:** `scheduling-and-queue → WoopSocial` (native PHOTO), `platform-specs-and-validation`
  (validate-before-publish + the postType/field rules), `content-calendar`.
- **Reuse/measure:** `content-recycling` (adapt to IG carousel / Pinterest), `analytics-and-reporting`
  (native completion/saves readout), `experimentation-and-ab-testing` (A/B hooks/first slide). **Images stay external.**
