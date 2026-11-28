# Scope, distinctions + connections

tiktok-video-publishing is a **format-execution skill** — it configures the TikTok VIDEO publish for reach
+ discovery + compliant disclosure **and** writes the search-optimized caption. The agent **configures the
publish + writes the caption**; the **video FILE is an input**; **WoopSocial publishes via `postType=VIDEO`.**

## Honest scope (never violate)
- **The agent configures the publish** (SEO caption, cover direction, reach settings, disclosure, mode) —
  the **video FILE is filmed by a human or made by the video skills** (`veo-3`/`luma`/`kling` + edited in
  CapCut). **WoopSocial doesn't film/edit/generate video.**
- **WoopSocial publishes via `postType=VIDEO`** (raw-bytes upload, server MIME detection) and
  **auto-discloses AI on TikTok.** **No update** → edit = delete + recreate (only while `NOT_STARTED`).
- **Native/human steps:** **pinned comment + replies** (no comment surface) and **licensed trending sound**
  (use `MEDIA_UPLOAD` to hand off to the TikTok inbox).
- **Disclosure truthful** (`isBrandedContent`/`isYourBrand`); **prohibited AI uses stay prohibited** (fake
  endorsements, unlabeled deepfakes of real people) even though WoopSocial auto-discloses AI; **branded
  content can't be private.**
- **Never promise the FYP / virality; never fabricate metrics** — native analytics only. **Verify-quarterly**
  (`references/video-publishing-2026-reality.md`).

## Distinct from its siblings
- **tiktok-video-publishing (this)** — the **VIDEO publish-config + search caption** layer (duet/stitch/
  cover apply; autoAddMusic ignored).
- **tiktok-script** — writes the **VIDEO SCRIPT** (the content/hook/beats); this skill **publishes** it.
- **tiktok-photo-mode** — the **PHOTO carousel** (`autoAddMusic` applies; duet/stitch/cover do **not**).
- **reels-script** / **youtube-shorts** — other **video** formats.
- **caption-writer** / **hook-writer** — generic copy; this skill owns the **TikTok-SEO caption + full config.**

## Where this connects
- **Reads first:** `brand-profile` (voice/niche), `goals-and-kpis` (reach/saves/shop), and the **script/
  video** (often from `tiktok-script`).
- **Content inputs (external to WoopSocial):** `tiktok-script` (the script), the **video** cluster
  (`veo-3`/`luma`/`kling`) + editing, `image-prompt`/`ideogram`/`nano-banana` (the cover frame),
  `hook-writer`/`caption-writer` (copy lines).
- **Publish:** `scheduling-and-queue → WoopSocial` (`postType=VIDEO`), `platform-specs-and-validation`
  (validate + the field rules), `content-calendar` (cadence/peak windows).
- **Engage/reuse/measure:** `community-management` (pinned comment + replies), `content-recycling`
  (→ Reels/Shorts), `analytics-and-reporting` (watch-time/search readout), `experimentation-and-ab-testing` (A/B caption/
  cover). **Filming/editing, trending sound, and comments stay external/native.**
