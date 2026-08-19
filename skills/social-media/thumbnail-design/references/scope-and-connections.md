# Scope, distinctions + connections

thumbnail-design is a **design-craft skill** — it writes the promise, designs the concept + text, produces
the **brief**, and plans the **A/B test**. The agent **designs + briefs + critiques + plans**; a **human or
image tool makes the raster**; the **YouTube thumbnail is uploaded + tested natively** (no WoopSocial field).

## Honest scope (never violate)
- **The agent** writes the **promise**, the **concept**, the **text**, and the **design brief**, and can
  **critique a mockup** — but a **HUMAN or an image/design tool** (`nano-banana`/`ideogram`/Canva/a designer)
  **creates the final raster.** The agent doesn't render the pixels.
- **The YouTube custom thumbnail is uploaded NATIVELY in YouTube Studio — WoopSocial has no thumbnail
  field** — and **Test & Compare is native** (WoopSocial can't run it). The agent plans the test; a **human
  uploads + runs** it.
- **TikTok / IG-Reel covers are different:** those are a frame/card set via **WoopSocial's `cover` field**,
  owned by `tiktok-video-publishing` / the IG format skill — not this skill's native-upload path.
- **No clickbait** — the thumbnail must keep the video's promise (the watch-time A/B test punishes
  deception). **No unconsented real-person faces or deepfakes.** **Never fabricate a CTR** — real testing/
  native analytics only. **Verify-quarterly** (specs + features move).

## Distinct from its siblings
- **thumbnail-design (this)** — the **thumbnail design craft + brief + A/B plan.**
- **youtube-publishing-and-metadata** — the **title/description/tags/category metadata**; it **points here**
  for the thumbnail (which it can't set via the API).
- **nano-banana** / **ideogram** — **generate the raster** this skill briefs (ideogram especially for in-image
  text).
- **tiktok-video-publishing** — owns the **TikTok `cover` field** (a video frame, not a designed YouTube
  thumbnail).
- **hook-writer** / **caption-writer** — supply **text lines**; this skill decides the thumbnail's words +
  composition.

## Where this connects
- **Reads first:** `brand-profile` (style/template) + the **video and its title** (often from
  `youtube-publishing-and-metadata` / `youtube-long-form` / `youtube-shorts`).
- **Pairs with:** `nano-banana`/`ideogram` (render the brief), `hook-writer` (the thumbnail words),
  `youtube-publishing-and-metadata` (title designed as a unit with the thumbnail), `experimentation-and-ab-testing` (the
  A/B discipline), `competitor-analysis` (preview against ranking thumbnails for the keyword).
- **Hands to:** a **human** to render (or an image tool), **upload the YouTube thumbnail natively**, and run
  **Test & Compare**; `analytics-and-reporting` reads CTR/watch-time back. **The raster, the native upload,
  and the native A/B test stay outside WoopSocial.**
