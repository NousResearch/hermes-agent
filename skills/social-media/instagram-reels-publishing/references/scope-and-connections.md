# Scope, distinctions + connections

instagram-reels-publishing is the **Reel publish-mechanics skill** — it gets a finished Reel **out** the way
Instagram displays it (specs, safe zone, cover, settings). The agent **specs the publish settings + the cover
brief + the specs/safe-zone checklist + advises audio/tags**; **WoopSocial publishes postType REEL + video +
caption + cover**; **trending audio, Share-to-Feed, tags, quality + Insights are native/in-app**, and the
video/cover/burned-in captions come from other tools.

## Honest scope (never violate)
- **The agent** specs the **publish settings** (postType REEL, 9:16, schedule), writes the **cover brief**,
  runs the **specs/safe-zone checklist**, and **advises** trending-audio/Share-to-Feed/tags.
- **WoopSocial publishes** **postType: REEL + the 9:16 video + the caption (`content.text`) + the cover
  (REEL-only field)** on a **schedule**, **spec-validated** via `platform-specs-and-validation`.
- **WoopSocial does NOT:** select **trending audio** (IG's licensed library — **not via API**; native/in-app),
  toggle **Share-to-Feed** (native), add **collaborator/product tags** (native), trim/edit the **video**, set
  the **media-quality** setting (native), or read **Reel Insights** (native). It also can't **render the
  video** (video tools/creator), **design the cover** (image tools), or **burn in captions**
  (`captions-and-clipping`).
- **9:16 1080×1920** or it crops/blurs; **keep content in the safe zone** (survives the 4:5 feed + 3:4 grid
  crops); **custom cover — can't edit after upload**; **remove other-platform watermarks**; **never fabricate
  reach.** **Verify-quarterly.**

## Distinct from its siblings
- **instagram-reels-publishing (this)** — the **publish MECHANICS** (specs, safe zone, cover, settings).
- **reels-script** — the **Reel script** (hook, beats, spoken content).
- **instagram-seo** — the **keyword caption** + first-3-sec spoken keyword + on-screen text (search). **They
  pair:** the caption is **published here.**
- **platform-specs-and-validation** — the **cross-platform validation** reference; this **uses it** (IG-Reel
  publish depth).
- **captions-and-clipping** — the **subtitle burn-in / clipping** (this specs *where* they sit; routes the
  burn-in there).
- **tiktok-video-publishing** / **youtube-publishing-and-metadata** — the **sibling publish skills** for other
  platforms. **thumbnail-design** — **YouTube** thumbnails (the Reel **cover** is the IG analogue).
  **instagram-growth** — the IG strategy.

## Where this connects
- **Reads first:** `brand-profile` (cover/caption style), the **finished Reel asset.**
- **Pulls from:** `reels-script` (the hook/script), `instagram-seo` (the keyword caption), the **video tools**
  (`veo-3`/`luma`/`kling`/`heygen` via `ai-video`) for the clip, `ideogram`/`nano-banana` for the **cover**,
  `captions-and-clipping` for the **burned-in captions.**
- **Validates with:** `platform-specs-and-validation` (9:16, ≤4GB, MP4/MOV).
- **Publishes via:** `scheduling-and-queue → WoopSocial` (postType REEL + video + caption + cover). **Measures
  with:** native **Reel Insights** + `analytics-and-reporting`. **Trending audio, Share-to-Feed, tags,
  quality, Insights + asset creation stay native/other-tools.**
