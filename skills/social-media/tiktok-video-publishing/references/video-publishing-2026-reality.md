# TikTok video publishing 2026 — verified

*Volatile. Re-verify quarterly (specs, disclosure rules, account caps move).*

## The caption is search (the most controllable reach layer)
TikTok is a **search engine** (~40% of Gen Z search TikTok/IG over Google), and the caption (`content.text`)
is the **most controllable layer** — metadata is set at upload, unlike spoken content. **Only the first
~100–150 chars show** before "more," so:
- **Lead with the primary keyword phrase** — the actual topic, not a teaser/question.
- Then a **value-prop sentence** (what the viewer gets), then **secondary keywords + niche hashtags.**
- **Long-tail** beats broad ("home workout no equipment" not "workout"); use search-bar autocomplete.
- Caption max **4,000 chars**; mirror the keyword in **on-screen text** (TikTok OCR reads it) and the first
  spoken seconds. Well-optimized videos also surface in **Google**. A single relevant hashtag ≈ **+5%
  views / +9% interactions**; skip generic #fyp.
- **Pinned-comment** ("second caption" with extra keywords) is a known tactic — but WoopSocial has **no
  comment surface**, so it's a native/human step after publish.

## Duet / Stitch / comments = reach levers
**Leaving `allowDuet` / `allowStitch` ON is free distribution** — others remixing your video grows the
engagement cluster around it ("internal video linking"). **`allowComment` ON** feeds the engagement engine
(and keyword-rich replies add SEO). Only restrict for genuine **brand-safety**, and know the reach cost.
**These apply to VIDEO** (not PHOTO).

## Cover / thumbnail
The **cover frame** wins the **profile grid + search results** (the video's discovery "second life").
WoopSocial's **`cover`** field sets the video's **cover/thumbnail** — treat it as a **legible title
card** (1080×1920). Keep key text centred (UI overlays sit on the right/bottom).

## Privacy + disclosure
- **`privacyLevel`**: `PUBLIC_TO_EVERYONE` for reach; `SELF_ONLY` / `MUTUAL_FOLLOW_FRIENDS` /
  `FOLLOWER_OF_CREATOR` restrict. Discover allowed values via **platform-inputs** (`privacyLevelOptions`).
  **Branded content cannot be private** (verify).
- **Commercial disclosure** = TikTok's toggle: **Branded Content** (third-party paid → `isBrandedContent`)
  vs **Your Brand** (own products → `isYourBrand`). Set **truthfully**; put **#ad/#sponsored at the start.**
- **AI**: TikTok requires **visible labels** for AI-generated/altered people/voices/realistic scenes;
  **AI endorsements and unlabeled deepfakes of real people are prohibited.** WoopSocial **auto-discloses AI
  on TikTok**, but that doesn't make prohibited uses allowed.

## Specs + length + mode
- **1080×1920, 9:16**, MP4/MOV, **≤1GB** for auto-publish; below **540×960** gets upscaled/blurry. Default
  upload cap **3 min** (10 min for eligible accounts); read the per-account cap from **`maxVideoPostDurationSec`.**
- Length to goal: **11–18s** viral, **21–34s** storytelling, **30–60s** educational, **60–90s** search-
  optimized, up to 3 min deep dives. Strong hook in the **first 2s**; **~96% of reach** in the first 10 days.
- **`postMode`**: `DIRECT_POST` publishes directly; **`MEDIA_UPLOAD`** sends to the TikTok **inbox** so the
  creator finishes in-app — the way to **add a licensed trending sound** (the API can't attach Commercial
  Music; `autoAddMusic` is **PHOTO-only**). **No edit after scheduling** → delete + recreate.
