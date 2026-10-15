# Title/description templates, publish config + two worked examples

## Title formula
```
[PRIMARY KEYWORD - early] : [modifier/power word] [year/qualifier]
```
- Primary keyword in **first ~40 chars**, total **~50–60** (max 100). Rank + click; clarity over clever.
- e.g. "YouTube SEO for Beginners: Complete 2026 Guide" · "Beat Malenia Solo - Elden Ring Boss Guide".

## Description template (= content.text)
```
Line 1   the value/hook (what they'll learn/see)  <- above the fold
Line 2-3 the primary keyword, naturally            <- above the fold
Body     300-500 words, natural keyword + secondary terms (no stuffing)
Chapters 00:00 Keyword-named  /  01:12 Keyword-named  (not "Part 1")
Links    3-7: playlist first, channel, website, resources; affiliate -> disclose
```

## Tag order
`[exact primary keyword]` (first, weighted most) → 2–4 secondary keywords → niche tags. Relevant-only.

## WoopSocial YouTube publish-config block (the exposed fields)
```
platform: YOUTUBE
title: required (the SEO title above)
content.text: the description (above-the-fold + chapters + links)
privacy: public | unlisted | private        # required
category: accurate category for the niche    # optional
tags: [exact primary keyword, secondary, niche]  # optional
madeForKids: true | false                     # COPPA - set TRUTHFULLY
media: the video file (raw-bytes upload; see tools/integrations/woopsocial.md)
```
**No thumbnail field** (native Studio). **No A/B title test** (native Studio). Then **validate**
(`POST /posts/validate` → `YOUTUBE_PRIVACY/CATEGORY/TAGS`, TITLE, DESCRIPTION, MEDIA). **No update** →
delete+recreate before publish; **edit metadata in Studio** after publish.

## Worked example 1 — long-form tutorial, own brand (blunt indie-founder voice)
```
TITLE: "Notion for Freelancers: the only setup you need (2026)"  (keyword early, 52 chars)
DESC: "Set up Notion to run your whole freelance business in 20 min." -> keyword line 2 -> 350 words + chapters + playlist link.
TAGS: ["notion for freelancers", "notion setup", "freelance productivity"]. CATEGORY: Education. PRIVACY: public. madeForKids: false.
THUMBNAIL: brief -> thumbnail-design, uploaded in Studio. Validate -> schedule.
```

## Worked example 2 — kids' content, truthful flag (warm studio voice)
```
TITLE: "Counting Fruit Song - Learn Numbers 1-10 | [Brand] Kids"  (keyword early)
DESC: value line -> "numbers song for toddlers" keyword -> chapters per number. CATEGORY: Education. PRIVACY: public.
madeForKids: TRUE (child-directed) -> comments/personalized ads disabled by YouTube; that's correct + legal. Validate -> publish.
```

Both: keyword-led title (rank + click); above-the-fold description + chapters; first-tag = primary keyword;
accurate category + privacy; madeForKids truthful; video file + thumbnail external/native; WoopSocial
publishes the YouTube fields; metrics native (never fabricated).
