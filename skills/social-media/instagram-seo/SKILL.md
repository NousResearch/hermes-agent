---
name: instagram-seo
description: >-
  The Instagram SEO / keyword-search skill -- IG-specific search depth, distinct from the cross-cutting
  social-seo. Use when someone wants an Instagram post/Reel to rank in search, needs IG keyword
  research, or wants a searchable caption/profile. Instagram shifted from hashtag-based discovery to
  keyword-based search (hashtags deprioritized to 3-5 topic labels). Search runs semantic analysis ->
  metadata matching -> visual recognition (checks alt text + visual MATCH the caption). Uses the MATCH
  framework. Reads brand-profile + the post's topic first. The agent researches the keyword cluster
  (never fabricates volume), writes the search-first caption (primary keyword first, no stuffing),
  specs on-screen text + spoken keyword, and advises profile fields/alt text. WoopSocial publishes the
  caption; profile fields, alt text, and IG Search/Insights stay native. No keyword stuffing; caption
  must match the visual. Distinct from social-seo, pinterest-seo, and hashtag-strategy.
version: 1.0.0
---

# instagram-seo

The **IG keyword-search layer** — research the queries and write the keyworded caption (and advise the profile
labels + visual signals) that make a post rank in Instagram search. The agent **researches + writes the
search-first caption + specs the on-screen/spoken keyword + advises the profile/alt text**; **WoopSocial
publishes the caption**; **the Name field, handle, bio, alt text, category, and auto-captions are native/
advised**, and on-screen text/audio come from other tools/the creator.

## The POV: Instagram is a keyword search engine — keywords replaced hashtags
Since late 2024 (official in 2026), IG shifted from **hashtag-based discovery to keyword-based search** —
hashtags are deprioritized to **3–5 topic labels**, and **a large and growing share of discovery/new followers
now starts from search** (verify-quarterly; never quote a precise share). It's a
**closed system** (no external crawl/backlinks) but **Google + Bing index IG captions.** Search runs
**semantic analysis → metadata matching** (Name field/bio/caption **keyword cluster**) **→ visual recognition**
(Computer Vision checks the **alt text + visual match the caption** — a mismatch lowers ranking). WoopSocial
publishes the **caption**; the profile labels, alt text, on-screen text, audio, and settings are native/
advised/other-tools. **Pairs with `caption-writer`** (the keyword goes in the searchable caption).

## Read these first
1. **brand-profile** — niche/voice.
2. the **post topic/destination.**

## The framework: MATCH
(Depth: `references/the-match-framework.md`.)
- **M — Mine the real queries:** scan top Reels' captions/on-screen text + competitor Name fields + Google
  "People Also Ask"; build a **3–5 keyword cluster.** *Method only; never fabricate volume.*
- **A — Anchor the caption to search:** lead the **first sentence / ~125 chars** with the primary keyword
  (natural long-tail); weave the cluster; 3–5 hashtag labels — the **publishable** surface (`content.text`).
- **T — Tag the profile labels:** Name field ("Name | Niche"), keyworded handle, indexable bio (niche/offer/
  location/CTA). *Profile-level — native/advised.*
- **C — Confirm the visual matches:** alt text + on-screen text + **first-3-sec spoken keyword** reinforce and
  **match the caption** (Computer Vision); auto-captions on; remove other-platform watermarks. *On-screen →
  image/video tool; script → `reels-script`; alt text → native.*
- **H — Harvest over time across search + Explore:** search = keyword-driven, Explore = behaviour-driven;
  optimize for saves/shares/watch-time; compounds over 90–120 days; no stuffing / irrelevant tags.

## The reality (verify-quarterly)
Keyword-based since late 2024 (official 2026); hashtags 3–5 labels; a growing share of discovery from search; closed
system but Google/Bing index captions; 3-step search (semantic → metadata cluster → Computer Vision alt-text/
visual match); surfaces (Name field high weight, handle, indexable bio, search-first captions [primary keyword
first sentence/125 chars, 150–300 words], alt text [must match], on-screen text transcribed, audio transcribed
[first-3-sec keyword], auto-captions lift watch time + are indexed, category/location, 3–5 cluster); search vs
Explore; engagement (saves/shares/watch-time/replays > likes; completion); Reels deprioritized (watermarks/
low-res); the Raw Content Revolution (human-made, retention/sends); research method + slow-burn timeline (2–4
wk / 30–60 day / 90–120 day): `references/instagram-seo-2026-reality.md`. The research method, the surface map
(publishable vs native vs other-tools), the caption template, the must-match checklist, the publish block +
worked examples: `references/keyword-research-and-surfaces.md`.

## Honest scope (never violate)
- **The agent** researches keywords (**advises the method**, **never fabricates volume**), writes the
  **search-first caption** (`content.text`, primary keyword first, no stuffing, 3–5 hashtag labels), **specs
  the on-screen text + first-3-sec spoken keyword**, and **advises** Name field/handle/bio/alt text/category.
- **WoopSocial publishes** the IG post's **caption** (POST/REEL/STORY via `content.text`).
- **WoopSocial does NOT:** set **Name field/handle/bio** (profile — native), **alt text** (native/at-post),
  **category/location** (native), toggle **auto-captions** (native), read **IG Search/Insights** (native),
  render the **on-screen text** (image/video tool), or record the **spoken audio** (creator/`reels-script`).
- **No keyword stuffing** (suppressed); **no irrelevant hashtags** (hurt credibility); the **caption must
  MATCH the visual** (Computer Vision — no misleading); **remove other-platform watermarks**; **never
  fabricate volume or rankings.** (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**instagram-seo (this)** = IG-specific keyword search (Name field, closed-system + Google-indexed, Computer
Vision caption-visual match, first-3-sec spoken keyword) · **social-seo** = the cross-cutting framework ·
**pinterest-seo** = Pinterest (boards/Trends/Rich Pins, an open web-indexed system) · **hashtag-strategy** =
deprioritized to 3–5 (they pair: tags label, keywords discover) · **caption-writer** = the craft (pairs) ·
**reels-script** = the Reel script.

## Where this connects
Reads first: **brand-profile** + the **post topic**. Pairs with: **caption-writer** (the caption),
**hashtag-strategy** (the 3–5 labels), **reels-script** (the first-3-sec spoken keyword), **ideogram**/the
video tools (the on-screen text overlay). Routes to/from: **instagram-growth** (the strategy), **social-seo**/
**ai-search-optimization** (the cross-platform + AI search picture), **content-recycling** (re-cut a winning
Reel into a keyworded carousel). Publishes via: **scheduling-and-queue → WoopSocial** (the caption; POST/REEL/
STORY) + **platform-specs-and-validation**. Measures with: native **IG Insights** + **analytics-and-reporting**
(profile visits from search, saves, non-follower reach). Profile, alt text, settings + IG Search stay native.

## Definition of done
Real queries mined via top-Reel scans + competitor Name fields + Google "People Also Ask" into a 3–5 keyword
cluster (method advised, no fabricated volume); a search-first caption with the primary keyword in the first
sentence (natural long-tail, no stuffing, 3–5 hashtag labels) written as the publishable surface; the Name
field ("Name | Niche"), handle, bio, and alt text advised; the on-screen text + first-3-sec spoken keyword
specced (to reels-script + an image/video tool) and confirmed to MATCH the caption with auto-captions on and no
watermark; the caption published via WoopSocial while profile/alt-text/category/auto-captions/IG-Search stay
native; no keyword stuffing, no irrelevant tags, no caption-visual mismatch, nothing fabricated; correctly
distinguished from social-seo and pinterest-seo and paired with caption-writer.
