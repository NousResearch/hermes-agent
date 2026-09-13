---
name: youtube-publishing-and-metadata
description: >-
  The YouTube publish-metadata skill. Use when someone wants to "write a YouTube title/description,"
  "set tags/category/privacy," "optimize YouTube SEO/metadata," set made-for-kids, or publish a
  YouTube video via WoopSocial. Writes the metadata (title, description+chapters, tags) and sets
  category/privacy/madeForKids — YouTube is a search engine, so metadata is the launchpad, and
  madeForKids is a COPPA/FTC legal flag set truthfully. Uses the INDEX framework. Reads
  brand-profile + goals-and-kpis + the script/video first. The video file + script are inputs (not
  WoopSocial); the thumbnail and A/B title testing are native YouTube Studio; WoopSocial publishes
  via the YouTube fields; metrics never fabricated. Distinct from youtube-long-form/youtube-shorts
  (the script) and thumbnail-design (the thumbnail).
version: 1.0.0
---

# youtube-publishing-and-metadata

A **format-execution skill** — write the YouTube publish metadata (title, description+chapters, tags) and
set category/privacy/madeForKids. The agent **writes the metadata + config**; the **video FILE + script are
inputs**; **WoopSocial publishes via the YouTube fields**; the **thumbnail + A/B testing are native Studio**.

## The POV: YouTube is a search engine; metadata is the launchpad
YouTube is the **2nd-largest search engine** (~27% of traffic from search, ~70% from recommendations — both
metadata-fed), and metadata is what it reads **before user behavior kicks in**: the **launchpad** that lets
a good video **compound for years**. But be honest — **retention keeps it ranking**; metadata gets you
discovered, it doesn't guarantee a rank. The **title does two jobs at once** (rank via the keyword + earn
the click), the **description's above-the-fold lines** carry the most weight, **tags are a minor signal now**
(get the first one right), and **`madeForKids` is a legal flag**, not a reach lever.

## Read these first
1. **brand-profile** — niche/voice + how the audience searches.
2. **goals-and-kpis** — the goal (search reach / subscribers / watch time).
3. the **script/video** (often from **youtube-long-form**/**youtube-shorts**).

## The framework: INDEX
(Depth: `references/the-index-framework.md`.)
- **I — Intent-match the title:** primary keyword in the first ~40 chars, ~50–60 total; rank + click;
  power word; clarity over cleverness.
- **N — Nail the description above-the-fold:** line 1 value → keyword in lines 2–3 → 300–500 words natural;
  keyword-named chapters; 3–7 links + affiliate disclosure. (Description = `content.text`.)
- **D — Direct the tags:** first tag = exact primary keyword; then secondary + niche; relevant-only.
- **E — Elect category + privacy:** accurate `category`; `privacy` (public/unlisted/private) per goal.
- **X — eXamine the kids-flag + validate:** set `madeForKids` **truthfully** (COPPA/FTC — it disables
  comments/personalized ads/etc.); never mislabel; **validate** before publish.

## The reality (verify-quarterly)
YouTube = search engine / metadata = launchpad (but retention ranks); title (keyword first ~40 chars, 50–60,
two jobs, power words, A/B native); description (above-the-fold, 300–500 natural, chapters, links +
disclosure); tags (minor now, first = primary keyword, relevant-only); category + privacy; **madeForKids
(COPPA/FTC legal flag, truthful)**; the WoopSocial gaps (no thumbnail field/no A/B → native Studio):
`references/youtube-metadata-2026-reality.md`. Title/description templates, tag order, the full publish-config
block + worked examples: `references/metadata-and-publish.md`.

## Honest scope (never violate)
- **The agent writes the metadata + config.** The **video FILE + script are inputs** (`youtube-long-form`/
  `youtube-shorts` + filming/editing) — **WoopSocial doesn't film/edit/generate video.** The **THUMBNAIL is
  native Studio** (no API field; it's a major CTR lever — brief it via `thumbnail-design`), and so is **A/B
  title testing.**
- **WoopSocial publishes via the YouTube fields** (`title`/`privacy`/`category`/`tags`/`madeForKids`;
  description = `content.text`; raw-bytes upload). **No update** → delete+recreate before publish; metadata
  edits in Studio after.
- **`madeForKids` truthful** (COPPA/FTC; mislabeling is illegal, not a lever). **Affiliate/AI disclosure**
  where relevant; **no keyword-stuffing**; **never promise a ranking; never fabricate metrics** (native
  analytics only). (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**youtube-publishing-and-metadata (this)** = the publish METADATA + config (title/description/tags/category/
privacy/madeForKids) · **youtube-long-form**/**youtube-shorts** = the SCRIPT/content · **thumbnail-design** =
the thumbnail (uploaded natively — no API field) · **caption-writer**/**hook-writer** = generic copy.

## Where this connects
Reads first: **brand-profile**, **goals-and-kpis**, the **script/video** (**youtube-long-form**/
**youtube-shorts**). Content inputs (external to WoopSocial): those script skills, the **video** cluster
(**veo**/**runway**/**kling**) + editing, **thumbnail-design** (native upload), **hook-writer**/
**caption-writer**. Publish: **scheduling-and-queue → WoopSocial** (YouTube fields),
**platform-specs-and-validation** (validate + field rules), **content-calendar**. Reuse/measure:
**content-recycling** (→ Shorts/clips), **analytics-and-reporting** (search/watch-time/CTR),
**experimentation** (title/thumbnail A/B — run natively in Studio). Video/thumbnail/A/B stay external/native.

## Definition of done
A keyword-led title (primary keyword in the first ~40 chars, ~50–60 total, rank + click, clarity); an
above-the-fold description (value line → keyword → 300–500 words natural, keyword-named chapters, 3–7 links +
disclosure) in `content.text`; tags led by the exact primary keyword, relevant-only; an accurate category
and a goal-fit privacy; `madeForKids` set truthfully per COPPA; the full WoopSocial YouTube config validated
before publish; the video file produced externally and the thumbnail + A/B testing left to native Studio;
edits via delete+recreate (or Studio post-publish), nothing fabricated and no promised ranking; correctly
distinguished from youtube-long-form/youtube-shorts and thumbnail-design.
