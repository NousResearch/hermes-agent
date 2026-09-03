---
name: social-seo
description: >-
  Use to make social content discoverable through SEARCH across social platforms — the
  social-SEO / keyword-discovery skill. Run when the user says "social SEO," "get found in search,"
  "TikTok SEO," "YouTube SEO," "keywords for social," "searchable captions," or wants search across
  platforms rather than just the feed. For single-platform depth, route "Instagram SEO" to
  instagram-seo and Pinterest asks to pinterest-seo; this skill owns the cross-platform framework.
  Reads brand-profile and audience first. All platforms are search engines; keywords, not hashtags,
  are the lever. Covers per-platform indexed surfaces, keyword research via autocomplete + native
  analytics (never fabricates volumes), the triple-mention technique, topical clusters, and the
  profile as a search asset. Routes hashtags to hashtag-strategy; hands the AI-search/LLM-citation
  (GEO) layer to ai-search-optimization. Measures via native search insights, not WoopSocial analytics.
metadata:
  version: 1.0.0
license: MIT
---

# Social SEO

Make content **findable when people search**, not just visible when the feed happens to show it. In
2026 every social platform is a search engine, and search is the highest-intent, most **evergreen**
discovery there is.

Four truths shape everything:

1. **All social platforms are search engines** (TikTok/IG/YouTube/Pinterest/LinkedIn) — and people
   search *different* platforms for different intents. If you only optimize for the feed, you're
   invisible to people actively looking.
2. **Keywords, not hashtags, are the lever.** Write for the search bar first; hashtags are a minor
   category signal → `hashtag-strategy`.
3. **Search optimization doubles as recommendation optimization** — the keywords that get you found in
   search are the same signals the algorithm uses to categorize you for the right feed/FYP audience.
4. **SEO turns posts into evergreen assets** — a search-optimized post keeps getting found for months;
   a feed post dies in ~48h.

(Full model + the AI-search/GEO boundary: `references/why-social-search.md`.)

## Step 0 — Read the foundation + the goal

Load `brand-profile.md` and `audience.md` (their exact search language matters). Get the **platform(s)**
and what the user wants to be **found for**.

## Step 1 — Keyword research (real demand, never invented)

Find the terms people actually search via **platform autocomplete**, **"others searched for"**,
related searches, **comments**, audience language, and the user's **native search analytics** (+ tools
like TikTok Creative Center / VidIQ). **The agent can't pull live search volumes — never fabricate
numbers**; give the method and work from intent/specificity. Favor long-tail, intent-rich queries you
can genuinely answer. See `references/keyword-research-and-technique.md`.

## Step 2 — Place keywords across the platform's SEO surface

Each platform indexes different things — optimize the right ones (full per-platform guide in
`references/platform-seo-surfaces.md`):

- **TikTok** — spoken audio (transcribed) + on-screen text + caption (the **triple mention**).
- **Instagram** — caption + the **searchable name/display field** + bio + alt text + captioned Reels.
- **YouTube** — keyword-led **title** + ~200–300w **description** + **transcript/captions** + chapters.
- **Pinterest** — keyword **title + description + board names + alt** (hashtags deprecated there).
- **LinkedIn** — searchable **headline/About** + on-topic post keywords → `linkedin-growth`.
- **Cross-cutting** — subtitles, alt text, descriptive file names, a consistent branded handle.

## Step 3 — The on-content technique

**Lead with keywords, support with hashtags.** State the primary phrase in **natural language, early,
once or twice**; align audio + on-screen + caption to **one** search intent; add 3–5 niche hashtags.
**Never keyword-stuff** (spam dilutes your topical signal). Resolve the search-vs-feed tension (clarity
vs hook) deliberately. See `references/keyword-research-and-technique.md`.

## Step 4 — Build topical clusters (evergreen authority)

One post rarely owns a query. **Cluster** evergreen content around a pillar topic so you become the
**answer/authority** — which compounds *and* sharpens recommendation categorization. Map a
`content-pillars` pillar → a keyword cluster → many evergreen posts.

## Step 5 — Optimize the profile as a search asset

The **name/display field, handle, headline/bio** should carry the searchable keyword/category (not just
a clever brand line). This is the *searchability* lens — `profile-optimization` owns the *conversion*
lens; both matter. Public profiles/content are increasingly **Google-indexable** too.

## Step 6 — Measure (honestly) + hand off GEO

Track **discoverability**: impressions from search, the search terms that found you, profile visits
from non-followers, saves, evergreen-post growth — via **native platform search analytics** (+ Search
Console/UTMs). **No WoopSocial analytics.** For getting cited by **AI search/LLMs (GEO)**, hand off to
`ai-search-optimization` — related but separate.

## Orchestration map

social-seo sets the discovery layer; it routes to / is routed from: `hashtag-strategy` (hashtag
specifics) · `caption-writer` (adds the keyword layer) · `profile-optimization` (conversion lens) ·
`content-pillars` (clusters) · `tiktok-growth` / `instagram-growth` / `linkedin-growth` (search as a
growth pillar) · `ai-search-optimization` (GEO hand-off) · `scheduling-and-queue` (publish).

## Quality bar — self-check

- Did I do **real keyword research** (autocomplete/native analytics) and **avoid inventing volumes**?
- Did I optimize the **platform-specific indexed surfaces** (not one generic answer)?
- Did I use **lead-with-keywords + the triple mention**, in **natural language** (no stuffing)?
- Did I think in **evergreen clusters / topical authority**, and note the **search = recommendation**
  payoff?
- Did I treat the **profile as a search asset**, route hashtags to `hashtag-strategy`, and **hand GEO**
  to `ai-search-optimization`?
- **Native-search-insights** measurement, **no WoopSocial analytics**, **searchability not guaranteed
  rankings**?

## Edge cases & pushback

- **"Give me exact search volumes"** → can't pull live volume; show the research method; don't
  fabricate.
- **"Stuff every keyword + 30 hashtags"** → refuse; spam dilutes the signal; natural language + a few
  niche tags.
- **"I optimized one post, done"** → build a cluster; one post rarely owns a query.
- **"Just use trending hashtags"** → hashtags categorize, keywords get found → `hashtag-strategy`.
- **"Make ChatGPT/Google-AI cite me"** → that's GEO → `ai-search-optimization`.
- **"Which post ranks best?"** → native platform search insights; no WoopSocial analytics.
- **Clever-but-vague title** → keep the searchable phrase; add intrigue after the keyword.

## Related skills

- `brand-profile`, `audience-research` — the search language and positioning.
- `instagram-seo`, `pinterest-seo` — single-platform search depth (route platform-specific asks there).
- `hashtag-strategy` — the hashtag layer (a minor signal); `caption-writer` — caption craft.
- `profile-optimization` — profile as conversion (this skill = profile as search).
- `content-pillars` — pillars become keyword clusters; the growth skills lean on search.
- `ai-search-optimization` — the AI-search/LLM-citation (GEO) hand-off; `scheduling-and-queue` — publish.

## References

- `references/why-social-search.md` — the 2026 reality, the three layers, search=recommendation, evergreen, the GEO boundary.
- `references/platform-seo-surfaces.md` — what each platform indexes and how to optimize it (TikTok/IG/YouTube/Pinterest/LinkedIn/X/FB + cross-cutting).
- `references/keyword-research-and-technique.md` — finding keywords (no fabrication), the triple-mention technique, clusters, no-stuffing, measurement.
- `references/examples.md` — worked optimizations per platform + a cluster plan + honest scope.
