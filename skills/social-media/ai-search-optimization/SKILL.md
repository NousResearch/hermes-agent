---
name: ai-search-optimization
description: >-
  Use to get a brand and its content CITED and RECOMMENDED by AI answer engines — the GEO (Generative
  Engine Optimization) / AI-search-visibility skill. Run when the user says "GEO," "get cited by
  ChatGPT/Perplexity/Google AI," "ChatGPT SEO," "LLM SEO / LLMO," "AI Overviews," "answer engine
  optimization (AEO)," "will AI recommend my brand," or wants to show up in AI-generated answers, not
  just the feed or Google links. Reads brand-profile and audience first. AI engines retrieve + fan-out
  and cite community/social sources heavily (Reddit, YouTube, Wikipedia); platforms disagree; earned
  media beats product pages; extractable, fresh content drives citation. Covers the four GEO levers
  (retrievable, earned mentions, extractable content, entity clarity), authentic social plays, and the
  manual prompt-audit method. Refuses astroturfing; never fabricates "share of voice." Sibling of
  social-seo (platform + Google search). Judges via the audit + GEO tools + AI referral traffic.
metadata:
  version: 1.0.0
license: MIT
---

# AI Search Optimization (GEO)

Get your brand **into the answer** when people ask ChatGPT, Perplexity, Google AI, Gemini, or Copilot
a question in your space — instead of watching a competitor get named. This is **GEO** (Generative
Engine Optimization; also AEO/LLMO): optimizing to be **cited and recommended** by AI engines. It
supplements search/SEO; it doesn't replace it.

Four truths shape everything:

1. **AI answers are built by retrieval + fan-out.** Engines retrieve live from search indexes
   (ChatGPT via OpenAI's own crawler/index, OAI-SearchBot — historically Bing-seeded; Google feeds
   AI Overviews/AI Mode) and split your topic into sub-queries — so **ranking in search feeds AI
   citation**, and you optimize for a constellation of questions.
2. **AI cites community/social sources most.** Reddit, YouTube, Wikipedia, LinkedIn, listicles and
   review sites dominate citations — **the cited pages are usually not your pages.** Earned mentions
   beat product pages.
3. **Platforms disagree.** ChatGPT skews Wikipedia, Perplexity skews Reddit, AI Overviews lean on
   E-E-A-T + the community web. Optimizing for one ≠ all.
4. **Extractable, fresh, well-sourced content gets quoted.** Quotations, statistics, citations, Q&A
   structure, and schema lift citation; stale content gets displaced.

(Full mechanics: `references/how-ai-engines-cite.md`.)

## Step 0 — Read the foundation + the goal

Load `brand-profile.md` and `audience.md` (entity clarity + the real questions matter). Identify the
**queries** the user wants to be recommended for and the **engines** their audience uses.

## Step 1 — Run the prompt-audit (always start here)

Ask the user's **10–30 buyer-intent queries** (plus fan-out sub-questions) across ChatGPT / Perplexity
/ Gemini in fresh sessions; document **whether the brand appears, how it's described, and which sources
are cited.** The cited sources *are* the strategy; the gaps are the content list. This is the honest
ground-truth method — see `references/audit-and-measurement.md`.

## Step 2 — Be retrievable (the foundation)

If you can't be found in search, you can't be cited: rank in **Google/Bing** and in platform search →
`social-seo` (the sibling). Same keyword/question research powers both. And **verify AI retrieval
crawlers can reach the site** — robots.txt and CDN/bot-protection defaults (e.g. Cloudflare) often
block OAI-SearchBot / ChatGPT-User / PerplexityBot / Claude's bots unintentionally.

## Step 3 — Earn brand mentions across cited sources (the social core)

Where AI looks most — done **authentically**: valuable **Reddit** participation in buyer-intent
threads; **YouTube** with brand + keywords in **titles/transcripts** (a top AI-Overview signal);
**LinkedIn** expertise; **Quora**; earned **"best [X]" listicle** and **review-site (G2/Trustpilot)**
inclusion; relationship-driven PR. The goal is a **web of mutual verification.** See
`references/the-geo-levers.md`.

## Step 4 — Make content extractable

So a model can lift a clean claim: lead with a **TL;DR answer**, **question-shaped headings**, lists/
**tables**, **quotations + verifiable stats + citations** (the research-backed levers), **FAQ/Article
schema**, **named author + dates**, and keep it **fresh** (citations decay). (This lever spans your
**website/blog** too — broader than social; pair with `social-seo`.)

## Step 5 — Build entity clarity

Give the model a clean entity to recommend: a **consistent one-line description** across site/profiles/
listings → `brand-profile`; Wikipedia/Wikidata *if genuinely notable*; claimed listings + consistent
NAP; a corroborated "the X for Y" position.

## Step 6 — Measure (honestly) + the boundary

Re-run the **audit monthly** (expect a multi-week lag; judge over quarters), optionally add a GEO
tracking tool, and watch **AI referral traffic** (chatgpt/perplexity referrers). **Never fabricate a
"share of voice" or citation %.** **No WoopSocial analytics.** Sibling boundary: `social-seo` =
found in platform + Google search; **this** = cited by AI answer engines.

## Orchestration map

ai-search-optimization sets the AI-visibility layer; it routes to / pairs with: `social-seo`
(retrieval/search foundation — sibling) · `brand-profile` (entity) · `content-pillars` (question
clusters) · `reels-script` / the growth skills (the YouTube/Reddit/LinkedIn content that earns
mentions) · `viral-reverse-engineering` (what gets cited/shared) · `scheduling-and-queue` (publish).

## Quality bar — self-check

- Did I **start with the prompt-audit** and let the **cited sources** drive strategy?
- Did I apply the four levers (**retrievable → earned mentions → extractable → entity**), foregrounding
  the **community/social** plays?
- Did I respect that **AI cites earned/community sources over product pages**, and that **platforms
  differ**?
- Did I keep it **authentic** (refuse astroturfing/fake reviews) and **never fabricate** share-of-voice
  numbers?
- Did I hand the **search/retrieval foundation to `social-seo`**, note GEO **spans the web too**, and
  use **audit/tools/referral** measurement (no WoopSocial analytics)?

## Edge cases & pushback

- **"Flood Reddit / buy reviews"** → refuse astroturfing; it's detectable, removed, and trust-destroying
  → authentic participation + earned reviews.
- **"Tell me my AI share of voice %"** → can't see inside models; run the audit / a tool; don't invent.
- **"Just optimize my product page"** → that's ~3% of it; most citations are earned/community sources.
- **"Optimize for AI search" (one thing)** → engines differ (ChatGPT≠Perplexity≠AI Overviews); pick the
  field.
- **"Is this my TikTok/Google SEO?"** → related but distinct → `social-seo` owns platform/Google search.
- **"Does WoopSocial track this?"** → no; measure via audit + GEO tools + referral analytics.
- **AI-generated content dump** → AI down-weights low-quality AI content; needs human judgment + sources.

## Related skills

- `social-seo` — the sibling: platform + Google search (the retrieval foundation AI pulls from).
- `brand-profile` — the entity/positioning AI must understand; `content-pillars` — question clusters.
- `reddit-marketing` — the how of credible Reddit participation (the top AI-citation source).
- `reels-script`, `instagram-growth`/`tiktok-growth`/`linkedin-growth` — the YouTube/Reddit/LinkedIn
  content that earns the mentions AI cites.
- `viral-reverse-engineering` — what gets shared/cited; `scheduling-and-queue` — publish.

## References

- `references/how-ai-engines-cite.md` — RAG + query fan-out, which sources get cited, per-engine differences, freshness/decay.
- `references/the-geo-levers.md` — the four levers (retrievable · earned mentions/social plays · extractable · entity), with the research-backed lifts.
- `references/audit-and-measurement.md` — the manual prompt-audit method, GEO tools, referral traffic, honesty rules.
- `references/examples.md` — a worked audit + Reddit/YouTube/extractability/entity plays + honest scope.
