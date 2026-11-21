# How AI Engines Cite

To get cited, you have to understand how AI answer engines actually choose their sources. This is the
model the whole skill rests on.

> GEO is the fastest-moving discipline in this library — re-verify quarterly. The mechanics below are
> well-evidenced as of 2026, but specifics (which engine cites what, the percentages) shift fast.

## The shift, in one line

When someone asks ChatGPT/Perplexity/Google AI "what's the best [X]" or "how do I [Y]", they don't
get ten links — they get a **synthesized answer that names specific brands and cites specific
sources.** GEO (Generative Engine Optimization; also AEO/LLMO) is the practice of being **in that
answer.** It supplements SEO, doesn't replace it. The stakes: AI search now handles a meaningful and
fast-growing share of informational queries, much of it **zero-click** — so being *mentioned* matters
even when nobody visits your site.

## How the answer gets built (RAG + fan-out)

1. **Retrieval-augmented generation (RAG).** Most AI search supplements training data with **live web
   retrieval**: ChatGPT retrieves via **OpenAI's own crawler + index (OAI-SearchBot)** — it was
   historically Bing-seeded and Bing rank still correlates; Google AI Overviews/AI Mode pull from
   Google's index; Perplexity searches the live web every query. **Implications (the levers most
   guides miss): search rank feeds AI citation, and the retrieval crawlers must be able to reach you
   — robots.txt or CDN bot-protection (Cloudflare et al. block AI bots by default on many sites)
   silently removes you from the candidate pool.** Classic + social search ranking **feeds** AI
   retrieval → this is why `social-seo` is the foundation, not a separate track.
2. **Query fan-out.** The engine breaks the question into **sub-queries** and searches each. "Best VPN
   for Netflix in Europe" → "best VPN 2026" + "VPN Netflix streaming" + "VPN Europe servers." So you
   optimize for the **constellation of sub-questions** around a topic, not one phrase. (This is the
   logic behind a citation audit — see `audit-and-measurement.md`.)
3. **Select → rank by credibility → merge.** The model picks candidate sources, ranks them by
   perceived credibility, and merges them into one answer while avoiding contradictions. So
   **corroboration across sources** (a "web of mutual verification") and **clear extractable claims**
   win.

## Which sources actually get cited (the social angle)

The pages AI cites are **often not your pages.** Optimizing only your product/marketing pages targets
a small slice of the problem. What gets cited:

- **Community platforms dominate** — community-driven sources are roughly **half** of citations across
  ChatGPT/Perplexity/Google AI Overviews combined. **Reddit** is the single most-cited source overall
  and ~**47% of Perplexity's** top sources (Perplexity is tuned to surface real people answering real
  questions). 
- **YouTube** is one of the most-cited AI-Overview sources — **brand mentions in video titles and
  transcripts are among the strongest correlates of AI-Overview visibility.**
- **Wikipedia** dominates **ChatGPT** for factual queries; **listicles, review sites (G2/Trustpilot),
  Quora, LinkedIn, news** all feed AI answers.
- **Brand-managed sources still matter too** — a large share of citations come from first-party sites
  and **business listings**; the mistake is doing *only* product pages, not doing them at all.

## Platforms disagree — optimize for the field

Citation logic differs sharply by engine, and overlap between them is low:

- **ChatGPT** (largest share) — own-index retrieval (OAI-SearchBot; historically Bing-seeded) +
  training data; skews **Wikipedia** + well-sourced authoritative content; rewards clear expertise
  signals.
- **Perplexity** — live search every query, **Reddit-dominant**, strong **freshness** preference
  (recent content), cites *many* sources per answer (more citation slots → lower competition per slot),
  rewards H2/H3 question headings + visible stats + named sources.
- **Google AI Overviews** — built on Google ranking + **E-E-A-T** + **structured data**; content that
  already ranks organically tends to appear; pulls broadly from the social/community web + YouTube.
- **Gemini / Copilot / Grok** — their own logic; brand citation rates vary enormously by engine.

"Optimize for AI search" as one thing is like running the same campaign on LinkedIn and TikTok — pick
the engines your audience uses and optimize per their citation logic.

## Freshness: citations decay

AI citations **decay fast** — a large share of cited content is only weeks old; Perplexity favors the
last ~90 days, AI Overviews cite recently-updated content several times more often. **Keep cited
content fresh** (update dates, "2026 update" sections, refreshed stats) or competitors displace you.
