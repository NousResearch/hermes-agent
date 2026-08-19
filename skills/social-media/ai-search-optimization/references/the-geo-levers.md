# The GEO Levers

The playbook: four levers that move AI citation, in priority order. The social/community plays sit in
Lever 2 because that's where AI engines look most — and where this library has an edge.

> Re-verify quarterly. Lifts cited below come from 2026 research and are directional, not guarantees.

## Lever 1 — Be retrievable (the foundation)

AI engines retrieve from search indexes, so **if you can't be found in search, you can't be cited.**

- **Rank in Google and Bing** for your topic — Google feeds AI Overviews/AI Mode; ChatGPT retrieves
  via OpenAI's own index (historically Bing-seeded, so Bing rank still helps). This is the
  often-missed prerequisite. (Submit sitemaps; basic technical SEO.)
- **Let the AI retrieval crawlers in.** Check robots.txt and CDN/bot-protection settings actually
  allow **OAI-SearchBot / ChatGPT-User / PerplexityBot / Claude's search bots** — Cloudflare and
  similar block AI crawlers by default on many sites, which silently removes you from the candidate
  pool. (Blocking *training* bots like GPTBot is a separate, legitimate choice.)
- **Be found in social/platform search** → `social-seo` (the sibling skill). The same keyword/question
  research powers both.
- Without this layer, the rest is shouting into a void.

## Lever 2 — Earn brand mentions across cited sources (the social core)

AI builds **category → brand** associations from how often and how consistently you're **mentioned**
across the sources it trusts — and it trusts **community/social** sources most. Earned mentions beat
your own product pages. The authentic plays:

- **Reddit** — the highest-leverage GEO surface (esp. Perplexity). Participate **genuinely** in the
  buyer-intent subreddits/threads in your niche: be the helpful, specific, expert answer under a
  question-shaped thread. That's exactly the signal RAG rewards. **Not** astroturf (see guardrails).
- **YouTube** — put your brand + topic keywords in **video titles and transcripts** (a top correlate
  of AI-Overview visibility). Make videos that answer the category's real questions. → `reels-script`,
  the growth skills.
- **LinkedIn** — expert/thought-leadership content gets pulled into B2B AI answers → `linkedin-growth`.
- **Quora / forums** — genuine expert answers to real questions.
- **"Best [X]" listicles & review sites (G2, Trustpilot, Capterra)** — earn inclusion and real reviews;
  these are heavily cited for buyer-intent queries.
- **PR / guest posts / expert commentary** in authoritative publications — relationship-driven, not
  blast-and-pray. The goal is a **web of mutual verification**: many trusted sources saying the same
  thing about you.

## Lever 3 — Make content extractable (so it can be quoted)

When the model does reach your content, it must be able to **lift a clean claim.** Research-backed
on-content moves (each associated with a meaningful citation lift):

- **Quotations, statistics, and citations** — the biggest levers; add verifiable stats *with sources*,
  quotable lines, and references to authoritative sources. **Fluency** (clear writing) also lifts.
- **Direct answers up front** — lead with a **TL;DR / one-sentence answer**, then support it.
- **Q&A format + question-shaped headings** — H2/H3s that match how people ask ("How much does X
  cost?"), each answered in a self-contained, extractable paragraph.
- **Lists, comparison tables, definitions** — structured chunks are easy to quote.
- **Schema / structured data** — **FAQPage** and **Article** schema correlate with materially higher
  citation/selection rates; add author credentials and dates.
- **E-E-A-T signals** — named author + bio, visible publish/update dates, inline references.
- **Freshness** — update content regularly; recency strongly increases citations (Lever recap from
  `how-ai-engines-cite.md`).

(Note: this lever spans **owned web content** — your site/blog — which is broader than social. Be
honest that a website/blog layer matters; this skill foregrounds the social + brand-mention
dimension and the universal mechanics.)

## Lever 4 — Build entity clarity (so AI knows who you are)

The model needs a clear **entity** to associate with your category:

- **Consistent brand description** everywhere (same one-line "who/what/for-whom" across profiles, site,
  listings) → `brand-profile`.
- **Wikipedia / Wikidata** *if genuinely notable* (you can't manufacture notability — don't try).
- **Google Business Profile / knowledge panel** and **consistent NAP** (name/address/phone) across
  listings.
- A clear, corroborated position ("the X for Y") repeated across sources so the model is confident
  recommending you for that query.

## Priority

Start with **Lever 1** (you must be retrievable), run the **audit** to see where you're missing, then
invest most in **Lever 2** (earned/community mentions — where AI looks and where you're likely weakest)
and **Lever 3** (extractability of the content you and others can cite). Lever 4 compounds everything.
