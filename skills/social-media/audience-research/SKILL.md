---
name: audience-research
description: >-
  Use to develop a deep, usable understanding of who a brand creates content for — sharp enough
  that every content skill resonates with them specifically. Run when the user says "who's my
  audience," "audience research," "target audience," "build a persona," "customer profile,"
  "ideal customer profile" / "ICP," "who am I talking to," "understand my followers," or
  before content work that needs more audience depth than the
  brand-profile sketch. Reads brand-profile first and goes deeper: jobs-to-be-done, pains,
  objections, and the audience's ACTUAL language (voice-of-customer), grounded in real sources
  where possible — never demographic theater. Produces an audience.md that content-pillars,
  batch-content-plan, and the content skills read. Works for any business.
metadata:
  version: 1.0.0
license: MIT
---

# Audience Research

Knowing the audience deeply is the highest-leverage input to good content. You can have perfect
voice and clean mechanics and still get ignored if the content doesn't speak to what the
audience actually wants, in words they actually use. This skill builds that understanding — and
writes it down so every other skill can use it.

It goes deeper than the audience sketch in `brand-profile`. Two ideas drive it:

1. **Jobs-to-be-done beat demographics.** "35-year-old marketing manager, urban, likes coffee"
   tells you nothing about what to post. "Wants to stop feeling stupid in budget meetings" tells
   you everything. Capture what the audience is *trying to do*, not what census box they tick.
2. **Mirror their language, not your jargon.** The single biggest lever in copy is using the
   audience's own words for their problem. So this skill collects the **voice of the customer** —
   real phrases from real sources — and builds a language bank the content skills draw from.

## When to use this

- After `brand-profile`, when content needs to land harder and the audience sketch isn't enough.
- Before `content-pillars` or a big `batch-content-plan` (audience pains feed both).
- When the user isn't sure who they're really talking to, or content keeps missing.

**When NOT to use it:** if a current `audience.md` exists, load it, summarize it, and move on
unless the user wants to revisit.

## Step 0 — Read the foundation first

Load `brand-profile.md`. It has the positioning, the audience sketch, and the POV. This skill
expands that sketch into something operational. If there's no brand profile, run it first.

## Step 1 — Gather evidence (don't invent)

The difference between research and guessing is **evidence**. Mine the audience's real words and
problems from whatever sources are available (see `references/voice-of-customer.md`):

- The brand's and competitors' **reviews**; **social comments** and replies.
- **Reddit / forums / communities** where the audience actually talks (to act on subreddit findings → `reddit-marketing`).
- **Support tickets, FAQs, sales-call notes, DMs** — the friction and objections, verbatim.
- **Search queries / "people also ask"** — how they phrase what they want.

Use what the user provides; if the agent can access public sources, mine those too. Where
evidence is thin, **flag the gap and mark assumptions as hypotheses to validate** — never
fabricate audience language or pains.

## Step 2 — Define sharp segments (1–3), and buyer vs follower

Pick the **1–3 segments** that matter most — not "everyone." For each, separate two roles when
they differ (especially in B2B):

- **Buyer** — who decides/pays.
- **Follower** — who actually follows, shares, and champions on social.

They're often different people with different needs; content usually leads with the follower.
See `references/jobs-to-be-done.md`.

## Step 3 — Capture the content-relevant dimensions

For each segment, capture only what changes how you'd create content:

- **Jobs-to-be-done** — functional, emotional, and social jobs.
- **Pains** (what blocks the job) and **desires** (the outcome they want).
- **What they already believe** — so content can meet or challenge it.
- **Objections** — why they hesitate; great content answers these pre-emptively.
- **Where they are** — platforms + mindset on each.
- **Who/what they follow** and **what content they engage with**.
- **Sophistication level** — beginner vs expert (changes vocabulary and depth).

Demographics only if they genuinely change the content (e.g., region for a local business).

## Step 4 — Build the language bank

From the evidence, collect the audience's **actual phrases** — how they describe the problem, the
desired outcome, and their objections — in *their* words, not paraphrased into marketing-speak.
This bank is what makes copy feel like it gets them. See `references/voice-of-customer.md`.

## Step 5 — Write the artifact

Produce `audience.md` using `references/audience-template.md`. Flag the primary segment and the
core **transformation** (before → after). Summarize back and invite edits. Content skills read
this on every task.

## Quality bar — self-check

- Is it built on **jobs-to-be-done and real language**, not demographic theater?
- Is the language bank made of the audience's **actual words**, grounded in sources (or honestly
  flagged as assumption)?
- Are there **1–3 sharp segments**, each usable — not "everyone"?
- Is **buyer vs follower** separated where they differ?
- Does every captured dimension actually **change how you'd write**?
- Could a writer who's never met this audience picture a real person and write to them?

If a field wouldn't change a single post, cut it. If you couldn't source a claim, flag it.

## Edge cases

- **No data / no access** → research what public sources you can; otherwise build a structured
  hypothesis from the brand profile and **clearly label it** as to-be-validated. Don't present
  guesses as findings.
- **B2B buyer ≠ user/follower** → capture both; note which content targets which.
- **Multiple distinct audiences** (e.g., nonprofit: beneficiaries + donors) → separate segments
  with different jobs and language.
- **Niche / technical audience** → sophistication and insider language matter most; get the
  jargon right (or right to avoid).
- **"Everyone is my audience"** → push for the one segment that moves the business most now;
  build for them first.
- **Aspirational audience** (who they *want* vs who they *have*) → note both; don't write to a
  fantasy audience that isn't there yet without saying so.

## Related skills

- `brand-profile` — read first; supplies the audience sketch and positioning.
- `content-pillars` — audience pains/jobs become content pillars.
- `batch-content-plan`, the content skills — use segments, pains, and the language bank.
- `voice-builder` — the brand's voice (distinct from the audience's language captured here).

## References

- `references/jobs-to-be-done.md` — the JTBD lens, pains/desires/objections, why it beats personas.
- `references/voice-of-customer.md` — mining real sources for the audience's language (the core).
- `references/audience-template.md` — the audience.md output schema.
- `references/examples.md` — audience profiles across business types vs the useless generic persona.
