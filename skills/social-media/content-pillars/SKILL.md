---
name: content-pillars
description: >-
  Use to define a brand's content pillars — the 3–5 recurring themes it posts about — the
  strategic backbone between the brand profile and the content calendar. Run when the user says
  "content pillars," "content buckets," "what should I post about," "content themes," "content
  strategy," "pillars," or before planning a batch or calendar. Reads brand-profile first and derives pillars from
  positioning, point of view, audience, and proof — real ownable themes, never generic
  "educational / behind-the-scenes / promotional" buckets (those are intents and formats, not
  pillars). Produces a content-pillars.md that batch-content-plan and the content skills read.
  Works for any business: B2B, ecommerce, local, personal brand, nonprofit.
metadata:
  version: 1.0.0
license: MIT
---

# Content Pillars

Content pillars are the few recurring **themes** a brand returns to — the strategic layer
between *who the brand is* (`brand-profile`) and *what it posts this week* (`batch-content-plan`).
Good pillars make a feed coherent, ownable, and infinitely easy to plan; their absence is why
so many brands post random things that add up to nothing.

The single most important idea in this skill — and the thing almost everyone gets wrong:

> **A pillar is a theme you own. It is NOT an intent or a format.** "Educational,"
> "behind-the-scenes," and "promotional" are not pillars — *educational* is an intent (the job a
> post does) and *behind-the-scenes* is a format/treatment. A real pillar is a subject area like
> "payroll without a specialist" or "the craft of coffee, demystified." Get this distinction
> right and everything downstream gets sharper. See `references/what-makes-a-pillar.md`.

## When to use this

- After `brand-profile`, before planning content at any scale.
- When the user doesn't know "what to post about," or their feed feels random.
- When `batch-content-plan` needs pillars and none are defined.

**When NOT to use this:** if a current `content-pillars.md` exists, load it, summarize it, and
move on unless the user wants to revisit it.

## Step 0 — Read the foundation first

Load `brand-profile.md` (and `voice.md`). Pillars are *derived from strategy* — positioning,
point of view, audience, proof, offers — not brainstormed from thin air. If there's no brand
profile, run `brand-profile` first; pillars built without it are guesses.

## Step 1 — Mine the brand profile for raw material

Pull the candidate sources (see `references/deriving-pillars.md`):

- **Positioning + point of view** → the *signature* pillar (the thing the brand wants to be
  known for; the hill it will die on).
- **Audience jobs-to-be-done / pains** → audience-relevant pillars (their problems, not your
  features).
- **Proof + expertise** → credibility pillars (what the brand can authoritatively teach).
- **The human / brand story** → a connection pillar.
- **Offers / product** → at most one conversion-linked pillar, used sparingly.

## Step 2 — Generate candidate pillars

From each source, draft candidate themes. Aim wide first (8–12 candidates), phrased as
*subjects*, not formats — "small-business money mistakes," not "educational posts."

## Step 3 — Test and select 3–5

Run each candidate through the **pillar test** (`references/what-makes-a-pillar.md`):

1. **Ownable** — can you say *why this brand specifically* is credible or distinctive here?
2. **Relevant** — does the target audience genuinely care?
3. **Strategic** — does it advance a real goal (awareness, trust, conversion)?
4. **Sustainable** — can you brainstorm **10+ post ideas** from it right now?

Keep the 3–5 strongest. A candidate that fails *ownable* or *sustainable* is not a pillar. Fewer,
sharper pillars beat more, vaguer ones — a feed with eight themes has none.

## Step 4 — Define each pillar

For every selected pillar, capture: a clear name, what it covers, **why the brand owns it**, who
it's for, the goal it serves, 5–8 example angles/sub-topics, and the proof/assets it draws on.
This is what makes pillars usable by the planning and content skills. See
`references/pillar-template.md`.

## Step 5 — Balance the mix

A healthy pillar set has:

- **One signature pillar** — the differentiator, the brand's distinct POV. Every brand needs
  one.
- **Supporting pillars** — audience problems, credibility, connection.
- **At most one conversion-linked pillar**, used as the minority (promotion rides on the value
  the others create — see the mix logic in `batch-content-plan`).

Spread the pillars across goals so the brand isn't all-teaching or all-selling.

## Step 6 — Write the artifact

Produce `content-pillars.md` using `references/pillar-template.md`. Summarize it back to the
user, flag the signature pillar, and invite one round of edits. `batch-content-plan` and the
content skills will read this on every task.

## Quality bar — self-check

- Are the pillars **themes**, not intents or formats? (No "educational" as a pillar.)
- Is each one **ownable** — with a reason *this* brand is credible/distinctive on it?
- Is each **audience-relevant**, not just brand-interesting?
- Could each generate **10+ post ideas** (sustainable)?
- Is there exactly **one signature pillar**, and is promotion a minority?
- Are there **3–5**, not eight vague ones?
- Does each pillar trace back to something real in the brand profile?

## Edge cases

- **User lists generic buckets** ("educational, BTS, promo") → explain those are intents/formats,
  and convert them into real themes derived from the brand.
- **Too many candidate topics** → narrow to the 3–5 that best pass the pillar test.
- **Brand with no obvious distinctive theme** → mine the POV and audience pains harder; if
  there's truly no differentiation, the signature pillar becomes *voice + a consistent angle*,
  and you say so honestly.
- **Personal brand / creator** → pillars are person-led (what they're known for, their journey,
  their craft); keep them tied to the person, not a faceless topic list.
- **Multiple audiences** (e.g., nonprofit: beneficiaries + donors) → some pillars may serve
  different audiences; note which.
- **Multi-product / broad business** → resist one pillar per product; pillars are audience
  themes, not a catalog.

## Related skills

- `brand-profile` — read first; supplies positioning, audience, POV, proof.
- `audience-research` — deepen audience pains that feed audience pillars.
- `batch-content-plan` — consumes pillars to build the calendar.
- The content skills (`caption-writer`, `reels-script`, …) — each post is tagged to a pillar.

## References

- `references/what-makes-a-pillar.md` — the definition, the pillars-vs-intents-vs-formats
  distinction, the pillar test, anti-patterns.
- `references/deriving-pillars.md` — the method for mining the brand profile into pillars.
- `references/pillar-template.md` — the content-pillars.md output schema.
- `references/examples.md` — ownable pillar sets across six business types.
