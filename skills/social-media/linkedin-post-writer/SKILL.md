---
name: linkedin-post-writer
description: >-
  Use to write a LinkedIn post — professional, thought-leadership, or founder/personal-brand
  content optimized for LinkedIn's specific mechanics (dwell time, early comments, the "see more"
  hook). Run when the user says "write a LinkedIn post," "LinkedIn," "post this on LinkedIn,"
  "thought leadership post," "founder post," or wants professional/B2B social content. Reads
  brand-profile and voice first, uses hook-writer for the opening line that must win the "see
  more" expand, formats for dwell, drives genuine comments, and routes links to the first comment
  to protect reach. Avoids LinkedIn cringe (broetry, engagement bait). This is the single-post
  craft: to "grow on LinkedIn" or set the account strategy use linkedin-growth; for company-page
  strategy use linkedin-company-pages; for general captions use caption-writer; for X threads use
  thread-writer; for the carousel slides use carousel-writer.
metadata:
  version: 1.0.0
license: MIT
---

# LinkedIn Post Writer

LinkedIn is its own game, and posts written like generic social captions underperform on it. The
algorithm rewards two things above all — **dwell time** (how long people actually spend on the
post) and **early, meaningful comments** — and the format has its own rules: the "…see more" hook,
whitespace that makes long posts readable, links that suppress reach, and personal voices that
beat company pages.

Two commitments shape every post here:

1. **Earn the read, then earn the comment.** Write the opening to win the "…see more" tap, format
   the body so people actually read it (dwell), and close in a way that invites a real comment.
2. **Substance over LinkedIn theatre.** The platform is full of broetry and "Agree? 👇"
   engagement bait — which both the algorithm and credible professionals increasingly punish.
   This skill does what genuinely works and refuses the cringe. See
   `references/avoiding-cringe.md`.

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md` (and `audience.md` for the professional segment). The
LinkedIn register is usually a notch more authority-leaning than other platforms, but still human
and first-person. Carry guardrails and any compliance rules.

## Step 1 — Lock the idea, goal, and who's posting

- **One idea** — the single point, story, or insight.
- **Goal** — authority, leads/warming, comments, or reach (shapes the close).
- **Person vs company** — personal/founder profiles consistently out-reach company pages and read
  as more trustworthy. Recommend a **personal** voice for thought leadership; reserve the company
  page for announcements/brand. (See `references/linkedin-mechanics.md`; the page's own strategy
  lives in `linkedin-company-pages`.)

## Step 2 — Write the "see more" hook

Use `hook-writer`. The first line (~140 chars on mobile before "…see more") must earn the expand —
that tap is the first dwell signal, and without it the post is invisible. Front-load tension; no
warm-up, no "I'm excited to announce."

## Step 3 — Structure for dwell

Pick a LinkedIn format (personal-story→lesson, contrarian/POV, how-to, case study,
observation — see `references/post-formats.md`), then format for readability: short sentences,
**one-to-two-line paragraphs, generous whitespace**, a clear build. A wall of text kills dwell.
(Whitespace as genuine readability — not the broetry gimmick of one dramatic fragment per line.)

## Step 4 — Write the body

Deliver the value or story the hook promised, in the brand voice at a LinkedIn register. Long is
fine **if it earns the read** (limit ~3,000 chars) — every line must pull to the next. Don't pad;
don't bait-and-switch the hook.

## Step 5 — Close for comments + one soft CTA

End with a **genuine discussion question** the audience can't help answering — relevant to their
work, easy but interesting, not "Agree?". Early comments drive reach, and the author replying to
them boosts it further. Add at most **one** soft CTA (follow for more, a resource) — LinkedIn
isn't the place to hard-sell.

## Step 6 — Mechanics

- **Links:** external links in the body suppress reach. Put the link in the **first comment** and
  say "link in comments," or post link-free. Native content wins. See
  `references/linkedin-mechanics.md`.
- **Hashtags:** 3–5 relevant, at the end.
- **Format options:** a **document/PDF carousel** is a high-dwell, high-save LinkedIn power format
  — if the idea suits it, write the post copy and hand the slides to `carousel-writer`.
- **AI disclosure:** disclose AI-generated media/posts per the brand's compliance rules.

## Deliver options, recommend one

Give **2–3 distinct posts** (different angles or formats), then recommend one with a one-line
reason tied to the goal.

## Quality bar — self-check

- Does the **first line win the "…see more" tap** within the cutoff?
- Is the body **formatted for dwell** (whitespace, scannable) and genuinely worth reading?
- Does it **sound like them** at a LinkedIn register (voice + "never" list)?
- Does it **close on a real question** that invites comments, with one soft CTA?
- Are **links routed to comments**, hashtags sensible, format native?
- Is it **free of cringe** — no broetry, no engagement bait, no fake vulnerability?
- Would you be comfortable if a respected peer read it?

## Edge cases & pushback

- **User asks for engagement bait** ("write a 'comment YES' post," broetry, a fake-emotional
  story) → decline that approach and offer the genuine alternative that actually performs and
  protects credibility. (`references/avoiding-cringe.md`.)
- **Idea is really a one-liner** → a short, sharp LinkedIn post or a poll can beat a forced essay;
  say so.
- **Company announcement** → can run from the page, but suggest a personal founder/employee post
  alongside for reach.
- **Regulated/sensitive** → carry compliance guardrails; no unverifiable claims; required
  disclaimers.
- **Hard-sell request** → reframe toward value-first; LinkedIn punishes overt selling.
- **No real story/insight** → don't manufacture a fake parable; sharpen the actual point or pick a
  format that fits what's true.

## Related skills

- `brand-profile`, `voice-builder`, `audience-research` — voice, register, professional audience.
- `hook-writer` — the "see more" opening line.
- `caption-writer` — general captions; `thread-writer` — X threads; `carousel-writer` — the slides
  for a document-carousel post.
- `linkedin-growth` — the personal-profile growth system (cadence, commenting, the golden hour);
  `linkedin-company-pages` — the company-page strategy. This skill stays on the single post.
- `scheduling-and-queue` — schedule the finished post.

## References

- `references/linkedin-mechanics.md` — dwell, comments, the see-more cutoff, links, formats, personal-vs-company.
- `references/post-formats.md` — LinkedIn post structures and the comment-driven close.
- `references/avoiding-cringe.md` — what NOT to do (broetry, engagement bait) and what works instead.
- `references/examples.md` — worked LinkedIn posts, formatted for dwell, vs a cringe version.
