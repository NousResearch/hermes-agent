---
name: hook-writer
description: >-
  Use to write the hook — the opening that earns attention — for any social content: a
  caption's first line, a video's first three seconds, a carousel cover slide, a thread opener,
  a YouTube title, or an email subject. Run when the user says "write a hook," "hook for this,"
  "opening line," "first three seconds," "cover slide," "make this scroll-stopping," or when
  good content keeps getting ignored. Reads brand-profile and voice first so hooks sound like
  the brand, not viral-bait templates. Hooks must be TRUE to the content that follows — this
  skill extracts the hook from the post's strongest element and never overpromises. For full
  captions use caption-writer; for full video scripts use the video skills. This writes the
  opening itself.
metadata:
  version: 1.0.0
license: MIT
---

# Hook Writer

The hook is the highest-leverage line in any piece of content. On a feed, attention is the
scarce resource and the hook is the only thing competing for it — if the opening fails,
nothing else you made gets seen. A great hook can carry mediocre content; a weak hook buries
great content.

Two ideas run through everything here:

1. **A hook works by opening a gap.** Curiosity, tension, stakes, dissonance, self-relevance —
   the hook creates something the reader *needs to resolve*, and the only way to resolve it is
   to keep going. Hooks are about the **mechanism**, not the template. "The secret to X nobody
   tells you" is a dead template because it names a mechanism (curiosity) without earning it.
2. **A hook is a promise, and the content must pay it off.** A hook that overpromises wins the
   tap and loses the trust — and platforms punish the bounce that follows. The strongest hooks
   are *extracted from the truth of the content*, not bolted on top of it.

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. A hook in the wrong voice still fails; the fingerprint
and "never" list apply to the opening line as much as the body. If neither exists, run
`brand-profile` (and `voice-builder` if samples exist) first.

## Step 1 — Pin the format and its cutoff

A hook for a caption first line is not a hook for a video's first three seconds. Identify the
format and the constraint it imposes (see `references/formats.md`):

- **Caption** — one line before the "…more" cutoff (~125 chars on Instagram).
- **Video** — the first 1–3 seconds, across *three channels at once*: what's on screen, the
  first spoken words, and on-screen text. Must work muted.
- **Carousel cover** — slide 1 is the hook; it must promise the swipe.
- **Thread opener** — the first post must promise the thread's payoff.
- **YouTube** — title + thumbnail together.
- **Email** — subject + preview text, mobile-truncated.

## Step 2 — Find the hook *in the content*

Don't invent a hook and hope the content matches. Mine the actual material for its strongest
raw element:

- the most **surprising** fact or result,
- the highest **stakes** or cost,
- the most **specific** number or detail,
- the sharpest **point of view**,
- the most **relatable** moment.

That element is the hook's raw material. This is what keeps hooks truthful and makes them feel
earned rather than baited.

## Step 3 — Generate many, across different mechanisms

Write **5–10 candidate hooks**, deliberately using *different mechanisms* from
`references/mechanisms.md` (curiosity gap, contrarian, stakes, specificity, identity, story,
mistake, transformation, question, timeliness, authority). Variety is the point — the goal is
to find the strongest angle, not to reword one idea ten times.

## Step 4 — Score and select

Run the candidates through the rubric in `references/scoring.md` and keep the best 1–2. The
rubric checks: does it open a real gap? is it specific? is it true to the content? does it fit
the format's cutoff? does it sound like them? would a stranger *need* the next beat? Discard
the rest without sentiment — most hooks you write should die.

## Step 5 — Tune to format and voice; confirm the payoff

Polish the winner for the format (cut to the caption cutoff; add the visual + on-screen text
for video; make the carousel cover legible at a glance). Re-check it sounds like the brand. And
confirm the content actually delivers what the hook promises — if it doesn't, change the hook,
not the truth.

## Deliver options, recommend one

Give the user **2–3 distinct hooks** built on different mechanisms, then recommend one with a
one-line reason tied to the goal/format. Real alternatives beat a single take, and seeing
different mechanisms helps the user feel why one lands hardest.

## Quality bar — self-check

- Does the hook open a **gap** a stranger needs to close?
- Is it **specific** (a real number, detail, or claim), not vague?
- Is it **true** — does the content pay it off without a bait-and-switch?
- Does it land **inside the format's cutoff** (chars, or the first 3 seconds)?
- Does it **sound like them** (fingerprint hit, "never" list respected)?
- Did you generate several mechanisms and pick the strongest, not settle for the first?

If it doesn't pass "would a stranger need the next beat?", it isn't a hook yet.

## Edge cases

- **Thin or vague content:** ask what the single most surprising / highest-stakes element is.
  Don't fabricate a hook the content can't support.
- **Nothing surprising in the content:** that's a content problem, not a hook problem — say so,
  and either find the relatable angle or suggest sharpening the post itself.
- **Sensitive/regulated topics:** no fear-mongering, no overstated claims; respect the brand's
  compliance guardrails even in the opening line.
- **Video, muted-first:** the on-screen text and the visual must hook on their own; assume sound
  off.
- **Tempted to overpromise for reach:** don't. A truthful, slightly less sensational hook that
  the content delivers beats a viral-bait line that burns trust and bounces.

## Related skills

- `brand-profile`, `voice-builder` — read first; supply voice and guardrails.
- `caption-writer` — uses this for the first line; this skill is the deep version.
- `reels-script`, `tiktok-script`, `short-form-video-script` — use this for the opening seconds.
- `thread-writer` — uses this for the thread opener.
- `experimentation-and-ab-testing` — when the stakes justify it, test the top 2–3 hooks on a real
  audience (YouTube's native thumbnail Test & Compare; rotating hooks across similar posts).
  The rubric picks your best guess; audience data outranks the rubric.

## References

- `references/mechanisms.md` — the hook mechanism taxonomy (why each one works).
- `references/formats.md` — hook craft by format and the cutoffs that govern each.
- `references/scoring.md` — the scoring rubric, the generate-and-select method, anti-patterns.
- `references/examples.md` — content → many hooks → scored → chosen, worked end to end.
