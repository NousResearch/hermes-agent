---
name: viral-reverse-engineering
description: >-
  Use to reverse-engineer why a piece of content went viral (or overperformed) — yours or someone
  else's — and extract the repeatable mechanism to apply to your own content. Run when the user says
  "why did this go viral," "break down this viral post/video," "reverse engineer," "what made this
  work," or wants to learn from viral content. Sources the observable signal first (intake,
  transcript, screenshots, top comments, visible stats — an agent usually can't watch a video from
  a link) and never fabricates what it can't see. Reads brand-profile and
  audience first, deconstructs the piece layer by layer, isolates the real driver, runs a
  replicability check, extracts the transferable principle, and applies it to the user's niche via
  the content skills. Mechanism, never a copy; flags non-replicable virality; visible signals only
  (no WoopSocial analytics). Single-POST teardown only: for the account-level competitive
  landscape use competitor-analysis; for riding a live trend use trend-jacking.
metadata:
  version: 1.0.0
license: MIT
---

# Viral Reverse-Engineering

Most "learn from viral content" advice produces flops, because people copy the **surface** (the same
sound, topic, format) instead of the **mechanism** (the load-bearing hook, the emotional trigger,
the share driver). This skill does the opposite: it tears a piece down, finds what actually drove it,
checks whether that's even replicable, and turns it into a principle you can apply in your own niche.

Two commitments:

1. **Mechanism, not surface.** Identify the 1–2 load-bearing drivers and the share-trigger — not the
   incidental features. Copying noise reproduces noise.
2. **Honest about luck and survivorship.** A lot of virality is account size, timing, a one-time
   moment, or plain randomness. When success isn't replicable, say so — a false formula is worse than
   none.

## Step 0 — Read the foundation

Load `brand-profile.md` and `audience.md` (for the "apply to your niche" step).

## Step 1 — Source the content (the step everyone skips)

**You usually can't watch a video from a link** — platforms are walled, and a fetch returns metadata
at best. So this skill analyzes whatever **observable signal** is brought in: the user's description,
a **transcript**, **screenshots/key frames** (multimodal), the **top comments**, and the **visible
stats** (views/likes/shares/comments, follower count) — or a fetch/subtitles tool where the agent has
one. Run the **structured intake** in `references/sourcing-the-content.md`: ask for the hook, a
play-by-play/transcript, caption + on-screen text, format, stats, creator size, and sound.

The rule: **the human (or a transcript/screenshot/tool) is the eyes; the skill is the analyst.**
Never fabricate frames or lines you weren't given — analyze what's provided and **name the gaps**.
Also: **patterns need multiple examples** — one viral post is an anecdote. (WoopSocial has no
analytics; work from visible/native signals or pasted data.)

## Step 2 — Deconstruct (the teardown)

Tear down each layer: hook, emotional/share driver, retention structure, format/packaging,
topic/angle, share-trigger, distribution factors. One line per layer; don't praise everything. See
`references/deconstruction-framework.md`.

## Step 3 — Isolate the real driver (counterfactual)

For each notable feature, ask **"remove this — does it still pop?"** Whatever it can't lose without
collapsing is a **driver**; what it can lose is **incidental**. Usually only 1–2 layers are
load-bearing (typically the hook + the emotional/share trigger). Most bad analysis credits the noise.

## Step 4 — Identify the share-trigger

Virality = shares, so name *why people sent it to someone else*: identity/self-expression,
high-arousal emotion (awe/anger/humor/inspiration), social currency, practical value, relatability,
story. A piece with no share-trigger gets views, not virality. See `references/why-things-spread.md`.
(The **top comments** are the best evidence here — see `references/sourcing-the-content.md`.)

## Step 5 — Replicability check

Screen for confounds before extracting anything: **account-size** advantage, **luck/variance**,
**one-time moments**, **survivorship bias**, **sample size**. If the success is mostly confound,
**flag it as non-replicable** and don't invent a principle. See `references/replicability-and-application.md`.

## Step 6 — Extract the principle + apply to your niche

State the mechanism in one line, translate it to the user's subject (same *mechanism*, your topic),
and hand execution to the content skills (`hook-writer`, `tiktok-script`, `reels-script`,
`caption-writer`, `carousel-writer`) in the brand voice. Output is "the lever is X; here's X applied
to you" — **never a copy**. Build a swipe file of recurring patterns over time.

## Quality bar — self-check

- Did I **source real input** (intake/transcript/screenshots/comments), and **not fabricate** what I
  couldn't see — naming the gaps?
- Did I find the **mechanism** (1–2 real drivers + the share-trigger), not the surface?
- Did the **counterfactual** rule out incidental features?
- Did I run the **replicability check** and flag confounds/luck/small-sample honestly?
- Is the output a **principle applied to the user's niche**, not a copy?
- Did I respect the **ethics line** (inspiration, not plagiarism/IP theft)?
- Did I use **visible/native signals** with no analytics claims, and make **no virality guarantees**?

## Edge cases & pushback

- **Bare link, nothing else** → explain you can't watch the video; run the intake (ask for
  transcript/screenshots/stats) or use a subtitles/fetch tool if available; don't pretend you saw it.
- **Partial input** (transcript only, screenshots only) → analyze what's there, **name what you can't
  assess** (e.g., pacing/edit, or the spoken layer).
- **"Copy it exactly with our product"** → mechanism + your own substance, not a surface copy
  (derivative + IP risk).
- **"It was the sound/topic"** → counterfactual-test it; usually the hook + trigger were the real
  lever.
- **Huge-account / one-time virality** → flag non-replicable; don't extract a false formula.
- **One example** → anecdote, not a pattern; tear down several to find recurring mechanisms.
- **"Guarantee us viral"** → no guarantees (luck/distribution); stack the odds via mechanisms.
- **No data to judge "viral"** → use visible signals; be clear about the limits.

## Related skills

- `brand-profile`, `audience-research` — relevance + the "apply to your niche" step.
- `hook-writer` — the most common load-bearing driver; `trend-jacking` — overlapping "why it spread."
- `tiktok-script`, `reels-script`, `caption-writer`, `carousel-writer` — execute the extracted principle.
- `competitor-analysis`, `analytics-and-reporting` (advisory) — broader performance analysis.

## References

- `references/sourcing-the-content.md` — how the content gets into context (intake, transcripts, screenshots, comments, tools) + graceful degradation. **Start here.**
- `references/deconstruction-framework.md` — the layer-by-layer teardown + the counterfactual driver test.
- `references/why-things-spread.md` — the share-trigger psychology (why people share).
- `references/replicability-and-application.md` — survivorship/luck/sample-size honesty; extract + apply; ethics.
- `references/examples.md` — worked teardowns, including a non-replicable case.
