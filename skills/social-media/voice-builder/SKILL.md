---
name: voice-builder
description: >-
  Use to build a reusable voice guide (voice.md) by analyzing a person's or brand's ACTUAL
  writing samples — so every other social skill can write in their exact voice instead of
  generic AI prose. Run this when the user says "build my voice," "sound like me," "capture my
  tone," "analyze my writing," "train on my posts," "ghostwrite in my voice," or when content
  keeps coming out off-voice or generic. Requires real writing samples; if there are none,
  use brand-profile's voice interview instead. Works for anyone: founders, creators, executives
  who are ghostwritten, and brands. Read brand-profile first; this skill produces the deeper
  voice.md that complements it. This skill DEFINES the voice; to apply it per piece — tone
  shifts, style edits, de-AI-ing a draft — use writing-style-and-tone.
metadata:
  version: 1.0.0
license: MIT
---

# Voice Builder

This skill turns real writing into a **reproducible voice** — a `voice.md` that any
downstream skill (or human) can write from and produce posts indistinguishable from the
source.

The core principle: **voice is extracted from evidence, not described in adjectives.**
"Punchy and authentic" is unwritable — it tells a writer nothing. "Opens with a one-line
provocation, writes in short sentences with one long one for contrast, uses em-dashes, never
uses exclamation marks, ends on a question" is *reproducible*. Your job is to convert someone's
writing into rules that specific.

## When to use this

- **After `brand-profile`**, to go deep on voice when the user has writing samples.
- When content keeps coming out generic, stiff, or "not me."
- For **ghostwriting at scale** — capture the voice once, write in it forever.

**When NOT to use this:** if there are no usable samples, don't invent a voice. Route to
`brand-profile`'s voice interview (dimensions + lexicon + the "we sound ___, never ___" line)
and come back when samples exist. A fabricated voice is worse than an honest interview.

## Step 0 — Read `brand-profile` first

Load `brand-profile.md` for identity, audience, and guardrails. `voice-builder` produces the
`voice.md` that sits alongside it. If a `voice.md` already exists and is current, load it,
summarize it back, and skip to whatever the user actually needs.

## Step 1 — Gather the right samples

Quality beats quantity. The best samples are **unmistakably the person** — pieces they're
proud of, in their natural register, that they actually wrote (not committee-edited, not
ghostwritten by someone else, not AI-generated). Aim for **5–10 substantial samples**; 3 is
the floor. **Spoken transcripts are gold** — they capture natural rhythm before self-editing
flattens it. See `references/samples-guide.md` for what to collect and what to exclude.

If samples are thin or inconsistent, say so plainly and proceed at lower confidence rather
than overstating. Honesty about confidence is part of the deliverable.

## Step 2 — Analyze across the six layers (with evidence)

Do not free-associate about "tone." Work the framework in
`references/analysis-framework.md`, layer by layer:

1. **Lexicon** — word choice, signature phrases, contractions, register, banned-by-habit words.
2. **Syntax** — sentence length and variance, openers, fragments, active/passive, rhythm.
3. **Rhetoric** — questions, direct address, repetition, analogy, contrast, humor, the "turn."
4. **Structure** — how they open, build, and close; paragraphing; formatting.
5. **Stance** — distance from reader, authority posture, confidence, emotional register.
6. **Negative space** — what they conspicuously *never* do. Often the most defining layer.

For every pattern you name, **cite the evidence** — quote the sample. A voice guide built on
assertions drifts; one built on quotes holds.

## Step 3 — Distill the voice fingerprint

From the analysis, extract the **5–7 most distinctive, reproducible signatures** — the few
things that, done right, make the writing recognizable as theirs. This is the heart of
`voice.md`. Pair it with the **negative space** (the hard "never"s). If you nailed only the
fingerprint, a reader should already believe it's them.

## Step 4 — Write `voice.md`

Use `references/voice-template.md`. Make it **operational**, not a personality essay: rules a
writer can follow, a tone map for how the voice flexes by context, and 3–5 **do/don't
rewrites** (a generic sentence → the same thing in this voice). Keep real sample snippets in
as calibration anchors.

## Step 5 — Validate with the voice test (do not skip)

A `voice.md` that hasn't been tested is a hypothesis. Run the loop in
`references/validation.md`:

1. Generate a fresh post on a *new* topic using only `voice.md`.
2. Place it beside a real sample. Could the user tell which is which?
3. Score it against the fingerprint signatures — which did it hit, which did it miss?
4. Fix the misses, refine `voice.md`, and repeat once or twice.

Show the user the test post and the result. The voice isn't done until it passes.

## What "great" looks like (self-check before finishing)

- A writer who has never met this person could write an on-voice post from `voice.md` alone.
- **Every claim cites a sample** — no unsupported adjectives.
- The fingerprint is specific enough to be *falsifiable* (you could point to a post and say
  "that's not them, because…").
- Negative space is captured, not just positive traits.
- It **passed the voice test**, and you showed the proof.

If any of these is missing, you're not done.

## Edge cases — handle explicitly

- **Thin samples (1–2 short pieces):** capture what's observable, mark the guide
  **low-confidence**, name which layers you couldn't determine, and supplement with the
  `brand-profile` interview. Don't bluff certainty.
- **Inconsistent samples (voice drift):** don't average them into mush. Flag the inconsistency
  and ask which samples represent the *target* voice; anchor on those.
- **Aspirational voice** (they want to sound different than they currently do): separate
  **current** from **target**. Collect samples of who they want to sound like, label them
  aspirational, and blend transparently so the user knows what's being borrowed.
- **Multiple voices** (company vs founder; several creators): one `voice.md` per voice, named.
  **Never blend voices** — it produces a person who exists nowhere.
- **Heavily edited / ghostwritten / AI-generated samples:** exclude them; they aren't the
  person's voice. Say why.
- **Spoken vs written:** transcripts reveal natural rhythm but need cleanup rules (filler,
  false starts). Capture the rhythm, note the cleanup.
- **Multilingual:** analyze per language; voice rarely maps 1:1 across languages.

## Related skills

- `brand-profile` — read first; provides identity, audience, guardrails.
- `writing-style-and-tone` — applies the `voice.md` per piece and per moment (tone map, edit
  passes, AI-tell sweep). This skill writes the constitution; that one governs by it.
- Every content skill (`caption-writer`, `reels-script`, `thread-writer`, `linkedin-post-writer`, …)
  reads the resulting `voice.md`.
- `audience-research`, `content-pillars` — complementary foundation skills.

## References

- `references/samples-guide.md` — how to source and vet the right writing samples.
- `references/analysis-framework.md` — the six-layer framework for deconstructing voice.
- `references/voice-template.md` — the exact `voice.md` structure to produce.
- `references/validation.md` — the voice test: generate, compare, score, refine.
- `references/examples.md` — two contrasting worked examples, end to end.
