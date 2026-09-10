---
name: carousel-writer
description: >-
  Use to write a carousel — a swipeable multi-slide post (Instagram carousel, LinkedIn
  document/PDF carousel, or TikTok photo carousel) — one of the highest-save, highest-dwell
  formats. Run when the user says "write a carousel," "carousel post," "slides for X," "LinkedIn
  document/PDF post," "swipe post," "turn this into a carousel," or wants multi-slide teaching or
  story content. Reads brand-profile and voice first; the cover slide hooks and promises the
  swipe (via hook-writer); each slide advances one idea and earns the next swipe; ends on a
  save-leaning CTA. Specifies the design direction and bundles an image-generation prompt pack
  (Nano Banana / Ideogram) for the slide visuals. Hands the caption to caption-writer and
  scheduling to scheduling-and-queue.
metadata:
  version: 1.0.0
license: MIT
---

# Carousel Writer

A carousel is a swipeable teaching-or-story post — and one of the best formats there is for
**saves** and **dwell time** (both strong ranking signals). But it only works if people swipe,
and they only swipe if each slide earns the next. So this skill is opinionated about the cover,
the one-idea-per-slide discipline, and keeping slides legible — not dumping paragraphs onto
images.

Three principles:

1. **The cover slide is the whole game.** Slide 1 must hook *and* promise the payoff — it decides
   whether anyone swipes past it. Treat it like a thread opener or a Reel hook.
2. **One idea per carousel, one idea per slide.** A cohesive arc across slides, each advancing it
   by one beat. A slide crammed with three points loses the swipe.
3. **It's copy *and* design.** Minimal, legible text per slide; a clear visual hierarchy. A
   carousel is read at a glance on a phone — walls of text kill it.

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. Voice applies to every slide (concise) and to the caption.
Carry guardrails and any compliance/disclosure rules.

## Step 1 — Lock the idea, goal, and platform

- **One idea** — the single thing this carousel teaches or tells.
- **Goal** — usually saves or authority (carousels excel here), sometimes comments.
- **Platform** — Instagram carousel, **LinkedIn document/PDF carousel** (very high dwell/saves),
  or TikTok photo carousel. Specs differ (see `references/platform-specs.md`).

## Step 2 — Write the cover slide (hook + promise)

Use `hook-writer`. The cover must hook (stop the scroll) **and** promise what the swipe delivers
("5 hooks that actually work →"). It must be legible at a glance and at thumbnail size. No
slow-build cover. See `references/slide-craft.md`.

## Step 3 — Architect the carousel

Pick a structure (how-to, listicle, story, before→after, myth-bust, framework — see
`references/architecture.md`) and outline the **spine**: the cover's promise and the ordered slides
that deliver it. One idea per slide. Decide slide count by the idea, not a target — a tight 6 beats
a padded 10.

## Step 4 — Write the slides

Each slide: a clear **headline** + at most a short supporting line, front-loaded and legible.
Maintain swipe momentum (progress, a small pull forward). Keep every slide in the brand voice and
ruthlessly minimal — if it needs a paragraph, it's two slides. See `references/slide-craft.md`.

## Step 5 — Land the payoff and a save-leaning CTA

End with a payoff/summary slide, then a CTA slide. Carousels win on **saves**, so the default CTA
is "save this for later" (plus optionally follow/comment). One ask. A link, if any, goes to the
caption/bio/first comment per platform.

## Step 6 — Specify the visuals (design direction + image pack)

A carousel is visual. Provide:

- **Design direction** — layout, hierarchy, palette, type, and a consistent template across slides
  (consistency is what makes it look pro).
- **The image-generation prompt pack** — bundled prompts (Nano Banana / Ideogram, which render
  legible in-image text) to produce on-brand slides, with the exact slide text specified. Or a
  clean template spec for Canva/a designer. Flag **AI-disclosure** for generated images. See
  `references/image-prompt-pack.md`. (For deeper visual systems, see `design-and-templates`.)

## Step 7 — Hand off

- **Caption:** the carousel post needs a caption that pulls people into slide 1 → `caption-writer`.
- **Schedule:** hand the finished carousel + caption to `scheduling-and-queue`.

## Quality bar — self-check

- Does the **cover hook AND promise** the swipe, legible at a glance?
- Is it **one idea**, with a clear spine — one idea per slide, no crammed slides?
- Does **each slide earn the next swipe** (momentum, no filler slides)?
- Is the text **minimal and legible** on mobile — not a wall?
- Does it **end on a payoff + one save-leaning CTA**?
- Is there a **consistent design direction** and an image pack (or template) to produce it?
- Is **AI-disclosure** handled for generated visuals; no fabricated stats on slides?
- On-voice throughout?

## Edge cases & pushback

- **"Make it 10 slides"** when the idea is 5 → resist padding; a tight carousel beats a bloated
  one. Offer the strongest N, not a quota.
- **Better as something else** → a single point is a single image; a dynamic story is often a Reel;
  a text argument may be a thread. Say so rather than forcing a carousel.
- **Text-heavy source** (a blog) → atomize into one idea per slide, don't paste paragraphs (see
  `cross-platform-repurposing`).
- **No design tool / no AI images** → provide a clean text-template spec that works on a simple
  branded background; don't block on production.
- **Regulated/sensitive** → carry compliance guardrails onto slides; only verifiable claims.

## Related skills

- `brand-profile`, `voice-builder` — voice + guardrails.
- `hook-writer` — the cover hook.
- `design-and-templates` — deeper visual design system for slides.
- `caption-writer` — the post caption; `cross-platform-repurposing` — source → carousel.
- `nano-banana`, `ideogram` (creative-tool mini-skills) — generate the slide visuals.
- `scheduling-and-queue` — schedule the finished carousel.

## References

- `references/architecture.md` — carousel structures, the spine, one-idea-per-slide, slide count.
- `references/slide-craft.md` — slide-level writing + visual hierarchy + the cover formula.
- `references/platform-specs.md` — IG / LinkedIn document / TikTok carousel specs.
- `references/image-prompt-pack.md` — the bundled image-generation prompt pack (Nano Banana/Ideogram).
- `references/examples.md` — worked carousels slide-by-slide, with prompts, vs a text-dump version.
