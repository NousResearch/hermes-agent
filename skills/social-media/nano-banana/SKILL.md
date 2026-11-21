---
name: nano-banana
description: >-
  Use to write great prompts for Nano Banana (Google's Gemini image model: Nano Banana Pro = Gemini 3
  Pro Image) to generate or edit images for social media — the image-prompt-craft mini-skill. Run when
  the user wants a nano-banana / Gemini image prompt, an AI image for a post (quote graphic, carousel
  slide, thumbnail, infographic, product shot, ad creative), or conversational image editing. Reads
  brand-profile for brand colours/style. Strengths: best-in-class legible in-image text (route
  typography/layout-led design to ideogram), multi-image consistency, conversational editing, and
  Google's photoreal lane (Imagen deprecated); surreal art stays Midjourney's. Teaches prompt anatomy,
  the text-first approach, reference-image consistency, and per-platform aspect ratios. Honest: SynthID
  watermark + AI disclosure; never real identifiable people or copyrighted characters/logos; verify
  spelling/data. The API/connection and WoopSocial upload flow live in tools/integrations/nano-banana.md.
metadata:
  version: 1.0.0
license: MIT
---

# Nano Banana (image prompt craft)

Write prompts that get studio-quality images out of **Nano Banana** — Google's Gemini image model
(**Nano Banana Pro = Gemini 3 Pro Image**; the Flash variant is faster/cheaper). This is the
**prompt-craft** layer of a three-layer setup:

- **Connection/API** (model IDs, auth, the *generate → upload to WoopSocial Media → attach* flow) →
  `tools/integrations/nano-banana.md`.
- **Prompt craft** (this skill) → how to ask for the right image well.
- **In-skill application** → e.g. `carousel-writer`'s image-prompt-pack.

> Fast-moving area — re-verify model names/capabilities quarterly.

## Reach for Nano Banana when… (match the job)

Its real strengths: **legible in-image text** (best in class), **multi-image/character consistency**,
**conversational editing**, and **Search-grounded infographics**. Reach **elsewhere** for
**typography/layout-led design** (→ `ideogram`) or **surreal/fine-art** aesthetics (Midjourney).
(**Imagen**, the former photoreal sibling, is **deprecated** — Gemini API shutdown Aug 17, 2026 — so
Google-side photoreal work now lands here.) Don't default it to every job. (Details:
`references/when-and-how-to-prompt.md`.)

## Step 0 — Read the brand + the job

Load `brand-profile.md` (colours, visual style, do/don't). Identify the **job** (quote graphic /
carousel / thumbnail / infographic / product shot / background / ad / edit) and the **platform
aspect ratio**.

## Step 1 — Write the prompt (natural language, not keyword spam)

Describe, don't pad. Include **subject · composition/camera · action/context · lighting/atmosphere ·
style/medium · constraints + exact text**. Drop "4k, masterpiece, trending on artstation" tokens —
modern Gemini models reason over natural language. Bake in the **brand** palette/style. Set the
**aspect ratio**. Iterate cheap at 1K, finalize at 2K/4K. See `references/when-and-how-to-prompt.md`.

## Step 2 — Use the superpowers deliberately

- **Text:** put the **exact words in quotes** + font + placement; settle the **copy first** (the
  text-first approach), keep it short, and **verify the spelling** in the output.
- **Consistency:** use **reference images** and a **locked style description**; generate a set **one at
  a time** ("everything the same except X"); hold the aspect ratio; review for drift.
- **Editing:** natural-language changes to a hero image for locale/seasonal/A-B variants.

See `references/text-and-consistency.md`.

## Step 3 — Pick the recipe

Adapt a pattern for the job (quote graphic, carousel set, thumbnail, infographic, product/lifestyle,
background, ad), with the right aspect ratio and brand grounding. See `references/social-recipes.md`.

## Step 4 — Verify, disclose, ship

- **Verify** rendered text and any infographic **data** before publishing (the model can misspell /
  misplace facts).
- **Disclose** AI generation per platform/region (Meta "Made with AI", EU AI Act); every image carries
  a **SynthID** watermark (+ C2PA) — don't strip it or pass the image off as a non-AI photo.
- **Ship:** generate per the integration guide → **upload bytes to WoopSocial Media → attach to the
  post** via `scheduling-and-queue`. WoopSocial doesn't generate images.

## Quality bar — self-check

- Did I **match the tool to the job** (and say when to use another model)?
- Is the prompt **natural-language and specific** (subject/composition/lighting/style/text), **brand-
  grounded**, with the **right aspect ratio**?
- For text: **exact words in quotes** + font + placement, **copy-first**, **spelling verified**?
- For sets: **references + locked style**, one-at-a-time, reviewed for drift?
- Did I handle **SynthID + disclosure**, **verify text/data**, and **refuse real people / IP**?
- Did I point to **`tools/integrations/nano-banana.md`** for the API + WoopSocial flow (no claim that
  WoopSocial generates images)?

## Edge cases & pushback

- **Photoreal headshot / surreal art** → suggest the better-fit model; don't force Nano Banana.
- **"logo, 4k, masterpiece, trending…"** → rewrite into a specific scene description; drop token spam.
- **Real person / copyrighted character / brand style** → refuse; offer an original alternative.
- **"Post it as a normal photo"** → SynthID + disclosure; don't disguise AI imagery.
- **Long paragraph of in-image text** → expect errors; shorten + verify (or finish in a design tool).
- **"Generate it in WoopSocial"** → WoopSocial doesn't generate; this skill prompts the model, then the
  image is uploaded to Media and attached.

## Related

- `tools/integrations/nano-banana.md` — API/model IDs, auth, pricing, the upload-to-WoopSocial flow.
- `carousel-writer` (image-prompt-pack), `reels-script` — consuming skills; `ideogram` — typography sibling.
- `brand-profile` — the visual brand; `caption-writer`/`hook-writer` — the in-image copy.
- `scheduling-and-queue` — attach the image to a post and publish.

## References

- `references/when-and-how-to-prompt.md` — when to reach for it vs other models, and the prompt anatomy.
- `references/text-and-consistency.md` — the two superpowers: legible in-image text and multi-image consistency/editing.
- `references/social-recipes.md` — adaptable prompt patterns per social job + aspect ratios.
- `references/examples.md` — weak→strong prompts, a consistent carousel, an edit, and honest scope.
