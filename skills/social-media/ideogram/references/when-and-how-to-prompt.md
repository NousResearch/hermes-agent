# When & How to Prompt Ideogram

When to reach for Ideogram over another image tool, and the prompt anatomy that gets design-grade
results. (The API/connection layer — model access, per-image pricing, the upload-to-WoopSocial flow —
is in `tools/integrations/ideogram.md`; this is prompt craft.)

> Fast-moving area — re-verify model/specs quarterly. As of 2026: **Ideogram 4.0** (released June 3,
> 2026) is current — a from-scratch 9.3B open-weight Diffusion Transformer with **structured JSON
> prompting**, native **2K** output, and best-in-class in-image text. (This skill covers both
> generation prompt-craft and the app-editor workflow — see `app-editor-and-canvas.md`.)

## Reach for Ideogram when the job is…

Its genuine strengths:

- **In-image text rendering** — the standout. Legible, correct typography inside the image (~0.97
  English OCR; multilingual; multi-line; varied weights; logos, signage, captions, watermarks). If
  the image must say words correctly, this is the reason to use it.
- **Deterministic layout** — **bounding-box** placement so headlines/elements land where the brief
  asks, repeatably across a campaign.
- **Brand-accurate color** — hex **color-palette** conditioning (up to ~16 colors).
- **2K photoreal + design output** — native 2048px, no separate upscaler; flexible aspect ratios.
- **Transparency** — Background Remover for clean alpha cutouts (logos/stickers/icons).

## Reach for something else when…

- **Conversational edits / iterative "change this" / pure photoreal scene generation** → **nano-banana**
  (Gemini image) is usually the better Google sibling.
- **Highly stylized art / specific aesthetic house looks** → a model tuned for that (e.g. the
  photoreal/midjourney branch via the image-prompt router).
- Match the job; don't default. The **image-prompt** router picks the tool.

## The prompt anatomy (design a layout, not just a scene)

Describe the image like a designer. Include:

- **Subject / scene / style** — what's depicted and the visual treatment (brand-grounded).
- **Text elements** — for each: the **literal string** + a **separate visual-styling description**
  (font feel, weight, case, color) + **placement** (bounding box). Don't bury copy in a vague sentence.
- **Color palette** — exact hex values per brand-profile.
- **Layout / composition** — where each element sits; hierarchy.
- **Format** — aspect ratio (multiples of 16) and 2K output.

Casual input works (magic-prompt expands it), but when **layout, text, and color must be exact**,
write **structured JSON** (see `references/json-prompting-and-layout.md`).

## Brand-grounded

Read `brand-profile.md` — bake the brand's palette (hex), type feel, and do/don't into the prompt so
the asset looks like the brand, not generic. Verify the rendered **text** before publishing.
