# When & How to Prompt Nano Banana

When to reach for Nano Banana over another model, and the prompt anatomy that gets studio-quality
results. (The API/connection layer is in `tools/integrations/nano-banana.md` — this is prompt craft.)

> Fast-moving model area — re-verify model names/capabilities quarterly. As of 2026: **Nano Banana
> Pro = Gemini 3 Pro Image** (highest quality); **Nano Banana / Nano Banana 2 = Gemini Flash Image**
> (faster, cheaper). Use Pro for final/text-heavy/complex work, Flash for fast iteration.

## Reach for Nano Banana when the job is…

Its genuine strengths (where it beats or matches the field):

- **Legible in-image TEXT** — the best model for correctly rendered, readable text in an image (short
  tagline → full paragraph, multiple languages, varied fonts). The killer use for social graphics.
- **Multi-image / character / brand CONSISTENCY** — keep the same character, product, or style across
  a set, using up to ~14 reference images. The killer use for carousels and series.
- **Conversational EDITING** — natural-language edits to an existing image ("make it night," "remove
  the background"), with lighting/reflections adjusted automatically; localize/translate text in-image.
- **Search-grounded INFOGRAPHICS / factual scenes** — real-world knowledge (and optional Search
  grounding) for diagrams, recipes, accurate scenes. (Verify the facts — see caveats.)
- **Speed + many aspect ratios + up to 4K** — covers every social format.

## Reach for something else when…

Be honest about the tool's edges (match the job, don't default):

- **Photoreal human portraits / luxury product realism** → dedicated photoreal models compete hard on
  skin/material detail (`flux`, Midjourney). Google's own photoreal specialist, Imagen 4, is
  **deprecated** (Gemini API shutdown Aug 17, 2026) — Google now routes that work here.
- **Surreal / fine-art / highly stylized aesthetics** → Midjourney-class models have more artistic range.
- **Pure typography-led layouts** → `ideogram` is a strong sibling (though Nano Banana Pro now rivals it
  on text).

## The prompt anatomy (natural language, not keyword spam)

Modern Gemini image models **reason over natural language** — drop the 2023-era "4k, masterpiece,
trending on artstation, ultra-detailed" token spam. Be **descriptive and specific** instead. Include:

- **Subject** — who/what, specifically ("a matte-black ceramic coffee mug," not "a mug").
- **Composition / camera** — framing, angle, distance ("close-up product shot, slight top-down").
- **Action / context** — what's happening / where ("resting on a wet city street at night").
- **Lighting / atmosphere** — ("soft radiant studio light"; "harsh directional chiaroscuro").
- **Style / medium** — ("clean vector art, pastel palette"; "commercial product photography").
- **Constraints / exact text** — aspect ratio, what to include/avoid, and any **text in quotes** (see
  `text-and-consistency.md`).

A workable shape: *"[Subject + adjectives] [action] in [context]. [Composition]. [Lighting]. [Style].
[Exact text + constraints]."*

## Work iteratively

- **Iterate at low res, finalize high.** Draft at 1K for quick/cheap passes; upscale to 2K/4K for the
  final. (Cost/latency notes live in the integration guide.)
- **Refine conversationally** — adjust one thing at a time ("keep everything, make the background
  warmer") rather than rewriting the whole prompt.
- **Set the aspect ratio for the platform** up front (see `social-recipes.md`).

## Brand-grounded

Read `brand-profile.md` and bake the brand into the prompt — **brand colours, visual style, do/don't**
— so the output looks like the brand, not generic stock. The image is the brand speaking visually.
