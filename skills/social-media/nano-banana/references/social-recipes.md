# Social Recipes

Adaptable prompt patterns for the common social jobs. Fill the brackets from `brand-profile.md`
(colours, style) and the post context. Set the aspect ratio for the platform; verify any text/data.

> Patterns, not magic strings — adapt to the brand and re-verify model behavior quarterly.

## Aspect ratios (set one explicitly)

- **1:1** — IG feed (square), general. **4:5** — IG/feed portrait (max feed real estate), carousels.
- **9:16** — Stories/Reels/TikTok/Shorts full-screen. **16:9** — YouTube thumbnail, LinkedIn/X landscape,
  presentations. **2:3 / 3:4** — Pinterest.

## Quote / text graphic (plays to the text superpower)

> "A [style, e.g. clean minimalist] social graphic, [brand background colour] background. Render the
> text "[exact quote]" in a [bold condensed sans-serif], [brand text colour], centred with clear
> hierarchy. Generous negative space. [Brand motif/logo space]. Aspect ratio 4:5."

Settle the copy first; keep it short; verify the spelling.

## Carousel set (plays to consistency)

> Slide 1: "[locked style description], [palette], aspect ratio 4:5. [Scene]. Render title "[slide 1
> text]" …" Then each next slide: "Same style, palette, and character as the previous image; now
> [slide N scene]; render "[slide N text]". Keep everything consistent except the scene and text."

Generate one at a time; review the set for drift. (→ `carousel-writer` owns the slide architecture;
this writes the image prompts.)

## Thumbnail (16:9, high-contrast)

> "A bold YouTube thumbnail, 16:9. [Subject/expression], high contrast, punchy [brand] colours,
> strong focal point with space for a short overlay. Render the text "[2–4 words]" large and legible,
> [font]. Leave the [left/right] third clear for a face/product."

## Infographic / diagram (grounding — then verify)

> "A clean infographic, [aspect ratio], [brand palette], vector style. Topic: [topic]. Include
> [the specific, correct data points/steps]. Label everything legibly. Clear visual hierarchy."

Provide the **correct data yourself** (or use Search grounding) and **fact-check the output** — the
model can misplace labels or numbers.

## Product / lifestyle shot (reference images)

> "Using the attached reference images of [product], a [lifestyle context] shot — [setting, time of
> day], [lighting], [style]. Maintain the exact logo placement and colourway. [Aspect ratio]."

For variants: edit the hero image ("same product, now on a marble surface / in autumn tones / for the
Korean market").

## Background / scene (for overlay)

> "A [style] background scene, [palette], [aspect ratio], with deliberate **negative space** in the
> [region] for text overlay. [Mood/lighting]. No text." (Add the text later in-image or via overlay.)

## Ad creative

> "A [style] ad visual, [aspect ratio], featuring [product/hero]. [Lighting/mood]. Render "[offer/
> headline]" in [font, placement] and "[CTA]" smaller beneath. [Brand colours/logo space]."

Verify the offer text; disclose as AI where required; never imply false claims.

## Always, before publishing

- **Verify** rendered text + any data.
- **Disclose** AI generation per platform/region; the image carries a **SynthID** watermark.
- **Right format** for the destination; **upload to WoopSocial Media → attach to the post** (see
  `tools/integrations/nano-banana.md` + `scheduling-and-queue`).
