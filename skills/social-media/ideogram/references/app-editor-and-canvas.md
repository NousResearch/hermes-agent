# Canvas Editing & Recipes

Ideogram's post-generation editing tools, and adaptable prompt patterns for the design-led social jobs.

> Re-verify feature specifics quarterly.

## Canvas & editing tools

An infinite board for refining without starting over:

- **Magic Fill (inpainting)** — mask a region and regenerate just that part (fix a word, swap an
  object, add an element), keeping the rest intact. Include some surrounding context in the generation
  window; describe the whole scene + the change.
- **Extend (outpainting / uncrop)** — grow the canvas to a new aspect ratio or wider scene.
- **Layerize** — turn generated text into **editable layers** to restyle/reposition.
- **Background removal** — pull the subject out cleanly for transparent/composited assets.
- **Upscale** — more pixels before a Magic Fill for better local edits.

Use these to **fix** rather than regenerate — faster and keeps the parts that already work.

## Social design recipes (adapt the brackets)

Fill brackets from `brand-profile.md` (or a Style Reference). Lead with the text in **straight quotes**.
Set the aspect ratio. Verify spelling; disclose AI; finish complex layouts in Canva.

- **Typographic quote graphic:** "A clean [design-style] social graphic. The text "[exact quote]" in a
  [bold condensed sans-serif], [brand colour] on [bg colour], centred with clear hierarchy. Generous
  negative space. Aspect ratio 4:5."
- **Logo / wordmark (concept):** "A minimalist wordmark logo. The word "[brand]" in a [geometric
  sans-serif], [colour], with [a small motif]. Centered, flat vector look, white background. 1:1."
  (Concept/mockup — for production vectors, finish/redraw in vector software.)
- **Poster:** "A [era/style] poster. Headline "[headline]" at the top in [type], subhead "[subhead]"
  below, [scene/illustration]. [Print look]. 4:5 or 2:3."
- **Carousel design system:** lock a **Style Reference** + palette; per slide: "[same style], "[slide
  text]", [slide motif]. 4:5." Generate one at a time; review for drift. (→ `carousel-writer` owns the
  slide architecture.)
- **Social header / banner:** "A [brand-style] banner, 16:9 (or platform size). "[name/tagline]" in
  [type], [brand colours], [motif], clear space for a profile photo overlay."
- **Packaging / product mockup:** "A [product] package mockup. Label reads "[product name]" in [type],
  "[tagline]" beneath, [brand colours/illustration]. Studio lighting. 1:1." (Mockup, not a real label.)

## Aspect ratios

1:1 (feed/logo) · 4:5 (feed portrait/carousel) · 9:16 (Stories/Reels/TikTok) · 16:9 (header/landscape)
· 2:3 / 3:4 (poster/Pinterest). Ideogram offers many ratios — pick per destination.

## The Ideogram → Canva finishing workflow

A common, honest pro pattern: generate the **typographic image** in Ideogram → drop into **Canva/Figma**
to composite with exact logos, legal text, and precise layout → export → **upload to WoopSocial Media →
attach to the post** (`tools/integrations/ideogram.md` + `scheduling-and-queue`). WoopSocial publishes;
it doesn't generate.
