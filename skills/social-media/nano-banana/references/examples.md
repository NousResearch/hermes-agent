# Examples — prompts in practice

Worked prompts showing weak→strong, the text/consistency superpowers, and an edit — plus honest scope.
Brand context: a SaaS social scheduler (blunt indie-founder brand — clean, mono, terminal-green
accent). Your output pulls the real palette/style from `brand-profile.md`.

---

## Weak → strong (drop the keyword spam)

**Weak:** "logo, 4k, masterpiece, trending on artstation, ultra detailed, best quality"
**Strong:** "A modern, minimalist wordmark for a social scheduler. Render the text "WOOP" in a bold,
geometric sans-serif, near-black on an off-white background, with a single small terminal-green dot as
an accent. Centered, generous spacing, flat vector style. Aspect ratio 1:1."
*(specific subject, exact text in quotes, font, palette, composition — no quality-token padding)*

## Text graphic (the superpower)

**Job:** an IG quote graphic. **Copy settled first:** "Schedule a week of posts in 20 minutes."
**Prompt:** "A clean minimalist social graphic, off-white background, aspect ratio 4:5. Render the text
"Schedule a week of posts in 20 minutes" in a bold condensed sans-serif, near-black, left-aligned in
the upper-middle with clear line breaks for hierarchy. A thin terminal-green underline beneath the
last line. Generous negative space below. No other text."
*(then verify the spelling in the output before publishing)*

## Consistent carousel (the other superpower)

**Slide 1:** "Flat vector illustration, off-white bg, near-black line art with terminal-green accents,
a calm solo founder at a laptop. Aspect ratio 4:5. Render title "Post less. Reach more."."
**Slide 2:** "Same style, palette, and character as the previous image; now the founder closing the
laptop and walking away while posts auto-publish in the background. Render "Batch once. Ship all week.".
Keep everything consistent except the scene and text."
*(generate one at a time; review the set for drift)*

## Conversational edit

**On a hero product shot:** "Keep the product, logo, and colourway exactly the same. Change the
background from a white studio to a wet city street at night with soft neon reflections. Aspect ratio
4:5." → a locale/seasonal variant from one image.

---

## Honest scope (say this)

- **SynthID watermark** — every Nano Banana image is invisibly watermarked (and carries C2PA
  credentials) as AI-generated; that's a feature, not something to strip.
- **Disclose AI** per platform/region (e.g. Meta "Made with AI", EU AI Act). Don't pass it off as a
  non-AI photo.
- **Verify text + data** — read the rendered text; fact-check infographic numbers/labels.
- **No real people / no IP** — don't generate real identifiable individuals, copyrighted characters, or
  brand styles/logos you don't own; use original characters/art direction.
- **Match the tool** — for photoreal headshots or surreal art, use another model.
- **It's prompt craft** — the API/model IDs and the *generate → upload to WoopSocial Media → attach*
  flow live in `tools/integrations/nano-banana.md`; WoopSocial doesn't generate images.

---

## What the examples share

- **Natural-language specificity** over quality-token spam.
- **Exact text in quotes** + font + placement; **copy settled first**, then verified.
- **Locked style + references** for consistent sets; **one change at a time** for edits.
- **Brand-grounded**, right aspect ratio, and **disclosed/verified** before publishing.
