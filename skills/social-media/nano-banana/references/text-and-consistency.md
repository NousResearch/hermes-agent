# Text & Consistency — the two superpowers

Nano Banana's two standout abilities for social, and how to prompt each. These are the reasons to
reach for it.

> Re-verify quarterly. Both are strong but **not infallible** — always review the output.

## 1. Legible in-image TEXT

Nano Banana Pro is the best model at rendering **correct, readable text inside an image** — the thing
every other generation has failed at. This unlocks quote graphics, carousel slides, thumbnails,
posters, infographics, logos, signage, and ad creative *without* finishing in Canva.

How to prompt text well:

- **Put the exact words in quotes.** Don't say "add a headline." Say: *Render the text "Schedule a week
  of posts in 20 minutes" …*.
- **Specify font character + placement.** "…in a bold, condensed sans-serif, white, centered in the
  lower third." You can specify multiple styled lines (e.g. a brush-script word over a blocky
  sub-line).
- **The text-first approach** (recommended for an LLM agent): settle the **copy first** (write/trim the
  exact words), *then* prompt the image with that locked text. Cleaner than improvising copy inside a
  visual prompt. (→ `caption-writer` / `hook-writer` for the words.)
- **Keep it short for reliability.** Fewer words render more reliably; for a paragraph, expect to
  verify. Give a clear hierarchy (headline vs sub).
- **Always verify spelling.** The model can still misspell or garble small/long text — **read the
  rendered text before publishing.** Never publish unchecked text-in-image.

## 2. Multi-image / character / brand CONSISTENCY (+ editing)

Keep the same character, product, or look across a **set** of images — the thing that makes carousels,
series, storyboards, and campaigns possible.

How to prompt consistency:

- **Use reference images.** Upload 3–5 (up to ~14) images of the product/character/brand and instruct:
  "Using the attached references of 'Sneaker X', …**maintain the exact logo placement and colorway**."
- **Lock a style description** and reuse it verbatim across every image in the set (palette, lighting,
  framing, medium).
- **Generate one at a time, referencing the prior** — "same character and style as the last image,
  now doing X" — rather than asking for a single composite. Keep "**everything the same except [the one
  change]**."
- **Hold the aspect ratio** constant across the set (e.g. all 4:5 for an IG carousel).
- **Consistency can still drift** — review the full set and regenerate the odd one out.

### Conversational editing

Edit an existing image in natural language: "change the sunny day to a rainy night," "remove the
person and replace with a potted plant," "translate the on-pack text to Korean, keep everything else."
The model reasons about the change and adjusts lighting/reflections. Great for variants (locale,
seasonal, A/B) from one hero image — **preserving product identity** with reasonable reliability
(verify identity-critical details).

## The honest caveat (both)

Strong ≠ perfect: small faces, fine details, long text, and complex infographic **data** can be wrong.
Treat Nano Banana as a fast, controllable drafting tool whose **text and facts you verify** before it
goes out. For data-heavy infographics, confirm the numbers/labels independently (see `social-recipes.md`).
