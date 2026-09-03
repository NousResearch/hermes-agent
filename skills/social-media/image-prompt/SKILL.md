---
name: image-prompt
description: >-
  Use as the foundation and router for any image a post needs — turn a social need into a clear image
  brief, describe it with model-agnostic prompt craft, and route to the right tool. Run when the user
  says "I need a visual/image for this post," "what image should I use," "make a graphic," "help me
  prompt an image," or is unsure which image tool/model to use. Reads brand-profile for the visual
  brand. Teaches the image brief, the universal anatomy of a strong prompt in natural language, and a
  tool router (typographic -> ideogram/nano-banana; photoreal -> nano-banana/flux; surreal ->
  Midjourney; editing -> nano-banana/flux; vector -> Recraft; video -> veo-3).
  Includes a "should this even be AI-generated?" gate (a real photo, screenshot, or chart often beats
  generic AI), accessibility, and honest scope (disclose AI, never real identifiable people or
  copyrighted IP, verify text/data). Hands off to the tool-specific mini-skills; WoopSocial publishes
  via Media and doesn't generate images.
metadata:
  version: 1.0.0
license: MIT
---

# Image Prompt (brief · craft · router)

The **foundation of the visual cluster.** When a post needs an image, this skill decides **what** image
(the brief), describes it with **model-agnostic** craft, and routes to the **right tool** — then hands
off. It sits *above* the tool-specific mini-skills (`nano-banana`, `ideogram`, `flux`,
`veo-3`), which own their model's specific tricks.

Think of it as the visual equivalent of `brand-profile`/`content-pillars`: the shared decision layer
the tool skills assume you've already worked through.

## Step 0 — Gate: should this even be AI-generated?

AI generation is one option, not the default. A **real photo** (your actual product/space/team), a
**screenshot**, **UGC**, or a **chart** often wins on authenticity and clarity — and AI must never
fabricate your real people/place. Reach for generation when you need something that doesn't exist, must
match a brand look, needs legible in-image text, or a consistent series — and no real asset fits. If a
real asset is better, **say so.** (See `references/the-image-brief.md`.)

## Step 1 — Write the brief

Read `brand-profile.md`, then answer four things: **job/purpose** (hero / explainer / quote graphic /
product / lifestyle / thumbnail / background / ad), **the one thing it must convey**, **format/aspect
ratio** for the destination, and the **brand look**. Output a one-line brief. See
`references/the-image-brief.md`.

## Step 2 — Describe it (model-agnostic)

Turn the brief into a **specific, natural-language** prompt — **subject · composition · lighting ·
colour · style · mood · detail · constraints** — and **drop quality-token spam** ("4k, masterpiece").
Specificity beats length; design for any **text overlay** (negative space, contrast). See
`references/prompt-anatomy.md`.

## Step 3 — Route to the tool

Pick by the image's **dominant requirement**:

- **Text / typographic** → `ideogram` or `nano-banana` (both strong — honest pick by tooling).
- **Photoreal portrait/product** → `nano-banana` or `flux` (Midjourney external). **Surreal/painterly** → Midjourney.
- **Editing** ("change one thing, keep the rest") → `nano-banana` (conversational) or `flux` (Kontext).
- **Consistent character/product set** → `nano-banana` (multi-image references) or `flux` (multi-reference).
- **Search-grounded infographics** → `nano-banana`. **Brand-exact hex / open weights / license control** → `flux`.
- **Vector/SVG** → Recraft. **Video** → `veo-3`.

When two fit, **say both work and pick by the deciding factor** — don't fake a winner. See
`references/choosing-the-tool.md`.

## Step 4 — Hand off, then ship

**Hand off to the tool's mini-skill** for model-specific craft (and `tools/integrations/<tool>.md` for
the API). After generation: **verify** in-image text/data, **disclose** AI per platform/region, then
**upload to WoopSocial Media → attach** via `scheduling-and-queue`. WoopSocial publishes; it **doesn't
generate**.

## Cross-cutting (always)

- **Disclose AI** per platform/region (Meta "Made with AI", EU AI Act). **Watermark behaviour varies by
  tool** (SynthID on Google tools; none enforced on Ideogram) — don't rely on it for disclosure.
- **No real people / no IP** — no real identifiable individuals, copyrighted characters, or trademarked
  logos; original directions only.
- **Verify** in-image text and data before publishing.
- **Accessibility** — contrast + legibility on mobile; leave room for overlays.

## Quality bar — self-check

- Did I **gate** (real asset vs AI) before generating?
- Did I write a **brief** (job + the one thing + format + brand) and a **specific, natural-language**
  prompt (no token spam)?
- Did I **route by the dominant requirement** with an honest which-tool read, and **hand off**?
- Did I design for **overlays/accessibility**, and cover **disclosure, IP, and verification**?
- Did I keep WoopSocial honest (publishes via Media; doesn't generate)?

## Edge cases & pushback

- **"Just write a cool image prompt"** → build the brief first; purpose before prompt.
- **"4k masterpiece ultra-detailed"** → rewrite into a specific scene; drop quality tokens.
- **"Generate my real cafe/staff"** → use a real photo; don't fabricate real people/place.
- **"Base it on [studio]'s characters" / a real person** → refuse IP/real people; original direction.
- **"Which model — A or B?"** → honest, tooling-based answer; both can be right.
- **"Generate it in WoopSocial"** → WoopSocial doesn't generate; route to a tool, then upload + attach.

## Related

- `nano-banana`, `ideogram`, `flux`, `veo-3` — the tool-specific mini-skills this routes/hands off to.
- `tools/integrations/*` — the connection/API for each tool; `brand-profile` — the visual brand.
- `carousel-writer`, `reels-script`, `caption-writer`/`hook-writer` — consumers (image-prompt-pack / overlay copy).
- `scheduling-and-queue` — upload the asset to Media and attach to a post.

## References

- `references/the-image-brief.md` — the "should this be AI?" gate and the four-question brief.
- `references/prompt-anatomy.md` — the universal, model-agnostic anatomy of a strong prompt.
- `references/choosing-the-tool.md` — the router/decision tree and honest which-tool calls.
- `references/examples.md` — end-to-end flows, weak→strong, the don't-generate gate, and honest scope.
