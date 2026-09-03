---
name: ideogram
description: >-
  Use to write great prompts for Ideogram (Ideogram 4.0) to generate design-grade images for social
  media — the image-prompt-craft mini-skill for typography + layout, sibling to nano-banana under
  the image-prompt router. Run when the user wants an Ideogram / AI image prompt where TEXT and
  LAYOUT matter: quote cards, carousel covers, posters, logos/wordmark concepts, signage, ads with
  headlines, or any image that must say words correctly. Reads brand-profile for style/palette.
  Ideogram's standouts: best-in-class in-image text rendering, structured JSON prompting (bounding
  boxes, hex palettes), native 2K, transparency, and Layerize Text. Also covers the Ideogram
  app-editor workflow (Canvas, Magic Fill, Extend, Style References). Honest: review rendered text,
  disclose AI images and sign with C2PA, never generate real identifiable people or copyrighted
  characters/IP/logos. The API/connection and the generate -> upload to WoopSocial Media -> attach
  flow live in tools/integrations/ideogram.md.
metadata:
  version: 1.0.0
  license: MIT
---

# Ideogram (image prompt craft)

Write prompts that get **design-grade images with correct text and deliberate layout** out of
**Ideogram** (**Ideogram 4.0**). This is the **prompt-craft** layer of a three-layer setup, and the
**typography/layout sibling to `nano-banana`** under the **image-prompt** router.

- **Connection/API** (per-image pricing, MCP, the *generate → upload to WoopSocial Media → attach*
  flow) → `tools/integrations/ideogram.md`.
- **Prompt craft** (this skill) → how to direct the right asset well.
- **Router** → `image-prompt` picks the tool for the job.

> This skill covers **both lanes**: *generation* prompt-craft (below) and the Ideogram
> **app-editor workflow** (Canvas, Magic Fill, Extend, Style References, Magic Prompt, Layerize) —
> for in-editor work, see `references/app-editor-and-canvas.md`. Re-verify model/specs quarterly.

## Reach for Ideogram when… (match the job)
Its real strengths: **in-image text rendering** (the standout — legible, correct typography),
**deterministic layout** (bounding-box placement), **brand-accurate color** (hex palettes), **native
2K**, and **transparency** (Background Remover). Reach **elsewhere** for **conversational edits / pure
photoreal scenes** (→ `nano-banana`) or specific **art house looks** (→ the photoreal branch). Don't
default it to every image. (Details: `references/when-and-how-to-prompt.md`.)

## Step 0 — Read the brand + the job
Load `brand-profile.md` (palette as hex, type feel, do/don't). Identify the **job** (quote card /
carousel cover / poster / logo concept / ad / signage) and the **aspect ratio** (4:5, 9:16, 1:1).

## Step 1 — Direct a layout, not just a scene
Describe like a designer: **subject/scene/style · text elements (exact string + styling + placement) ·
hex palette · layout/hierarchy · format (2K, aspect)**. Don't bury copy in a vague sentence. Ground in
the brand. See `references/when-and-how-to-prompt.md`.

## Step 2 — Use the superpower: structured JSON
Ideogram 4.0 is JSON-native. For exact, repeatable **layout/text/color**, write **structured JSON**:
per-element **bounding boxes**, **hex palette** (up to ~16), and each text element's **literal string
+ visual styling**. Casual input works via **magic-prompt** (it expands to JSON). See
`references/json-prompting-and-layout.md`.

## Step 3 — Text, transparency, recipes
- **Text:** ~0.97 OCR, multilingual, multi-line, logos/signage — but **review every render.**
- **Transparency:** Background Remover → clean alpha cutout. **Layerize Text** (live, beta) turns
  rendered text into **editable layers** — restyle/rewrite copy without regenerating; works best on
  clear, straight type. **2K** output; aspect ratios in multiples of 16.
- Pick a **recipe** (quote card, carousel cover, logo concept, ad, poster). See
  `references/text-rendering-and-recipes.md`.

## Step 4 — Iterate cheaply, then verify, disclose, ship
- **Iterate on Turbo / magic-prompt** to lock layout; **finalize on Quality** with refined JSON.
  Per-image pricing — **prompt/layout precision is the cost + quality lever.**
- **Verify** the rendered **text** and composition before publishing.
- **Disclose** AI images (**EU AI Act Article 50**, **California AB 853**) and sign with **C2PA**;
  never pass an AI image off as a real photo. Open weights are **non-commercial** (commercial → plan/API).
- **Ship:** generate per the integration guide → **upload to WoopSocial Media → attach** via
  `scheduling-and-queue`. WoopSocial doesn't generate images.

## Quality bar — self-check
- Did I **match the tool to the job** (route conversational/photoreal to `nano-banana`)?
- Are **text elements typed** (exact string + styling + placement), with a **hex palette** and **2K/
  aspect** set, **brand-grounded**?
- Did I use **JSON** where layout/text precision matters (and **magic-prompt** for casual)?
- Did I plan **iterate-cheap → finalize**, and **verify the rendered text**?
- Did I handle **disclosure + C2PA**, **non-commercial license**, and **refuse real people / IP / logos**?
- Did I point to **`tools/integrations/ideogram.md`** for the API + WoopSocial flow (no claim WoopSocial
  generates images)?

## Edge cases & pushback
- **"Exact headline + brand colors + placement"** → structured JSON (bounding boxes + hex palette).
- **Conversational "now change the background"** → route to `nano-banana`; Ideogram is layout-first.
- **Real person / copyrighted IP or logo / "post as a real photo"** → refuse; disclose + C2PA; offer
  an original/owned alternative.
- **"Editable text layers"** → **Layerize Text is live (beta**, all plans + API endpoint**)** — best on
  clear, straight standard type; curved/decorative text may not be detected.
- **"Generate it in WoopSocial"** → WoopSocial doesn't generate; this prompts Ideogram, then the image
  is uploaded to Media and attached.
- **Long paragraph of text in-image** → keep lines short, hierarchy clear; verify the render.

## Related
- `tools/integrations/ideogram.md` — API/pricing/MCP, the upload-to-WoopSocial flow.
- `image-prompt` — the router; `nano-banana` — the conversational/photoreal image sibling.
- `brand-profile` — the visual brand; `carousel-writer`, `caption-writer`, `hook-writer` — copy that
  pairs with the asset; `veo-3` — animate a still into video (image-to-video).
- `scheduling-and-queue` — attach the image to a post and publish.

## References
- `references/when-and-how-to-prompt.md` — when to reach for Ideogram vs other tools, and the prompt anatomy.
- `references/json-prompting-and-layout.md` — the 4.0 differentiator: structured JSON, bounding boxes, palettes.
- `references/text-rendering-and-recipes.md` — the text superpower, transparency/2K, social recipes.
- `references/app-editor-and-canvas.md` — the in-app editor workflow: Canvas, Magic Fill, Extend, Style References, Magic Prompt, brand models.
- `references/examples.md` — weak→strong, a JSON layout, a text-heavy card, and honest scope.
