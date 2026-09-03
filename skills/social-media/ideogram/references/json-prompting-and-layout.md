# JSON Prompting & Layout — the 4.0 differentiator

Ideogram 4.0 was trained on **structured JSON captions**, so JSON is the native prompt format. It's
what gives you layout-level control plain-text prompts on other models can't match: **per-element
bounding boxes, hex color palettes, and per-element text styling.**

> Re-verify the exact schema/fields quarterly against developer.ideogram.ai. Shapes below are
> illustrative.

## When to write JSON vs plain text
- **Plain text + magic-prompt** — casual/exploratory; magic-prompt auto-expands it into JSON. Good for
  ideation and quick variants.
- **Structured JSON** — when **layout, typography, and color must be exact and repeatable** (campaign
  assets, brand templates, anything where copy and placement matter). This is the design-grade path.

## The pieces that matter
- **Text elements** — each carries a **literal string** (the exact words) and a **separate visual
  description** (font feel, weight, case, color), plus a **bounding box** (top/left/bottom/right) for
  placement. Separating string from styling is why Ideogram renders text correctly.
- **Object elements** — described subjects with their own bounding boxes for deterministic composition.
- **Color palette** — up to ~16 **hex** values to lock brand color.
- **Format** — resolution/aspect (native 2K; sides 256–2048, multiples of 16).

## Illustrative shape
```json
{
  "prompt": "Calm minimalist quote card, off-white background, subtle grain",
  "color_palette": ["#0B0B0B", "#F4F4F0", "#1FAA59"],
  "elements": [
    { "type": "text", "string": "Post less. Reach more.",
      "style": "bold geometric sans, near-black, tight tracking",
      "box": { "top": 120, "left": 110, "bottom": 320, "right": 900 } },
    { "type": "text", "string": "a weekly batch beats daily scrambling",
      "style": "light sans, muted grey, sentence case",
      "box": { "top": 360, "left": 110, "bottom": 430, "right": 760 } }
  ],
  "aspect": "4:5", "resolution": "2K"
}
```

## Practical tips
- One **clear hierarchy** — a hero line, then support. Don't crowd the canvas with competing text.
- Keep strings **short and exact**; long paragraphs still risk small errors — **review the render.**
- Reuse a **locked JSON template** across a campaign and swap strings/palette for consistency (pairs
  with `batch-content-plan` / `content-calendar`).
- Lock layout cheaply on **Turbo/magic-prompt**, then finalize on **Quality** with the refined JSON.

## MCP / automation
Ideogram 4.0 supports **MCP** for agent workflows — useful when wiring generation into an automated
pipeline. Connection details: `tools/integrations/ideogram.md`.
