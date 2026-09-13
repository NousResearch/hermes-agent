# Choosing the Tool

The router. Match the job to the model, then hand off to that tool's mini-skill for the model-specific
craft. This is a decision layer, not a tool tutorial.

> Fast-moving landscape — re-verify the "best for X" picks quarterly. The principle (match the job to
> the tool) is stable; the specific winners shift.

## The decision tree

Pick by the **dominant requirement** of the image:

- **In-image TEXT / typographic design** (quote graphics, posters, logos, packaging, text-heavy
  layouts) → **`ideogram`** or **`nano-banana`**. Both are excellent at text now — **pick by
  tooling/ecosystem**, not a fake "best text" claim (Ideogram = design tooling, structured layout
  control; Nano Banana = Gemini reasoning, multi-image consistency, editing). Honest either-way.
- **Photoreal human portraits / luxury product realism** → **`nano-banana`** or **`flux`** (Midjourney externally). Avoid
  Ideogram for real faces.
- **Surreal / painterly / editorial art** → **Midjourney**-class (more artistic range).
- **Search-grounded infographics / accurate scenes / conversational editing**
  → **`nano-banana`** (Gemini reasoning + references up to ~14 + Magic-Fill-style editing).
- **Editing an existing image** ("change one thing, keep the rest" / inpaint/outpaint/fix) →
  **`nano-banana`** (conversational editing) or **`flux`** (Kontext instruction-based edits).
- **A consistent character/product across a set** → reference-image models (**`nano-banana`**
  references up to ~14, or **`flux`** multi-reference).
- **Brand-exact colors (hex parameters) / license control / open weights you can run/tune** →
  **`flux`** (or Ideogram's open weights).
- **Vector / SVG for brand work** → **Recraft** (vector-native).
- **Moving image / video** → **`veo-3`** (and animate a still via the image→video pipeline).

## Honest "which of two?" calls

When two tools both fit (common with text graphics), **say both work and pick by the deciding factor** —
the workflow you're already in, the controls you need (Style References vs reference images vs editing),
the ecosystem (Gemini vs Ideogram), cost, or open-weights/MCP access. Don't manufacture a clean winner
where there isn't one.

## Then hand off

Once routed, **hand off to the tool's mini-skill** for the model-specific craft:
- `nano-banana` — Gemini image prompt craft (text, consistency, editing).
- `ideogram` — typography/layout prompt craft (exact text in quotes, structured layout).
- `flux` — variant + license picks, instruction-based edits, multi-reference consistency, hex colors.
- `veo-3` — video prompt craft (audio, camera, image-to-video).

The **connection/API** for each lives in `tools/integrations/<tool>.md`. After generation, the image/
video is **uploaded to WoopSocial Media and attached to the post** via `scheduling-and-queue` —
**WoopSocial publishes; it doesn't generate.**

## If no tool fits

If the best answer is a **real photo, screenshot, chart, or UGC** (see `the-image-brief.md`), route
there instead of forcing generation.
