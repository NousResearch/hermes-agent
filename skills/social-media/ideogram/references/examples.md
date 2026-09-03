# Examples — prompts in practice

Worked Ideogram prompts showing weak→strong, structured JSON layout, a text-heavy card, and honest
scope. Brand context: a calm SaaS social scheduler (clean, off-white, terminal-green accent). Your
output pulls the real style from `brand-profile.md`.

---

## Weak → strong

**Weak:** "a motivational quote poster, nice typography, 4k"
**Strong (plain text, magic-prompt will structure it):** "Minimalist quote card, off-white background
with subtle grain. Hero line 'Post less. Reach more.' in a bold geometric near-black sans, top-left,
tight tracking. Small muted-grey support line 'a weekly batch beats daily scrambling' lower-left.
Terminal-green underline accent. 4:5, 2K."
*(exact strings, styling per line, placement, palette, format — no quality spam)*

## Structured JSON (when layout/text/color must be exact)
```json
{ "prompt": "calm carousel cover, off-white, subtle grain",
  "color_palette": ["#0B0B0B", "#F4F4F0", "#1FAA59"],
  "elements": [
    { "type": "text", "string": "The 20-minute content week",
      "style": "bold geometric sans, near-black",
      "box": { "top": 140, "left": 120, "bottom": 360, "right": 920 } },
    { "type": "text", "string": "1/7",
      "style": "small mono, terminal green",
      "box": { "top": 60, "left": 60, "bottom": 110, "right": 180 } } ],
  "aspect": "4:5", "resolution": "2K" }
```
*(reuse this template across the carousel; swap the hero string + index per slide)*

## Text-heavy (the superpower)
"Event poster: title 'CREATOR OPS LIVE' large serif engraving feel, centered top; 'June 14 · 18:00 ·
Rotterdam' as a clean sans line beneath; small 'free, RSVP in bio' footer. High-contrast, 2K, 4:5."
*(multiple distinct typed elements; verify each line renders exactly before publishing)*

---

## Honest scope (say this)

- **Review the text** — even at ~0.97 OCR, dense/complex typography can slip; verify before publishing.
- **Disclose AI images** — EU AI Act Article 50 and California AB 853; recommend **C2PA** content
  signing. Never pass an AI image off as a real photo.
- **No real people / no IP** — no real identifiable individuals, copyrighted characters/logos, or
  brand styles you don't own.
- **Cost lever** — iterate on **Turbo/magic-prompt**, finalize on **Quality**; per-image pricing.
  Open weights are **non-commercial** (commercial use needs an Ideogram plan/API/license).
- **Layerize Text is live (beta)** — rendered text extracts to editable layers (all plans + API);
  best on clear, straight type. Movable *image* layers remain roadmap.
- **Prompt craft only** — API/pricing/MCP + the *generate → upload to WoopSocial Media → attach* path
  live in `tools/integrations/ideogram.md`; WoopSocial doesn't generate images.

---

## What the examples share

- **Typed text elements** (exact string + styling) and **deterministic layout** — not vague scenes.
- **Brand palette in hex**; **format set up front** (2K, social aspect).
- **Verified text, disclosed, IP-safe** before publishing.
