# The PIXEL framework — FLUX images that are on-brand and on-license

FLUX's leverage is control (references, editing, exact color, typography) — but the expert separator is knowing
**which variant you're legally using.** Each letter is a gate; most failures happen at P (the license) or by
skipping E (shipping the first roll).

## P — Pick the variant + license
Match the job AND the rights before generating:
- **Hosted API** ([pro]/[flex]/[max] via BFL, fal, Replicate, Together) = easiest commercial path — license
  handled, provenance metadata applied, pay-per-image.
- **Self-host free-and-commercial:** FLUX.1 [schnell] or FLUX.2 **[klein] 4B (Apache 2.0**, ~13GB VRAM,
  sub-second; the 9B klein is non-commercial).
- **[dev] weights:** outputs commercial-OK, but a **self-hosted commercial service needs a paid BFL tier**
  (agency tier: 3 clients included, then per-client fees) and the license **requires filters/manual review.**
- [max] when **web-grounded** content matters; [flex] for fine control; verify terms at bfl.ai (verify-quarterly).

## I — Instruct in scenes, not tags
FLUX reads **natural language, at length** (32k-token context): describe the scene — subject, setting, action,
lighting, mood, camera/lens — as a brief, not keyword soup. Multi-part prompts work ("In the foreground… behind
her… the light source is…"). Use **hex codes for brand-exact color** (#8FBC8F beats "sage green"). [dev] gains
notably from **prompt upsampling.** Put exact in-image text **in quotes.** Route model-agnostic prompt craft to
`image-prompt`; the brand system from `brand-profile`/`design-and-templates`.

## X — eXact references + edits
The 2026 workflow is reference-first, edit-second:
- **Multi-reference (up to ~8–10 images, one call):** lock the character/product/style refs once; vary only the
  scene per image — campaign consistency **without fine-tuning.** (A consistent "character" must never be a real
  person's unpermitted likeness.)
- **Edit, don't re-roll:** a 95%-right image gets a Kontext/FLUX.2 **in-context edit** ("make the jacket blue",
  "change the sign to read 'OPEN'") — the rest stays intact; chain small successive edits, watch for drift.
- **Typography:** legible headlines/UI copy are now production-viable — but **verify every character** (models
  still misspell; a wrong headline is a published error).

## E — Evaluate + iterate
The **human judges every image** — the agent writes prompts and can call APIs, but never pretends to see or
approve a render. Iterate deliberately: fix one variable per roll (prompt → refs → seed), keep a seed when
composition is right and refine by editing, and review campaign sets **side-by-side** for drift. No fabricated
"that looks great."

## L — License + label, then publish
Before shipping: rights confirmed for the variant actually used ✓ · no unpermitted likeness or cloned trade
dress ✓ · **AI-disclosure** where platform/region requires (EU AI Act; C2PA — the API already signs provenance
metadata; never strip it) ✓ · brand QA (color, type, alt text) ✓ → export → `scheduling-and-queue` →
**WoopSocial publishes** the finished image (it does not generate media).
