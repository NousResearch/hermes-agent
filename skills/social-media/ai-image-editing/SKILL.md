---
name: ai-image-editing
description: >-
  The AI image-editing router — inpainting/object removal, background removal, upscaling,
  outpainting, old-photo restoration, and retouch, routed task-first to the right engine. Use when
  someone wants to remove an object/person from a photo, cut out backgrounds, upscale an image,
  extend an image to new aspect ratios, restore an old photo, fix a generated image, or asks which
  editing tool to use. Uses the TOUCH framework. Reads brand-profile + design-and-templates first.
  The agent names the task, routes to the right engine, writes the spec, and can call APIs where
  connected; the HUMAN judges every result at 100%; WoopSocial publishes. Honesty spine: an edited
  real photo is an edited claim — creative upscalers hallucinate detail (never on
  products/documents), no defect concealment, body-retouch disclosure honored, no
  watermark/provenance stripping. Distinct from image-prompt/flux/nano-banana (generation), canva
  (the design workflow), and before-after-and-transformation (the claim rules).
version: 1.0.0
---

# ai-image-editing

The **edit router** — name the task, route the engine, one change at a time, uphold the real, check the seams,
hand off with rights. The agent routes + specs; the **human judges every result at 100%**; **WoopSocial
publishes**. (Ships with `tools/integrations/ai-image-editing.md`.)

## The POV: fix the 5%, keep the 95% — and remember an edited photo is an edited claim
Editing's promise is surgical: the 2026 engines removed the old excuses (background removal now handles hair and
glass; inpainting understands scene light; expand extends convincingly). The top-1% operator holds three lines
the tool marketing won't. **(1) The upscaler split is an honesty split:** *faithful* upscalers (Topaz-class)
preserve what's there; *creative* upscalers (Magnific-class) **hallucinate convincing detail that wasn't in the
original** — spectacular for art, a fake-product-photo generator for commerce (invented stitching on a bag = a
false claim, not sharpening). In-image text gets mangled — re-typeset it; no upscaler beats a reshoot.
**(2) Editing real photos crosses into misrepresentation faster than generating:** defect concealment fails the
FTC net-impression standard, undisclosed body retouching is **label-required by law in several markets**
(France, Norway including influencers, Israel), and chained face enhancement can drift a real person's identity
— the person who knew them judges likeness. **(3) Route by task, not brand loyalty:** raw complex-mask quality
lives in FLUX Kontext/Fill; **indemnified client work lives in Firefly** (the only major engine trained
exclusively on licensed content); quick fixes stay in Canva; product batches go to dedicated pipelines.

## Read these first
1. **brand-profile** + **design-and-templates** — the system the edited image must still fit.
2. **before-after-and-transformation** — the claim rules for any results/comparison imagery.

## The framework: TOUCH
(Depth: `references/the-touch-framework.md`.)
- **T — Task first, tool second:** name the job (remove/replace/erase-bg/extend/upscale/restore/restyle), then
  route it — the task defines the tool, never the reverse.
- **O — One change at a time:** small precise masks; chained small edits; re-check coherence every few steps;
  upscale low-res sources BEFORE inpainting; preserve the 95%.
- **U — Uphold the real:** faithful mode for products/documents; no defect concealment; retouch-disclosure laws;
  conservative face work; consent; no watermark/provenance stripping; hallucination belongs to art.
- **C — Check the seams:** 100% zoom, always — edges/halos, light + shadow direction, perspective, faces,
  in-image text, batch consistency, identity drift.
- **H — Hand off with rights:** source license + engine terms (Firefly indemnification vs FLUX
  outputs-vs-service) + disclosure labels; master as PNG/TIFF → WoopSocial.

## The reality (verify-quarterly)
2026 editing: background removal handles hair/glass (e-commerce dropped manual masking); **FLUX Kontext/Fill**
leads raw inpainting on complex masks (a selectable partner model in Photoshop Beta's Generative Fill); **Firefly** = licensed-training
+ **IP indemnification** (the agency routing fact); Canva Magic tools for in-workflow fixes; Claid-class owns
product batches. **Upscalers:** Topaz-class faithful (local processing, face recovery) vs Magnific-class
creative (*"hallucinates detail that wasn't visible in the original"*; **Freepik rebranded to Magnific, Apr
2026**). Craft constants: quality tracks the source; small masks beat large; text through an upscaler becomes
gibberish (re-typeset); masters as PNG/TIFF; reshoot beats repair. Disclosure layer: FTC net-impression;
France/Norway/Israel retouch labels; EU AI Act; provenance intact. *"The editing lane still rewards judgment
over blind automation."* **Attribute all; verify-quarterly.** Full detail:
`references/ai-image-editing-2026-reality.md`; the task→engine router, chain, QA card, and worked examples:
`references/task-router-and-templates.md`.

## Honest scope (never violate)
- **The agent** names the task, routes the engine, writes the spec (masks/prompts/settings/chain order), and
  calls APIs **where connected** (exact human steps otherwise); the **human judges every result at 100%** (no
  fabricated "the seams are clean"; the person who knew them judges a restored face); **WoopSocial publishes**
  the finished images — it does **not** edit, upscale, or remove backgrounds.
- **The honesty spine:** an edited real photo is an edited claim — no defect concealment, faithful upscaling for
  accuracy-critical images, retouch-disclosure laws honored (verify per market), conservative face work,
  consent for others' images, no watermark removal or provenance stripping, AI-disclosure where required,
  source licenses confirmed. **Never fabricate** engine capabilities, benchmarks, or terms — test on your own
  images; counsel for high-stakes claims. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**ai-image-editing (this)** = editing images that exist (the router) · **image-prompt / flux / nano-banana /
ideogram / nano-banana** = generation (FLUX Kontext + nano-banana serve both lanes — this routes their *editing*
use; their skills own the tools) · **canva** = the design workflow (Magic tools = the quick-fix lane) ·
**infographic-and-data-viz / quote-cards-and-text-graphics** = graphics built from scratch ·
**before-after-and-transformation** = the FTC rules any edited "result" must meet · **capcut / ai-video** =
motion (video cleanup routes there).

## Where this connects
Reads first: **brand-profile** + **design-and-templates** + **before-after-and-transformation** (results
imagery). Pulls sources from: real photography, **flux**/the generators (fixing generated images is half the
2026 workload), archives (restoration). Feeds: **canva** (edited assets into layouts), **thumbnail-design** /
**carousel-writer** / **pinterest-pin-design**, the **platform publishing skills.** Publishes via: master +
derivatives → **scheduling-and-queue → WoopSocial.** Tool file: **`tools/integrations/ai-image-editing.md`.**
Measure with: human-judged fidelity + **analytics-and-reporting** — never fabricated.

## Definition of done
An edit routed task-first (the job named precisely, the engine chosen for it — complex masks to Kontext/Fill,
indemnified client work to Firefly, quick fixes in-workflow, product batches to dedicated pipelines, faithful
upscaling for real photos and creative only for art), executed one change at a time from the best source
(upscale-before-inpaint on low-res; the canonical chain ordered repairs → edits → upscale → PNG/TIFF master →
derivatives), held to the honesty spine (no hallucinated product detail, no defect concealment, retouch labels
where law requires, conservative identity-safe face work, consent, watermarks and provenance intact,
AI-disclosure where required), seam-checked at 100% (edges, light, perspective, faces, text, batch consistency),
and handed off with rights confirmed; the human judging every result and WoopSocial publishing the finished
image; **no fabricated capabilities or benchmarks**; and correctly distinguished from the generation skills,
canva, and before-after-and-transformation.
