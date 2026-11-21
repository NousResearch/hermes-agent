---
name: flux
description: >-
  The FLUX craft skill (Black Forest Labs) — generate and edit on-brand images with the right
  variant and license. Use when someone wants to generate images with FLUX/FLUX.2/Kontext, edit a
  generated image, keep a character or product consistent across a campaign, render legible text
  in images, get brand-exact colors, pick between FLUX variants, or asks if their FLUX use is
  commercially licensed. Uses the PIXEL framework. Reads image-prompt + brand-profile +
  design-and-templates first. The agent writes prompts/edit instructions and can call the API
  where connected; the HUMAN judges every image; WoopSocial publishes. License spine: [dev]
  outputs are commercial-OK but self-hosting for a commercial service needs a paid BFL tier;
  Apache-2.0 paths are [schnell]/[klein] 4B. Never use unpermitted likeness, clone trade dress,
  strip provenance, or invent stats. Distinct from image-prompt, ideogram/nano-banana/Midjourney
  (sibling tools), ai-image-editing (the edit router this feeds), and canva.
version: 1.0.0
---

# flux

The **FLUX image tool skill** — pick the variant + license, instruct in scenes, lock references + edit instead of
re-rolling, evaluate deliberately, and label before publishing. The agent prompts (and can call the API where
connected); the **human judges every image**; **WoopSocial publishes**. (Ships with `tools/integrations/flux.md`.)

## The POV: control is the product — and the license is the trap
FLUX's 2026 edge isn't just quality; it's **control**: multi-reference consistency (up to ~8–10 images in one
call — a campaign-consistent character or product with **no fine-tuning**), in-context editing ("change the
jacket, keep everything else"), **hex-code brand colors** as parameters, 32k-token scene prompts, and typography
clean enough that ad headlines are production-viable. Two top-1% edges most users miss. **(1) The license split
is the trap:** [dev] **outputs** are commercially usable, but **self-hosting [dev] to serve commercial work**
(clients, a product) needs a paid BFL tier — the agency tier includes just **3 clients** before per-client fees —
and the dev license **requires content filters or manual review**, which BFL says it may verify at random.
Apache-2.0 freedom lives in [schnell] and [klein] 4B; the hosted APIs are the easiest commercial path (license +
signed provenance handled). **(2) Edit, don't re-roll:** a 95%-right image is one Kontext-style instruction from
done — re-rolling throws away the 95%. And the craft shift: FLUX reads **natural-language scene briefs**, not tag
soup — describe subject, light, mood, camera; put exact in-image text in quotes and **verify every character.**

## Read these first
1. **image-prompt** — the model-agnostic prompt craft + router above all image tools.
2. **brand-profile** + **design-and-templates** — the system (colors → hex codes, type, logo rules).

## The framework: PIXEL
(Depth: `references/the-pixel-framework.md`.)
- **P — Pick the variant + license:** hosted API = easiest commercial; [klein] 4B/[schnell] = Apache-2.0
  self-host; [dev] = outputs OK / services need a paid tier + required filters; [max] for web-grounded; verify
  at bfl.ai.
- **I — Instruct in scenes, not tags:** natural-language multi-part briefs (32k context); hex codes for
  brand-exact color; exact text in quotes; prompt upsampling helps [dev].
- **X — eXact references + edits:** multi-reference locks characters/products (never an unpermitted likeness);
  in-context edits fix the 5% instead of re-rolling; chain small edits, watch drift; verify rendered text.
- **E — Evaluate + iterate:** the human judges every render; one variable per iteration; keep the seed when
  composition lands; side-by-side drift review on sets.
- **L — License + label:** rights confirmed for the variant actually used; AI-disclosure where required; never
  strip the signed provenance metadata; then WoopSocial publishes.

## The reality (verify-quarterly)
FLUX.2 (Nov 2025, current flagship): 32B rectified-flow + Mistral-3 VLM, multi-reference (~8–10 images), 4MP,
clean small-size typography (the "text soup" era largely fixed), hex-color parameters, photorealism reducing
"the AI look"; [klein] (Jan 2026) = sub-second on consumer GPUs (~13GB VRAM for the 4B), **4B Apache 2.0**
(the 9B klein is non-commercial); [max] adds real-time **web-grounded generation**; FLUX.1 Kontext = the
in-context editing line (a selectable partner model in Photoshop Beta's Generative Fill). Licensing (bfl.ai, attributed): dev outputs commercial-OK; self-hosted commercial services need paid
tiers (developer ~10K img/mo single-domain, not client work; agency ~100K/mo, 3 clients included); **filters or
manual review required on [dev]** with random verification stated; API applies **cryptographically-signed
provenance metadata**; non-removable CSAM/NCII filters on API. Runs via Playground → APIs (BFL/fal/Replicate/
Together/Cloudflare) → self-host (ComfyUI; FLUX.2 [dev] quantized on an RTX 4090). **Attribute all;
verify-quarterly.** Full detail: `references/flux-2026-reality.md`. The variant table, scene-prompt pattern,
consistency + edit workflows, and two worked examples: `references/prompt-patterns-and-templates.md`.

## Honest scope (never violate)
- **The agent** prompts, plans references/edits, and calls the API **where connected** (exact human steps
  otherwise); the **human judges every image** (no fabricated "that looks great"; side-by-side set reviews);
  **WoopSocial publishes** finished exports — it does **not** generate or edit media.
- **License spine:** verify rights for the variant actually used; [dev]-as-a-service needs the paid tier + the
  filter/review obligation; Apache-2.0 = [schnell]/[klein] 4B; high-stakes → counsel (not legal advice).
  **Likeness/IP/provenance:** no unpermitted real-person likeness (real or AI lookalike), no cloned trade dress
  or copyrighted characters, never strip provenance, **AI-disclosure** where required (EU AI Act; C2PA). **Data
  honesty:** never invent stats (data viz → infographic-and-data-viz). **Never fabricate** benchmarks/prices/
  capabilities. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**flux (this)** = the FLUX-specific lane (photorealism + typography + multi-reference + editing + open-weight
control) · **image-prompt** = the model-agnostic router/craft (read first) · **ideogram** = graphic-design/
text-layout lane · Midjourney = distinct aesthetic lane (external — no skill in this library) ·
**nano-banana** = its documented lane (
strengths shift per release — test, don't trust leaderboards) · **ai-image-editing** = the edit router FLUX
editing will serve · **canva** = the design workflow output drops into · **ai-video / veo-3 / kling** = video
(BFL's video model is announced — verify before claiming).

## Where this connects
Reads first: **image-prompt** + **brand-profile** + **design-and-templates.** Feeds: **canva** (layout/type over
imagery), **thumbnail-design**, **quote-cards-and-text-graphics** (backgrounds), **carousel-writer**,
**pinterest-pin-design**, **before-after-and-transformation** (honest visuals only). Publishes via: export →
**scheduling-and-queue → WoopSocial.** Tool file: **`tools/integrations/flux.md`.** Measure with: native +
**analytics-and-reporting** — never fabricated.

## Definition of done
A FLUX workflow that is on-brand and on-license: the variant chosen with its actual rights verified (hosted API
for easy commercial; Apache-2.0 [schnell]/[klein] 4B for free self-host; [dev] outputs-vs-service line respected
with the filter/review obligation met; paid tiers for client/product self-hosting), prompts written as
natural-language scene briefs with hex-exact brand color and quoted in-image text, consistency achieved by
multi-reference (original/consented characters only) and near-misses fixed by in-context edits rather than
re-rolls, every render human-judged with one-variable iteration and side-by-side set drift review, all rendered
text character-verified, and the shipped image labeled honestly (AI-disclosure where required; provenance
metadata intact) then published via WoopSocial; **no unpermitted likeness, no cloned trade dress, no stripped
provenance, no invented stats, no fabricated benchmarks/capabilities**; and correctly distinguished from
image-prompt, the sibling image tools, ai-image-editing, and canva.
