# The reality of FLUX in 2026 (verify-quarterly)

Model variants and license terms shift with releases — attribute, **verify-quarterly**, and check bfl.ai +
Hugging Face model cards before relying on a term.

## The family map (Black Forest Labs — the ex-Stable-Diffusion founders)
- **FLUX.1 era:** [dev] (open weights, Non-Commercial License), [schnell] (**Apache 2.0** — fully open),
  [pro]/1.1[pro]/Ultra (API-only; Ultra = 4MP + Raw photographic mode), **FLUX.1 Kontext** (mid-2025:
  text-to-image **plus in-context editing** — change objects/colors/text by instruction; character/style/object
  consistency **without fine-tuning**; successive edits with minimal drift; [dev] non-commercial, [pro]/[max]
  API; a selectable partner model in Adobe Photoshop Beta's Generative Fill).
- **FLUX.2 (Nov 2025 — current flagship):** a 32B rectified-flow transformer paired with a Mistral-3 24B VLM.
  What it changes: **32k-token prompts** (detailed multi-part briefs), **multi-reference — up to ~8–10 images in
  one call** (consistent characters/products/styles, no LoRA), **clean typography even at small sizes** (the
  "text soup" era largely fixed → production ad creatives/UI mockups), up to **4MP**, **hex color codes as
  direct parameters** (brand-exact color), photorealistic lighting/physics reducing "the AI look." Variants:
  **[pro]** (API flagship), **[flex]** (API, fine-grained control), **[max]** (API, highest quality + real-time
  **web-grounded generation**), **[dev]** (32B open weights, non-commercial), **[klein]** (Jan 2026: **4B =
  Apache 2.0**, sub-second on consumer GPUs — ~13GB VRAM, RTX 3090/4070-class; the 9B sits under the
  non-commercial license). [dev]
  benefits significantly from **prompt upsampling** (a small LLM expanding your prompt).

## The license reality (the trap this skill exists to defuse)
- **Outputs vs the service:** [dev] **generated outputs can be used commercially** per the license — but
  **self-hosting [dev] weights to provide a commercial service** (client work, a product feature) requires a
  **paid BFL license.** Tiers (bfl.ai/licensing, attribute): developer (~10K img/mo, single-domain internal/
  marketing use, **not client work**), product (~100K/mo), **agency (~100K/mo, first 3 clients included, then
  per-client fees).**
- **The overlooked obligation:** the dev Non-Commercial License **requires content filters or manual review** in
  deployment — BFL states it **may approach known deployers at random to verify.**
- **Fully-open paths:** FLUX.1 [schnell] and FLUX.2 [klein] 4B (Apache 2.0). **Easiest commercial path:** the
  hosted APIs (BFL, fal, Replicate, Together, Cloudflare) — license handled, pay-per-image.
- **Provenance:** the FLUX API applies **cryptographically-signed metadata** (C2PA-aligned) to outputs; the dev
  repo ships pixel-layer watermarking examples. Filters for CSAM/NCII (Hive/Microsoft) are non-removable on API.

## Where it runs
BFL Playground (browser, zero setup) → BFL API / third-party APIs → self-host (ComfyUI/diffusers; FLUX.2 [dev]
quantized runs on an RTX 4090 with a remote text encoder). Signals of maturity: Adobe integration, Mistral Le
Chat, and Scorsese-adopted storyboarding (June 2026) — attribute.

## House metrics
Human-judged fitness for the brief + downstream post performance (native) — never a fabricated benchmark; model
strengths shift per release, so test on your own use case.
