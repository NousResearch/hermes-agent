# Scope, distinctions & connections

## Honest scope (never violate)
- **The agent** picks the variant/license path, writes the scene prompts, reference plans, and edit
  instructions, and — **where an API connection exists** (BFL / fal / Replicate / Together) — can call
  generation/editing programmatically. Where it doesn't, exact steps for the human (Playground/ComfyUI). No
  pretended automation.
- **The human judges every image** — the agent can't see renders and never fabricates "that looks great";
  campaign sets get a side-by-side drift review.
- **WoopSocial publishes** the finished exports (measurement: the platforms' native analytics); it does **not** generate or edit
  images.
- **The license spine:** rights verified for the variant actually used ([dev] outputs commercial-OK; [dev]
  self-hosted *services* need a paid tier; filters/manual review required on [dev] deployments — BFL may verify;
  Apache-2.0 paths = [schnell]/[klein] 4B); terms shift — verify at bfl.ai (verify-quarterly); high-stakes →
  counsel (not legal advice).
- **Likeness + IP + provenance:** no real-person likeness (real or AI lookalike) without permission; no cloned
  competitor ads/trade dress or copyrighted characters; **never strip provenance metadata**; **AI-disclosure**
  where platform/region requires (EU AI Act; C2PA). **Data honesty:** FLUX renders visuals — it never invents
  statistics (data viz routes to infographic-and-data-viz rules). **Never fabricate** benchmarks, prices, or
  capabilities.

## Distinct from its siblings (route correctly)
- **image-prompt** = the model-agnostic prompt/router skill above all image tools (read it first; this is the
  FLUX-specific lane).
- **ideogram / nano-banana** = sibling image tools with their own lanes (Ideogram =
  graphic-design/text-layouts; nano-banana = photoreal + edits; verify current
  strengths — they shift per release; test on your use case). Midjourney = distinct aesthetic lane
  (external — no skill in this library).
- **ai-image-editing** = the inpaint/upscale/bg-removal router; FLUX Kontext/FLUX.2 editing is a
  primary engine it routes to.
- **canva** = the brand-aware design workflow FLUX output drops into (layout/type system).
- **design-and-templates** = the brand visual system (colors → hex codes, type, logo rules).
- **ai-video / veo-3 / kling** = video generation (BFL's video model is announced — verify before
  claiming).

## Where this connects
- **Reads first:** image-prompt (the craft/router) + brand-profile + design-and-templates (the system, the hex
  codes).
- **Feeds:** canva (layout/type over generated imagery), thumbnail-design, quote-cards-and-text-graphics
  (backgrounds), before-after-and-transformation (honest visuals only), carousel-writer, pinterest-pin-design.
- **Publishes via:** export → scheduling-and-queue → **WoopSocial.**
- **Tool file:** `tools/integrations/flux.md` (variants, license tiers, API/self-host layers, provenance).
- **Measure with:** native + analytics-and-reporting on the published posts — never fabricated.
