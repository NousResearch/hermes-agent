# The reality of AI image editing in 2026 (verify-quarterly)

Engines and terms shift per release — attribute, **verify-quarterly**, and test on your own images (benchmarks
age fast).

## The 2026 state (attribute)
Editing "fixes images you already have" — and the 2026 shift is that the gap between "good enough" and
"indistinguishable from a reshoot" closed for the core tasks: **background removal** is now sharp enough around
hair and glass that most e-commerce sellers stopped paying for manual masking; **inpainting** (diffusion-based
mask-fill) understands scene context, lighting, and texture; **generative expand/outpainting** extends the
canvas convincingly. The standing caveat from the same reviews: *"generative tools can hallucinate, so the
editing lane still rewards judgment over blind automation."*

## The upscaler split (the honesty edge of this whole skill)
Two families, different jobs (attribute — LetsEnhance, aiphotogenerator, tooldirectory):
- **Faithful/true-to-source** (Topaz Gigapixel/Photo AI-class; ESRGAN lineage): adds resolution while
  preserving what's there — the choice for **product, documentary, restoration, anything accuracy-critical.**
  Topaz processes **locally** (privacy; no upload).
- **Creative/generative** (Magnific, Topaz Bloom-class; diffusion): **"hallucinates convincing new detail —
  textures, facial features, environmental elements that weren't visible in the original."** Spectacular for
  art/renders; a **misrepresentation risk** for real products and people. Magnific: creativity/fractality/
  resemblance sliders, prompt-guided upscaling, ~10k×10k ceiling; **Freepik rebranded its whole platform to
  Magnific (Apr 2026).**
- **The checks:** faces at 100% (eyes/teeth/skin); **in-image text gets mangled into "AI-hallucinated
  gibberish"** — OCR + re-typeset documents instead; masters exported PNG/TIFF, never recompressed JPEG; print
  math: 1024px ≈ 3.4" at 300 DPI → 4× ≈ 13.6". And the honest baseline: **"no upscaler beats having the
  original high-resolution file — if you can reshoot, do that."**

## The inpainting engine map (attribute; verify)
- **FLUX Kontext / FLUX.1 Fill** — best raw inpainting quality on complex/large/irregular masks (the stack's
  primary editor; see `flux`); a selectable partner model in Photoshop Beta's Generative Fill.
- **Adobe Firefly Generative Fill** — the notable engine **trained exclusively on licensed content → IP
  indemnification** for commercial/agency work (the routing fact with legal weight).
- **Nano-Banana / Gemini-class editing** — conversational multi-step edits (see `nano-banana`).
- **Canva Magic Eraser/Expand/Grab** — quick fixes inside the design workflow (see `canva`); **Clipdrop**
  (Stability → Jasper) — bundled utilities; **Claid-class** — product-photo-optimized batch pipelines.
- Craft constants: quality tracks the **source** (well-lit, high-res, clean separation; studio > phone
  snapshots); **small precise masks beat large irregular ones**; low-res sources get upscaled **before**
  inpainting; object replacement needs perspective/lighting checks.

## The disclosure/consent layer (stable law + evolving norms)
Commercial **body-shape retouching requires labels in several markets** — France's retouched-photo decree,
Norway's Marketing Act amendment (explicitly covering influencer posts), Israel's 2012 law — verify
applicability per market. The **FTC net-impression** standard means an edited product/result photo IS the claim
(defect concealment = deception; ties to `before-after-and-transformation`). **AI-disclosure** for
synthetic/edited content per platform + the EU AI Act; leave provenance metadata intact.

## House metrics
Human-judged fidelity at 100% + downstream post performance (native) — never a fabricated benchmark.
