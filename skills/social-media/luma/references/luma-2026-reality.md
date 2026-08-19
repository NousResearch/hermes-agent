# The reality of Luma Dream Machine in 2026 (verify-quarterly)

Models, credit rates, and tier structures move fast here — attribute everything, **verify-quarterly** at
lumalabs.ai/pricing, and treat any single review's tier table as provisional (they genuinely conflict).

## What it is (and its honest gaps)
Luma Labs' Dream Machine (web + iOS + API) generates cinematic 5–10s video shots from text or images. Its
2026 position (attribute — eesel, pasqualepillitteri, tooljunction): the cinematic/HDR frontier — Ray3's
photorealism and 3D-informed camera motion compete with Runway Gen-4 — and a natural migration after OpenAI's
consumer Sora shutdown (Apr 2026). The honest gaps: **no native audio** (Veo 3.1/Kling generate sound; Luma is
post-production sound), duration caps (shots, not films), and **same-prompt variance** — independent testing
found identical prompts produce noticeably different results **more than half the time** (goenhance) — a real
client-work concern that reference workflows mitigate.

## The model line (attribute; verify — it keeps moving)
- **Ray3** (Sept 2025): the first "reasoning" video model — it interprets, generates, self-evaluates, and
  retries before showing output. Native **16-bit HDR** (three exposure curves) + **EXR export** for grading,
  **keyframes** (start/end frames), **character reference** (identity across shots), **visual annotation**
  (draw motion directly on the image), **Modify** (V2V — restyle real footage, keep the performance).
- **Ray3.14** (Jan 26, 2026): the volume default — native 1080p, ~4× faster, ~3× cheaper at 720p; **drops
  character reference and HDR/EXR** (fall back to Ray3 for identity-locked or film-delivery work).
- A newer Ray already appears on the official pricing page — the line moves; check before assuming.
- **Photon** = Luma's image model; **Luma Agents** (2026) = a multi-model bundle (Ray + Veo, Kling, image +
  ElevenLabs audio models; brief-to-video from DOCX/PDF) — worth it if you'd pay for 2–3 bundled models
  separately; overpaying if you use one (eesel).

## The economics (the workflow IS the cost control)
- **Draft Mode → Hi-Fi:** iterate prompt/composition/camera at draft cost, then Hi-Fi refine only the winning
  shot to production quality — "shoot in draft, develop only the good frame." The single best cost habit.
- **The multipliers:** HDR ≈ 2× and HDR+EXR ≈ 3× the SDR rate; 1080p HDR lands ~**16×** standard 720p credits
  (checkthat); resolution jumps ≈ 4×. Reserve HDR for graded final deliverables.
- **Iteration baseline:** ~3 attempts per usable clip; 4–5 for complex/consistency shots — budget it.
- **Credit traps:** monthly plan credits **expire at reset** (plan sessions, don't hoard); **top-ups roll over**
  (~12-month validity reported); **API credits are a separate wallet** — they do not transfer from the app.
  Trustpilot sentiment skews negative on billing specifically (attribute).

## Rights + tiers (sources conflict — reconcile honestly)
**Free = watermarked + explicitly non-commercial** (all sources agree). Whether the cheapest paid tier includes
commercial rights **conflicts across sources**; the uncontested commercial floor is the **Plus-level plan
(~$30/mo; annual ~20% off)**. Two tier generations coexist (legacy Dream Machine tiers still billable + the
2026 Agents plans ~$30/$90/$300) — **verify in-app.** The overlooked clause: you own outputs, but **Luma
retains a broad license to use your generations for service improvement AND marketing** (your clip may appear
in their promos); **Enterprise** adds a data-privacy guarantee (not used for training) — the tier for NDA work.

## House metrics
Human-judged fitness per shot + downstream post performance (native) — never a fabricated benchmark; test on
your own briefs.
