# Runway 2026 capabilities — verified

*Volatile. Re-verify quarterly against runwayml.com/pricing and docs.dev.runwayml.com. Confirm model
IDs/prices before building.*

## What it is
The **control / director's** video platform — built for reproducible shots, camera paths, branded
characters, and edit-grade tools that slot into a production pipeline ("approvals, not just vibes").
Targets filmmakers, agencies, and brand teams. Web editor + mobile + API. (Rebranded from "Runway
ML" to "Runway"; Gen-2 naming is outdated.)

## Models (pick by job)
- **Gen-4.5** — flagship text-to-video AND image-to-video; top-ranked for fidelity, physical
  accuracy, prompt adherence, stylistic range. *Default for hero/final shots.*
- **Gen-4 Turbo** — image-to-video, faster/cheaper for iteration and storyboard tests (animate a
  still with camera/motion control).
- **Gen-4 Image / Gen-4 Image Turbo** — the stills lane (reference-image inputs). (Gen-3 Alpha =
  legacy.)

## Signature control features (why you pick Runway)
- **Gen-4 References** — character/object/style **consistency across shots** (improved object
  consistency + prompt adherence). Solves AI video's biggest weakness.
- **Aleph 2.0** — **video-to-video editing** by text prompt: relight, add/remove objects, style
  transfer, reframe — preserving motion, **without regenerating.** Takes 2–30s of 1080p input, up
  to 5 keyframe anchor images (pinned to "first"/"last" or a timestamp), and edits across multi-cut
  clips (up to ~10 scene changes). Ships with **Edit Studio** in the web app. (The older Gen-4
  Aleph is deprecated — sunset July 30, 2026.)
- **Act-Two** — motion capture (head/face/body/hand) from **any camera** + a character reference;
  transpose a performance onto a character; can change the voice in-tool. **Consent-gated.**
- **Motion Brush** (direct motion in painted areas) and **Camera Control** (defined moves/paths;
  static toggle) — in-app control surfaces that evolve fast (verify in-app), **4K upscaling**, lip
  sync, background removal.

## Limits (design around these)
- Clip length **2–10s per generation** (Gen-4.5 durations: pick 5/8/10s; Veo runs 8s, Kling ~15s
  multi-shot) → **think in shots, stitch in the edit.**
- **No native audio** — pair **ai-voiceover** + sound design in post (this is veo-3's edge, not
  Runway's).
- **4K is an upscale**, not native generation (native 4K is kling's lane); base output is
  720p/1080p.
- Model artifacts (causal-reasoning slips, object permanence, success bias) like all video models.
  Credits deplete fast on top models.

## Multi-model marketplace
One Runway subscription also reaches **Veo 3.1, Kling 3.0, Seedance 2.0, Seedream, Nano Banana,
Gemini, and ElevenLabs audio** inside one dashboard — useful as a single control room across
producers.

## API + pricing (verify-quarterly)
- API: docs.dev.runwayml.com; **API key via `Authorization: Bearer` + a required dated
  `X-Runway-Version` header**; model IDs like `gen4.5` (text/image-to-video), `gen4_turbo`
  (image-to-video), `aleph2` (video-to-video; `gen4_aleph` deprecated), `act_two` (performance).
- **Billed credits per second per model**, credits ~$0.01 each: Gen-4.5 ~12 credits/sec
  (~$0.12/sec), Gen-4 Turbo ~5 (~$0.05/sec), Aleph 2.0 ~28 (~$0.28/sec, 56-credit minimum),
  Act-Two ~5. Poll task status (5s+ intervals with jitter); self-serve usage tiers cap concurrency
  (excess tasks sit `THROTTLED`).
- Plans: Free (125 one-time credits, basic models only, watermarked — a demo) · **Standard ~$12/mo
  annual (~$15 monthly)** (625 credits; unlocks Gen-4.5, Aleph, Veo/Kling, watermark removal, 4K
  upscaling) · **Pro ~$28/mo** (2,250 credits, custom voices, 500GB) · **Max ~$76/mo** (9,500
  credits, 1-month credit rollover, first access to new models).

## Rights + provenance (verify-quarterly; not legal advice)
- **You own your outputs**, and Runway's usage-rights page allows **commercial use on all plans** —
  but Free output is watermarked, so treat it as a demo tier for client work.
- **Runway trains on your inputs/outputs by default** on self-serve plans — flag for NDA/client
  footage (Enterprise for guarantees; verify current terms in-app).
- Outputs carry an invisible **C2PA provenance watermark** (AI origin is detectable — disclose
  regardless).

## How it compares
**Veo** leads photorealism + **native audio**; **Kling** leads native 4K, multi-shot, and
price-per-second; **Luma** leads HDR/mood; **Runway** leads **control + consistency** (References,
Aleph, Act-Two) and pipeline integration. Pick Runway when control beats one-shot quality. (Sora is
discontinued — never route to it.)
