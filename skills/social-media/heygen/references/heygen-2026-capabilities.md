# HeyGen 2026 capabilities — verified

*Volatile. HeyGen ships major updates monthly — re-verify quarterly against heygen.com and
developers.heygen.com. Confirm exact API schema/prices before building.*

## What it is
Script → photorealistic talking-head video: paste a script, pick an avatar, pick a voice, render
a clip with synced lips, gestures, intonation, and micro-expressions. Best-in-class at avatar
quality, language coverage, custom-avatar accessibility, and API.

## Avatar engines (pick by job)
- **Avatar V** (released Apr 8 2026) — highest fidelity; builds a **Digital Twin from a single
  ~15-second clip**; holds identity across ~10-minute videos without drift; ~0.840 face similarity.
  Digital Twins only; Creator plan and up. *Use for a personal-brand clone.*
- **Avatar IV** — the expressive default (v3); reads the script's emotional register; timing-aware
  gestures. *Use for most stock/twin work.*
- **Avatar III** — legacy v1/v2, existing customers only; unlimited minutes on paid plans. Avoid for
  new builds.
- **Cinematic Avatar** — flat rate per short video (4–15s). **LiveAvatar / Realtime** — real-time
  streaming avatar over WebRTC (720p), connectable to any LLM for 24/7 live sales/support.

## Avatar types
500+ **stock avatars**; **Instant Avatar / Digital Twin** (from ~15s–2min footage; **consent
verification required**); **Talking Photo** (single still — fine under ~15s, uncanny beyond).
Text-prompt avatars exist but are inconsistent — stock or twins are safer.

## Localization (HeyGen's strongest use case)
**Video Translator: 175+ languages** with lip-sync re-sync; **voice cloning** preserves the
original speaker's voice; **Voice Mirroring** matches pacing/emotion. As of Feb 2026, **audio
dubbing is unlimited on paid plans** (no premium credits); lip-synced translation uses credits.

## Production surface
**Video Agent 2.0** (prompt → full video: script/avatar/voice/B-roll/transitions); **BrandKit**
(logos/fonts/colors); 75+ **templates**; **variable insertion** for personalized outreach at scale;
**SCORM/LMS + interactive video/quizzes** (Business+); screen recording; PPT/PDF import. Native
B-roll is generated via Veo 3.1 (route generative scenes through veo-3 / ai-video; do **not** rely
on the Sora path — Sora is discontinued).

## API + pricing (verify-quarterly)
- REST: `POST https://api.heygen.com/v3/videos`; auth via **`X-Api-Key`** (Settings → API). Three
  paths: **MCP** (OAuth, no key, bills web-plan credits), **Skills**, **Direct API** (key, separate
  API wallet). TTS, translation, lipsync, LiveAvatar endpoints too.
- **Pay-as-you-go from $5** (Feb 2026 restructure; no free API credits; credits expire 12 months).
  Avatar V ≈ $3/min (API), Avatar IV ≈ $4/min 1080p, translation ≈ $2/min.
- Web plans: Free (3 videos/mo, 1 min each, 1 twin, 500+ avatars); Creator (200 credits ≈ ~10 min
  Avatar IV, since Avatar IV/V = 20 credits/min); Business $149/mo (1,500 credits, 5 twins, SCORM).
  Credit gotcha: "if you don't use it you lose it" — limited rollover.

## Compliance + how it compares
SOC 2 Type II, GDPR, CCPA, **EU AI Act alignment**, **consent-verified avatar creation** (verbal
consent for likeness). Note: researchers flag HeyGen's consent checks as **less strict than
Synthesia** — so enforce consent yourself. **HeyGen** leads on photoreal clones, languages, API;
**Synthesia** on regulated-enterprise trust/training; **D-ID** on interactive; **Argil** on
faceless short-form speed.

## Honest limits
Avatars lack spontaneity and parasocial warmth; emotional storytelling is still human territory;
single-twin feeds flatten (CPM creep, CTR decay within weeks) — diversify. HeyGen is one input
into the creative mix, not the whole system.
