---
name: luma
description: >-
  The Luma Dream Machine craft skill — generate cinematic AI video shots (Ray3: text-to-video,
  image-to-video, keyframes, character reference, Modify V2V, HDR) with draft-first economics and
  the right rights. Use when someone wants to generate video with Luma/Dream Machine/Ray3, animate
  a still into video, keep a character consistent across shots, restyle real footage, control
  credit burn, pick a Luma model or plan, or build a multi-shot AI brand film. Uses the DREAM
  framework. Reads ai-video + brand-profile + short-form-video-script first. The agent writes shot
  briefs and can drive the API where connected; the HUMAN judges every clip; WoopSocial publishes.
  Generations are shots not films (no native audio); Draft Mode first, HDR only for finals; verify
  tiers in-app. Never use unpermitted likeness, generate fake-event footage, or state credit rates
  as fact. Distinct from ai-video, veo-3/kling/runway, flux (the stills this animates),
  heygen/synthesia, and capcut.
version: 1.0.0
---

# luma

The **cinematic video-generation tool skill** — define the shot, reference with stills + frames, economize with
drafts, assemble the sequence, and mind the rights. The agent briefs (and can drive the API where connected);
the **human judges every clip**; **WoopSocial publishes**. (Ships with `tools/integrations/luma.md`.)

## The POV: shots, not films — and drafts, not gambles
Dream Machine's 2026 edge is per-shot cinematic quality: Ray3 was the first "reasoning" video model (it
interprets, generates, self-evaluates, retries), with 16-bit HDR, keyframes, character reference, and Modify
(restyle real footage, keep the performance). The top-1% operator holds four truths the marketing softens.
**(1) Generations are shots, not films:** 5–10 seconds each, Extend degrades past the initial clip — a brand
film is a shot list, generated shot-by-shot and **assembled in an editor** with **sound added in post** (no
native audio; that's Veo/Kling territory). **(2) Draft first, always:** iterate cheap in Draft Mode, Hi-Fi
master only the winner — because HDR at 1080p runs ~16× standard credits and monthly credits expire at reset.
**(3) Variance is real:** the same prompt differs run-to-run more than half the time (independent testing) — so
client-grade consistency comes from **references** (still-first image-to-video, character reference on base
Ray3, keyframes), never from luck. **(4) The rights fine print:** free = watermarked non-commercial, the
commercial floor is the Plus-level plan (tier tables genuinely conflict — verify in-app), and **Luma keeps a
marketing license on your generations** — the clause client work needs to know about.

## Read these first
1. **ai-video** — the model-agnostic router/craft above the video tools.
2. **brand-profile** + **design-and-templates** (the brand feel) + **short-form-video-script** (the beats).

## The framework: DREAM
(Depth: `references/the-dream-framework.md`.)
- **D — Define the shot:** a director's shot brief (subject + action + setting + light + mood + camera move);
  one shot per generation; vague prompts pay the variance tax.
- **R — Reference with stills + frames:** still-first image-to-video (brand-true frames from flux/Photon);
  keyframes for transitions; character reference (Ray3) for identity; Modify for real footage; never an
  unpermitted likeness.
- **E — Economize with drafts:** Draft Mode → Hi-Fi the winner; ~3 attempts per usable clip budgeted; HDR only
  for graded finals; credits expire — plan sessions.
- **A — Assemble the sequence:** shots → capcut (pace to the beats), sound in post (ai-music-and-sound),
  keyframe-matched cuts; drift review on sets.
- **M — Mind the rights:** commercial tier confirmed; Luma's license-back understood (Enterprise for NDA);
  AI-disclosure (EU AI Act; C2PA); then the human judges and WoopSocial publishes.

## The reality (verify-quarterly)
2026 Luma: **Ray3** (reasoning model; 16-bit HDR + EXR; keyframes; character reference; visual annotation;
Modify V2V) · **Ray3.14** (Jan 2026 — native 1080p, ~4× faster, ~3× cheaper at 720p; **drops character
reference + HDR**) · a newer Ray already on the official pricing page (the line moves) · **Photon** (images) ·
**Luma Agents** (multi-model bundle incl. Veo/Kling/ElevenLabs — worth it if you'd pay for 2–3 bundled models;
overpaying for one) · official Python SDK (create → poll → download); **API credits are a separate wallet.**
Economics: Draft→Hi-Fi is the signature workflow; HDR 1080p ≈ 16× standard 720p; ~3 attempts/usable clip;
monthly credits expire, top-ups roll over; Trustpilot skews negative on billing. Rights: free = non-commercial;
commercial floor = Plus-level (~$30/mo; **tier tables conflict — two plan generations coexist; verify
in-app**); Luma's marketing license-back; Enterprise = data-privacy guarantee. Variance: same prompt differs
>50% of runs (goenhance). Sora's consumer shutdown (Apr 2026) made Luma a migration target. **Attribute all;
verify-quarterly.** Full detail: `references/luma-2026-reality.md`. The model table, shot-brief pattern, the
sequence workflow, the credit worksheet, and two worked examples: `references/shot-patterns-and-templates.md`.

## Honest scope (never violate)
- **The agent** briefs shots, plans references/keyframes/credits, and drives the SDK **where connected** (exact
  human steps otherwise; no unreviewed auto-publish); the **human judges every clip** (no fabricated "that shot
  works"; drift reviews on sets); **WoopSocial publishes** the exports — it does **not** generate or edit video.
- **Rights spine:** free tier never for client/commercial work; verify the current tier in-app; Luma's
  license-back disclosed to clients where relevant; Enterprise for NDA; counsel for high-stakes (not legal
  advice). **Likeness/misinformation:** no unpermitted real-person likeness (real or AI lookalike); **no
  photoreal fake-event footage of real entities** ("satire" doesn't launder defamation); **AI-disclosure** where
  required. **Sound in post** with proper licenses. **Never fabricate** rates, tiers, benchmarks, capabilities.
  (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**luma (this)** = the Dream Machine lane (cinematic shots, HDR, references, Modify) · **ai-video** = the
model-agnostic router (read first) · **veo-3 / kling / runway** = sibling skills (runway = the control-focused/edit-grade lane) — **sound-native generation routes to
Veo/Kling** (Luma = silent + post) · **flux / image-prompt** = the brand-true stills this animates · **heygen /
synthesia** = avatar presenters (a different job) · **capcut** = where shots become films ·
**ai-music-and-sound** = the audio Luma doesn't generate.

## Where this connects
Reads first: **ai-video** + **brand-profile** + **design-and-templates** + **short-form-video-script.** Pulls
stills from: **flux / image-prompt.** Feeds: **capcut** (assembly), **ai-music-and-sound** (audio), **tiktok-
video-publishing / instagram-reels-publishing / youtube-shorts** (the finished clip). Publishes via: export →
**scheduling-and-queue → WoopSocial.** Tool file: **`tools/integrations/luma.md`.** Measure with: native +
**analytics-and-reporting** — never fabricated.

## Definition of done
A Luma workflow that ships shots on-budget and on-rights: every generation briefed as a director's shot (one
shot, concrete subject/action/light/camera), brand control achieved still-first (image-to-video from canonical
flux/Photon frames; character reference on Ray3 for identity; keyframes for transitions; Modify for real
footage; no unpermitted likeness), credits governed by the draft-first loop (Draft iterations → one Hi-Fi
master; HDR reserved for graded finals; ~3-attempt budget; sessions planned around expiring credits; API wallet
funded separately where used), sequences assembled in capcut to the script's beats with licensed sound added in
post and a side-by-side drift review, and rights closed before publishing (commercial tier verified in-app;
Luma's license-back known; AI-disclosure where required; no fake-event footage of real entities); the human
judging every clip and WoopSocial publishing the finished file; **no fabricated rates/tiers/benchmarks, no
one-generation "film" promises**; and correctly distinguished from ai-video, veo-3/kling, flux,
heygen/synthesia, and capcut.
