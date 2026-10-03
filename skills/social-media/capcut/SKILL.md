---
name: capcut
description: >-
  The CapCut craft skill — edit short-form social video (TikTok, Reels, Shorts) fast and safely:
  retention-paced cuts, auto-captions, beat-synced sound, exports. Use when someone wants to
  edit in CapCut, cut a reel/short, set up auto-captions, use CapCut's AI tools, pick a plan, or
  asks whether CapCut music/assets or client work are safe commercially. Uses the EDITS framework.
  Reads short-form-video-script + brand-profile/design-and-templates + voice-builder first. The
  agent plans the edit and flags the licensing/ToS gates; the HUMAN executes in CapCut and approves;
  WoopSocial publishes the finished export. Safety spine: client/NDA work edits locally, never
  synced (the ToS licenses uploads), and music licensing has fine print — never pirated assets,
  non-consensual lip-sync, or uncorrected captions. Distinct from short-form-video-script (the
  script),
  captions-and-clipping/opus-clip (clipping long video), descript (long-form), canva (graphics),
  and ai-video/veo-3/kling/runway (generation).
version: 1.0.0
---

# capcut

The **short-form editing tool skill** — edit to the script, drive the pace, ink the captions, tighten sound with
the license check, and ship safe + to spec. The agent plans and flags the hazards; the **human executes and
approves**; **WoopSocial publishes** the finished file. (Ships with `tools/integrations/capcut.md`.)

## The POV: the edit serves the script — and "free" has fine print
CapCut earned its place as the default short-form editor: auto-captions in seconds, beat-sync, background removal,
trend templates, phone-to-desktop sync — with a **~15-minute ceiling by design** (it's a short-form tool; route
long-form to Descript/Resolve). Two top-1% edges most creators learn the hard way. **(1) The licensing fine
print:** the June 2025 ToS grants ByteDance a perpetual, transferable, sublicensable license to anything
**uploaded/synced** (you keep copyright — it's a license, not ownership, stated precisely) — so **client/NDA
material edits locally, never synced**; and the music trap: free-library assets are personal-use only, and even
tracks labeled **"Commercial use" are reported cleared for TikTok/CapCut platforms only** — YouTube/IG claims
still happen, verify per track. **(2) Check the tier before you build:** features migrate behind Pro over time,
and the documented failure is hours of editing ending at a paywalled export. The craft line: the edit **executes
what `short-form-video-script` designed** — pacing, captions, and sound serve the beats; and auto-captions always
get an accuracy pass, because a wrong claim in a caption is a published error.

## Read these first
1. **short-form-video-script** — the beats this edit executes.
2. **brand-profile** + **design-and-templates** (caption style/type/color) + **voice-builder** (any copy/TTS).

## The framework: EDITS
(Depth: `references/the-edits-framework.md`.)
- **E — Edit to the script:** load the beats; cut everything that isn't in them; the hook lands in the first
  seconds; the last cut is the loop.
- **D — Drive the pace:** visual change every ~2–4s (jump cuts, punch-ins/keyframes, B-roll, text pops);
  beat-sync; scrub-test for dead stretches.
- **I — Ink the captions:** auto-captions + the mandatory accuracy pass; style once, save the preset; sound-off
  first; SRT/VTT export when copy travels.
- **T — Tighten sound + effects (the license gate):** license-check music BEFORE export (the "Commercial
  use"-is-TikTok-only trap); platform-native audio, licensed tracks, or ai-music-and-sound; effects sparingly.
- **S — Ship safe + to spec:** right resolution/fps per destination; ToS hygiene (local-only for client work);
  likeness/AI-disclosure handled; export → scheduling-and-queue → WoopSocial.

## The reality (verify-quarterly)
2026 CapCut: auto-captions (free ~10-min/project cap; Pro unlimited; SRT/VTT/ASS export), real-time background
removal, motion/camera tracking, vocal isolation + flicker removal (Pro), TTS/avatars, Long-Video-to-Shorts,
keyframes/beat-sync, MP4/MOV to 4K (Pro). Plans restructured early 2026 — **Standard ≈ $9.99 / Pro ≈ $19.99
(~$15 annual); sources conflict (regional/legacy $7.99–9.99 figures) — verify in-app**, and features migrate
behind Pro over time (check tiers before building). Documented billing/cancellation complaints (manage via app
store). The ToS + music-license spines as above. India ban since 2020. **Attribute all; verify-quarterly.** Full
detail: `references/capcut-2026-reality.md`. The daily loop, the pre-edit tier check, the client-work hygiene
checklist, long-form routing, and two worked examples: `references/workflows-and-templates.md`.

## Honest scope (never violate)
- **The agent** plans the edit (cut plan, pacing, caption spec, sound plan + license gates, export spec) and
  flags every ToS/licensing hazard; the **human executes in CapCut**, judges footage, approves the cut (no CapCut
  agent connector — never pretend to see or edit video; never fabricate "that cut works"). **WoopSocial
  publishes** the finished export (TikTok's required fields validate there) — it does **not** edit video; CapCut's
  direct-to-TikTok path exists and the split is a stated choice.
- **Licensing/ToS spine:** personal-use free assets; "Commercial use" music verified per track; uploaded-content
  license → **client/NDA work local-only, never synced** (counsel for high-stakes; not legal advice); users bear
  responsibility for uploads. **Likeness + AI-disclosure:** no non-consensual lip-sync/deepfakes; disclose
  TTS/AI voices where required (EU AI Act; C2PA). **Never** pirated packs, unlicensed music, uncorrected
  captions, or fabricated prices/gates/metrics. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**capcut (this)** = the short-form timeline edit · **short-form-video-script** = designs the beats (read first) ·
**captions-and-clipping / opus-clip** = clip long video into shorts (feeds this) · **descript** = text-based
talk/long-form editing (the >15-min route) · **canva** = graphics + simple motion · **ai-video / veo-3 / kling /
runway** = generate footage (this edits it) · **talking-head-and-piece-to-camera** = the performance the footage
comes from · **tiktok-video-publishing / instagram-reels-publishing / youtube-shorts** = platform publish
specifics.

## Where this connects
Reads first: **short-form-video-script** + **brand-profile/design-and-templates** + **voice-builder.** Pulls
footage from: **talking-head-and-piece-to-camera**, **ai-video/heygen** (disclosed), **captions-and-clipping.**
Feeds: **tiktok-video-publishing**, **instagram-reels-publishing**, **youtube-shorts**, **before-after-and-
transformation**, **content-recycling.** Publishes via: export → **scheduling-and-queue → WoopSocial.** Tool
file: **`tools/integrations/capcut.md`.** Measure with: native + **analytics-and-reporting** on watch-through /
3s hold — never fabricated.

## Definition of done
A short-form edit that executes the script's beats (hook inside the first seconds, dead air gone, the loop cut
last), paced with a visual change every ~2–4s and beat-synced sound, captioned via auto-captions plus the
mandatory accuracy pass in a saved on-brand preset inside safe zones, with every asset license-checked before
export (personal-use free assets avoided for commercial work; "Commercial use" music verified per destination
platform; no pirated packs or unlicensed songs), the ToS hygiene held (client/NDA material local-only, never
synced), likeness/AI-disclosure handled (no non-consensual lip-sync; TTS/avatars disclosed where required), and
the export to spec per destination; tiers checked before the edit was built (no paywall surprise at export), the
human executing and approving the cut, and the finished file published via WoopSocial; measured on watch-through/
3s hold natively; **no fabricated prices, feature gates, capabilities, or metrics**; and correctly distinguished
from short-form-video-script, captions-and-clipping/opus-clip, descript, canva, and the video generators.
