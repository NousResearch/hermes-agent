# The EDITS framework — fast short-form editing that's safe to publish

CapCut's job is speed-to-published-short-form; the failure modes are edits that don't serve retention and assets
that aren't cleared. Each letter is a gate; most edits fail at D (flat pacing) or T (the license check).

## E — Edit to the script
The edit executes what `short-form-video-script` designed — it doesn't improvise structure. Load the beats (hook /
body / payoff / loop), then cut everything that isn't in them: dead air, filler words, the warm-up seconds before
the hook. The first cut's job is to make the hook land inside the first seconds; the last cut's job is the loop.
If there was no script, extract the beats *before* touching the timeline.

## D — Drive the pace
Retention is pacing made visible: a **visual change every ~2–4s** — jump cuts, punch-ins (scale keyframes), angle
switches, B-roll inserts, on-screen text pops. **Beat-sync** cuts to the audio so the rhythm feels intentional.
Keyframes over static frames; speed-ramps sparingly. The test: scrub the timeline — any stretch longer than a few
seconds without a change is a drop-off risk (read the craft in `short-form-video-script`'s WATCH).

## I — Ink the captions
Sound-off first (a large share of social video is watched muted — design for it): **auto-captions, then the
mandatory accuracy pass** — auto transcription
mangles names, accents, and technical vocabulary, and a wrong claim in a caption is a published error. Style once
(legible size, high contrast, 1–2 lines, safe zones, brand type) and **save it as a preset/template** so daily
captioning stops eating time. Export SRT/VTT when the copy needs to travel. Free tier caps captioned minutes;
verify in-app.

## T — Tighten sound + effects (the license gate)
SFX and music carry pacing — but **run the license check before export:** free-library assets are personal-use
only; even "Commercial use"-labeled tracks are reported cleared for **TikTok/CapCut platforms only** (YouTube/IG
claims still happen — verify per track). Safer routes: platform-native licensed audio added in-app at publish,
properly licensed external music, or `ai-music-and-sound`. Effects and trend templates **sparingly** — trends
date fast, and the effect is seasoning, not the meal. Vocal isolation/flicker removal for cleanup (Pro).

## S — Ship safe + to spec
- **Spec:** right resolution/frame rate per destination (4K is Pro), MP4 H.264/H.265, vertical safe zones
  respected, captions inside them.
- **Safe:** the ToS hygiene — client/NDA/confidential material edits **locally, never synced** (the uploaded-
  content license); assets tier-checked before the edit was built; likeness/AI-disclosure handled (TTS/avatar
  voices disclosed where required; never lip-sync a real person without consent).
- **Publish:** export the finished file → `scheduling-and-queue` → **WoopSocial** (TikTok's required fields —
  privacyLevel, allowComment/Duet/Stitch, isYourBrand, isBrandedContent, autoAddMusic — validate at that layer).
  CapCut's direct-to-TikTok path exists; the stack split is a stated choice. The human approves the cut —
  WoopSocial doesn't edit video, and the agent doesn't judge footage it can't see.
