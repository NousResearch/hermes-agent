---
name: ai-music-and-sound
description: >-
  The AI music + sound-design skill for social -- original/licensed audio beds and sound design for
  Reels/TikToks/Shorts/videos. Use when someone needs background music, a track, or sound effects for a
  social video, asks which AI music tool is safe to use, or asks "can I use this trending sound/song on
  my brand video?". The real brief is "audio that won't get muted, claimed, or sued," so it picks the
  safest licensed source and never uses copyrighted or trending music without a license. Uses the SCORE
  framework. Reads brand-profile + the video it scores first. The agent briefs the music + sound
  design, picks the safest licensed source (ElevenLabs Music/SFX or stock libraries over Suno/Udio;
  paid tier for commercial rights), and advises licensing/Content-ID/disclosure. The tool
  generates/licenses the audio; the creator bakes it in; WoopSocial publishes the video and does NOT
  generate music. Pure AI music may not be copyrightable; never "100% legally safe." Pairs with
  ai-voiceover.
version: 1.0.0
---

# ai-music-and-sound

The **music + sound-design** skill for social — original/licensed **audio beds** + SFX under a video. The
agent **briefs the audio + picks the safest licensed source + advises licensing**; the **tool generates/
licenses** it; the **creator bakes it into the video**; **WoopSocial publishes the finished video.**

## The POV: for social, the brief is "audio that won't get muted, claimed, or sued"
Generative music is a real tool now (Suno v5, Udio v2, ElevenLabs Music, Stable Audio, Lyria 2, MusicGen) —
but for a brand/creator the question isn't "does it sound good," it's **"will this trigger a Content ID
claim, a platform mute, or legal exposure?"** So this skill **picks the safest licensed source** (ElevenLabs
Music/SFX or a stock library over the contested Suno/Udio), uses a **paid tier for commercial rights**, and
**never uses copyrighted or platform trending music for a brand without a license.** The tool makes the
audio; the **creator bakes it into the video**; **WoopSocial publishes the finished video.**

## Read these first
1. **brand-profile** — genre/vibe fit.
2. the **video/asset it scores** — length, cuts, beats.

## The framework: SCORE
(Depth: `references/the-score-framework.md`.)
- **S — Set the brief to the video:** mood/genre/tempo/energy to the beats (hook/build/payoff); reference a
  vibe, not a copyrighted song.
- **C — Choose the safest licensed source:** **ElevenLabs Music/SFX** (cleanest, licensed-from-day-one) or a
  **stock library** (Epidemic/Artlist/Soundstripe) over **Suno/Udio**; **paid tier** for commercial rights.
- **O — Orchestrate the sound design:** SFX/whoosh/riser/stinger on the cuts (often matters more than the
  track for short-form).
- **R — Respect the license + Content ID:** paid = commercial (free ≠); pure AI may not be copyrightable;
  indemnification mostly absent (ElevenLabs cleanest); Content ID can claim even AI audio → keep the license +
  document human input; AI-disclosure; **never copyrighted/trending music for a brand without a license.**
- **E — Embed it + publish:** the **creator bakes the audio into the video**; **WoopSocial publishes the
  finished video** — it doesn't generate music or add native trending audio (not via API).

## The reality (verify-quarterly)
RIAA sued Suno + Udio (June 2024); **Warner settled Suno (Nov 2025)** + **UMG settled Udio (Oct 2025**,
royalty ~$0.002–0.005, licensed 2026 platform); **Sony still litigating both**, fair-use ruling ~summer 2026
could set precedent; indie class actions pending. **ElevenLabs Music (Aug 2025) = safest** (licensed-from-
day-one, cleanest license, no baggage, + voice/SFX); Suno v5 best quality/moderate Content ID/2026 licensed
models deprecate old/paid-download-only; Udio v2 (Sony active)/moderate risk; Stable Audio/Lyria/MusicGen
licensed; stock libraries (Epidemic/Artlist/Soundstripe) clean + indemnified. Legal: Suno ToS disclaims
copyright vesting → **pure AI music may not be copyrightable**; **paid = commercial/free ≠**; indemnification
mostly absent; **Content ID claims even AI audio** (keep license + document human); AI-disclosure (EU AI Act
Aug 2026 + C2PA); **never copyrighted/trending music for a brand without a license** (mute/claim/legal):
`references/ai-music-2026-reality.md`. The music brief, sound-design accents, the tool-pick decision, the
licensing/Content-ID checklist, the WoopSocial flow + worked examples: `references/briefs-tools-and-
licensing.md`.

## Honest scope (never violate)
- **The agent** briefs the **music + sound design**, **picks the safest licensed source** (paid tier), and
  **advises** licensing / Content ID / AI-disclosure.
- **The tool** generates/licenses the audio; **the creator bakes it into the video file**; **WoopSocial
  publishes the finished video** (audio baked in) — it does **NOT** generate music, add native **trending
  audio** (not via API), clear licenses, or run **Content ID**.
- **Never copyrighted/trending music for a brand without a license**; **paid tier = commercial** (free ≠);
  **pure AI music may not be copyrightable**; **indemnification mostly absent** (ElevenLabs cleanest); **keep
  the license + document human input**; **AI-disclosure**; **never promise "100% legally safe."** (Scope,
  distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**ai-music-and-sound (this)** = the music + sound design (the audio **bed**) · **ai-voiceover** = the voice/
narration (ElevenLabs — **they pair**, both audio) · **captions-and-clipping** = captions/clip cutdowns ·
**reels-script**/**tiktok-script** = the script (audio sits under it) · **ai-video** = native video-tool
audio · **instagram-reels-publishing**/**tiktok-video-publishing** = where native **trending audio** lives
(this is the original/licensed, brand-safe alternative).

## Where this connects
Reads first: **brand-profile** + the **video it scores.** Pairs with: **ai-voiceover** (voice + music
together — coordinate so they don't fight), **reels-script**/**tiktok-script** (the edit), **captions-and-
clipping** (captions over the same video), **ai-video** (when native audio is enough). Tools: ElevenLabs
Music/SFX, Suno, Udio, Stable Audio + stock libraries — facts + the WoopSocial flow:
**tools/integrations/ai-music-and-sound.md** (+ **tools/integrations/elevenlabs.md**). Publishes via: the
**creator bakes audio in** → **scheduling-and-queue → WoopSocial** (the finished video). Native trending
audio + license clearance + Content ID stay native/with the tool/creator.

## Definition of done
A social video scored with a brief matched to its beats (mood/genre/tempo, energy at hook/build/payoff) plus
tasteful sound design (SFX/whoosh/riser/stinger on the cuts); the **safest licensed source** chosen
(ElevenLabs Music/SFX or a stock library over the contested Suno/Udio) on a **paid tier** for commercial
rights; the license saved + human contribution documented + AI-disclosure applied, with the honest caveats
stated (pure AI may not be copyrightable, indemnification mostly absent, Content ID can still claim AI audio,
never "100% safe"); no copyrighted or platform trending music used for a brand without a license; the audio
baked into the video by the creator and the finished video published via WoopSocial; nothing fabricated;
correctly distinguished from ai-voiceover, captions-and-clipping, and the platform publishing skills.
