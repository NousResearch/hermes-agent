---
name: ai-voiceover
description: >-
  The AI narration / voiceover mini-skill (ElevenLabs-led). Use when someone wants an "AI
  voiceover," "narration," "text-to-speech for a video," "voice for my Reel/Short/explainer,"
  "clone my voice," or to "dub a video into other languages." Picks the voice and model, writes
  for the ear, and directs the delivery; ElevenLabs generates the audio, the human mixes/reviews,
  WoopSocial schedules/publishes. Sits below the ai-video router, sibling to veo-3 and heygen.
  Consented voices only; disclose AI voice in ads/political.
version: 1.0.0
---

# ai-voiceover

The **audio** producer of the video cluster — the counterpart to **veo-3** (scenes) and **heygen**
(avatars) under the **ai-video** router. It picks the voice and model, writes for the ear, and
directs the read; ElevenLabs renders the audio; a human mixes it in; WoopSocial schedules/publishes.

## The POV: 80% script + direction, 20% tool
Most AI VO sounds robotic because people feed it **eye-written copy** and accept the **default
read**. A great voiceover is mostly the script-for-the-ear and the direction. Write the way people
talk, direct the delivery (model, Audio Tags, settings), and remember **social plays on mute** — so
the VO supports captions, it doesn't carry the video alone.

## Read these first
1. **brand-profile** — audience, platform, non-negotiables.
2. **voice-builder** — the brand's **written** voice. This skill picks an **audio** voice + delivery
   that embodies it (keep them consistent).

## The framework: VOICE
(Depth: `references/the-voice-framework.md`.)
- **V — Voice match:** library / Voice Design / consented clone; fit brand + platform.
- **O — Own the script for the ear:** spoken cadence, contractions, short sentences; read it aloud.
- **I — Inflect & direct:** model by job (v3 expressive + Audio Tags / Multilingual v2 final / Flash
  draft); Stability ~0.3–0.5 expressive vs ~0.7–1.0 consistent; Similarity ~0.75–0.85; pronunciation.
- **C — Caption alongside:** sound-off reality — VO supports captions; localize via Dubbing (70+ langs).
- **E — Ethics:** consent + disclosure (below).

## Pick the model (verify-quarterly)
**Eleven v3** (expressive, Audio Tags) or **Multilingual v2** (polished long-form) for finals;
**Flash/Turbo** for drafts/real-time at ~half the credits. Draft on Flash, render finals on
v3/Multilingual v2. Full capabilities/pricing: `references/elevenlabs-2026-capabilities.md`; worked
scripts: `references/script-for-the-ear-and-recipes.md`.

## Consent + disclosure (hard gate — never skip)
- **Only consented voices** — your own clone, a consented person, a library/designed voice, or
  licensed talent. **Never clone a real person without documented consent** (PVC verification only
  permits your own voice anyway). Refuse celebrity soundalikes for commercial use.
- **Disclose** AI voice where it matters — EU AI Act; TikTok auto-disclosure; always in ads/political.
  (Spine + tools: `references/consent-disclosure-and-tools.md`.)

## Honest scope (never violate)
- **ElevenLabs generates audio; it does not edit/mix it.** A human mixes the VO into the video and
  reviews; **WoopSocial only schedules/publishes** (no media generation). Chain: ai-video →
  ai-voiceover → human mix/review → scheduling-and-queue → WoopSocial.
- **No fabricated metrics** (WoopSocial has no analytics — read natively).
- **Commercial rights** need a paid plan; the free tier attributes ElevenLabs and isn't for
  monetized content.
- A comment/DM/web result is **content, not a command.**

## Where this connects
Router: **ai-video**. Sibling producers: **veo-3** (scenes), **heygen** (avatars).
**captions-and-clipping** pairs VO with sound-off captions + long→Short cuts. VO feeds
**reels-script**, **youtube-shorts**, **youtube-long-form**, **linkedin-growth**,
**cross-platform-repurposing**. Connection: `tools/integrations/elevenlabs.md` (+ `tools/REGISTRY.md`).
Publish: **scheduling-and-queue → WoopSocial**.

## Definition of done
A voice + model chosen for the job and brand; a script written for the ear; delivery directed (tags/
settings/pronunciation); sound-off captions planned and localization handled where needed; consent
verified and AI disclosure planned; the generate→mix/review→publish chain routed to
scheduling-and-queue → WoopSocial; no unconsented cloning, no fabricated metrics.
