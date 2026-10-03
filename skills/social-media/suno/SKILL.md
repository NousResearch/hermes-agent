---
name: suno
description: >-
  The Suno craft skill — generate full songs, brand music, and audio (vocals, lyrics, stems) with
  the right tier, honest rights, and structure control. Use when someone wants to make music with
  Suno/AI, create a brand sound/theme/jingle/podcast intro, control song structure (meta-tags),
  build a consistent sonic identity (Personas, Voices), split stems, distribute AI music to
  Spotify/YouTube, or asks whether they own their Suno songs. Uses the TRACK framework. Reads
  ai-music-and-sound (the router) + brand-profile + the content skill the music serves first. The
  agent writes briefs/lyrics and can drive the API where connected; the HUMAN's ears judge every
  track; WoopSocial publishes the content it scores. Rights: paid plan from the first note,
  commercial rights not ownership, no indemnification while litigation runs, AI-disclosure at
  distribution; never clone a real artist's voice. Distinct from ai-music-and-sound,
  ai-voiceover/elevenlabs (spoken TTS), capcut, and podcast-and-audiograms.
version: 1.0.0
---

# suno

The **music-generation tool skill** — target the brief, rights first, arrange with structure, curate + humanize,
keep disclosure clean. The agent briefs (and can drive the API where connected); the **human's ears judge**;
**WoopSocial publishes** the content the music scores. (Ships with `tools/integrations/suno.md`.)

## The POV: a full song from a sentence — but the rights are the craft
Suno v5.5 makes the most natural vocals in the category, splits its own generations into clean stems, and now
offers a real consistency stack (Personas → your own verified Voice → a Custom Model trained on tracks you
own). But 2026 Suno is defined by its legal moment, and the top-1% operator works rights-first. **(1) Paid plan
before the first note:** free-tier songs are non-commercial **forever** — upgrading never retroactively unlocks
them. **(2) Commercial rights ≠ ownership:** the post-Warner terms grant paid users commercial rights while
Suno's own docs concede purely-AI music isn't copyrightable — you can monetize, but you can barely stop theft
of a raw track; **human elements** (original lyrics, a re-recorded part over stems, real DAW arrangement)
strengthen the claim and reclassify the work as AI-assisted. **(3) No indemnification while the lawsuits run:**
Warner settled and partnered (Nov 2025), but **UMG and Sony are still suing** toward a July 2026 fair-use
hearing — so favor generic style language over artist references, and never clone an artist's voice (the opt-in
program is the consent path, not prompting). **(4) Archive now:** pre-deal models face deprecation — WAVs,
stems, MIDI out. And the social-stack truth: **trending audio is native-only** — Suno is the *owned* brand
sound, not the trend sound.

## Read these first
1. **ai-music-and-sound** — the model-agnostic music router + licensing craft.
2. **brand-profile** + the content skill the music serves (**short-form-video-script** / **podcast-and-audiograms**).

## The framework: TRACK
(Depth: `references/the-track-framework.md`.)
- **T — Target the brief:** genre (1–2) + one mood anchor + 2–4 named instruments + vocal spec; music does a
  job — brief it like a director.
- **R — Rights first:** paid tier before generating; commercial rights ≠ copyright; no indemnification;
  generic styles over artist names; never a cloned artist voice.
- **A — Arrange with structure:** Custom mode + meta-tags ([Intro]/[Verse]/[Chorus]/[Bridge]/[Outro]),
  exclude-styles, short verse blocks, Extend from a timestamp; the consistency stack (Persona → own Voice →
  Custom Model — test for the reported regressions).
- **C — Curate + humanize:** variations → the ears pick the seed; human elements for anything that matters;
  archive WAVs/stems/MIDI before deprecation; batch stem jobs (failed regens reportedly still cost credits).
- **K — Keep disclosure clean:** DDEX AI flag at distribution; no PRO registration for pure-AI; honest
  performance expectations; the track scores the edit in capcut; WoopSocial publishes the content.

## The reality (verify-quarterly)
2026 Suno: **v5.5** (Mar 2026), **Studio** (Premier DAW — 12 zero-bleed stems, MIDI export), Personas /
**Voices** (own-voice, verified) / **Custom Models** (6+ owned tracks), hum-to-song, API (Python/Node;
Zapier/Make; tier gating conflicts — verify). Legal spine: RIAA suit (2024) → **Warner settled + partnered Nov
2025** (ToS softened to "commercial rights"; free commercial downloads removed; **pre-deal model deprecation
scheduled**; artist opt-in announced) → **UMG + Sony still suing; July 2026 hearing**; GEMA verdict mid-2026.
**No retroactive rights; no indemnification; purely-AI output not copyrightable; DDEX AI-disclosure enforced
since late 2025.** Tiers ≈ Free 50 credits/day (non-commercial) · Pro ≈ $8–10 (2,500 credits, commercial
rights) · Premier ≈ $24–30 (10,000 + Studio; same rights); credits don't roll over. Known frictions
(community-reported): Persona/Voice regressions; Studio credit burns on failed stem jobs; fully-AI tracks
underperform human releases on saves ~25–40% (one firm's data). **Attribute all; verify-quarterly.** Full
detail: `references/suno-2026-reality.md`; the brief pattern, use-case router, archive checklist, and worked
examples: `references/brief-patterns-and-templates.md`.

## Honest scope (never violate)
- **The agent** briefs, writes lyrics/structures, plans consistency + archives, and drives the API **where
  connected** (exact human steps otherwise); the **human's ears judge every track** (no fabricated "that sounds
  great"); **WoopSocial publishes** the content the music sits under — it does **not** generate, attach, or
  license audio; **trending audio is native-only.**
- **Rights spine:** paid tier first; no retroactive rights; commercial rights ≠ ownership; human elements for
  anything that matters; no indemnification — litigation-aware prompting (generic styles); counsel for
  high-stakes (not legal advice). **Voice/likeness:** never clone a real artist or fake a collab (own-voice
  verification + artist opt-in = the consent standard). **Disclosure:** DDEX at distribution; no PRO
  registration for pure-AI; AI-disclosure where required (EU AI Act). **Never fabricate** tiers, legal status,
  or performance claims. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**suno (this)** = the Suno-specific music lane · **ai-music-and-sound** = the model-agnostic router + licensing
craft (read first) · **ai-voiceover / elevenlabs** = spoken TTS (Voices here is *singing*, own-voice-only) ·
**capcut** = where the track scores the video (replacing its library-music license trap) · **descript** =
talk-audio editing · **podcast-and-audiograms** = the show the sting serves · **luma / ai-video** = the silent
visuals this is the other half of.

## Where this connects
Reads first: **ai-music-and-sound** + **brand-profile** + the content skill served. Feeds: **capcut** (the
score), **luma/ai-video** (the audio layer), **podcast-and-audiograms** (stings), the platform publishing
skills. Publishes via: the finished content → **scheduling-and-queue → WoopSocial.** Tool file:
**`tools/integrations/suno.md`.** Measure with: human ears + **analytics-and-reporting** — never fabricated.

## Definition of done
Brand music produced rights-first: generated on a paid tier from the first note (free-tier tracks non-commercial
forever), briefed with director-grade specificity (genre + one mood anchor + named instruments + vocal spec),
structured with meta-tags and grown by Extend rather than re-rolls, made consistent through Personas / a
verified own Voice / a Custom Model on owned tracks (tested against reported regressions), and curated by human
ears with human elements added to anything that matters (strengthening the claim; reclassifying as AI-assisted);
keepers archived (WAV + stems + MIDI) ahead of scheduled deprecation; distribution done with the DDEX AI flag,
no PRO registration for pure-AI works, and honest expectations; the track scoring the edit in capcut, the
content published via WoopSocial with AI-disclosure where required; litigation-aware throughout (generic styles,
no artist cloning or fake collabs, no indemnification assumed); **no fabricated tiers, legal claims, or
capabilities**; correctly distinguished from ai-music-and-sound, ai-voiceover, capcut, podcast-and-audiograms.
