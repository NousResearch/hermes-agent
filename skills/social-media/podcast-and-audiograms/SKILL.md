---
name: podcast-and-audiograms
description: >-
  The audio-to-social pipeline skill -- turn a podcast episode into short, branded, captioned clips and
  audiograms that drive people back to the full episode. Use when someone wants to repurpose a
  podcast, make podcast clips, create audiograms, or promote an episode on social. Uses the WAVES
  framework. Reads brand-profile + the episode first. Select clips by listener-retention data +
  timestamped transcript, not just an AI virality score, plus a strategic CTA back to the episode. The
  agent plans the clip strategy, briefs the clip/audiogram creation, plans the cadence, and writes
  hooks/CTAs; a clip or audiogram tool produces the clips; the creator finalizes; WoopSocial publishes
  on a schedule and does NOT record/edit/transcribe the podcast, generate clips, or host it. Captions
  are non-optional (85% sound-off). Never fabricates virality or clips someone's copyrighted podcast.
  Distinct from captions-and-clipping, content-recycling, and youtube-long-form.
version: 1.0.0
---

# podcast-and-audiograms

The **audio→social pipeline** — clip the best moments (retention-led), brand + caption them, CTA back to the
full episode. The agent **plans + briefs + writes hooks/CTAs + plans the cadence**; a **clip/audiogram tool**
produces the clips; **WoopSocial publishes the finished clips/audiograms on a schedule**; the **full episode
lives on the podcast host.**

## The POV: a podcast is long-form; social is short, visual, sound-off
The skill isn't "make a podcast" — it's turning the best **30–90s** moments into **branded, captioned**
clips/audiograms that **drive people back to the full episode** (the episode is the pizza; a clip is one
irresistible slice). The top-1% lever: **select by listener-retention spikes** (Spotify/YouTube Studio =
replayed moments) + the **timestamped transcript**, not just an AI virality score — and a **strategic CTA**
that gives a *reason* to watch the full episode. **Audiograms** (cover art + waveform + captions) make
audio-only shows visual. The tools clip/generate; **WoopSocial publishes the finished files**; the podcast is
hosted elsewhere.

## Read these first
1. **brand-profile** — voice/branding.
2. the **episode** — recording + transcript + retention data.

## The framework: WAVES
(Depth: `references/the-waves-framework.md`.)
- **W — Winnow to the best moments:** retention spikes (Spotify/YouTube Studio) + the timestamped transcript →
  5–8 moments/episode; don't re-listen; AI virality scores are a starting point.
- **A — Aim the format to what you recorded:** video-podcast clip (9:16, reframe to active speaker) /
  audiogram (cover art + waveform + captions) if audio-only / quote card (→ `design-and-templates`).
- **V — Voice the hook + captions:** first-3-sec hook; captions on every clip (85% sound-off — non-optional);
  video-clip burn-in → `captions-and-clipping`/`clipping.md`.
- **E — Endcap with a reason to listen:** strategic CTA (a teaser/cliffhanger, not "check out the full
  episode") + link to the episode on the host; SEO keywords.
- **S — Spread on a schedule:** WoopSocial publishes the finished clips/audiograms across platforms on a
  cadence (episode → clips → quote cards → carousel); one episode → a week+ (→ `content-recycling`). The full
  episode stays on the host.

## The reality (verify-quarterly)
55% of Americans listen monthly (Edison 2025); **85% watch sound-off → captions non-optional.** Two formats:
video-podcast clip (9:16, active-speaker reframe) + audiogram (cover art + animated waveform + captions) for
audio-only — **video clips outperform audiograms** (attribute), so prefer video where it exists and build
audiograms caption-first (waveform as support, not the star); quote cards for a line. Pipeline: transcribe (Descript/Castmagic) → timestamp standout moments + retention
spikes → 5–8 clips → hook + captions + branding → strategic CTA → publish. Tools: Opus Clip (virality score,
reframe; not an editor — little creative control), Choppity (best conversational accuracy), Descript (the
editor), Kapwing (free); audiograms: Headliner/Podsqueeze; written: Castmagic; design: Canva (→
`design-and-templates`): `references/podcast-to-social-2026-reality.md`. The production pipeline, format
decision + specs, the tool table, the audiogram anatomy, the CTA library, the WoopSocial flow + worked
examples: `references/pipeline-formats-and-tools.md`.

## Honest scope (never violate)
- **The agent** plans the **clip strategy** (retention-based), **briefs the clip/audiogram creation**, **writes
  hooks + CTAs**, and **plans the cadence.**
- **A clip/audiogram tool** produces the clips; **the creator finalizes** them; **WoopSocial publishes the
  finished clips/audiograms on a schedule.** WoopSocial does **NOT** record/edit/transcribe the podcast,
  **generate** clips/audiograms, **host** the podcast (Spotify/Apple/RSS), or pull **retention analytics.**
- **Captions non-optional** (85% sound-off); **strategic CTA** to the full episode; **the full episode lives on
  the host**; **never fabricate virality**; **never clip someone else's copyrighted podcast** for a brand.
  (Scope, distinctions + connections: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**podcast-and-audiograms (this)** = the audio→social pipeline (retention-led selection + the **audiogram**
format + CTA-back-to-episode) · **captions-and-clipping** = the general clip/caption-burn-in craft (a
video-podcast clip's burn-in routes there) · **content-recycling**/**cross-platform-repurposing** = the
general reuse axes (the episode is one source) · **youtube-long-form**/**youtube-publishing-and-metadata** =
the full episode on YT · **ai-voiceover** = a synthetic voice (this repurposes a real recording) ·
**ai-music-and-sound** = the music bed · **carousel-writer**/**email-and-newsletter** = the quote-card +
newsletter routes.

## Where this connects
Reads first: **brand-profile** + the **episode** (recording + transcript + retention). Pulls from: the
**podcast host** (the full episode + retention), the **clip/audiogram tools**, **captions-and-clipping**/
`clipping.md` (burn-in), **design-and-templates** (audiogram templates + quote cards), **carousel-writer** (a
moment → carousel), **email-and-newsletter** (episode → newsletter), **youtube-publishing-and-metadata**
(titles/keywords). Publishes via: the **finished clips/audiograms** → **scheduling-and-queue → WoopSocial**
(cross-platform cadence). Reuse: **content-recycling**/**cross-platform-repurposing**. Measure with: native +
**analytics-and-reporting** (never fabricated). The full episode stays on the podcast host.

## Definition of done
An episode repurposed into a week of social: 5–8 moments chosen from retention spikes + the timestamped
transcript; the right format per moment (9:16 video clip with active-speaker reframe / audiogram with cover
art + waveform + captions / quote card); a first-3-second hook and captions on every asset (non-optional); a
strategic CTA that gives a reason to watch the full episode, linked to the host; an on-brand template; the
clips produced by a clip/audiogram tool and finalized by the creator, then published by WoopSocial across
platforms on a cadence; recording/editing/hosting/transcription left with the tools + the podcast host; no
virality fabricated and no one else's copyrighted podcast clipped; correctly distinguished from
captions-and-clipping, content-recycling, and youtube-long-form.
