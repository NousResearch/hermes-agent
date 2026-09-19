---
name: tiktok-script
description: >-
  Use to write a TikTok script — a native, sound-aware short-video script optimized for TikTok's
  discovery engine (the FYP), built around a brutal 1-second hook, watch-time, and rewatches. Run
  when the user says "TikTok script," "TikTok video," "script for TikTok," "make a TikTok,"
  "storytime," "TikTok hook," or wants short-form video for TikTok specifically. Reads
  brand-profile and voice first; rides a trending sound/format where it genuinely fits (live trend
  research goes to trend-jacking); structures for completion + rewatch; optimizes for TikTok search;
  outputs a production doc (shot - on-screen text - spoken - sound - timing) plus a Veo prompt pack
  for faceless/b-roll shots (native filming often beats AI on TikTok). For Instagram Reels use
  reels-script; for YouTube Shorts use youtube-shorts; for the platform-agnostic scripting craft use
  short-form-video-script; hands caption to caption-writer and scheduling to scheduling-and-queue.
metadata:
  version: 1.0.0
license: MIT
---

# TikTok Script

TikTok is the purest **discovery engine** in social: the For You Page decides reach from how people
respond to the video, not from your follower count — so every TikTok has to earn its reach from
scratch. That makes the rules specific, and different from Reels:

1. **The hook is brutal — ~1 second.** Stop the scroll instantly on three channels (verbal +
   on-screen text + visual) or the video is invisible.
2. **Watch-time and rewatches are everything.** Design for completion and a loop, not just a hook.
3. **Make TikToks, not ads.** Native, authentic, slightly-raw content beats polished/ad-like
   content. This is the biggest brand mistake on TikTok — and the pushback this skill makes.
4. **Sound and trends are part of discovery** — ride them where they fit, never forced.
5. **TikTok is also a search engine** — keywords (spoken + on-screen + caption) get you found.

(Full mechanics in `references/tiktok-mechanics.md`.)

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. Voice carries into the spoken script and the caption. Carry
guardrails and compliance rules.

## Step 1 — Lock the concept, goal, and format

- **One idea** — a TikTok does one thing.
- **Goal** — reach, saves, follows, or comments (shapes the close/CTA).
- **Format** — talking head, storytime, POV, tutorial, listicle, green-screen, Stitch/Duet, or
  faceless voiceover. Match it to the message *and* who's making it. See `references/formats-and-sound.md`.

## Step 2 — Decide the sound

Ride a **trending sound** only where it genuinely fits; otherwise use **original audio** (most
brand/founder content) or a commercial-safe music bed. A forced trend hurts. For live trend research
and timing, hand off to `trend-jacking`. See `references/formats-and-sound.md`.

## Step 3 — Write the 1-second hook

Use `hook-writer`. Fire the hook on all three channels at once in the first ~1–3 seconds — no intro,
no logo, no "hey guys." It must be true (no clickbait the video won't pay off). See
`references/hooks-and-retention.md`.

## Step 4 — Structure for watch-time and the loop

Front-load value, keep an open loop, cut all dead air, and design a **loop or rewatch trigger** so
the ending feeds the start (rewatches are one of the strongest FYP signals). Re-hook mid-video for
longer scripts. See `references/hooks-and-retention.md`.

## Step 5 — Write the production doc

Output a shootable doc — a row per beat with **shot · on-screen text · spoken · sound · timing** —
so it can be filmed or generated as-is, not a vague paragraph.

## Step 6 — TikTok SEO + caption + hashtags

Place target **keywords in the spoken script, on-screen text, and caption** (TikTok reads all
three). Use keyword + niche hashtags, not generic high-volume ones. Hand the caption to
`caption-writer`.

## Step 7 — Production: film-first, or the Veo pack

Default to **native filming** for faces/voices (it out-trusts AI on TikTok). For **faceless niches,
b-roll, or unfilmable concepts**, use the bundled Veo prompt pack
(`references/veo-prompt-pack.md`; API layer in `tools/integrations/veo.md`) — generated
"native/handheld," not glossy. **AI-disclosure** is auto on TikTok via WoopSocial.

## Step 8 — Hand off

- **Caption →** `caption-writer`.
- **Schedule →** `scheduling-and-queue` (upload the video to WoopSocial Media, then schedule).

## Quality bar — self-check

- Does the **hook land in ~1 second** on all three channels, with no intro tax?
- Is it built for **watch-time** (fast, no dead air) with a **loop/rewatch trigger**?
- Is the **sound** chosen intentionally and does it actually fit?
- Is it **native, not ad-like**?
- Are **TikTok-search keywords** in spoken + on-screen + caption?
- Is the output a **shootable production doc**?
- On-voice; truthful hook; AI-disclosure where generated?

## Edge cases & pushback

- **"Make it a polished ad with our tagline"** → push back: ad-style underperforms on TikTok;
  deliver a native version that still hits the goal.
- **"Put the trending dance sound on our B2B explainer"** → don't force a mismatched trend; pick a
  sound/format that fits, or original audio.
- **"Generate it with AI instead of filming"** → fine for faceless/b-roll, but flag that filming
  often wins on TikTok; recommend honestly.
- **Thin concept** → shape the one idea; don't pad to fill time.
- **Repurposing a Reel** → re-fit for TikTok (native tone, sound, search keywords), don't cross-post
  the watermarked identical file — see `cross-platform-repurposing`.
- **Regulated/sensitive** → carry guardrails; only verifiable claims; required disclaimers.

## Related skills

- `brand-profile`, `voice-builder` — voice + guardrails.
- `short-form-video-script` — the master scripting craft this specializes for TikTok.
- `hook-writer` — the 1-second hook.
- `trend-jacking` — live trend/sound research and timing.
- `reels-script` — Instagram Reels; `youtube-shorts` — Shorts; `cross-platform-repurposing`
  — adapt across them.
- `caption-writer` — the caption; `scheduling-and-queue` — schedule the post.
- `tools/integrations/veo.md` — the Veo API/connection layer.

## References

- `references/tiktok-mechanics.md` — the FYP discovery engine, signals, native rule, TikTok SEO, length.
- `references/hooks-and-retention.md` — the 1-second hook, watch-time tactics, loop/rewatch design.
- `references/formats-and-sound.md` — TikTok formats + sound/trend strategy and the fit test.
- `references/veo-prompt-pack.md` — AI-video pack for faceless/b-roll, with the native-beats-AI caveat.
- `references/examples.md` — worked production docs, with loops + SEO, vs a polished-ad version.
