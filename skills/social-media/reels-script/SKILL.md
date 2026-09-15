---
name: reels-script
description: >-
  Use to write an Instagram Reels script — a native, sound-off-first short-video script optimized
  for Instagram's discovery engine, built around a 3-second hook, watch-through, and sends (DM
  shares). Run when the user says "Reels script," "Instagram Reel," "make a Reel," "Reel hook,"
  "trial reel," or wants short-form video for Instagram specifically. Reads brand-profile and voice
  first; writes the muted version first; structures for watch-through + sends; places IG-search
  keywords; picks original vs trending audio (trending audio is added natively in-app — not via
  WoopSocial); briefs the cover frame; outputs a production doc (shot - on-screen text - spoken -
  audio - timing) plus a Veo prompt pack for faceless/b-roll shots. The generic scripting craft
  lives in short-form-video-script; for TikTok use tiktok-script; for Shorts use youtube-shorts;
  hands the caption to caption-writer, publishing to instagram-reels-publishing, and scheduling to
  scheduling-and-queue.
metadata:
  version: 1.0.0
license: MIT
---

# Reels Script

Reels are Instagram's **discovery engine**: most Reel reach comes from **non-followers**, decided by
how viewers respond — not by follower count. But Reels are not TikToks with a different logo:

1. **The hook window is ~3 seconds** — they decide whether a Reel is pushed wider or throttled.
   Layer it: verbal + on-screen text + visual.
2. **Watch-through and sends rule.** Watch time is the top signal; **sends-per-reach (DM shares)**
   are weighted roughly 3–5× more than likes for reaching new people. Script for both.
3. **Write the muted version first.** Most Reels viewers watch on mute — if the on-screen text
   doesn't carry the story, the script fails before the audio matters.
4. **Eligibility gates come before ranking:** original, **no TikTok/CapCut watermarks**, has audio,
   under the length cap, clear niche. A watermarked reupload is throttled at the gate.
5. **The Reel also lives on your grid** — plan the cover frame; it can't be edited after publish.

(Full mechanics in `references/reels-mechanics.md`.)

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. Voice carries into the spoken script, on-screen text, and
caption. Carry guardrails and compliance rules.

## Step 1 — Lock the concept, goal, and length tier

- **One idea** — a Reel does one thing.
- **Goal** — reach, sends, saves, follows, or comments (shapes the close/CTA).
- **Length tier** — ~15–35s is the reach sweet spot; up to ~90s for tutorials/storytelling that
  earns it; up to ~3 min stays recommendation-eligible. Let the idea pick the length; never pad.
  See `references/reels-mechanics.md`.

## Step 2 — Decide the audio (and write sound-off anyway)

**Original audio** (your voice) for most founder/brand content; a **trending audio** where it
genuinely fits — but Instagram's licensed library is **native-only**: the script names the audio
direction, the human adds the track in-app (WoopSocial can't attach trending audio). Either way, the
Reel must work **on mute**. Live trend research → `trend-jacking`. See `references/formats-and-audio.md`.

## Step 3 — Write the 3-second hook

Use `hook-writer`. Fire on all three channels at once — verbal + bold on-screen text + a striking
first frame — with no intro, no logo, no "hey guys." The on-screen text hook must work silently, and
the hook must be true: a bait-and-switch creates a mid-video retention cliff the algorithm punishes.
See `references/hooks-and-retention.md`.

## Step 4 — Structure for watch-through and sends

Front-load value, open a loop early, change the visual every ~2–4 seconds, cut all dead air, and
engineer a **loop or rewatch trigger** so the ending feeds the start. Then ask the sends question:
**who would DM this to whom, and why?** Relatable, useful-reference, or conversation-starting beats
travel; generic tips don't. See `references/hooks-and-retention.md`.

## Step 5 — Write the production doc + cover brief

Output a shootable doc — a row per beat with **shot · on-screen text · spoken · audio · timing** —
plus a one-line **cover-frame brief** (which frame, or a custom cover with bold text in the
grid-safe center). Keep hook text, faces, and CTAs inside the safe zone — exact zones live in
`instagram-reels-publishing`.

## Step 6 — Instagram SEO + caption + hashtags

Instagram search is keyword-based now: put the target keyword in the **first ~3 seconds spoken**,
the **on-screen text**, and the **caption's first sentence** — and the visual must match the
caption. Keyword research + the search-first caption → `instagram-seo`; caption craft →
`caption-writer`; 3–5 topic-label hashtags, not thirty.

## Step 7 — Test with Trial Reels; produce film-first or with the Veo pack

For unproven hooks/formats, recommend a **Trial Reel** (shown to non-followers first — a free test
that doesn't touch follower-facing stats). Default to **native filming** for faces and voices. For
**faceless niches, b-roll, or unfilmable shots**, use the bundled Veo prompt pack
(`references/veo-prompt-pack.md`; API layer in `tools/integrations/veo.md`) — and follow Meta's
AI-labeling rules for generated media (disclosure on Instagram is native/manual, not automatic).

## Step 8 — Hand off

- **Caption →** `caption-writer` (keyworded via `instagram-seo`).
- **Publish (specs, safe zone, cover, settings) →** `instagram-reels-publishing`.
- **Schedule →** `scheduling-and-queue` (upload the video to WoopSocial Media, then schedule).

## Quality bar — self-check

- Does the **hook land in ~3 seconds** on all three channels, with no intro tax?
- Does the script **work on mute** (on-screen text carries the story)?
- Is it built for **watch-through** (fast, no dead air) with a **loop/rewatch trigger**?
- Can I name **who sends this to whom, and why**?
- Is the **audio choice** intentional — and flagged as native add-in-app if trending?
- Does it **clear the eligibility gates** (original, watermark-free, niche-clear)?
- Keywords in **spoken + on-screen + caption**; a **cover-frame brief** included?
- Is the output a **shootable production doc** — on-voice, truthful hook, no fabricated stats?

## Edge cases & pushback

- **"Just repost my TikToks"** → the watermark + originality gates throttle reach; re-cut a clean
  native version → `cross-platform-repurposing`.
- **"Add the trending audio for me"** → the script names the audio direction; the human adds IG's
  licensed audio in-app. Never claim WoopSocial attaches it.
- **"Make it a polished brand ad"** → authentic, person-first content out-travels glossy spots;
  deliver a native version that still hits the goal.
- **"Guarantee this goes viral"** → no guarantees; a strong hook + send-worthy idea stacks the
  odds, and a Trial Reel tests it risk-free.
- **Thin concept** → shape the one idea; don't pad to fill a length tier.
- **Regulated/sensitive** → carry guardrails; only verifiable claims; required disclaimers.

## Related skills

- `short-form-video-script` — the master scripting craft this specializes for Instagram.
- `brand-profile`, `voice-builder` — voice + guardrails; `hook-writer` — the 3-second hook.
- `instagram-seo` — keywords + the search-first caption; `caption-writer` — caption craft.
- `trend-jacking` — live trend/audio research; `instagram-growth` — the growth strategy above this.
- `tiktok-script` — TikTok; `youtube-shorts` — Shorts; `cross-platform-repurposing` — adapt across.
- `instagram-reels-publishing` — specs/safe zone/cover/publish; `scheduling-and-queue` — schedule
  (via WoopSocial); `tools/integrations/veo.md` — the Veo API/connection layer.

## References

- `references/reels-mechanics.md` — the Reels discovery engine, signals, gates, length tiers, Trial Reels, the grid.
- `references/hooks-and-retention.md` — the 3-second hook, sound-off design, watch-through tactics, the sends lever.
- `references/formats-and-audio.md` — Reels formats + original vs trending audio (and the native-only limit).
- `references/veo-prompt-pack.md` — AI-video pack for faceless/b-roll, with disclosure and film-first caveats.
- `references/examples.md` — worked production docs with cover briefs, sends design, and IG keywords.
