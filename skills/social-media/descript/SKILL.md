---
name: descript
description: >-
  The Descript craft skill — edit talk content (podcasts, interviews, talking-head video) by
  editing the transcript instead of the timeline. Use when someone wants to edit in Descript, edit
  a podcast or interview, remove filler words/silences, clean up audio (Studio Sound), fix a
  flubbed word without re-recording (Overdub), auto-cut between speakers, turn one recording into
  clips + show notes + chapters, or asks about Descript's plans, credits, or Underlord. Uses the
  WORDS framework plus the interview rule: concision yes, meaning-flips never. Reads the
  recording's content skill + brand-profile/voice-builder first. The agent plans the edit (API/MCP
  where connected); the HUMAN verifies by ear and approves; WoopSocial publishes the exports.
  Overdub is consent-verified own-voice-only; tiers/credits are verified in-app. Distinct from
  capcut (visual
  short-form), captions-and-clipping/opus-clip (clip selection at scale), ai-voiceover (dedicated
  TTS), and podcast-and-audiograms (the strategy).
version: 1.0.0
---

# descript

The **talk-content editing tool skill** — write the edit in the transcript, Overdub with consent, refine the
sound, dress the visuals, and ship the cuts. The agent plans (and can drive Underlord via API/MCP where
connected); the **human verifies by ear and approves**; **WoopSocial publishes** the exports. (Ships with
`tools/integrations/descript.md`.)

## The POV: the transcript is the timeline — decide in text, verify by ear
For dialogue-heavy content, editing the transcript beats scrubbing a timeline: delete the sentence, the clip
disappears; move the paragraph, the footage follows — reviews report ~60–70% editing-time cuts for talk content.
But the paradigm has two sharp edges the top 1% respect. **(1) The voice spine:** Overdub's consent-verified,
**own-voice-only** design is the model, not an obstacle — it exists so nobody types words into someone else's
mouth; and the craft truth is it shines on flubbed *words*, not paragraphs (long Overdub drifts synthetic —
re-record those). **(2) The meaning spine:** text-editing makes it dangerously easy to rearrange a guest into
saying something they didn't — **concision yes, meaning-flips never**, and the human owns the final cut of anyone
else's words. Operationally: the **accuracy pass is mandatory** (transcript errors become wrong edits AND wrong
captions), and since the Sept 2025 overhaul, the workflow must be **credit-aware** — media minutes count
everything you import, and formerly-unlimited AI features are metered.

## Read these first
1. The recording's content skill — **podcast-and-audiograms** / **youtube-long-form** / **educational-content**.
2. **brand-profile** + **voice-builder** (written outputs) + **design-and-templates** (captions/layout).

## The framework: WORDS
(Depth: `references/the-words-framework.md`.)
- **W — Write the edit:** accuracy pass first; then cut tangents/bad takes, one-step filler+silence removal
  (Underlord), restructure by moving paragraphs — decide in text, **verify by ear.**
- **O — Overdub with consent:** own-voice-only, consent-verified; single words/short phrases (paragraphs =
  re-record); vocabulary/credit limits; disclose synthetic speech where required.
- **R — Refine the sound:** Studio Sound once per source (credits; cleaner audio also improves the transcript);
  level speakers; extreme noise is a re-record, not a rescue.
- **D — Dress the visuals:** Automatic Multicam (record separate tracks on purpose), captions from the corrected
  transcript, human-approved B-roll, Eye Contact used honestly; beat-sync/color route elsewhere.
- **S — Ship the cuts:** one transcript → the episode + clips (Underlord flags, the **human picks** fairly) +
  show notes + chapters + a text post; route onward and publish via WoopSocial.

## The reality (verify-quarterly)
2026 Descript: **Underlord** (agentic co-editor — filler/silence in one step, bad-take flags, B-roll suggestions,
clips, show notes) now triggerable via the **2026 public API (open beta) incl. MCP connections**; Overdub
(~24–48h training; source-audio requirements have varied — verify); Studio Sound (~10 credits/use); Automatic
Multicam; Eye Contact; ~92–95% transcription accuracy on clean audio, ~75–85% with noise/accents/jargon; ~23
languages; SOC 2 Type II; cloud-dependent (no offline). **Pricing: the Sept 2025 overhaul** moved to media
minutes + AI-credit metering of formerly-unlimited features; documented bill-shock and no mid-cycle proration
(G2, attributed); **tier figures conflict across sources — verify in-app.** Full detail:
`references/descript-2026-reality.md`. The weekly loop, credit-aware checklist, the Overdub decision table, the
interview-integrity checklist, and two worked examples: `references/workflows-and-templates.md`.

## Honest scope (never violate)
- **The agent** plans the edit and can trigger Underlord/media actions **via the API/MCP where connected**
  (exact human steps otherwise — no pretended automation); the **human verifies by ear** (pacing/tone/fairness
  don't live in text) and approves — the agent never fabricates "that cut sounds great." **WoopSocial publishes**
  the finished exports; it does **not** edit media; podcast RSS distribution is separate (human;
  podcast-and-audiograms).
- **Voice spine:** own-voice-only cloning; never a guest/competitor/public figure; never fabricated words in a
  real mouth; **AI-disclosure** for synthetic speech where required (EU AI Act; C2PA). **Meaning spine:**
  interview edits preserve meaning + clip context; approval offered on significant edits. **Never fabricate**
  tiers, credits, or metrics — verify in-app. (Full scope: `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**descript (this)** = text-based talk-content editing · **capcut** = beat-synced visual short-form (the hybrid:
master here, style cuts there) · **captions-and-clipping / opus-clip** = clip selection at scale (this feeds
them the master) · **podcast-and-audiograms** = the strategy this tool serves · **ai-voiceover (elevenlabs)** =
dedicated TTS/narration (Overdub = own-voice corrections) · **talking-head-and-piece-to-camera** = the
performance (Eye Contact patches a read, doesn't replace delivery) · **youtube-long-form** = the structure the
recording follows.

## Where this connects
Reads first: **podcast-and-audiograms** / **youtube-long-form** + **brand-profile/voice-builder** +
**design-and-templates.** Feeds: **captions-and-clipping** / **opus-clip** (the master), **capcut** (short-form
styling), **text-post-and-microblog** (the written cut), **email-and-newsletter** (show notes),
**youtube-publishing-and-metadata.** Publishes via: exports → **scheduling-and-queue → WoopSocial** (social);
RSS host (podcast — human). Tool file: **`tools/integrations/descript.md`.** Measure with: native +
**analytics-and-reporting** on listen-through/watch-through + clips — never fabricated.

## Definition of done
A talk-content edit made at the speed of text and verified by ear: transcript corrected first (names/jargon —
errors become wrong edits and captions), tangents/bad takes cut and filler+silences removed in batched Underlord
passes, restructure done in text and the result listened through; flubs fixed with consent-verified own-voice
Overdub at word/phrase length only (paragraphs re-recorded; synthetic speech disclosed where required); Studio
Sound run once per source; multicam/captions/layout dressed on-brand; the interview rule held (meaning + clip
context preserved, approval offered, the human owning the final cut of anyone else's words); one corrected
transcript shipped as the episode + human-picked clips + show notes + chapters, routed onward and published via
WoopSocial; the workflow credit-aware under the post-Sept-2025 model (import only what you'll edit; verify tiers
in-app); API/MCP automation only where actually connected; **no cloned third-party voices, no meaning-flips, no
fabricated tiers/credits/metrics**; and correctly distinguished from capcut, captions-and-clipping/opus-clip,
ai-voiceover, and podcast-and-audiograms.
