# Scope, distinctions & connections

## Honest scope (never violate)
- **The agent** plans the edit (the cut list against the transcript, the Underlord prompts, the Overdub decision,
  the credit-aware sequence, the repurpose set) and — **where the 2026 API/MCP connection is available** — can
  trigger Underlord actions and media imports programmatically. Where it isn't, the agent writes exact steps and
  the **human executes.** No pretended automation.
- **The human verifies by ear and approves** — pacing, tone, and whether a cut is fair don't live in the
  transcript; the agent never fabricates "that cut sounds great," and the human owns the final cut of anyone
  else's words.
- **WoopSocial publishes** the finished exports (episode clips, social cuts) (measurement: the platforms' native analytics). It does
  **not** edit media, host podcasts, or judge a cut. Podcast distribution (RSS hosts) is outside WoopSocial —
  the human publishes there; route strategy to `podcast-and-audiograms`.
- **The voice spine:** Overdub is **consent-verified, own-voice-only** — never clone a guest, competitor, or
  public figure; never fabricate words into a real person's mouth; **AI-disclosure** for synthetic speech where
  platform/region requires (EU AI Act; C2PA). **The meaning spine:** interview edits preserve meaning and clip
  context (concision yes, meaning-flips never).
- **Never fabricate** tier prices, credit allowances, or metrics — the Sept 2025 overhaul metered
  formerly-unlimited features and sources conflict; **verify in-app** (verify-quarterly).

## Distinct from its siblings (route correctly)
- **capcut** = beat-synced visual short-form editing; **this** = the talk-content/long-form text-based edit (the
  hybrid pipeline: master here, style the cuts there).
- **captions-and-clipping / opus-clip** = clip selection at scale from long video (this feeds them the master;
  Underlord's clip flags overlap — either can find candidates; opus-clip specializes the volume pipeline).
- **podcast-and-audiograms** = the podcast strategy/format skill (this is its editing tool).
- **ai-voiceover (elevenlabs)** = dedicated TTS/voice generation (higher quality for narration; Overdub is for
  own-voice corrections inside the edit).
- **short-form-video-script / youtube-long-form** = the scripts/structures the recording follows.
- **talking-head-and-piece-to-camera** = the performance craft (Eye Contact AI patches a read; it doesn't replace
  delivery).

## Where this connects
- **Reads first:** the content skill for the recording (podcast-and-audiograms / youtube-long-form /
  educational-content-and-how-to) + brand-profile/voice-builder (written outputs) + design-and-templates
  (captions/layout).
- **Feeds:** captions-and-clipping / opus-clip (the master for clipping), capcut (short-form styling),
  text-post-and-microblog (the written cut), email-and-newsletter (show notes), youtube-publishing-and-metadata.
- **Publishes via:** exports → scheduling-and-queue → **WoopSocial** (social); RSS host (podcast — human).
- **Tool file:** `tools/integrations/descript.md` (API/MCP layer, plan gates, the voice-consent spine).
- **Measure with:** native + analytics-and-reporting on listen-through/watch-through + clip performance — never
  fabricated.
