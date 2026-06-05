# UGC Stage-Gated Production (BeSoul workflow)

## Why this reference exists
Session learning: user requested a complete generic UGC pipeline and explicitly asked to review progress by stages before finalizing publication.

## Durable workflow

### Stage 1 — Creative definition
- Define: target audience, narrator persona, location, emotional arc.
- Keep script under 45s.

### Stage 2 — Dual-provider still generation
- Generate avatar/location still with two providers (e.g., FAL + Higgs).
- Evaluate against:
  - realism (UGC credibility),
  - emotional fit to message,
  - framing suitability for talking-head/lipsync.
- Pick one winner and record why.

### Stage 3 — Voice
- Generate TTS with ElevenLabs matching persona and message tone.
- Validate final duration is within campaign constraint.

### Stage 4 — Lip-sync capability probe
Before attempting animation, validate model I/O support:
- `higgsfield model list --video --json`
- `higgsfield model get <job_set_type> --json`
- Check if selected model accepts media inputs (`--image`, `--audio`) for the intended operation.

If unsupported, do not stall. Use fallback video assembly.

### Stage 5 — Fallback assembly (if no lip-sync)
- Build motion from still image (subtle zoom/pan), sync with TTS.
- Add mobile-safe caption layers and CTA.
- Export 9:16 MP4.

### Stage 6 — Reel copy
- Hook + value + CTA + hashtags aligned to the hook.
- Keep one language per test post when measuring hook performance.

### Stage 7 — Publish + verify
- Publish via Instagram operational flow.
- Verify permalink before claiming success.

## Reporting format (recommended)
For each stage, report one of:
- `ok`
- `blocked` (with exact reason)
- `fallback` (with produced artifact path/URL)

This avoids hidden failures and keeps user in control of quality decisions.
