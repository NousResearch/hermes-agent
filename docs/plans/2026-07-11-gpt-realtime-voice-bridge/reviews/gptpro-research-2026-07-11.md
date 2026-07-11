# GPT Pro Research — 2026-07-11

## Verdict: NEEDS_ARCHITECTURE_REVISION
## Score: 6/10 original proposal; recommended hybrid is implementable

supporting_only: true
can_approve: false
can_dispatch: false
can_close: false

## Accepted P0 findings

1. `gpt-realtime` is a conversational model, not deterministic TTS; Hermes text may be paraphrased.
2. Realtime conversation + Hermes conversation creates dual ownership and synchronization risk.
3. Transcription completion does not provide API-level exactly-once semantics; Hermes must deduplicate.
4. Beta event names in the Google Meet client must not be reused for GA.
5. Audio barge-in must not be treated as Hermes tool cancellation.

## Adopted architecture

`Browser mic → OpenAI Realtime transcription/VAD over WebRTC → Hermes exactly-once turn → existing/streaming TTS → Browser`

- Hermes owns transcript acceptance, conversation, tools, memory, and policy.
- Realtime response generation is disabled/not used.
- Standard API key remains backend-only.
- Legacy STT/TTS remains fallback.

## Required tests

- transcript ordering and dedupe; duplicate turn rate 0
- key exposure 0
- reconnect/session cleanup
- barge-in/audio-stop latency, without implicit tool cancellation
- legacy voice regression

## Source

GPT Pro conversation: https://chatgpt.com/c/6a51c56a-754c-83e8-8909-f19ebd80b96c
Raw capture: `/tmp/hermes-gpt-realtime-research.raw.txt`
