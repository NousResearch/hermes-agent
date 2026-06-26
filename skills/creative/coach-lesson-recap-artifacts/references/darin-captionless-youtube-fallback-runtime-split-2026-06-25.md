# Darin captionless YouTube fallback runtime split (2026-06-25)

## Durable lesson
For coach-agent lesson recap flows, a patched skill alone does not restore behavior if the active runtime processor remains transcript-only.

This session re-established a two-layer fix requirement:
1. **Knowledge layer** — shared/profile-local `youtube-content` skill must encode the fallback order:
   - transcript/captions first
   - direct audio-only via `yt-dlp --extractor-args "youtube:player_client=android" -f "bestaudio[ext=m4a]/bestaudio"`
   - Android-client muxed fallback via `-f 18`
   - local `faster-whisper` transcription
2. **Runtime layer** — the actual processor script invoked by the agent must implement that same order. If the live processor still just calls `fetch_transcript.py` and blocks on failure, the skill update is operationally inert.

## Exact operational failure mode found
- Darin had local STT configured.
- Shared `youtube-content` skill had previously been patched with the correct fallback order.
- But the active processor (`darin_youtube_public_processor.py`) still did only:
  - transcript fetch
  - success => proceed
  - failure => blocker
- Darin profile-local `youtube-content` skill copy was also stale and missing the fallback text.

## Verified recovery pattern
A bounded test against `https://youtu.be/WpjPcUYFAjs` verified:
- direct audio-only `bestaudio[ext=m4a]/bestaudio` remained unavailable in this runtime
- Android-client muxed fallback produced `source.mp4`
- direct local `faster-whisper` transcription of that muxed media succeeded
- acquisition/transcription state should be reported as `local_transcription_complete_from_muxed_media`

## Important implementation detail
For this recovery, direct local `faster-whisper` invocation in the Hermes venv was the correct production fix for the fallback lane.

Reason:
- the generic Hermes local STT wrapper still had a 25MB size gate that blocked large local media before transcription
- the known-good operational goal was restoring Darin’s real fallback path immediately
- therefore the minimal safe fix was to bypass the size-gated wrapper in the live fallback lane and call local `faster-whisper` directly

## Scale-out rule
When restoring or introducing coach-agent YouTube transcription fallback for Darin/Sergio/future agents:
- patch the shared/umbrella skill
- patch the active runtime processor/wrapper
- sync profile-local skill copies if those profiles maintain their own copies
- verify with a forced no-captions test, not just a captions-available case

## What to verify
- captions-available path still returns transcript cleanly
- forced transcript failure triggers direct-audio attempt, then Android muxed fallback
- fallback media file is actually written and non-empty
- local transcription produces real text
- debug/proof artifact captures exact state instead of a vague blocker
