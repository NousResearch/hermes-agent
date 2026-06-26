# Darin local transcription runtime path lesson (2026-06-24)

## Why this note exists
A live Bryan YouTube lesson request looked like Darin was "going through 3 hours of video." The deeper check showed the gateway was alive, but the specific job was stuck in a degraded local-transcription fallback loop.

## Durable lesson
For coach-agent lesson recap workflows, the important question is not "is Whisper installed somewhere?" The important question is whether the **exact runtime path the target profile uses** can actually perform bounded local transcription.

## What happened
Observed pattern in the Darin profile:
- gateway remained alive and continued receiving Bryan DM traffic
- the specific YouTube request never produced a normal `response ready` completion
- the agent hit repeated `whisper: command not found` errors
- later attempts partially ran Whisper from a profile-local Python user site
- those later attempts spent multiple 600-second terminal runs in CPU transcription attempts without cleanly closing the task

This is the dangerous half-wired state:
- the service is alive
- some Whisper components exist
- but the task still degrades into long retries instead of a clean success or a fast truthful blocker

## Root-cause shape
The live check found three distinct runtime realities:
1. Hermes main venv Python did **not** have `whisper` importable.
2. Darin profile-local Python user site **did** have `openai-whisper` importable.
3. The fallback lane still lacked a reliable PATH contract for external helpers like `ffmpeg`.

This is the durable class-level lesson: mixed environments create false confidence. "Installed" is not enough.

## Verification pattern to reuse
When validating a coach-agent local transcription fallback:

1. Identify the exact runtime the target profile will use.
2. Check module import in that exact runtime.
3. Check PATH visibility for `ffmpeg` in that exact profile context.
4. Run a bounded smoke transcription on a tiny local audio file.
5. Verify a real output artifact exists.
6. Distinguish clearly:
   - service alive
   - job in progress
   - job stuck/degraded
   - local transcription healthy

## Practical patch pattern that worked here
A pragmatic unblocking move was to ensure the target profile's terminal environment had a PATH that included the Homebrew/system binary locations needed for `ffmpeg`, then re-run a bounded Whisper smoke test in the profile-local HOME context.

Important: the durable lesson is **not** the specific PATH string. The durable lesson is to verify the exact profile runtime + PATH + helper visibility together, then prove the lane with a smoke transcription.

## Reporting rule
For operator communication, separate these truths:
- "the agent service is alive"
- "this specific lesson request is stuck"
- "the local transcription fallback lane is healthy/unhealthy"

Do not collapse them into a single yes/no status.
