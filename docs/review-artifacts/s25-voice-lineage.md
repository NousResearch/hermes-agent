# S25 WebUI voice donor lineage

This directory preserves the independently reviewable donor for the browser voice slice.

## Immutable donor

- Commit: `791ca2b9f8677568c3d49b38c97549aa981e0706`
- Parent: `a79255d574bbf7279fadd2759782a4641c7976d9`
- Subject: `fix(web): use live speech for mobile voice input`
- Durable local ref: `refs/archive/s25-voice-donor-791ca2b9`
- Deterministically compressed binary format-patch: `docs/review-artifacts/791ca2b9.patch.gz`
- Compressed artifact SHA-256: `88fe5f0f1648cd3694de1c8dd76be1b61acd619b59433952581c2459b9610f0f`
- Uncompressed patch SHA-256: `b8b5ac97c9384601d7ad62b21d24e4e63f88c60a1832a5525a950ef1605e9bec`
- Stable patch-id: `79794abce4eb3e1a12eeb05e9a3bcf5c6626ddbd`

After decompression, the stable patch-id computed from the committed patch equals the patch-id computed directly from commit `791ca2b9`, including the two donor screenshots. The committed patch remains sufficient if the non-published local archive ref is later removed.

## Current-main port delta

The candidate is a manual current-main port, not a cherry-pick. It retains the donor's browser `SpeechRecognition` direction, final/interim transcript accumulation, Android `onend` restart, fatal permission stop, tap-to-submit interaction, and typed Gboard fallback. Relative to the donor it:

- ports onto `origin/main` at `f751a8c5467c41500e505d90cb0eb8b70929080f`;
- bounds restart attempts with exponential backoff and protects callbacks with a capture generation;
- normalizes and submits a non-empty transcript exactly once through a dedicated PTY helper;
- adds explicit disconnected, cancel, unavailable, start-failure, retry-budget, and no-empty-submit behavior tests;
- integrates the control into the current `ChatPage`/`ChatSidebar` layout;
- consumes structured `message.complete` events for browser speech output and cancels playback on barge-in;
- deliberately removes the unproven optional native bridge rather than selecting it from object shape alone; and
- omits the donor screenshots from the product commit because they are preserved byte-for-byte in the binary donor patch and do not constitute real-device proof.

## Truth boundary

Gboard is typed fallback only. Chrome Web Speech supplies browser transcript events. This slice does not claim private Gboard audio/VAD access, on-device or offline recognition, continuous or hands-free conversation, or verified behavior on a physical S25. Those claims remain gated on real-device evidence.
