## Summary

Barge-in teardown for desktop voice playback was kept in a single module-level slot that two overlapping playbacks could not both fit. This PR makes teardown deterministic: `stopVoicePlayback()` now drains **every** live speech session (streaming WebSocket + AudioContext, and data-URL audio), a stream start that crosses a stop is abandoned instead of resurrecting playback, and a settled session can no longer write stale state over a newer turn.

## Root cause

`apps/desktop/src/lib/voice-playback.ts` tracked the current playback's barge-in stop in one slot — `let currentStop: (() => void) | null` — with five writers, none of which checks ownership:

| writer | what it does |
| --- | --- |
| `settle` (~L173) | unconditionally nulls the slot, even if it now belongs to a *newer* session |
| `openSpeechStream` (~L199) | unconditionally installs its hook over whatever is live |
| `playSpeechDataUrl` `cleanup` (~L381) | unconditionally nulls the slot |
| `playSpeechDataUrl` stop install (~L405) | installs |
| `playSpeechText` catch (~L484) | unconditionally nulls the slot |

`stopVoicePlayback()` was `currentStop?.(); currentStop = null` — it reaches **at most one** session. The comment next to the install states the invariant this breaks: "stopVoicePlayback() → immediate barge-in: kill the socket … and the audio context" — true of one session at a time, silently false of two. For a streaming session the stop hook is the **only** teardown path: a session whose hook is overwritten/nullified keeps its WebSocket open (the server keeps synthesising), keeps scheduling PCM buffers into its AudioContext, its `done` never resolves (the hook's feed timer leaks), and its queued audio re-emerges audibly on a later turn — "reads an old reply from several turns ago" — while a manual stop only ever settles the (possibly stale) slot holder, which is why "only a manual interruption struggles".

### Overlap entry condition (from code)

`startSpeechStream()` is the only playback entry point whose `stopVoicePlayback()` runs **after** its async gap — `resolveSpeakStreamUrl()` awaits `desktop.getConnection(profile)`, a credential-mint + gateway round trip. The barge-in flow re-points the caller's state inside that gap: `onSpeech` → `stopVoicePlayback()`; `onUtterance` → `submitCapturedUtterance` → `dropSpeechSession()` / `consumePendingResponse()` + `onSubmit` → the next response → `openLiveSpeech` → a second `startSpeechStream`. Whichever gap resolves second then:

- installs its session **after** the stop — playback "resumes" after the user barged — and
- its own post-gap `stopVoicePlayback()` settles **the current turn's live session** through the shared slot (a stop targeting a session other than the caller's); the caller's `responseIdRef` guard (`use-voice-conversation.ts` ~L506) then settles the fresh stale session too, and the extra sequence bumps trip `stoppedDuringStart` (~L536) for the next turn — a current reply cut plus a silently dropped reply.

With two hooks in flight, a single slot cannot represent both — which hook survives is arbitrary, so teardown is exactly non-deterministic after a barge-in.

## Fix (minimal, `voice-playback.ts` only)

1. **Registry instead of a slot.** Every playback registers its idempotent stop in a `liveStops` set and unregisters on settle/cleanup; `stopVoicePlayback()` drains the whole set (snapshot iteration), so a barge closes every socket and AudioContext no matter how playbacks overlap.
2. **Stale-start guard.** `startSpeechStream()` captures the sequence at entry; if any stop landed during stream discovery it returns `null` — an already-stopped playback can never resume (same `isCurrent()` pattern `playSpeechText` already uses). `openLiveSpeech` already handles a null session via its `responseIdRef` and sequence checks.
3. **Stale-completion guard.** The `session.done` completion callback is sequence-scoped so a settled session never writes `idle` over a newer turn's state (which could let the next playback start on top of a live one).

No changes to `use-voice-conversation.ts`; the existing `responseIdRef` / `stoppedDuringStart` / `isCurrent` guards remain as-is.

Out of scope / follow-up: an independent review of this patch identified one pre-existing hook-layer window (generation-phase barge re-invoking `openLiveSpeech` for the barged reply before it is consumed) that the module-level fix deliberately does not address — every session it opens is still torn down deterministically by the next start's drain. That belongs with the voice state-machine family (#74337, #80257, #85770), not this playback-teardown fix.

## Repro level

**Code-path analysis + unit tests — not a hardware reproduction.** This is a Windows hands-free voice path (mic VAD barge-in, Edge TTS); the failing interleaving depends on event-loop ordering of async gaps and cannot be reproduced live on this machine. What the tests do is pin the teardown contract directly with fakes: the stale-start race, the stale state-stomp, and every barge teardown path (live session, drain window, data-URL audio) are exercised.

## Tests

New files:

- `apps/desktop/src/lib/voice-playback.test.ts` — 5 unit tests (fake WebSocket/AudioContext/Audio).
- `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation-overlap.test.tsx` — 1 integration test: the REAL `voice-playback` module driven through the REAL `useVoiceConversation` hook; two turns' stream discoveries overlap around a barge-in, resolved out of order.

Red before / green after (recorded on the pre-fix tree):

- "abandons a stream start that crosses a stop instead of resuming playback" — RED: returned a live session and opened a socket after a stop. GREEN: returns null, opens no socket, state stays idle.
- "does not let a settled session reset the playback state of a newer turn" — RED: `$voicePlayback.status` ended `idle` (the settled session's completion callback stomped the newer session's `preparing`). GREEN: stays `preparing`.
- integration — RED: turn 1's stale discovery settled turn 2's live session (`status` fell to `idle`, its socket was closed, a second socket was opened). GREEN: the current reply keeps speaking, no extra socket, and after the session settles naturally no further text is sent to any socket (stale text cannot feed the next turn).

Suites run (all green):

- 12 voice-related files: `voice-playback.test.ts`, `voice-stop-word.test.ts`, `speech-text.test.ts`, `wake-sound.test.ts`, `thinking-sound.test.ts`, `spoken-reply.test.ts`, `voice-prefs.test.ts`, `use-voice-conversation.test.tsx`, `use-voice-conversation-rearm.test.tsx`, `use-voice-conversation-overlap.test.tsx`, `voice-field-visible.test.ts`, `voice-provider-fields.test.ts` — **77/77**.
- `assistant-message.test.tsx` (read-aloud consumer of `playSpeechText`) — **4/4**.
- Broad sweep `src/lib` + `src/store` + `src/app/chat/composer` — **2478 tests**; the only failures were 1–2 timeout flakes in unrelated files (`markdown-blocks.test.ts` property fuzz, `session-unread-tile.test.ts`) under batch CPU load — both pass standalone (9/9). All voice tests green in both sweeps.
- Typecheck (`tsc -p .`, `tsc -p tsconfig.electron.json`, `tsc -p tsconfig.e2e.json`, all `--noEmit`) — clean.
- ESLint on the three changed/added files — clean.

## Verification

- The two red tests were confirmed failing on the pre-fix tree (outputs captured), then green after the fix; the full voice suite was run twice more after the fix with no flakes.
- `stopVoicePlayback()` semantics for the single-playback case are unchanged (existing `use-voice-conversation*.test.tsx` behavior preserved — 17/17 across those plus the new files, run repeatedly).

Closes #91991
