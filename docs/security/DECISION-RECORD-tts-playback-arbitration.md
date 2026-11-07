# DECISION-RECORD: TTS Playback Arbitration Design

**PR:** fix: global playback arbitration for TTS (multi-session 叠音)
**Date:** 2026-08-14
**Status:** Accepted (user-approved, three-layer scope)

## Context

Hermes Desktop renders multiple sessions (tabs/windows) against a single
backend `serve` process; a separate `gateway` process may also run. Each
surface can trigger TTS playback ("read aloud", voice-conversation mode,
gateway audio). Pre-change, a second playback request overwrote the global
`_active_playback` reference **without interrupting the first player**, so two
`afplay` streams layered on top of each other — 叠音 (audio stacking). The
renderer's sequence-based guard ran *after* backend playback had already
started, so it could not prevent the overlap.

## Decision 1: Interrupt-before-spawn (barge-in) vs queue

**Chosen: latest-request-wins (barge-in).**

- A queue (FIFO serialization) would prevent overlap but creates unbounded
  latency when several surfaces request speech in quick succession, and
  "the last thing the user asked to hear" is the natural priority.
- Barge-in matches the pre-existing `stop_playback()` semantics (used by
  `/api/audio/stop`), so the mental model is uniform.
- The interrupt is atomic with the spawn under `_playback_lock`, closing the
  window where two players coexist.

## Decision 2: In-process arbitration placement

**Chosen: inside `_playback_audio_file_impl` player loop, under the existing
`_playback_lock`.**

- One choke point covers all callers: desktop `/api/audio/speak?play=true`,
  CLI voice mode, TUI, and any future surface that calls `play_audio_file`.
- `proc.wait()` stays outside the lock so `stop_playback()` can still
  terminate mid-play; only the kill+spawn+record sequence is atomic.
- A bounded `prev.wait(timeout=0.5)` after `prev.terminate()` eliminates the
  tens-of-ms overlap window from async SIGTERM delivery.

## Decision 3: Cross-process arbitration via PID file vs flock vs daemon

**Chosen: shared PID file with verified-comm kill; flock rejected; daemon
rejected.**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **PID file + `ps` comm check** | ~60 LOC, no new infra, works across unrelated processes, PID-reuse guard | Stale-file residue (harmless), needs `ps` | ✅ Chosen |
| `flock` (blocking) | Simple | Queue semantics (undesired), a hung player blocks everyone, still needs coordination to *interrupt* | ❌ Wrong semantics |
| Unified audio daemon | Cleanest arbitration | New long-running service, state management, startup/health/failure modes — disproportionate for playback | ❌ Overkill |

Key safety property: kill only after verifying the PID still names a system
audio player (`afplay`/`ffplay`/`aplay`). This guards against PID reuse
killing an unrelated process — the only dangerous primitive in the design.

## Decision 4: No fall-through after interruption

**Chosen: a superseded or signal-killed player must NOT fall through to the
next player in the list.**

- Pre-change, a non-zero exit fell through (`afplay → ffplay → aplay`) to
  recover from player failures. After an *interrupt*, falling through would
  **re-spawn the exact stream we were interrupted on** — defeating arbitration
  and re-layering audio.
- Two guards: `was_current` (superseded by newer request or `stop_playback`)
  and `rc < 0` (terminated by signal, incl. cross-process kill). Both return
  `False`. Genuine player failures (exit 0/positive non-zero) still fall
  through.

## Decision 5: PID file ownership

- `_record_playback_pid()` writes after spawn; `_clear_playback_pid()` removes
  only if the file still names our PID (owner-checked). `stop_playback()`
  clears its own PID. Stale files from crashed processes are tolerated: the
  next `_interrupt` reads them, the comm check fails or the PID is dead, and
  the file is overwritten by the next player.

## Decision 6: Per-session serial queue vs global barge-in (follow-up, #23065)

After review triage cross-linked #23065 (serial TTS playback for auto_tts),
we added `play_audio_file_queued(file_path, key)` as a **complementary**
layer rather than replacing barge-in:

- **Same key** (a chat/session id) → requests play in arrival order, one at
  a time: nothing overlaps, nothing is dropped (the #23065 auto_tts
  contract).
- **Different keys / no key** → the global barge-in arbitration still rules:
  a user-initiated read-aloud in another session interrupts, latest intent
  wins.
- **Interrupted queue** → when a queued item is barged-in (returns False),
  the rest of that key's queue is drained (all report False) and the worker
  pauses. Auto-continuing would immediately re-fight the newer playback.

Implementation: `_PlaybackQueue` — one daemon worker per active key, worker
exits after 5s idle (many gateway chats don't pile up threads), restarted on
next submit. Callers that want serialized auto-tts pass their session/chat id
as `key`; the existing desktop read-aloud path stays key-less (barge-in).

## Consequences

- **Positive:** no audio stacking within or across Hermes processes; barge-in
  UX; no new services; bounded (~0.5s max) interruption latency.
- **Negative:** a genuinely crashing player is no longer masked by fall-through
  when it dies by signal — acceptable, rare, and arguably more honest.
- **Trade-off accepted:** cross-process arbitration is best-effort (swallows
  all `ps`/file errors) — playback never blocks on it.
