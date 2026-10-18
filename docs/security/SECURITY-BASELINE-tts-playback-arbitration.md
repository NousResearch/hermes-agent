# SECURITY-BASELINE: TTS Playback Arbitration

**PR:** fix: global playback arbitration for TTS (multi-session 叠音)
**Date:** 2026-08-14

## Purpose

Baseline the pre-change behavior so the fix can be verified against a known
starting point, and document the behavior contract the fix must preserve.

## Pre-Change Baseline (upstream `main` @ f52feed1ef)

### Behavior
- `play_audio_file()` spawns a system player (`afplay` on macOS) and stores it
  in the module-global `_active_playback` under `_playback_lock`.
- A second concurrent playback request **does not interrupt the first**: it
  overwrites `_active_playback` and both players run → **audio layering
  (叠音)** across sessions/windows/processes.
- `stop_playback()` terminates the *current* `_active_playback` (last writer
  wins) — a request that was superseded by a newer one can no longer be
  stopped by its own caller.
- On macOS, sounddevice output is skipped (`kTCCServiceMediaLibrary` prompt);
  playback goes through `afplay` subprocesses.
- Player fall-through: if a player exits non-zero, the next player in the
  list is tried (`afplay → ffplay → aplay`).

### Baseline Test Results (macOS, before change)
```
tests/tools/test_voice_mode.py: 57 passed, 3 failed, 2 skipped, 2 deselected
Failures (pre-existing, environment-only):
  - TestPlayBeep::test_beep_calls_sounddevice_play        (sounddevice env)
  - TestWSL2PowerShellFallback::test_powershell_pipeline_preserves_real_exit_status (WSL-only)
  - TestWSL2PowerShellFallback::test_wsl2_unique_temp_filename (WSL-only)
```

## Post-Change Baseline (same host, same harness)

```
tests/tools/test_voice_mode.py: 63 passed, 3 failed, 1 skipped, 2 deselected
New tests: 7 (TestPlaybackArbitration 3 + TestCrossProcessArbitration 4)
Remaining failures: identical 3 pre-existing environment failures.
```

**Zero regression.** Every previously passing test still passes; the only
new failures would be our own new tests (all pass).

## Behavior Contract (must hold after the fix)

1. **One sound at a time per process** — at most one live player process per
   Hermes process at any instant (in-process lock + interrupt-before-spawn).
2. **One sound at a time across processes** — at most one live player across
   Hermes processes sharing `HERMES_HOME` (PID-file arbitration).
3. **Latest request wins (barge-in)** — a newer playback interrupts an older
   one; the interrupted call returns `False` and does NOT re-spawn.
4. **`stop_playback()` still works** — external stop interrupts the current
   player and clears both the in-process slot and the cross-process pidfile.
5. **Normal completion** — a player that exits 0 is reported `True`; the slot
   is cleared; the pidfile is removed only by its owner.
6. **No credential leak** — `hermes_subprocess_env(inherit_credentials=False)`
   unchanged; players never inherit gateway tokens/API keys.
7. **Graceful degradation** — if `ps` is unavailable, the pidfile is stale,
   or `HERMES_HOME` is unwritable, playback still works (arbitration is
   best-effort, never a blocker).

## Verification Evidence

- Real concurrent playback (in-process): two threads play 4s/2s WAVs →
  `afplay` peak count **1**, first call interrupted, second completes.
- Real cross-process playback (two independent Python processes): first
  process's player interrupted by second process's play; `afplay` peak **1**;
  no residue; pidfile cleaned.
- `pkill -x afplay` between runs to guarantee a clean start (a stale player
  process from a previous test polluted early peak measurements).
