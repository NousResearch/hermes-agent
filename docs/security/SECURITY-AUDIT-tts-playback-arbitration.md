# SECURITY-AUDIT: Global Playback Arbitration

**PR:** fix: global playback arbitration for TTS (multi-session 叠音)
**Files:** `tools/voice_mode.py`, `tests/tools/test_voice_mode.py`
**Date:** 2026-08-14

## Scope

This audit covers the addition of in-process and cross-process playback
arbitration to `tools/voice_mode.py`:

1. **In-process arbitration** — a new playback request interrupts the previous
   one atomically (under `_playback_lock`) before spawning the system player.
2. **Cross-process arbitration** — a shared PID file under `HERMES_HOME`
   records the current player; a new playback terminates a *verified live
   player* owned by another Hermes process (desktop `serve` vs `gateway` vs
   CLI) before starting.

## Attack Surface Analysis

| # | Surface | Risk | Mitigation | Verdict |
|---|---------|------|-----------|---------|
| 1 | `os.kill(pid, SIGTERM)` on a PID read from a shared file | **PID reuse → killing an unrelated process** | Before any kill, `ps -p <pid> -o comm=` must return one of `{"afplay","ffplay","aplay"}`. A non-player PID (or a failed/stale lookup) is never killed. | ✅ Mitigated |
| 2 | Shared PID file path | Path traversal / symlink attack if attacker controls `HERMES_HOME` | File lives under `get_hermes_home()/tts/.audio-playback.pid` — same trust domain as `state.db`, logs, and config. An attacker who can write `HERMES_HOME` already owns the process. | ✅ Acceptable |
| 3 | PID file content injection (`int()` parse) | Malformed content crashing playback | `int(...)` wrapped in `try/except (OSError, ValueError)`; `pid <= 0` short-circuits. Non-numeric content is ignored. | ✅ Mitigated |
| 4 | Signal handling | `SIGTERM` vs `SIGKILL` — graceful stop vs hard kill | Arbitration uses `SIGTERM` (same as pre-existing `stop_playback()`); `SIGKILL` is only used on 300s player timeout (pre-existing). Player exit is awaited (bounded 0.5s) so no overlap window. | ✅ Safe |
| 5 | Re-entrancy / lock nesting | `_playback_lock` deadlock | The new code never calls `stop_playback()` from inside the lock (which would deadlock on the non-reentrant `threading.Lock`); it performs the terminate inline under the same lock. `proc.wait()` stays OUTSIDE the lock so `stop_playback()` can still interrupt mid-play. | ✅ Audited |
| 6 | `ps` availability | Arbitrary command execution | `ps` is invoked with a fixed argv (`ps -p <pid> -o comm=`), `stderr=DEVNULL`, `timeout=2`; output is decoded and compared against a fixed allowlist. No shell involved. `CalledProcessError`/`OSError`/`TimeoutExpired` are swallowed (arbitration is best-effort). | ✅ Safe |
| 7 | Subprocess env credential leak | Player inheriting gateway tokens | Pre-existing `hermes_subprocess_env(inherit_credentials=False)` scrub is unchanged and still wraps every player spawn. | ✅ Pre-existing, unchanged |
| 8 | Fall-through re-spawn after interrupt | **Defeat of arbitration**: interrupted player re-spawned by next player in list | Two guards: `was_current` check (superseded by newer request / `stop_playback`) and `rc < 0` (killed by signal) both return `False` without trying the next player. | ✅ Mitigated |

## PID-Reuse Guard (Detailed)

The only dangerous primitive is killing a PID that is no longer the player we
recorded. The check is:

```
pidfile → pid
ps -p pid -o comm=  →  must equal "afplay" | "ffplay" | "aplay"
os.kill(pid, SIGTERM)
```

If the original player has exited and the OS has recycled the PID to an
unrelated process, `ps` returns that process's comm → not in the allowlist →
no kill. If the player exited and the PID is free, `ps` fails → no kill.
The worst case is a *missed* interrupt (stale pidfile), never a wrong kill.

## Test Coverage

- `test_interrupt_ignores_non_player_pid` — non-player comm ⇒ no kill
- `test_interrupt_ignores_stale_pidfile` — dead PID / ps failure ⇒ no crash
- `test_interrupt_kills_live_player` — real child process killed only when
  comm matches
- `test_record_then_clear` — pidfile ownership semantics
- `test_new_playback_terminates_previous_atomically` — concurrent plays,
  latest wins, no fall-through
- `test_superseded_playback_does_not_fall_through` — stop_playback ⇒ no
  re-spawn
- `test_sequential_playback_clears_slot` — normal completion clears slot

All pass on macOS (`7 passed` arbitration + cross-process suite).

## Conclusion

**No new exploitable surface.** The only privileged operation (cross-process
`SIGTERM`) is gated behind a verified-comm allowlist and a bounded wait; the
fall-through guards preserve arbitration invariants; credential scrubbing is
unchanged. Signed-off for merge.
