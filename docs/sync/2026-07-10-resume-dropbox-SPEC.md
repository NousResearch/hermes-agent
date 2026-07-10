# SPEC — Resume-request dropbox (fixes the sessions.json clobber race)

**Date:** 2026-07-10 · **Author:** Apollo · **Incident:** SGR-6EA95669

## Problem

External tools (safe-restart watcher) need to mark a gateway session
`resume_pending=True` so the NEW gateway auto-resumes it at boot. Today they
edit `sessions.json` directly. Two structural failures:

1. **Clobber race:** sessions.json is the gateway's OWN persisted state, saved
   from memory on every mutation. During a drain window the old gateway's final
   `_save()` overwrites the watcher's flag with in-memory `False`. Observed
   live 2026-07-10 (3-minute drain; flag written at quiescence, clobbered
   before shutdown completed).
2. **Boot-read-once:** the new gateway loads sessions.json once at boot
   (`_ensure_loaded_locked`). A flag re-asserted by the watcher AFTER boot is
   invisible — and the gateway's next save clobbers it again.

Root cause shape: **two writers, one file, last-writer-wins, one reader-once.**
The watcher-side mitigation (session-specific verification + fallback ping,
shipped 2026-07-10) makes the failure loud, not absent.

## Fix: single-writer dropbox with boot + periodic sweep

New module `gateway/resume_requests.py`:

- **Dropbox dir:** `<HERMES_HOME>/gateway/resume_requests/`
- **`submit_resume_request(home, session_key, reason)`** — for EXTERNAL
  writers. Writes `<sanitized-key>-<ts>.json` atomically (tmp+rename):
  `{"session_key": ..., "reason": ..., "requested_at": iso}`. Never touches
  sessions.json. Idempotent: duplicate requests for the same key are fine
  (sweep dedups).
- **`sweep_resume_requests(home)`** — gateway-side. Lists the dir, parses each
  file, returns deduped `[(session_key, reason)]`, deletes consumed files.
  Malformed file → moved to `<name>.rejected` (never crash the sweep, never
  re-parse forever). Empty/missing dir → `[]` fast path (one listdir).
  Requests older than `max_age_seconds` (default 3600) are dropped as stale
  (a resume request from yesterday must not wake a session today).

Gateway integration (`gateway/run.py`):

1. **Boot sweep:** at the top of `_schedule_resume_pending_sessions`, sweep
   the dropbox and `session_store.mark_resume_pending(key, reason)` each hit
   (respects the existing suspended-wins rule; unknown keys logged+skipped).
   Then the existing candidate enumeration proceeds unchanged — dropbox
   requests flow through the SAME allowlist/adapter/loop-breaker gates.
2. **Periodic sweep:** the housekeeping loop (already 60s cadence) calls the
   same sweep + a scoped `_schedule_resume_pending_sessions` when it consumed
   ≥1 request. This closes boot-read-once: a request landing after boot is
   honored within ~60s.

Watcher change (`skills-shared/.../watcher.py`):

- `set_resume_flag()` now ALSO calls the dropbox submit (vendored inline —
  the watcher must not import from the runtime tree it is restarting).
  The direct sessions.json write stays for one release (belt-and-suspenders
  for a not-yet-updated gateway), then can be removed.

## Invariants

- INV-1: the gateway remains the ONLY writer of sessions.json.
- INV-2: a dropbox request never bypasses the existing auto-resume gates
  (`_AUTO_RESUME_REASONS`, allowlist auth, suspended-wins, restart-loop
  breaker, freshness window). It only sets the same flag the gateway's own
  shutdown path sets.
- INV-3: sweep is fail-open: an unreadable dropbox never blocks boot resume of
  gateway-marked sessions.
- INV-4: reasons submitted must be in `_AUTO_RESUME_REASONS` to auto-fire
  (unknown reasons keep the existing warn-and-skip path — visible, not silent).
- INV-5: no new config/env surface; dir derived from HERMES_HOME. Max-age is a
  module constant (not config) until someone actually needs to tune it.

## Acceptance

- Unit: submit→sweep roundtrip; dedup; stale-drop; malformed→.rejected;
  missing-dir fast path; boot sweep marks + resumes an idle session; periodic
  sweep honors a post-boot request; suspended session NOT resumed by request.
- Live E2E: with the new gateway running, `submit_resume_request` for an idle
  session from a subprocess → session wakes within one housekeeping tick,
  `boot_resume_scheduled`-equivalent log line present (`PHASE=dropbox_resume`).
- Watcher E2E: full safe-restart on an idle session resumes it (the exact
  SGR-6EA95669 scenario).
