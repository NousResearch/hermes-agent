/**
 * Repair-loop guard for the desktop bootstrap.
 *
 * Why this exists
 * ───────────────
 * Hermes desktop can request a "repair" of its bundled backend when the
 * renderer observes a transient backend failure (see issue #74874). The
 * classic failure fingerprint:
 *
 *   1. Backend Python process hits a transient GIL stall (e.g. heavy
 *      import, MCP discovery, a long-running agent turn).
 *   2. The renderer's WebSocket can't deliver the `gateway.ready` frame
 *      in time and treats the socket as dead.
 *   3. Renderer calls `hermes:bootstrap:repair`.
 *   4. Bootstrap unconditionally force-reinstalls the venv, restarting
 *      the backend — which stalls again for the same reason.
 *   5. Renderer reports dead backend → another repair → infinite loop.
 *
 * The desktop should distinguish:
 *   - "the venv/install is genuinely broken" → hard reinstall is correct
 *   - "the runtime is healthy but temporarily stalled" → restart only,
 *     NOT a destructive reinstall that drops the venv
 *
 * What this module does
 * ─────────────────────
 * A pure decision helper. Given the current repair attempt count and a
 * hint about whether the live backend process still looks alive, return
 * whether the next repair should:
 *   - `hardReinstall: true`  → run the installer, recreate the venv
 *   - `hardReinstall: false` → restart the existing backend, keep the venv
 *
 * Cap on soft restarts is bounded so an actually-corrupted install still
 * eventually escalates to a hard reinstall after repeated stalls — the
 * guard prevents the *unbounded* reinstall loop, not all reinstalls.
 *
 * The module is intentionally pure (no I/O, no logging, no global state)
 * so it is unit-testable in isolation. Wiring into `main.ts` lives there.
 */

export type RepairDecision =
  | {
      /** Run the installer (recreate venv). Caller bypasses the active runtime. */
      hardReinstall: true
      /** Human-readable rationale for the desktop log. */
      reason: string
      /** 1-indexed repair attempt number for diagnostics. */
      attempt: number
    }
  | {
      /** Skip the installer; restart the existing backend only. */
      hardReinstall: false
      reason: string
      attempt: number
    }

export type RepairDecisionInput = {
  /**
   * 1-indexed count of how many repair attempts have happened in this
   * failure episode. The first repair is `attempt === 1`; a successful
   * boot resets the counter (see `main.ts`'s bootstrap completion path).
   */
  attempt: number
  /**
   * Soft-restart budget before escalation to a hard reinstall. Defaults
   * to 3: three "just restart" attempts, then a real reinstall. Bounded
   * so a corrupt install still gets fixed; high enough that a GIL
   * stall no longer loops the user into a 30-minute reinstall cycle.
   */
  maxSoftAttempts?: number
  /**
   * Whether the live backend process (the one we are about to tear down
   * to honour the repair request) still looks alive. A process whose
   * `exitCode !== null` or `signalCode !== null` has actually exited;
   * a process with both null is either still running or stalled — and a
   * stall is exactly the case the soft-restart path is for.
   */
  primaryBackendAlive: boolean
}

/**
 * Decide the next repair action.
 *
 * Decision matrix:
 *   attempt ≤ maxSoftAttempts AND alive   → soft restart (don't reinstall)
 *   attempt ≤ maxSoftAttempts AND dead    → soft restart (process exited,
 *                                            but we don't yet trust that
 *                                            the install is corrupt; restart
 *                                            once to confirm)
 *   attempt >  maxSoftAttempts            → hard reinstall (give up on the
 *                                            current install)
 *
 * "Alive" being true does NOT force a soft restart on every call: the
 * attempt counter still increments, so an actually-broken install that
 * keeps respawning a child but never announces READY still escalates
 * after `maxSoftAttempts` cycles.
 */
export function decideBootstrapRepair(input: RepairDecisionInput): RepairDecision {
  const maxSoftAttempts = input.maxSoftAttempts ?? 3
  const attempt = Math.max(1, Math.floor(input.attempt))
  const alive = Boolean(input.primaryBackendAlive)

  if (attempt > maxSoftAttempts) {
    return {
      hardReinstall: true,
      attempt,
      reason:
        `repair attempt ${attempt} exceeds soft-restart budget ` + `(${maxSoftAttempts}); escalating to hard reinstall`
    }
  }

  return {
    hardReinstall: false,
    attempt,
    reason: alive
      ? `repair attempt ${attempt}/${maxSoftAttempts}: primary backend process ` +
        `still alive (likely transient stall, see #74874); restarting only, ` +
        `skipping installer`
      : `repair attempt ${attempt}/${maxSoftAttempts}: primary backend process ` +
        `has exited; restarting before escalating to reinstall`
  }
}
