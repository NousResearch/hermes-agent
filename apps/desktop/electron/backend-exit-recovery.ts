/**
 * Supervisor decision for a primary backend child's post-ready exit (#112344).
 *
 * The child's `exit` handler classifies the exit as "current" (this child
 * still owned the connection slot) or "stale" (the slot was already cleared
 * or moved on). A stale exit is normally harmless: a replacement owns the
 * slot or a start is already in flight. But the same classification also
 * fires when the slot was emptied and NOTHING followed — the connection was
 * invalidated without a replacement start, or the child's own `error`
 * handler cleared the slot first — and then the UI keeps running with no
 * engine until the user relaunches the app (9 h observed).
 *
 * `claim` answers "does the supervisor own the respawn for this exit?" from
 * the primary slot's state alone. Pool children never enter the decision:
 * they do not own the window backend and must not suppress its recovery.
 */
export type BackendExitRecoveryState = {
  /** A live primary (local child or remote descriptor) or a published attempt still holds the slot. */
  hasCurrentOwner: boolean
  /** startHermes() is running but has not published its attempt yet. */
  hasPendingStart: boolean
  /** The slot was emptied on purpose (re-home, quit, hand-off, latched boot failure). */
  intentionalTeardown: boolean
}

export type BackendExitRecoveryOptions = {
  /** Respawns the supervisor grants per `windowMs` before it stops and lets the user relaunch. */
  maxRespawns?: number
  windowMs?: number
  now?: () => number
}

export function createBackendExitRecoveryLatch({
  maxRespawns = 3,
  windowMs = 120_000,
  now = Date.now
}: BackendExitRecoveryOptions = {}) {
  let claimed = false
  let respawnedAt: number[] = []
  let crashLooping = false

  const blocked = (state: BackendExitRecoveryState) =>
    state.hasCurrentOwner || state.hasPendingStart || state.intentionalTeardown

  const claim = (state: BackendExitRecoveryState): boolean => {
    if (claimed || blocked(state)) {
      return false
    }

    const at = now()
    respawnedAt = respawnedAt.filter(t => at - t < windowMs)
    crashLooping = respawnedAt.length >= maxRespawns

    if (crashLooping) {
      return false
    }

    respawnedAt.push(at)
    claimed = true

    return true
  }

  return {
    /**
     * True exactly once per empty slot; `reset()` when a backend becomes ready
     * again. A backend that dies again shortly after every ready re-arms the
     * latch each time, so the grant is also bounded: more than `maxRespawns`
     * within `windowMs` is a crash loop, and the supervisor stops respawning
     * (`isCrashLooping()`) instead of cycling child + error toast forever.
     */
    claim,
    /**
     * Re-arm only the recovery attempt that already owns this latch and failed
     * before reaching ready. A concurrent owner/start or intentional teardown
     * keeps the existing claim intact; its eventual ready transition owns the
     * normal reset. A real retry consumes the same crash-loop budget as every
     * other supervisor respawn.
     */
    retryAfterFailedStart(state: BackendExitRecoveryState): boolean {
      if (!claimed || blocked(state)) {
        return false
      }

      claimed = false

      return claim(state)
    },
    /** True when the last claim attempt was refused because the respawn budget for the window is spent. */
    isCrashLooping(): boolean {
      return crashLooping
    },
    reset(): void {
      claimed = false
    }
  }
}

export type SupersededExitNoticeState = {
  /** The dying child had completed boot and served renderer traffic. */
  ready: boolean
  /** The slot was emptied on purpose (re-home, quit, hand-off, shutdown). */
  intentionalTeardown: boolean
  /** The supervisor took the exit over (respawn, or its crash-loop notice). */
  recovered: boolean
}

/**
 * Whether a STALE exit — the slot moved on before this child's exit event was
 * processed — should still tell the renderer the backend died (#118784).
 *
 * The current-owner branch sends `hermes:backend-exit` whenever the supervisor
 * does not take over, but the stale branch sent nothing: a backend that died
 * post-ready and was re-dialed by the renderer replaced itself with no user
 * notice, and the open chat lost its pinned state silently. Notify exactly
 * when the death was real (ready, not deliberate) and nobody else notified:
 * the respawn and crash-loop paths already send their own exit payload, and
 * boot-time exits stay silent so the boot overlay keeps owning failure UX.
 */
export function shouldNotifySupersededBackendExit(state: SupersededExitNoticeState): boolean {
  return state.ready && !state.intentionalTeardown && !state.recovered
}
