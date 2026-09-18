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

  return {
    /**
     * True exactly once per empty slot; `reset()` when a backend becomes ready
     * again. A backend that dies again shortly after every ready re-arms the
     * latch each time, so the grant is also bounded: more than `maxRespawns`
     * within `windowMs` is a crash loop, and the supervisor stops respawning
     * (`isCrashLooping()`) instead of cycling child + error toast forever.
     */
    claim(state: BackendExitRecoveryState): boolean {
      if (claimed || state.hasCurrentOwner || state.hasPendingStart || state.intentionalTeardown) {
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
    },
    /** True when the last `claim` was refused because the respawn budget for the window is spent. */
    isCrashLooping(): boolean {
      return crashLooping
    },
    reset(): void {
      claimed = false
    }
  }
}
