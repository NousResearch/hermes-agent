/**
 * Replay policy for the Quick Entry window's pushed gateway truth.
 *
 * The quick window carries NO gateway connection of its own: the primary
 * renderer pushes connection state + recent sessions through main, which
 * caches the latest payload. A window that loads AFTER a push must be replayed
 * the cached copy, but the historical blind `webContents.send` from
 * `did-finish-load` raced React's mount — a payload sent before the composer
 * wired its `onState` listener vanished and the composer stayed stuck at its
 * initial "Not connected" placeholder even while the primary window chatted
 * away (#95132).
 *
 * The fix is an acknowledged, bounded retry: every delivery is retried on a
 * short backoff until the quick window ECHOES the adopted payload back (its
 * store keeps the payload verbatim, so an echo proves a mounted composer
 * accepted exactly this state). Newest cached truth supersedes an in-flight
 * replay; a dead window or an exhausted budget abandons the retry instead of
 * spinning forever. Everything here is Electron-free so the policy is
 * unit-testable; main.ts owns the actual windows and IPC channels.
 */

/** Default cadence: short enough to outlive React's mount, bounded forever. */
const DEFAULT_RETRY_MS = 200
const DEFAULT_MAX_SENDS = 5

/** Structural shape of a pushed payload (kept loose — policy never inspects fields). */
export type QuickEntryStatePayloadLike = null | undefined | Record<string, unknown>

export interface QuickEntryStateRelayOptions {
  /**
   * Equality over gateway truth. Used both to detect supersession (a newer
   * cached payload replaced the one being replayed) and to match acks.
   */
  equals: (a: QuickEntryStatePayloadLike, b: QuickEntryStatePayloadLike) => boolean
  /** The latest payload main has cached; replays of anything older are stale. */
  latest: () => QuickEntryStatePayloadLike
  /** False once the target window is gone. */
  isTargetAlive: () => boolean
  /** Deliver the payload to the window. */
  send: (payload: Record<string, unknown>) => void
  retryMs?: number
  maxSends?: number
}

export interface QuickEntryStateRelay {
  /**
   * Deliver a payload, replacing any in-flight replay it supersedes. Retries
   * continue (bounded) until {@link acknowledge} matches it or the target dies.
   */
  deliver: (payload: Record<string, unknown>) => void
  /** The window echoed an adopted payload; settle a matching in-flight replay. */
  acknowledge: (payload: QuickEntryStatePayloadLike) => void
  /** Stop retrying (target window closed, feature torn down). */
  cancel: () => void
}

export function createQuickEntryStateRelay(options: QuickEntryStateRelayOptions): QuickEntryStateRelay {
  const retryMs = options.retryMs ?? DEFAULT_RETRY_MS
  const maxSends = options.maxSends ?? DEFAULT_MAX_SENDS

  let outstanding: null | Record<string, unknown> = null
  let attempts = 0
  let timer: ReturnType<typeof setTimeout> | null = null

  const clearTimer = () => {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
  }

  const sendNow = () => {
    if (!options.isTargetAlive() || !outstanding) {
      outstanding = null
      clearTimer()

      return
    }

    options.send(outstanding)
    attempts += 1
    clearTimer()

    // Budget spent: settle so a FUTURE delivery request (window reload,
    // re-summon, fresh push) can start a clean cycle instead of inheriting a
    // dead counter.
    if (attempts >= maxSends) {
      outstanding = null

      return
    }

    timer = setTimeout(() => {
      timer = null

      if (!outstanding) {
        return
      }

      // Superseded meanwhile: the newer truth owns the channel now.
      const current = options.latest()

      if (!current || !options.equals(outstanding, current)) {
        outstanding = null

        return
      }

      sendNow()
    }, retryMs)
  }

  return {
    deliver(payload) {
      if (!payload || !options.isTargetAlive()) {
        return
      }

      const current = options.latest()

      // Stale request: something newer is already cached.
      if (!current || !options.equals(payload, current)) {
        return
      }

      // Already being retried for this exact truth — the in-flight loop owns
      // the delivery; a duplicate request must not add sends.
      if (outstanding && options.equals(outstanding, payload)) {
        return
      }

      // A genuinely new cycle (different truth, or the previous one settled):
      // the payload gets its own full retry budget.
      attempts = 0
      outstanding = payload
      sendNow()
    },
    acknowledge(payload) {
      if (outstanding && options.equals(payload, outstanding)) {
        outstanding = null
        clearTimer()
      }
    },
    cancel() {
      outstanding = null
      clearTimer()
    }
  }
}

/** True when two pushed payloads carry the same gateway truth. */
export function sameQuickEntryState(a: QuickEntryStatePayloadLike, b: QuickEntryStatePayloadLike): boolean {
  const leftSessions = Array.isArray((a as { sessions?: unknown[] } | null)?.sessions)
    ? (a as { sessions: Array<{ id?: unknown; title?: unknown }> }).sessions
    : []

  const rightSessions = Array.isArray((b as { sessions?: unknown[] } | null)?.sessions)
    ? (b as { sessions: Array<{ id?: unknown; title?: unknown }> }).sessions
    : []

  return (
    ((a as { connected?: unknown } | null)?.connected ?? false) ===
      ((b as { connected?: unknown } | null)?.connected ?? false) &&
    leftSessions.length === rightSessions.length &&
    leftSessions.every(
      (session, index) => session.id === rightSessions[index]?.id && session.title === rightSessions[index]?.title
    )
  )
}
