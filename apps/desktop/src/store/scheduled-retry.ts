import { atom, computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'

// ── Scheduled retry of a failed turn ───────────────────────────────────────
// A rate-limited turn ("The usage limit has been reached") leaves an error
// card whose only way forward is a manual Retry click. This store holds the
// user's promise to retry later ("in 3 hours", "at 12:00") so the app can
// fire the same reload the Retry button fires, without anyone at the keyboard.

export interface ScheduledRetry {
  /** Epoch ms when the failed turn should be re-run. */
  at: number
  /** The failed assistant message to reload. */
  messageId: string
  /** The session the failed message belongs to. */
  sessionId: string
}

/** One scheduled retry per session — a new schedule replaces the previous. */
export const $scheduledRetries = persistentAtom<Record<string, ScheduledRetry>>(
  'hermes.desktop.scheduledRetries',
  {},
  {
    decode: decodeScheduledRetries,
    encode: value => (Object.keys(value).length === 0 ? null : JSON.stringify(value))
  }
)

/** Storage sanitizer: only well-formed retry records survive a load. */
export function decodeScheduledRetries(raw: string): Record<string, ScheduledRetry> {
  const parsed = JSON.parse(raw) as unknown
  const out: Record<string, ScheduledRetry> = {}

  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    for (const [sessionId, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (
        value &&
        typeof value === 'object' &&
        typeof (value as ScheduledRetry).at === 'number' &&
        typeof (value as ScheduledRetry).messageId === 'string' &&
        typeof (value as ScheduledRetry).sessionId === 'string'
      ) {
        out[sessionId] = value as ScheduledRetry
      }
    }
  }

  return out
}

/** The scheduled retry for a session, or null when none is pending. */
export function sessionScheduledRetry(sessionId: null | string | undefined) {
  return computed($scheduledRetries, retries => (sessionId ? (retries[sessionId] ?? null) : null))
}

/** Schedule (or reschedule) a retry for a session. A `null` at cancels. */
export function setScheduledRetry(sessionId: string, retry: ScheduledRetry | null): void {
  const next = { ...$scheduledRetries.get() }

  if (retry === null) {
    delete next[sessionId]
  } else {
    next[sessionId] = retry
  }

  $scheduledRetries.set(next)
}

/** Drop every pending retry whose target message no longer exists. */
export function pruneScheduledRetries(liveMessageIds: ReadonlySet<string>): void {
  const current = $scheduledRetries.get()
  const next: Record<string, ScheduledRetry> = {}
  let changed = false

  for (const retry of Object.values(current)) {
    if (liveMessageIds.has(retry.messageId)) {
      next[retry.sessionId] = retry
    } else {
      changed = true
    }
  }

  if (changed) {
    $scheduledRetries.set(next)
  }
}

/** Monotonic clock for scheduling — injectable so tests don't need to wait. */
export const $now = atom<number>(Date.now())
