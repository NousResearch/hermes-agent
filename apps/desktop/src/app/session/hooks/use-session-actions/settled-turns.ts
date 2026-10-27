/**
 * When this renderer last settled a turn of each runtime session (message.complete),
 * so a resume that was in flight meanwhile can tell its snapshot is older than the
 * reply on screen (resume-reconcile.ts). Bounded: a resume only asks about the
 * seconds its own RPC was in flight.
 */

const settledAt = new Map<string, number>()
const MAX_TRACKED_SESSIONS = 64

export function noteTurnSettled(runtimeSessionId: string, at = Date.now()): void {
  settledAt.delete(runtimeSessionId)
  settledAt.set(runtimeSessionId, at)

  if (settledAt.size > MAX_TRACKED_SESSIONS) {
    settledAt.delete(settledAt.keys().next().value as string)
  }
}

/** Whether a turn of this runtime session settled at or after `sinceMs`. */
export function turnSettledSince(runtimeSessionId: string, sinceMs: number): boolean {
  return (settledAt.get(runtimeSessionId) ?? -Infinity) >= sinceMs
}
