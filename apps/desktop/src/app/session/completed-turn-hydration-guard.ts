import type { SessionMessage } from '@/types/hermes'

export interface CompletedTurnHydrationGuard {
  acknowledgedAt?: number
  finalAssistantRowId: number
}

export type CompletedTurnHydrationGuards = Map<string, CompletedTurnHydrationGuard>

// Keep the acknowledged row around long enough to cover transcript reads that
// were already in flight when the first current snapshot arrived.
export const COMPLETED_TURN_HYDRATION_GUARD_RETENTION_MS = 60_000

export function completedTurnHydrationGuardKey(storedSessionId: string, runtimeSessionId: string): string {
  return `${runtimeSessionId}:${storedSessionId}`
}

export function registerCompletedTurnHydrationGuard(
  guards: CompletedTurnHydrationGuards,
  storedSessionId: string,
  runtimeSessionId: string,
  finalAssistantRowId: number
): void {
  guards.set(completedTurnHydrationGuardKey(storedSessionId, runtimeSessionId), { finalAssistantRowId })
}

/**
 * Gate every persisted-transcript publication for a locally completed turn.
 * Background, status and session-info reads share this state with the direct
 * post-turn fallback, so a successful older page cannot erase the reply.
 */
export function storedTranscriptCanReplaceCompletedTurn(
  messages: readonly SessionMessage[],
  guards: CompletedTurnHydrationGuards,
  storedSessionId: string,
  runtimeSessionId: string,
  requiredFinalAssistantRowId?: number,
  now = Date.now()
): boolean {
  const key = completedTurnHydrationGuardKey(storedSessionId, runtimeSessionId)
  const entry = guards.get(key)
  // A newer completion registered while an older read was in flight wins over
  // that read's call-local row id.
  const finalAssistantRowId = entry?.finalAssistantRowId ?? requiredFinalAssistantRowId

  if (finalAssistantRowId === undefined) {
    return true
  }

  const includesExpectedRow = messages.some(
    message => message.id === finalAssistantRowId || message.row_id === finalAssistantRowId
  )

  if (includesExpectedRow) {
    if (entry) {
      if (entry.acknowledgedAt === undefined) {
        entry.acknowledgedAt = now
      } else if (now - entry.acknowledgedAt >= COMPLETED_TURN_HYDRATION_GUARD_RETENTION_MS) {
        guards.delete(key)
      }
    }

    return true
  }

  if (
    entry?.acknowledgedAt !== undefined &&
    now - entry.acknowledgedAt >= COMPLETED_TURN_HYDRATION_GUARD_RETENTION_MS &&
    requiredFinalAssistantRowId === undefined
  ) {
    guards.delete(key)

    return true
  }

  return false
}
