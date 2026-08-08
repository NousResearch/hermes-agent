import { useEffect } from 'react'

import type { ContextBreakdown, UsageStats } from '@/types/hermes'

/**
 * Eagerly seeds `context_max` for the active session so the context-usage
 * statusbar item is visible from the start of a session.
 *
 * Without this the item stays hidden until the user opens the context panel —
 * which can't happen while the item is hidden (the Catch-22 in #78936). The
 * panel still refetches on open, so this is a one-time seed: once `contextMax`
 * is known we stop, and a failed fetch fails open (the item stays hidden
 * rather than showing bad data).
 */
export function useContextUsageSeed({
  activeSessionId,
  contextMax,
  publishContextUsage,
  requestGateway
}: {
  activeSessionId: string | null
  contextMax: number | undefined
  publishContextUsage: (snapshot: Pick<UsageStats, 'context_max' | 'context_percent' | 'context_used'>) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}): void {
  useEffect(() => {
    if (!activeSessionId || contextMax) {
      return
    }

    let cancelled = false

    void requestGateway<ContextBreakdown>('session.context_breakdown', { session_id: activeSessionId })
      .then(data => {
        if (!cancelled) {
          publishContextUsage({
            context_max: data.context_max,
            context_percent: data.context_percent,
            context_used: data.context_used
          })
        }
      })
      .catch(() => {
        // Fail open — leave the item hidden rather than showing bad data.
      })

    return () => {
      cancelled = true
    }
  }, [activeSessionId, contextMax, publishContextUsage, requestGateway])
}
