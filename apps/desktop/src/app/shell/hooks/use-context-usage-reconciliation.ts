import { useEffect } from 'react'

import type { UsageStats } from '@/types/hermes'

interface ContextUsageReconciliationOptions {
  gatewayState: string
  onUsage: (usage: UsageStats) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: null | string
}

interface FocusedUsageTarget {
  primaryFocused: boolean
  updateCachedUsage: (usage: UsageStats) => void
  updatePrimaryUsage: (usage: UsageStats) => void
  usage: UsageStats
}

export function reconcileFocusedContextUsage({
  primaryFocused,
  updateCachedUsage,
  updatePrimaryUsage,
  usage
}: FocusedUsageTarget): void {
  updateCachedUsage(usage)

  if (primaryFocused) {
    updatePrimaryUsage(usage)
  }
}

/** Reconcile the focused session's cached usage whenever its live runtime is available. */
export function useContextUsageReconciliation({
  gatewayState,
  onUsage,
  requestGateway,
  sessionId
}: ContextUsageReconciliationOptions): void {
  useEffect(() => {
    if (gatewayState !== 'open' || !sessionId) {
      return
    }

    let cancelled = false

    void requestGateway<UsageStats>('session.usage', { session_id: sessionId })
      .then(usage => {
        if (!cancelled && usage) {
          onUsage(usage)
        }
      })
      .catch(() => {
        // Keep the last streamed snapshot through a transient reconnect race.
      })

    return () => {
      cancelled = true
    }
  }, [gatewayState, onUsage, requestGateway, sessionId])
}
