import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { gatewayRegistrySnapshot } from '@/store/gateway'
import { reconcileActiveSubagents } from '@/store/subagent-liveness'
import { $subagentsBySession } from '@/store/subagents'

export const SUBAGENT_LIVENESS_POLL_MS = 15_000

interface GatewayLike {
  request: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}

interface GatewayEntry {
  gateway: GatewayLike
  profile: string
}

type GatewaySnapshot = () => readonly GatewayEntry[]

function activeIdsFromStatus(value: unknown): null | string[] {
  if (!value || typeof value !== 'object') {
    return null
  }

  const active = (value as { active?: unknown }).active

  if (!Array.isArray(active)) {
    return null
  }

  const activeIds: string[] = []

  for (const item of active) {
    if (!item || typeof item !== 'object') {
      return null
    }

    const id = (item as { subagent_id?: unknown }).subagent_id

    if (typeof id !== 'string' || !id) {
      return null
    }

    activeIds.push(id)
  }

  return activeIds
}

/** Poll every live profile gateway while event-derived rows look active.
 * Missing terminal events are thereby bounded instead of pinning the Desktop
 * Agents indicator forever. Failed or malformed profiles remain untouched,
 * while valid profile snapshots reconcile independently. */
export function useSubagentLiveness(getGateways: GatewaySnapshot = gatewayRegistrySnapshot): void {
  const bySession = useStore($subagentsBySession)

  const hasActive = Object.values(bySession).some(list =>
    list.some(item => item.status === 'queued' || item.status === 'running')
  )

  useEffect(() => {
    if (!hasActive || typeof window === 'undefined') {
      return
    }

    let disposed = false
    let inFlight = false

    const refresh = async () => {
      if (inFlight) {
        return
      }

      const gateways = getGateways()

      if (gateways.length === 0) {
        return
      }

      inFlight = true

      try {
        // Reconcile against the instant this snapshot was requested. Events that
        // arrive while a slow RPC is in flight are newer than the snapshot and
        // must not be pruned by its stale view.
        const requestedAt = Date.now()
        const settled = await Promise.allSettled(gateways.map(entry => entry.gateway.request('delegation.status', {})))

        const snapshots = settled.flatMap((result, index) => {
          if (result.status !== 'fulfilled') {
            return []
          }

          const activeIds = activeIdsFromStatus(result.value)

          return activeIds ? [{ activeIds, profile: gateways[index]!.profile }] : []
        })

        const allAuthoritative = snapshots.length === gateways.length

        if (!disposed && snapshots.length > 0) {
          reconcileActiveSubagents(snapshots, requestedAt, allAuthoritative)
        }
      } finally {
        inFlight = false
      }
    }

    void refresh()
    const interval = window.setInterval(() => void refresh(), SUBAGENT_LIVENESS_POLL_MS)

    return () => {
      disposed = true
      window.clearInterval(interval)
    }
  }, [getGateways, hasActive])
}
