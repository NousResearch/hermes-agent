import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $gatewayState } from '@/store/session'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'
import {
  $subagentsBySession,
  reconcileSubagentRoster,
  type SubagentPayload,
  type SubagentProgress
} from '@/store/subagents'

const POLL_MS = 5000

const rejectUnownedRequest = async <T>(): Promise<T> => {
  throw new Error('Subagent owner unavailable')
}

const hasRunning = (items: readonly SubagentProgress[]) => items.some(item => item.status === 'running')

/** Reconcile every displayed running row with its exact owner backend while
 * the all-session spawn tree is mounted. Per-composer polling only covers the
 * focused session, so a missed completion in a background session otherwise
 * leaves a spinner ticking forever. */
export function useSubagentRosterReconciliation() {
  const gatewayState = useStore($gatewayState)

  useEffect(() => {
    let cancelled = false
    const pending = new Set<string>()
    const failures = new Map<string, number>()

    const refresh = async () => {
      const bySession = $subagentsBySession.get()

      await Promise.all(
        Object.entries(bySession).map(async ([sessionId, before]) => {
          if (!hasRunning(before) || pending.has(sessionId) || (failures.get(sessionId) ?? 0) >= 3) {
            return
          }

          pending.add(sessionId)
          const owner = JSON.stringify(knownOwnerForSession(sessionId))

          try {
            const snapshot = await requestForOwnedSession<{ active: SubagentPayload[] }>(
              sessionId,
              rejectUnownedRequest,
              'delegation.status'
            )

            if (
              !cancelled &&
              owner === JSON.stringify(knownOwnerForSession(sessionId)) &&
              before === $subagentsBySession.get()[sessionId] &&
              Array.isArray(snapshot.active)
            ) {
              reconcileSubagentRoster(sessionId, snapshot.active)
            }

            failures.delete(sessionId)
          } catch {
            failures.set(sessionId, (failures.get(sessionId) ?? 0) + 1)
          } finally {
            pending.delete(sessionId)
          }
        })
      )
    }

    const retry = () => {
      failures.clear()
      void refresh()
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    window.addEventListener('focus', retry)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('focus', retry)
    }
  }, [gatewayState])
}
