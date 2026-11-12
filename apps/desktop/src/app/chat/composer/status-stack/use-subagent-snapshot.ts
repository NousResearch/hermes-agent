import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { $gatewayState } from '@/store/session'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'
import { $subagentsBySession, reconcileSubagentSnapshot, type SubagentPayload } from '@/store/subagents'

export const rejectUnownedSubagentRequest = async <T>(): Promise<T> => {
  throw new Error('Subagent owner unavailable')
}

/** Hydrate even an empty composer; live events remain authoritative over reads.
 *  `poll` keeps the 5s safety-net refresh — off when nothing on screen shows
 *  the answer, the one-shot hydrate still lands. */
export function useSubagentSnapshot(sessionId: string | null, poll = true) {
  const gatewayState = useStore($gatewayState)
  const paneVisible = usePaneVisible()
  useEffect(() => {
    if (!sessionId || !paneVisible) {
      // Keep-alive tiles stay mounted; only poll while this pane is the visible tab (reveal re-runs the effect).
      return
    }

    let cancelled = false
    let pending = false
    let failures = 0

    const refresh = async () => {
      if (cancelled || pending || failures >= 3) {
        return
      }

      pending = true
      const before = $subagentsBySession.get()[sessionId]
      const owner = JSON.stringify(knownOwnerForSession(sessionId))

      try {
        const snapshot = await requestForOwnedSession<{
          delegations?: SubagentPayload[]
          subagents: SubagentPayload[]
        }>(sessionId, rejectUnownedSubagentRequest, 'subagent.list', { session_id: sessionId })

        if (
          !cancelled &&
          owner === JSON.stringify(knownOwnerForSession(sessionId)) &&
          before === $subagentsBySession.get()[sessionId] &&
          Array.isArray(snapshot.subagents)
        ) {
          reconcileSubagentSnapshot(sessionId, snapshot.subagents, snapshot.delegations ?? [])
        }

        failures = 0
      } catch {
        // Older backends retain their event-fed frame; don't hot-loop a missing RPC.
        failures++
      } finally {
        pending = false
      }
    }

    void refresh()

    if (!poll) {
      return () => {
        cancelled = true
      }
    }

    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh()
      }
    }, 5000)

    const retry = () => {
      failures = 0
      void refresh()
    }

    window.addEventListener('focus', retry)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('focus', retry)
    }
  }, [sessionId, gatewayState, paneVisible, poll])
}
