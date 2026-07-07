import { type MutableRefObject, useCallback } from 'react'

import type { HermesGateway } from '@/hermes'
import { normalizeProfileKey } from '@/store/profile'
import { $sessions } from '@/store/session'

import type { ClientSessionState } from '../../types'

// A dropped socket swallows the terminal events (message.complete / error) of
// any in-flight turn, leaving its ClientSessionState latched busy — the
// "composer stuck on Thinking… until manual Stop" bug. After a successful
// reconnect we ask the gateway which runtime sessions are GENUINELY still
// occupied (session.active_list reports live status per in-memory session) and
// clear the stale busy/awaitingResponse latches for everything else. Turns the
// backend kept running through the drop stay busy — their next events stream
// in over the fresh socket.

// Live statuses that keep the client-side busy latch: the turn is still
// running ('working'), the agent is still building ('starting'), or the turn
// is blocked on a user prompt ('waiting'). Anything else — including a session
// the gateway no longer knows (grace-reaped or backend restarted) — means the
// turn's completion can never arrive, so busy must be cleared.
const LIVE_BUSY_STATUSES = new Set(['starting', 'waiting', 'working'])

interface LiveSessionRow {
  id?: string
  session_key?: string
  status?: string
}

interface ReconnectReconciliationOptions {
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/** Profile that owns a cached session, resolved through the sidebar list (the
 *  runtime cache doesn't record it). Null when the row isn't loaded — treated
 *  as belonging to the reconciling gateway, since sessions started in this
 *  window always ride the profile socket that created them. */
function cachedSessionProfile(storedSessionId: string | null): null | string {
  if (!storedSessionId) {
    return null
  }

  const row = $sessions.get().find(s => s.id === storedSessionId || s._lineage_root_id === storedSessionId)

  return row ? normalizeProfileKey(row.profile) : null
}

export function useReconnectReconciliation({
  sessionStateByRuntimeIdRef,
  updateSessionState
}: ReconnectReconciliationOptions) {
  return useCallback(
    async (gateway: HermesGateway, profile: string) => {
      const profileKey = normalizeProfileKey(profile)

      const stuck = [...sessionStateByRuntimeIdRef.current.entries()].filter(
        ([, state]) => state.busy || state.awaitingResponse
      )

      if (stuck.length === 0) {
        return
      }

      // If the probe itself fails we keep every latch: better a lingering
      // spinner (the next reconnect retries) than clearing a genuinely
      // running turn on a half-open socket.
      let rows: LiveSessionRow[]

      try {
        const result = await gateway.request<{ sessions?: LiveSessionRow[] }>('session.active_list', {})
        rows = Array.isArray(result?.sessions) ? result.sessions : []
      } catch {
        return
      }

      const statusByRuntimeId = new Map<string, string>()
      const statusByStoredKey = new Map<string, string>()

      for (const row of rows) {
        const status = typeof row.status === 'string' ? row.status : ''

        if (row.id) {
          statusByRuntimeId.set(row.id, status)
        }

        if (row.session_key) {
          statusByStoredKey.set(row.session_key, status)
        }
      }

      for (const [runtimeId, state] of stuck) {
        // Sessions owned by another profile's socket are that socket's problem
        // — its own reconnect runs this reconciliation with its profile key.
        const owner = cachedSessionProfile(state.storedSessionId)

        if (owner !== null && owner !== profileKey) {
          continue
        }

        const status =
          statusByRuntimeId.get(runtimeId) ??
          (state.storedSessionId ? statusByStoredKey.get(state.storedSessionId) : undefined)

        if (status && LIVE_BUSY_STATUSES.has(status)) {
          continue
        }

        updateSessionState(runtimeId, current => ({
          ...current,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          pendingBranchGroup: null,
          streamId: null,
          turnStartedAt: null
        }))
      }
    },
    [sessionStateByRuntimeIdRef, updateSessionState]
  )
}
