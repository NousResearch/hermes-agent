import type { GatewayEvent } from '@hermes/shared'

import { activeGatewayConnectionId } from './gateway'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  lineageAliases,
  setCronSessions,
  setMessagingSessions,
  setSessions
} from './session'
import { recordProfileSessionRemovals } from './session-removal'
import { $sessionStates, runtimeSessionOwner } from './session-states'

/** Remove exact sidebar IDs without refreshing a transcript or touching a live
 * runtime. Profile-owned removal generations fence older in-flight pages. */
export function notifySessionsDeleted(payload: GatewayEvent<'sessions.deleted'>['payload'] | undefined): void {
  if (!payload) {
    return
  }

  const ids = new Set(payload.session_ids)
  const live = new Set<string>()
  const rows = [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()]

  for (const [runtimeId, state] of Object.entries($sessionStates.get())) {
    if (state.busy || state.awaitingResponse || state.turnLive || state.needsInput) {
      const owner = runtimeSessionOwner(runtimeId)
      const profile = typeof owner === 'string' ? owner : owner?.profile

      if (profile && profile !== payload.profile) {
        continue
      }

      if (owner && typeof owner === 'object' && owner.connectionId !== (activeGatewayConnectionId() ?? 'local')) {
        continue
      }

      // Unknown owners remain protected; proven foreign writers do not block
      // an owned twin. Include runtime-only and compression lineage aliases.
      const ownedRows = profile ? rows.filter(row => (row.profile ?? 'default') === profile) : rows

      for (const id of lineageAliases(state.storedSessionId ?? runtimeId, ownedRows)) {
        live.add(id)
      }
    }
  }

  // Fence unseen IDs too, but only their owner: a foreign twin's first page
  // may already be in flight even when that twin is not loaded yet.
  const removed = new Set([...ids].filter(id => !live.has(id)))

  if (!removed.size) {
    return
  }

  recordProfileSessionRemovals([...removed], payload.profile)

  const keep = (rows: typeof $sessions.value) => {
    const next = rows.filter(row => !removed.has(row.id) || (row.profile ?? 'default') !== payload.profile)

    return next.length === rows.length ? rows : next
  }

  setSessions(keep)
  setCronSessions(keep)
  setMessagingSessions(keep)
}
