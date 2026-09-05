import { computed } from 'nanostores'

import { stableArray } from '@/lib/stable-array'

import { agentOwnerKey, type AgentRow, type Attention, overviewCache } from './agent-overview'
import { isGatewayOpenForAgent } from './gateway'
import { $sessionDotStateById, type SessionDotState, sessionStatusBucket } from './session-dot-state'
import { $sessionStates, runtimeSessionOwner } from './session-states'

interface ObservedStatus {
  connectionId: string
  profile: string
  id: string
  runtimeId: string
  status: SessionDotState
  socketOpen: boolean
}

const attentionFor = (status: SessionDotState): Attention => {
  const bucket = sessionStatusBucket(status)

  return bucket === 'needs-input' ? 'needs-you' : bucket === 'draft' ? 'idle' : bucket
}

export function overlayObservedStatuses(
  rows: AgentRow[],
  observed: ObservedStatus[],
  previous: readonly AgentRow[] = []
): AgentRow[] {
  const byOwner = new Map(observed.map(item => [agentOwnerKey(item.connectionId, item.profile, item.id), item]))
  const oldRows = new Map(previous.map(row => [row.key, row]))

  const next = rows.map(row => {
    const observation =
      byOwner.get(row.key) ?? byOwner.get(agentOwnerKey(row.connectionId, row.profile, row.resolvedId))

    // The event ledger survives disconnects and cannot establish source freshness.
    if (row.stale) {
      const old = oldRows.get(row.key)

      const lastKnownAttention =
        row.lastKnownAttention ??
        (old?.attention === 'working' || old?.attention === 'needs-you' ? old.attention : old?.lastKnownAttention)

      // Remember only previously presented attention, not untrusted stale events.
      return lastKnownAttention === row.lastKnownAttention ? row : { ...row, lastKnownAttention }
    }

    if (observation?.socketOpen !== true) {
      return row
    }

    const attention = attentionFor(observation.status)

    if (attention === row.attention && observation.runtimeId === row.runtimeId) {
      return row
    }

    return { ...row, attention, runtimeId: observation.runtimeId }
  })

  return next.every((row, index) => row === rows[index]) ? rows : next
}

let stableRows: readonly AgentRow[] = []
// Subscribe to coarse status edges, NOT the token-bearing runtime state atom.
export const $overviewRows = computed([overviewCache.state, $sessionDotStateById], (cache, dots) => {
  const rows = cache.data?.rows ?? []
  const aliasCounts = new Map<string, number>()

  for (const row of rows) {
    for (const id of new Set([row.id, row.resolvedId])) {
      aliasCounts.set(id, (aliasCounts.get(id) ?? 0) + 1)
    }
  }

  const observed: ObservedStatus[] = []

  for (const [runtimeId, state] of Object.entries($sessionStates.get())) {
    // The inbound event ledger proves the socket owner; a navigation hint does not.
    const owner = runtimeSessionOwner(runtimeId) ?? state.ownerRoute

    if (!owner || !state.storedSessionId) {
      continue
    }

    const id = state.storedSessionId

    // A bare-id dot is usable only when its alias is unambiguous. Runtime bits
    // remain useful under collisions because their inbound socket proved owner.
    const status = state.needsInput
      ? 'needs-input'
      : state.busy || state.awaitingResponse
        ? 'working'
        : (aliasCounts.get(id) ?? 0) <= 1
          ? (dots[id] ?? 'idle')
          : 'idle'

    observed.push({
      connectionId: owner.connectionId,
      profile: owner.profile,
      id,
      runtimeId,
      status,
      socketOpen: isGatewayOpenForAgent(owner.connectionId, owner.profile)
    })
  }

  const next = overlayObservedStatuses(rows, observed, stableRows)
  const old = new Map(stableRows.map(row => [row.key, row]))
  stableRows = stableArray(
    stableRows,
    next.map(row => {
      const previous = old.get(row.key)

      return previous && JSON.stringify(previous) === JSON.stringify(row) ? previous : row
    })
  )

  return stableRows
})
